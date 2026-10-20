"""Tools and perception modules for the UAV-CodeAgents multi-agent framework.

Provides ReAct tools for aerial visual grounding, aspect-ratio-invariant coordinate
projection, multi-UAV flight simulation, and scenario resolution.
"""

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from smolagents import tool

import config

# ==============================================================================
# Global Swarm Synchronization State
# ==============================================================================

# Global buffer to synchronize concurrent flight tracks across active UAVs
_SWARM_TRAJECTORIES: Dict[int, List[Tuple[int, int]]] = {}
SWARM_FLIGHT_TIMES: Dict[int, float] = {}
# Confirmed detections of the current mission, keyed by drone index
SWARM_DETECTIONS: Dict[int, List[Dict[str, Any]]] = {}
_SWARM_LOCK = threading.Lock()
CURRENT_OUTPUT_DIR: Optional[str] = None


def reset_swarm_state() -> None:
    """Clears trajectories, flight times, and detections before a new mission."""
    with _SWARM_LOCK:
        _SWARM_TRAJECTORIES.clear()
        SWARM_FLIGHT_TIMES.clear()
        SWARM_DETECTIONS.clear()


def record_detections(drone_id: int, detections: List[Dict[str, Any]]) -> None:
    """Stores confirmed detections of one drone for mission reporting.

    Args:
        drone_id: Index of the reporting UAV.
        detections: Dictionaries with 'frame', 'coords' and 'probability'.
    """
    with _SWARM_LOCK:
        SWARM_DETECTIONS.setdefault(drone_id, []).extend(detections)


def get_swarm_detections() -> Dict[int, List[Dict[str, Any]]]:
    """Returns a snapshot of the detections recorded during the current mission."""
    with _SWARM_LOCK:
        return {drone_id: list(items) for drone_id, items in SWARM_DETECTIONS.items()}


# Pixel keypoints and map size of the latest `pixelpoint_objects` call
_LAST_GROUNDING: Dict[str, Any] = {}


def ensure_pixel_keypoints(keypoints: List[Any]) -> List[Any]:
    """Returns keypoints in map pixels, undoing a rescaling to the normalized grid.

    All tools work in pixel coordinates of the map. If the agent converted the
    output of `pixelpoint_objects` to the 0..GROUNDING_GRID_SIZE range, the
    original pixel keypoints are restored and a warning is printed.
    """
    if not _LAST_GROUNDING:
        return keypoints
    restored, changed = undo_normalization(
        keypoints, _LAST_GROUNDING["points"], _LAST_GROUNDING["size"]
    )
    if changed:
        print(
            "⚠️ [Keypoints] Received keypoints rescaled to the normalized grid;"
            " restored the pixel coordinates returned by pixelpoint_objects."
            " Pass keypoints unchanged."
        )
    return restored


def resolve_output_dir(output_dir: Optional[str] = None) -> str:
    """Resolves the artifact directory: active experiment, then argument, then default."""
    return (
        os.environ.get("CURRENT_EXP_DIR") or output_dir or config.DEFAULT_OUTPUT_DIR
    )


def get_headers() -> Dict[str, str]:
    """Generates standard authorization headers for upstream API requests."""
    return {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }


def get_coords(kp: Any) -> Tuple[float, float]:
    """Extracts (x, y) coordinates from point dictionaries, lists, or tuples.

    Args:
        kp: Waypoint representation (dict containing 'point_2d' or 'x'/'y', or coordinate tuple/list).

    Returns:
        Tuple of (x, y) float coordinates.
    """
    if isinstance(kp, dict):
        if "point_2d" in kp:
            return float(kp["point_2d"][0]), float(kp["point_2d"][1])
        return float(kp.get("x", 0)), float(kp.get("y", 0))
    return float(kp[0]), float(kp[1])


# ==============================================================================
# Scenario Resolution
# ==============================================================================
# Maps free-form operator requests and incident names onto `config.SCENARIOS`,
# so every module agrees on scenario naming.

def canonical_incident(name: Optional[str]) -> str:
    """Normalizes an incident name or alias to its canonical scenario key.

    Args:
        name: Incident type, data folder name, or alias (e.g. 'cars_accident').

    Returns:
        Canonical key of `config.SCENARIOS`, or `config.UNKNOWN_INCIDENT`.
    """
    key = re.sub(r"[\s\-]+", "_", str(name or "").strip().lower())
    if key in config.SCENARIOS:
        return key
    for scenario_key, scenario in config.SCENARIOS.items():
        if key == scenario["folder"] or key in scenario.get("aliases", []):
            return scenario_key
    return config.UNKNOWN_INCIDENT


def _keyword_hits(text: str, keyword: str) -> int:
    # Whole words only (with an optional plural), so 'car' does not match 'careful'
    pattern = rf"(?<![a-z]){re.escape(keyword.lower())}(?:e?s)?(?![a-z])"
    return len(re.findall(pattern, text))


def match_scenario(text: str) -> Optional[str]:
    """Finds the scenario whose keywords occur most often in the text.

    Args:
        text: Operator request or mission description.

    Returns:
        Canonical scenario key, or None when no keyword is present. Ties are
        resolved in favour of the scenario declared first in `config.SCENARIOS`.
    """
    text = str(text or "").lower()
    best_key, best_score = None, 0
    for key, scenario in config.SCENARIOS.items():
        required = scenario.get("required_keywords")
        if required and not any(_keyword_hits(text, kw) for kw in required):
            continue
        score = len(
            [kw for kw in scenario.get("keywords", []) if _keyword_hits(text, kw)]
        )
        if score > best_score:
            best_key, best_score = key, score
    return best_key


def detection_threshold(incident_type: Optional[str]) -> float:
    """Probability above which the onboard verifier reports a frame for a scenario."""
    scenario = config.SCENARIOS.get(canonical_incident(incident_type), {})
    return float(scenario.get("detection_threshold", config.DETECTION_CONF_THRESHOLD))


def scenario_info(key: str) -> Dict[str, str]:
    """Returns the fields of a scenario that the mission pipeline works with."""
    scenario = config.SCENARIOS[key]
    return {
        "request": scenario["request"],
        "folder": scenario["folder"],
        "target_object": scenario["target_object"],
        "incident_type": key,
    }


def resolve_scenario(user_input: str) -> Dict[str, str]:
    """Resolves console input (scenario key or free text) to a scenario description.

    Args:
        user_input: Scenario key, alias, or a natural language request.

    Returns:
        Dictionary with 'request', 'folder', 'target_object' and 'incident_type'.
        A multi-word input replaces the default request text of the scenario.
    """
    raw_text = str(user_input or "").strip()

    # 1. Default scenario if input is empty (user pressed Enter)
    if not raw_text:
        return scenario_info(config.DEFAULT_SCENARIO)

    # 2. Direct match with a scenario key or alias (e.g. 'fire' or 'cars_accident')
    direct_key = canonical_incident(raw_text)
    if direct_key != config.UNKNOWN_INCIDENT:
        return scenario_info(direct_key)

    # 3. Keywords inside the entered sentence, then fallback to the default scenario
    key = match_scenario(raw_text)
    if key is None:
        print(
            f"⚠️ Keywords not recognized in '{user_input}'. Defaulting to"
            f" '{config.DEFAULT_SCENARIO}' scenario."
        )
        key = config.DEFAULT_SCENARIO

    scenario = scenario_info(key)
    if len(raw_text.split()) > 1:
        scenario["request"] = raw_text
    return scenario


# ==============================================================================
# Keypoint Grounding Filters
# ==============================================================================
# A vision-language model occasionally answers without looking at the image and
# returns fabricated coordinates (typically a regular row or column of points).
# These helpers reject such answers and choose, among several independent
# answers, the one that the others confirm best.

def normalize_points(raw_points: Any, default_label: str) -> List[Dict[str, Any]]:
    """Keeps well-formed points that lie on the grounding grid.

    Args:
        raw_points: Parsed model output (list of dicts or coordinate pairs).
        default_label: Label for points that come without one.

    Returns:
        List of {'point_2d': [x, y], 'label': str, 'kind': 'object' | 'area'} in
        grid coordinates. 'area' marks a point of a spread over a large region.
    """
    grid = config.GROUNDING_GRID_SIZE
    points = []
    for item in raw_points if isinstance(raw_points, list) else []:
        if isinstance(item, dict) and "point_2d" not in item and not {"x", "y"} <= set(item):
            continue
        try:
            x, y = get_coords(item)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if not (0 <= x <= grid and 0 <= y <= grid):
            continue
        label = item.get("label", default_label) if isinstance(item, dict) else default_label
        is_area = isinstance(item, dict) and str(item.get("kind", "")).lower() == "area"
        points.append(
            {"point_2d": [x, y], "label": str(label), "kind": "area" if is_area else "object"}
        )
    return points


def looks_synthetic(points: Sequence[Dict[str, Any]]) -> bool:
    """Detects fabricated answers: an axis-aligned line of evenly spaced points.

    Real objects on aerial imagery are never both exactly aligned with an image
    axis and equally spaced, while a model that ignores the image produces
    exactly that pattern. Points of kind 'area' are not checked.

    Args:
        points: Points in grid coordinates.

    Returns:
        True when at least `config.GROUNDING_SYNTHETIC_MIN_POINTS` points share
        one coordinate and are evenly spaced along the other.
    """
    # An even spread over a large area is requested from the model and is legitimate
    coords = [get_coords(p) for p in points if p.get("kind") != "area"]
    tolerance = config.GROUNDING_SYNTHETIC_AXIS_TOLERANCE
    min_points = config.GROUNDING_SYNTHETIC_MIN_POINTS

    for axis in (0, 1):
        for anchor in coords:
            line = sorted(
                c[1 - axis] for c in coords if abs(c[axis] - anchor[axis]) <= tolerance
            )
            if len(line) < min_points:
                continue
            gaps = [b - a for a, b in zip(line, line[1:])]
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap <= 0:
                return True  # Many copies of the same point
            # The median deviation ignores a single irregular gap
            deviations = sorted(abs(g - mean_gap) for g in gaps)
            if deviations[len(deviations) // 2] <= config.GROUNDING_SYNTHETIC_GAP_VARIATION * mean_gap:
                return True
    return False


def select_best_answer(
    runs: Sequence[Sequence[Dict[str, Any]]],
    radius: Optional[float] = None,
    max_points: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Picks the grounding answer that the other answers confirm best.

    Whole answers are compared instead of intersecting their points. Compact
    objects (buildings) are found at the same spots by every honest answer, so a
    fabricated answer gets no support and loses. Area targets (water, forest)
    are marked at different spots by every answer; intersecting them would
    discard valid points, while here one complete answer is still returned.

    Args:
        runs: One list of points per independent model answer (grid coordinates).
        radius: Distance within which points of two answers are the same object.
            Defaults to `config.GROUNDING_MATCH_RADIUS`.
        max_points: Maximum number of points returned. Defaults to
            `config.MAX_KEYPOINTS`.

    Returns:
        Tuple of (points of the chosen answer, confirmed points first; number of
        its points that another answer reproduces). Empty when no answer has
        points or at least half of the answers found nothing.
    """
    radius = config.GROUNDING_MATCH_RADIUS if radius is None else radius
    max_points = config.MAX_KEYPOINTS if max_points is None else max_points

    non_empty = [run for run in runs if run]
    if not non_empty or 2 * len(non_empty) <= len(runs) and len(runs) > 1:
        return [], 0

    def confirmed(run_idx: int) -> List[bool]:
        others = [get_coords(p) for k, run in enumerate(runs) if k != run_idx for p in run]
        flags = []
        for point in runs[run_idx]:
            x, y = get_coords(point)
            flags.append(any(math.hypot(x - ox, y - oy) <= radius for ox, oy in others))
        return flags

    best_idx, best_flags = None, []
    for run_idx, run in enumerate(runs):
        if not run:
            continue
        flags = confirmed(run_idx)
        key = (sum(flags), len(run))
        if best_idx is None or key > (sum(best_flags), len(runs[best_idx])):
            best_idx, best_flags = run_idx, flags

    points = [dict(p) for p in runs[best_idx]]
    ordered = [p for p, ok in zip(points, best_flags) if ok] + [
        p for p, ok in zip(points, best_flags) if not ok
    ]
    return ordered[:max_points], sum(best_flags)


def undo_normalization(
    keypoints: Sequence[Any],
    reference_points: Sequence[Dict[str, Any]],
    image_size: Tuple[int, int],
    grid: Optional[int] = None,
    tolerance: float = 2.0,
) -> Tuple[List[Any], bool]:
    """Restores pixel keypoints that were rescaled to the normalized grid.

    Grounding returns pixel coordinates of the map. An agent may convert them to
    the 0..grid range before passing them on, which shifts every waypoint. When
    the passed keypoints equal the grounding output scaled by grid / image size,
    the original pixel points are returned instead.

    Args:
        keypoints: Keypoints received by a tool.
        reference_points: Pixel keypoints last returned by the grounding tool.
        image_size: (width, height) of the map the reference points belong to.
        grid: Side of the normalized grid. Defaults to `config.GROUNDING_GRID_SIZE`.
        tolerance: Allowed rounding difference in grid units.

    Returns:
        Tuple of (keypoints to use, whether a rescaling was undone).
    """
    grid = config.GROUNDING_GRID_SIZE if grid is None else grid
    width, height = image_size
    if not keypoints or not reference_points or (width == grid and height == grid):
        return list(keypoints), False

    references = [get_coords(r) for r in reference_points]

    def find(x: float, y: float, scale_x: float, scale_y: float) -> Optional[int]:
        for idx, (rx, ry) in enumerate(references):
            if abs(rx * scale_x - x) <= tolerance and abs(ry * scale_y - y) <= tolerance:
                return idx
        return None

    try:
        coords = [get_coords(k) for k in keypoints]
    except (TypeError, ValueError, IndexError, KeyError):
        return list(keypoints), False

    as_pixels = [find(x, y, 1.0, 1.0) for x, y in coords]
    as_grid = [find(x, y, grid / width, grid / height) for x, y in coords]
    pixel_hits = sum(m is not None for m in as_pixels)
    grid_hits = sum(m is not None for m in as_grid)

    # Rescaled input: most points match the scaled reference and not the original
    if grid_hits < 0.8 * len(coords) or grid_hits <= pixel_hits:
        return list(keypoints), False

    restored = []
    for keypoint, match in zip(keypoints, as_grid):
        if match is None:
            restored.append(keypoint)
            continue
        point = dict(reference_points[match])
        if isinstance(keypoint, dict) and "label" in keypoint:
            point["label"] = keypoint["label"]
        restored.append(point)
    return restored, True


# ==============================================================================
# Aspect-Ratio Letterbox & Coordinate Inversion Helpers
# ==============================================================================

def _prepare_letterbox(
    img: Image.Image, target_size: int = config.GROUNDING_GRID_SIZE
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Fits an image onto a standardized square canvas while preserving native aspect ratio.

    Args:
        img: Input PIL Image with arbitrary dimensions.
        target_size: Target square canvas size (default: config.GROUNDING_GRID_SIZE).

    Returns:
        Tuple of (padded square PIL Image, metadata dictionary with scaling parameters).
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    square_img = Image.new("RGB", (target_size, target_size), (128, 128, 128))

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    square_img.paste(resized_img, (pad_x, pad_y))

    meta = {
        "orig_w": w,
        "orig_h": h,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return square_img, meta


def _restore_points(
    raw_points: List[Any], meta: Dict[str, Any], default_label: str
) -> List[Dict[str, Any]]:
    """Inverts normalized 1000x1000 coordinates back to the native high-resolution image space.

    Args:
        raw_points: List of candidate waypoint detections in normalized letterbox coordinates.
        meta: Geometric scaling and padding metadata returned by _prepare_letterbox.
        default_label: Fallback semantic category label for detected targets.

    Returns:
        List of dictionaries with restored 2D pixel coordinates and semantic labels.
    """
    restored = []
    scale = meta["scale"]
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    for kp in raw_points:
        kx, ky = get_coords(kp)
        label = (
            kp.get("label", default_label)
            if isinstance(kp, dict)
            else default_label
        )

        # 1. Subtract canvas padding offsets
        img_x = kx - pad_x
        img_y = ky - pad_y

        # 2. Invert scaling factor
        real_x = int(round(img_x / scale))
        real_y = int(round(img_y / scale))

        # 3. Clamp strictly within valid native image boundaries
        real_x = max(0, min(orig_w - 1, real_x))
        real_y = max(0, min(orig_h - 1, real_y))

        restored.append({"point_2d": [real_x, real_y], "label": label})

    return restored


def _safe_bbox(
    cx: int, cy: int, half_size: int, max_w: int, max_h: int
) -> Tuple[int, int, int, int]:
    """Safely calculates bounding box coordinates, preventing axis inversion and canvas overflow.

    Args:
        cx: Center x-coordinate.
        cy: Center y-coordinate.
        half_size: Half of the bounding box edge length.
        max_w: Maximum image width.
        max_h: Maximum image height.

    Returns:
        Tuple of (x1, y1, x2, y2) validated integer bounding box coordinates.
    """
    # 1. Clamp center coordinate within canvas limits
    clamped_cx = max(0, min(max_w, cx))
    clamped_cy = max(0, min(max_h, cy))

    # 2. Compute initial extents
    x1 = max(0, clamped_cx - half_size)
    x2 = min(max_w, clamped_cx + half_size)
    y1 = max(0, clamped_cy - half_size)
    y2 = min(max_h, clamped_cy + half_size)

    # 3. Enforce strictly positive dimensions (x2 > x1, y2 > y1)
    if x2 <= x1:
        if x1 >= max_w:
            x1 = max(0, max_w - 1)
            x2 = max_w
        else:
            x2 = min(max_w, x1 + 1)

    if y2 <= y1:
        if y1 >= max_h:
            y1 = max(0, max_h - 1)
            y2 = max_h
        else:
            y2 = min(max_h, y1 + 1)

    return int(x1), int(y1), int(x2), int(y2)


# ==============================================================================
# Agent Tools for Smolagents Runtime
# ==============================================================================

@tool
def describe_satellite_image(image: Image.Image) -> str:
    """Analyzes a high-resolution aerial or satellite image using a multimodal foundation model.

    Args:
        image: A PIL.Image.Image object of the aerial scene to inspect.

    Returns:
        str: Comprehensive textual description of the scene topology, hazards, and objects.
    """
    api_key = config.LLM_API_KEY
    if not api_key:
        return "Error: No API key provided in configuration."

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=config.IMAGE_JPEG_QUALITY)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": """
        Describe this image in very detailed manner.
        Name each object you observe and its approximate location.
        Example: "I see 6 buildings in the upper part of the image. A road crosses from left to right."
        """,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        },
    ]

    payload = {
        "model": config.MODEL_ID,
        "max_tokens": config.VLM_DESCRIBE_MAX_TOKENS,
        "temperature": config.VLM_TEMPERATURE,
        "top_p": 1,
        "top_k": 40,
        "messages": [{"role": "user", "content": content}],
    }

    response = requests.post(
        config.LLM_CHAT_COMPLETIONS_URL, headers=get_headers(), json=payload
    )

    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}"

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"JSON Parse Error: {e}. Raw response: {response.text[:500]}"


@tool
def pixelpoint_objects(
    image: Optional[Image.Image] = None,
    objects: Optional[str] = None,
    images: Optional[Image.Image] = None,
    target_object: Optional[str] = None,
) -> List[dict]:
    """Detects target objects on satellite imagery and extracts metric rooftop centroids.

    Args:
        image: Primary PIL Image object of the operational terrain.
        objects: String description of semantic target categories to locate.
        images: Alternative parameter for image input.
        target_object: Alternative parameter for target description.

    Returns:
        List[dict]: Detected targets with labels and 2D pixel coordinates of the input image
            (not normalized). Pass them to other tools unchanged.
    """
    target_str = objects or target_object or "target object"
    if isinstance(target_str, list):
        target_str = ", ".join(map(str, target_str))

    img = image if image is not None else images
    if isinstance(img, (list, tuple)):
        img = img[0]

    if img is None:
        raise ValueError("No valid image provided to pixelpoint_objects.")

    if img.mode != "RGB":
        img = img.convert("RGB")

    # 1. Project input onto standardized square canvas with aspect ratio preservation
    grid = config.GROUNDING_GRID_SIZE
    square_img, meta = _prepare_letterbox(img, target_size=grid)

    buffered = io.BytesIO()
    square_img.save(buffered, format="JPEG", quality=config.IMAGE_JPEG_QUALITY)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 2. Structured prompt: one centred point per compact object, an even spread
    #    of points over large areas such as water or forest
    prompt = f"""
    This is the input satellite image. You need to detect keypoints of {target_str} for drone navigation.
    A reconnaissance drone will fly to every keypoint and inspect the ground around it.

    COORDINATE SPECIFICATIONS:
    - The image canvas is defined on a normalized {grid}x{grid} integer grid.
    - Top-left corner is [0, 0]. Bottom-right corner is [{grid}, {grid}].
    - Both X and Y must be integers strictly within range [0, {grid}].
    - Do not output points on empty background or padding areas.

    There are two kinds of targets. Decide the kind of every target yourself.

    KIND "object" - compact objects such as buildings, vehicles, tanks, junctions:
    - Output exactly one point per object, at the geometric center of the object.
    - Focus strictly on the illuminated physical roof surfaces.
    - Ignore cast shadows entirely: do not place coordinates between the building and its shadow.
    - Center rule: Coordinates must be placed strictly on the centroid of the roof perimeter, not on edges or driveways.

    KIND "area" - large continuous regions such as open water, sea, lakes, rivers, forest, fields, open terrain:
    - A single point in the middle is NOT enough: the drone must survey the whole region.
    - Spread points evenly over the ENTIRE visible extent of the region, about {config.GROUNDING_AREA_SPACING} grid units apart, in every direction.
    - Cover the far parts of the region too, including the parts close to its shores or edges.
    - Keep every point inside the region and at least {config.GROUNDING_BORDER_MARGIN} grid units away from the image border.
    - Use the whole point budget when the region is large.

    Return maximum {config.MAX_KEYPOINTS} points in total.
    Do not confuse with surrounding noise or background textures.
    For EVERY point, identify its SPECIFIC visual type from the {target_str} category.
    Provide only the answer in json. Return them STRICTLY in the form:
    ```json
    [
      {{"point_2d": [x1, y1], "label": "Here is the specific label of the detected target", "kind": "object"}},
      {{"point_2d": [x2, y2], "label": "Here is the specific label of the detected target", "kind": "area"}}
    ]
    ```
    """

    headers = get_headers()

    data_payload = {
        "model": config.MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "You are an autonomous aerial reconnaissance target extraction assistant.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        },
                    },
                ],
            },
        ],
        "temperature": config.GROUNDING_TEMPERATURE,
    }

    def request_points() -> Optional[List[Dict[str, Any]]]:
        """One grounding request; None when the API did not return an answer."""
        for attempt in range(config.GROUNDING_MAX_RETRIES):
            try:
                response = requests.post(
                    config.LLM_CHAT_COMPLETIONS_URL,
                    headers=headers,
                    json=data_payload,
                    timeout=config.GROUNDING_TIMEOUT_S,
                )

                if response.status_code == 200:
                    raw_content = response.json()["choices"][0]["message"]["content"]
                    json_str = re.sub(r"```json\s*|\s*```", "", raw_content).strip()

                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        objects_found = re.findall(
                            r'\{[^{}]*"point_2d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*\][^{}]*\}',
                            json_str,
                        )
                        parsed = (
                            json.loads("[\n" + ",\n".join(objects_found) + "\n]")
                            if objects_found
                            else []
                        )
                    return normalize_points(parsed, target_str)

                if response.status_code in (400, 401, 403):
                    print(
                        f"❌ Critical API error ({response.status_code}): {response.text}"
                    )
                    return None

                print(
                    f"⚠️ Attempt {attempt + 1} returned status code {response.status_code}. Retrying..."
                )
                time.sleep(config.GROUNDING_RETRY_DELAY_S)

            except Exception as e:
                print(f"DEBUG: Attempt {attempt + 1} failed: {e}")
                time.sleep(config.GROUNDING_RETRY_DELAY_S)
        return None

    # 3. Ask several times in parallel and use the answer the others confirm best.
    #    A model that answers without looking at the image returns fabricated
    #    coordinates; such answers are dropped or outvoted here.
    votes = max(1, config.GROUNDING_VOTES)
    grid_points: List[Dict[str, Any]] = []

    def usable_answers(count: int) -> List[List[Dict[str, Any]]]:
        with ThreadPoolExecutor(max_workers=count) as executor:
            answers = list(executor.map(lambda _: request_points(), range(count)))
        usable = []
        for answer in answers:
            if answer is None:
                continue
            if looks_synthetic(answer):
                print("⚠️ [Grounding] Discarded an answer with fabricated, evenly spaced points.")
                continue
            usable.append(answer)
        return usable

    for round_idx in range(max(1, config.GROUNDING_MAX_ROUNDS)):
        runs = usable_answers(votes)
        if 0 < len(runs) < votes:
            # Replace discarded answers once, so the vote keeps its full size
            runs += usable_answers(votes - len(runs))

        if not runs:
            print(f"⚠️ [Grounding] Round {round_idx + 1}: no usable answer. Retrying...")
            continue

        grid_points, confirmed = select_best_answer(runs)
        print(
            f"📍 [Grounding] {len(grid_points)} keypoints from the best of {len(runs)}"
            f" answers ({confirmed} reproduced by another answer)."
        )
        break
    else:
        print("❌ [Grounding] No usable answer; returning an empty list.")

    # 4. Invert transformation back to native metric pixel coordinates
    pixel_points = _restore_points(grid_points, meta, target_str)
    _LAST_GROUNDING.clear()
    _LAST_GROUNDING.update(size=img.size, points=[dict(p) for p in pixel_points])
    return pixel_points


@tool
def visualize_keypoints_from_image(
    image: Image.Image,
    keypoints: list,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
) -> str:
    """Visualizes candidate inspection targets with clean frameless text on the base terrain image.

    Args:
        image: Base PIL image representing the operational map.
        keypoints: Target keypoints in pixel coordinates of the image, exactly as returned by
            `pixelpoint_objects`.
        output_dir: Target directory to save the visualization plot. Defaults to 'uav_logs'.

    Returns:
        str: Confirmation message detailing the destination file path.
    """
    target_dir = resolve_output_dir(output_dir)
    os.makedirs(target_dir, exist_ok=True)
    keypoints = ensure_pixel_keypoints(keypoints)

    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # Pick the first available font
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, 11)
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Alternate label offsets so that nearby labels do not overlap
    offsets = [
        (6, -14),
        (6, 4),
        (-6, -14),
        (-6, 4),
    ]

    for idx, kp in enumerate(keypoints):
        x, y = get_coords(kp)
        x, y = int(round(x)), int(round(y))

        # Clean up the object label
        if isinstance(kp, dict):
            raw_label = str(kp.get("label", "Target")).strip()
            clean_word = re.split(r"[,;/\n]", raw_label)[0].strip()
            clean_word = clean_word.replace("_", " ").title()
        else:
            clean_word = "Target"

        label_text = f"#{idx + 1} {clean_word}"

        # 1. Minimal point marker (coral centre with a white outline)
        r = 3
        draw.ellipse(
            [(x - r, y - r), (x + r, y + r)],
            fill="#FF5252",
            outline="#FFFFFF",
            width=1,
        )

        # 2. Compute the label position
        dx, dy = offsets[idx % len(offsets)]
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]

        # Labels shifted to the left are right-aligned against the point
        text_x = x - text_w + dx if dx < 0 else x + dx
        text_y = y + dy

        # 3. Frameless text with a soft contrasting stroke
        draw.text(
            (text_x, text_y),
            label_text,
            fill="#FDE047",           # Warm amber, readable on vegetation and soil
            stroke_width=1,
            stroke_fill="#0F172A",     # Dark navy halo for readability
            font=font,
        )

    output_path = os.path.join(target_dir, config.TRAJECTORY_FILENAME)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title(f"Target Waypoints ({len(keypoints)} inspection sites)", fontsize=13, pad=10)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=config.TRAJECTORY_DPI)
    plt.close(fig)

    return f"Saved clean trajectory map with {len(keypoints)} labeled targets to {output_path}."


@tool
def uav_simulation(
    image: Image.Image,
    labeled_points: Optional[List[Any]] = None,
    waypoints: Optional[List[Any]] = None,
    drone_id: int = 0,
    total_drones: int = config.COUNTER_DRONES,
    speed_px: float = config.UAV_SPEED_PX,
    crop_size: int = config.CAMERA_CROP_SIZE,
    home_point: Tuple[int, int] = config.DEFAULT_HOME_POINT,
    output_dir: str = config.DEFAULT_OUTPUT_DIR,
) -> Dict[str, Tuple[Image.Image, Tuple[int, int]]]:
    """Simulates continuous UAV flight trajectories and renders a combined swarm MP4 video.

    Args:
        image: Operational target terrain image (PIL Image) to patrol over.
        labeled_points: List of target waypoints or keypoints to visit.
        waypoints: Alias for labeled_points. List of target coordinates to visit.
        drone_id: Unique integer index of this UAV.
        total_drones: Total number of active drones in the swarm.
        speed_px: Flight speed in pixels per step.
        crop_size: Pixel dimension (FOV) of onboard camera.
        home_point: Base station coordinates (x, y) where swarm starts and lands.
        output_dir: Target folder to save flight videos and visual artifacts.

    Returns:
        Dictionary mapping frame identifiers to tuples of (cropped PIL Image, (x, y) coordinates).
    """
    global _SWARM_TRAJECTORIES, SWARM_FLIGHT_TIMES, CURRENT_OUTPUT_DIR

    # Priority: Environment variable -> parameter override -> fallback default
    target_dir = resolve_output_dir(output_dir)
    CURRENT_OUTPUT_DIR = target_dir
    os.makedirs(target_dir, exist_ok=True)

    if drone_id == 0:
        with _SWARM_LOCK:
            _SWARM_TRAJECTORIES.clear()
            SWARM_FLIGHT_TIMES.clear()

    points = labeled_points if labeled_points is not None else waypoints
    if not points:
        raise ValueError(
            "No waypoints provided to uav_simulation (expected 'labeled_points' or 'waypoints')."
        )

    img_w, img_h = image.size
    half_crop = max(1, crop_size // 2)

    # Base station boundary validation
    home_x = max(0, min(img_w - 1, int(home_point[0])))
    home_y = max(0, min(img_h - 1, int(home_point[1])))
    home_point_clean = (home_x, home_y)

    # Parse and sanitize waypoint coordinates
    clean_points = []
    for p in points:
        pt = None
        if isinstance(p, dict):
            if "point_2d" in p and len(p["point_2d"]) >= 2:
                pt = (p["point_2d"][0], p["point_2d"][1])
            elif "x" in p and "y" in p:
                pt = (p["x"], p["y"])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            pt = (p[0], p[1])

        if pt is not None:
            px, py = float(pt[0]), float(pt[1])
            # Auto-scale normalized relative coordinates [0.0, 1.0]
            if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and img_w > 1 and img_h > 1:
                px *= img_w
                py *= img_h
            # Clamp strictly within map bounds
            px = max(0, min(img_w - 1, int(round(px))))
            py = max(0, min(img_h - 1, int(round(py))))
            clean_points.append((px, py))

    if not clean_points:
        raise ValueError("Valid target coordinates could not be parsed from input.")

    drone_colors = config.DRONE_COLORS

    # 1. Construct continuous trajectory waypoints
    full_path = [home_point_clean] + clean_points + [home_point_clean]
    continuous_coords = [full_path[0]]

    for i in range(len(full_path) - 1):
        p_start = full_path[i]
        p_end = full_path[i + 1]
        dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
        dist = (dx**2 + dy**2) ** 0.5
        steps = max(1, int(round(dist / speed_px)))
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            cx = int(round(p_start[0] + alpha * dx))
            cy = int(round(p_start[1] + alpha * dy))
            continuous_coords.append((cx, cy))

    total_dist_px = sum(
        np.hypot(
            continuous_coords[k][0] - continuous_coords[k - 1][0],
            continuous_coords[k][1] - continuous_coords[k - 1][1],
        )
        for k in range(1, len(continuous_coords))
    )
    flight_sec = (
        total_dist_px * config.GROUND_SAMPLING_DISTANCE_M
    ) / config.CRUISE_SPEED_M_PER_S

    with _SWARM_LOCK:
        SWARM_FLIGHT_TIMES[drone_id] = flight_sec
        _SWARM_TRAJECTORIES[drone_id] = continuous_coords

    print(
        f"[UAV #{drone_id}] Path planned: {len(clean_points)} targets + RTL. "
        f"Total frames: {len(continuous_coords)} | Flight duration: {flight_sec:.1f}s"
    )

    # 2. Safe camera field-of-view (FOV) frame extraction
    def extract_frame(item):
        idx, (cx, cy) = item
        x1, y1, x2, y2 = _safe_bbox(cx, cy, half_crop, img_w, img_h)
        cam_crop = image.crop((x1, y1, x2, y2))
        frame_key = f"drone{drone_id}_frame_{idx:04d}"
        return frame_key, (cam_crop, (cx, cy))

    with ThreadPoolExecutor(max_workers=config.SIMULATION_WORKERS) as executor:
        frames_dict = dict(executor.map(extract_frame, enumerate(continuous_coords)))

    # 3. Synchronized swarm flight video rendering
    should_render_video = False
    with _SWARM_LOCK:
        if len(_SWARM_TRAJECTORIES) >= total_drones:
            should_render_video = True
            trajectories_snapshot = dict(_SWARM_TRAJECTORIES)

    if should_render_video:
        print(f"[SWARM] All {total_drones} UAV trajectories ready. Generating MP4 flight video...")
        max_steps = max(len(track) for track in trajectories_snapshot.values())
        time_steps = list(range(0, max_steps, 1))

        def render_video_frame(t):
            swarm_vis = image.copy()
            d_all = ImageDraw.Draw(swarm_vis)

            # Draw Ground Control Station
            hx, hy = home_point_clean
            d_all.rectangle([hx - 7, hy - 7, hx + 7, hy + 7], fill="#FFFF00", outline="#000000")
            d_all.text((hx + 10, hy - 6), "BASE STATION", fill="#FFFF00")

            # Draw UAV positions, FOV projections, and flight ribbons
            for d_idx, track in trajectories_snapshot.items():
                c_color = drone_colors[d_idx % len(drone_colors)]
                cur_step = min(t, len(track) - 1)
                cur_pos = track[cur_step]
                track_history = track[: cur_step + 1]

                if len(track_history) > 1:
                    d_all.line(track_history, fill=c_color, width=2)

                cx_i, cy_i = cur_pos
                x1_i, y1_i, x2_i, y2_i = _safe_bbox(cx_i, cy_i, half_crop, img_w, img_h)

                d_all.ellipse([cx_i - 5, cy_i - 5, cx_i + 5, cy_i + 5], fill=c_color, outline="#FFFFFF", width=2)
                d_all.rectangle([x1_i, y1_i, x2_i, y2_i], outline=c_color, width=2)
                d_all.text((cx_i + 7, cy_i - 7), f"UAV-{d_idx}", fill=c_color)

            return cv2.cvtColor(np.array(swarm_vis), cv2.COLOR_RGB2BGR)

        video_path = os.path.join(target_dir, config.VIDEO_FILENAME)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        video_writer = cv2.VideoWriter(
            video_path, fourcc, config.VIDEO_FPS, (img_w, img_h)
        )

        if video_writer.isOpened():
            # Render and write chunk by chunk so only a few frames stay in memory
            chunk = max(1, config.VIDEO_CHUNK_FRAMES)
            with ThreadPoolExecutor(
                max_workers=config.SIMULATION_WORKERS
            ) as video_executor:
                for start in range(0, len(time_steps), chunk):
                    for frame in video_executor.map(
                        render_video_frame, time_steps[start : start + chunk]
                    ):
                        video_writer.write(frame)
            video_writer.release()
            print(f"[SWARM SUCCESS] Saved MP4 flight video to: {video_path}")
        else:
            print(f"[ERROR] Could not initialize OpenCV VideoWriter for path: {video_path}")

        with _SWARM_LOCK:
            _SWARM_TRAJECTORIES.clear()

    return frames_dict

@tool
def get_folder_by_query(query: str) -> dict:
    """Analyzes an operational request to resolve incident type, data folder, and target label.

    Args:
        query: Operator prompt or mission specification string.

    Returns:
        dict: Mapping with 'folder', 'incident_type' and 'target_object'. Unrecognised
            requests return the baseline folder and incident type 'unknown'.
    """
    unknown = {
        "folder": config.BASELINE_FOLDER,
        "incident_type": config.UNKNOWN_INCIDENT,
        "target_object": config.UNKNOWN_TARGET_OBJECT,
    }

    def describe(key: str) -> dict:
        info = scenario_info(key)
        return {
            "folder": info["folder"],
            "incident_type": info["incident_type"],
            "target_object": info["target_object"],
        }

    # 1. Handle dictionary-formatted string inputs
    if isinstance(query, str) and query.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(query)
            if isinstance(parsed, dict) and "folder" in parsed:
                key = canonical_incident(
                    parsed.get("incident_type") or parsed.get("folder")
                )
                if key != config.UNKNOWN_INCIDENT:
                    return describe(key)
                return {
                    "folder": parsed.get("folder", unknown["folder"]),
                    "incident_type": config.UNKNOWN_INCIDENT,
                    "target_object": parsed.get(
                        "target_object", unknown["target_object"]
                    ),
                }
        except Exception:
            pass

    # 2. Scenario key / alias, then whole-word keyword classification
    key = canonical_incident(query)
    if key == config.UNKNOWN_INCIDENT:
        key = match_scenario(str(query))

    return describe(key) if key else unknown


@tool
def extract_inspection_targets(user_request: str) -> list:
    """Extracts semantic inspection categories from unstructured human instructions.

    Args:
        user_request: High-level natural language prompt from the human operator.

    Returns:
        list: Structured list of semantic target categories to localize on imagery.
    """
    system_prompt = """
    You must understand the problem and determine exactly which specific objects are needed to locate the issue using a reconnaissance drone.
    Adapt the selection of objects based on the context:
    - House or building fire -> ["residential building", "warehouse", "storage facility"]
    - Forest fire or logging -> ["forest", "large trees", "cluster of trees", "dense thicket"]
    - Oil spill -> ["open water", "body of water", "pipeline junction", "service vehicle"]
    - Illegal construction -> ["open terrain", "potential construction site", "cleared plot"]

    Return a strictly valid JSON array of strings in English.
    Output format example: ["building", "smoke source", "parking lot"]
    """

    headers = get_headers()

    data = {
        "model": config.MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'User Request: "{user_request}"'},
        ],
        "temperature": config.VLM_TEMPERATURE,
    }

    try:
        response = requests.post(
            config.LLM_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=data,
            timeout=config.TARGET_EXTRACTION_TIMEOUT_S,
        )
        response.raise_for_status()

        result = response.json()
        raw_text = result["choices"][0]["message"]["content"]
        json_str = re.sub(r"```json\s*|\s*```", "", raw_text).strip()

        targets = json.loads(json_str)
        if isinstance(targets, list):
            return targets
        return [str(targets)]

    except Exception as e:
        print(f"Error during semantic target extraction: {e}")
        return ["target object", "surrounding structure", "access route"]