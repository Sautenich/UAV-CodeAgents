"""Tools and perception modules for the UAV-CodeAgents multi-agent framework.

Provides ReAct tools for aerial visual grounding, aspect-ratio-invariant coordinate
projection, multi-UAV flight simulation, edge-assisted target verification,
and multimodal incident classification.
"""

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import io
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from google import genai
from google.genai import Client, types
from google.oauth2 import service_account
import imageio
from IPython.display import Image as IPImage, display
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
_SWARM_LOCK = threading.Lock()
CURRENT_OUTPUT_DIR: Optional[str] = None

# Inference parameter defaults
Temp = 0.5
model = config.MODEL_ID
path = config.OPENROUTER_BASE_URL


def get_headers() -> Dict[str, str]:
    """Generates standard authorization headers for upstream API requests."""
    return {
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        "HTTP-Referer": "https://colab.research.google.com",
        "X-Title": "UAV_AGENT",
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
# Aspect-Ratio Letterbox & Coordinate Inversion Helpers
# ==============================================================================

def _prepare_letterbox(
    img: Image.Image, target_size: int = 1000
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Fits an image onto a standardized square canvas while preserving native aspect ratio.

    Args:
        img: Input PIL Image with arbitrary dimensions.
        target_size: Target square canvas size (default: 1000x1000).

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
    api_key = config.FIREWORKS_API_KEY
    if not api_key:
        return "Error: No API key provided in configuration."

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
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
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 40,
        "messages": [{"role": "user", "content": content}],
    }

    headers = {
        "Accept": "application/json",
        "X-Title": "Satellite Analysis App",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(path, headers=headers, json=payload)

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
        List[dict]: Detected targets with labels and 2D pixel coordinates in native resolution.
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

    # 1. Project input onto standardized 1000x1000 canvas with aspect ratio preservation
    square_img, meta = _prepare_letterbox(img, target_size=1000)

    buffered = io.BytesIO()
    square_img.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 2. Structured prompt enforcing centroid grounding and shadow invariance
    prompt = f"""
    CRITICAL COORDINATE SPECIFICATIONS:
    - Focus strictly on the illuminated physical roof surfaces.
    - Ignore cast shadows entirely: do not place coordinates between the building and its shadow.
    - Center rule: Coordinates must be placed strictly on the centroid of the roof perimeter, not on edges or driveways.
    - The image canvas is defined on a normalized 1000x1000 integer grid.
    - Top-left corner is [0, 0]. Bottom-right corner is [1000, 1000].
    - Both X and Y must be integers strictly within range [0, 1000].
    - Place coordinates exactly at the geometric center of each '{target_str}'.
    - Do not output points on empty background or padding areas.

    This is the input satellite image. You need to detect all keypoints of {target_str} for drone navigation.
    Return maximum 15 points of '{target_str}'.
    Do not confuse with surrounding noise or background textures.
    Provide only the answer in json. Return them STRICTLY in the form:
    ```json
    [
      {{"point_2d": [x1, y1], "label": "{target_str}"}},
      {{"point_2d": [x2, y2], "label": "{target_str}"}}
    ]
    ```
    """

    headers = {
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

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
        "temperature": getattr(config, "Temp", 0.0),
    }

    raw_points = []
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.post(
                config.OPENROUTER_BASE_URL,
                headers=headers,
                json=data_payload,
                timeout=120,
            )

            if response.status_code == 200:
                raw_content = response.json()["choices"][0]["message"]["content"]
                json_str = re.sub(r"```json\s*|\s*```", "", raw_content).strip()

                try:
                    raw_points = json.loads(json_str)
                except json.JSONDecodeError:
                    objects_found = re.findall(
                        r'\{[^{}]*"point_2d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*"label"\s*:\s*"[^"]+"\s*\}',
                        json_str,
                    )
                    raw_points = (
                        json.loads("[\n" + ",\n".join(objects_found) + "\n]")
                        if objects_found
                        else []
                    )
                break

            if response.status_code in (400, 401, 403):
                print(
                    f"❌ Critical API error ({response.status_code}): {response.text}"
                )
                break

            print(
                f"⚠️ Attempt {attempt + 1} returned status code {response.status_code}. Retrying..."
            )
            time.sleep(5)

        except Exception as e:
            print(f"DEBUG: Attempt {attempt + 1} failed: {e}")
            time.sleep(5)

    # 3. Invert transformation back to native metric pixel coordinates
    return _restore_points(raw_points, meta, target_str)


@tool
def visualize_keypoints_from_image(
    image: Image.Image,
    keypoints: list,
    output_dir: str = "uav_logs",
) -> str:
    """Visualizes candidate inspection targets on the base terrain image and saves the plot.

    Args:
        image: Base PIL image representing the operational map.
        keypoints: Target keypoints list (dicts with 'point_2d' or (x, y) tuples).
        output_dir: Target directory to save the visualization plot. Defaults to 'uav_logs'.

    Returns:
        str: Confirmation message detailing the destination file path.
    """
    # Priority: Environment variable -> parameter override -> fallback default
    target_dir = os.environ.get("CURRENT_EXP_DIR") or output_dir or "uav_logs"
    os.makedirs(target_dir, exist_ok=True)

    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for idx, kp in enumerate(keypoints):
        x, y = get_coords(kp)
        label = (
            f"{idx + 1}. {kp.get('label', 'point')}"
            if isinstance(kp, dict)
            else f"{idx + 1}. target"
        )

        # Coordinates are already projected into native image scale
        x, y = int(round(x)), int(round(y))

        if idx == 0:
            draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill="red")
        else:
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], outline="red", width=2)

        draw.text((x + 8, y - 8), label, fill="blue", font=font)

    output_path = os.path.join(target_dir, "uav_trajectory.png")

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return f"Saved visualization with {len(keypoints)} targets to {output_path}."


@tool
def uav_simulation(
    image: Image.Image,
    labeled_points: Optional[List[Any]] = None,
    waypoints: Optional[List[Any]] = None,
    drone_id: int = 0,
    total_drones: int = 3,
    speed_px: float = 6.0,
    crop_size: int = 128,
    home_point: Tuple[int, int] = (50, 50),
    output_dir: str = "uav_logs",
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
    target_dir = os.environ.get("CURRENT_EXP_DIR") or output_dir or "uav_logs"
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

    drone_colors = [
        "#FF3333",  # Drone 0: Red
        "#3388FF",  # Drone 1: Blue
        "#33CC33",  # Drone 2: Green
        "#FF9900",  # Drone 3: Orange
        "#9933FF",  # Drone 4: Purple
    ]

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
    flight_sec = (total_dist_px * 1.35) / 6.0

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

    with ThreadPoolExecutor(max_workers=8) as executor:
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

        with ThreadPoolExecutor(max_workers=8) as video_executor:
            rendered_frames = list(video_executor.map(render_video_frame, time_steps))

        video_path = os.path.join(target_dir, "uav_flight_all_drones.mp4")
        fps = 13
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_w, img_h))

        if video_writer.isOpened():
            for frame in rendered_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"[SWARM SUCCESS] Saved MP4 flight video to: {video_path}")
        else:
            print(f"[ERROR] Could not initialize OpenCV VideoWriter for path: {video_path}")

        with _SWARM_LOCK:
            _SWARM_TRAJECTORIES.clear()

    return frames_dict


@tool
def detect_and_display(
    frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]],
    target_object: str,
    output_dir: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Detects target objects across UAV camera frames and saves annotated plots.

    Args:
        frames_dict: Dictionary mapping frame identifiers to tuples of (PIL Image, (x, y) coordinates).
        target_object: Name or semantic label of the object to search for (e.g., 'fire', 'car accident').
        output_dir: Target folder to save detection images and summary artifacts.

    Returns:
        List of (x, y) coordinate tuples where the target was confirmed.
    """
    # Priority: Environment variable -> parameter override -> fallback default
    target_dir = os.environ.get("CURRENT_EXP_DIR") or output_dir or "uav_logs"
    os.makedirs(target_dir, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    # 1. Subsample every 4th frame to balance inspection fidelity and API rate limits
    selected_frames = [
        (frame_name, img, coords)
        for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items())
        if idx % 4 == 0
    ]

    print(
        f"🔍 Running parallel VLM verification for '{target_object}' across "
        f"{len(selected_frames)} candidate frames into directory: {target_dir}..."
    )

    safe_target_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in target_object
    )
    plot_lock = threading.Lock()

    # 2. Parallel frame evaluation via ThreadPoolExecutor
    def process_single_frame(frame_data):
        frame_name, img, coords = frame_data
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"Do you see any {target_object} on the picture? Answer ONLY with 'YES' or 'NO'."
        data = {
            "model": config.MODEL_ID,
            "messages": [
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
                }
            ],
            "temperature": getattr(config, "TEMPERATURE", 0.5),
        }

        try:
            resp = requests.post(
                config.OPENROUTER_BASE_URL,
                headers=headers,
                json=data,
                timeout=30,
            )
            if resp.status_code == 200:
                desc = resp.json()["choices"][0]["message"]["content"].strip()
                if "YES" in desc.upper():
                    print(
                        f"🎯 Target '{target_object}' confirmed at coordinates {coords} in {frame_name}"
                    )
                    save_path = os.path.join(
                        target_dir, f"detected_{safe_target_name}_{frame_name}.png"
                    )

                    # Save figure with title and coordinates matching the reference layout
                    with plot_lock:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(img)
                        ax.set_title(
                            f"Coords: {coords}"
                        )
                        ax.axis("off")
                        fig.savefig(save_path, bbox_inches="tight")
                        plt.close(fig)

                    return coords
        except Exception:
            pass
        return None

    # 3. Execute thread pool and aggregate confirmed detection points
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_frame, selected_frames))

    detected_coordinates = [c for c in results if c is not None]

    print(
        f"\n✅ Visual inspection completed. Confirmed target locations: {len(detected_coordinates)}"
    )
    return detected_coordinates


@tool
def get_folder_by_query(query: str) -> dict:
    """Analyzes an operational request to resolve incident type, data folder, and target label.

    Args:
        query: Operator prompt or mission specification string.

    Returns:
        dict: Mapping containing resolved folder key, target object, and incident type.
    """
    # 1. Handle dictionary-formatted string inputs
    if isinstance(query, str) and query.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(query)
            if isinstance(parsed, dict) and "folder" in parsed:
                return {
                    "folder": parsed.get("folder", "default"),
                    "target_object": parsed.get("target_object", "unknown"),
                    "incident_type": parsed.get("incident_type", "unknown"),
                }
        except Exception:
            pass

    # 2. Semantic keyword classification
    query_lower = str(query).lower()

    keywords_map = {
        "fire": [
            "fire",
            "smoke",
            "flame",
            "burn",
            "burning",
            "on right now",
            "heat",
            "blaze",
        ],
        "oil_spill": [
            "oil",
            "spill",
            "leak",
            "slick",
            "pollution",
            "pipeline",
        ],
        "cars_accident": [
            "car",
            "accident",
            "crash",
            "collision",
            "vehicle",
            "police",
        ],
        "hogweed_thickets": [
            "hogweed",
            "weed",
            "thicket",
            "plant",
            "heracleum",
            "growth",
        ],
        "illegal_buildings": [
            "illegal",
            "building",
            "construction",
            "unauthorized",
            "shed",
            "house",
        ],
    }

    for incident, markers in keywords_map.items():
        if any(marker in query_lower for marker in markers):
            return {"folder": incident, "incident_type": incident}

    return {"folder": "default", "incident_type": "unknown"}


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

    headers = {
        "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": config.MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'User Request: "{user_request}"'},
        ],
        "temperature": Temp,
    }

    try:
        response = requests.post(
            config.OPENROUTER_BASE_URL,
            headers=headers,
            json=data,
            timeout=180,
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
