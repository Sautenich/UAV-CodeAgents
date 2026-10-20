"""Data loading and terrain imagery ingestion module for UAV-CodeAgents.

Provides tool interfaces to retrieve aerial imagery from remote GitHub repositories
and load calibrated benchmark map pairs (archival reference baseline and real-time
post-incident terrain) for multi-UAV reconnaissance missions.
"""

import io
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import requests
from smolagents import tool

import config
from src.tools import canonical_incident


@tool
def fetch_images_from_github(
    repo_owner: str,
    repo_name: str,
    folder_path: str,
    branch: str = config.GITHUB_BRANCH,
) -> List[Image.Image]:
    """Downloads aerial images from a GitHub repository directory and returns them as PIL Images.

    Args:
        repo_owner: GitHub username or organization identifier (e.g., 'grishakalinin2014-alt').
        repo_name: Target repository name (e.g., 'UAV-Agent').
        folder_path: Path to the target directory inside the repository (e.g., 'data/default').
        branch: Target Git branch name. Defaults to 'main'.

    Returns:
        List[Image.Image]: List of downloaded PIL Images sorted numerically by filename (up to 10).
    """
    api_url = (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
        f"{folder_path}?ref={branch}"
    )
    headers = {"User-Agent": "UAV-Agent-App"}

    print(f"📡 Querying GitHub API: {api_url}")
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()

    files = response.json()
    if not isinstance(files, list):
        raise ValueError(
            f"Expected a file list from GitHub API, received {type(files)}"
        )

    # Filter for standard image file extensions
    image_files = [
        f
        for f in files
        if isinstance(f, dict)
        and f.get("name", "").lower().endswith(config.IMAGE_EXTENSIONS)
    ]

    def extract_number(name: str) -> float:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else float("inf")

    # Sort files numerically and cap batch size
    image_files.sort(key=lambda x: extract_number(x["name"]))
    image_files = image_files[: config.GITHUB_MAX_IMAGES]

    loaded_images = []
    for file_info in image_files:
        print(f"📥 Downloading remote asset: {file_info['name']}...")
        img_response = requests.get(file_info["download_url"], headers=headers)
        img_response.raise_for_status()

        img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
        loaded_images.append(img)

    print(f"✅ Successfully loaded {len(loaded_images)} images from GitHub repository.")
    return loaded_images


@tool
def load_local_map_images(
    folder_name: str,
    index: int = 1,
    base_dir: str = config.DATA_DIR,
) -> Tuple[Image.Image, Image.Image]:
    """Loads archival reference base map and corresponding real-time target map for an experiment.

    Args:
        folder_name: Incident subfolder name (e.g., 'fire', 'oil_spill', 'cars_accident').
        index: Map and experiment numerical index (1 to 30). Defaults to 1.
        base_dir: Root dataset directory on the local filesystem. Defaults to config.DATA_DIR.

    Returns:
        Tuple[Image.Image, Image.Image]:
            - base_image: Archival baseline map used strictly for structure/keypoint extraction.
            - target_image: Real-time ground image used strictly for onboard camera simulation.
    """

    def _number(filename: str) -> Optional[int]:
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else None

    def _list_sorted(full_path: str) -> List[str]:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Local directory not found: {full_path}")

        files = [
            f
            for f in os.listdir(full_path)
            if f.lower().endswith(config.IMAGE_EXTENSIONS)
        ]
        if not files:
            raise FileNotFoundError(f"No valid images found in directory: {full_path}")

        # Stable numeric ordering; only the selected files are opened afterwards
        return sorted(files, key=lambda f: (_number(f) or 0, f))

    def _pick_by_index(files: List[str], target_num: int) -> str:
        # 1. Exact numeric pattern match in filename (e.g., index 1 in '1_square.jpg' or '1.jpg')
        for fname in files:
            if _number(fname) == target_num:
                return fname

        # 2. Sequential fallback based on 1-based index ordering
        pos = target_num - 1
        if 0 <= pos < len(files):
            return files[pos]

        # 3. Default fallback to first element in list
        return files[0]

    def _open(full_path: str, filename: str) -> Image.Image:
        return Image.open(os.path.join(full_path, filename)).convert("RGB")

    default_path = os.path.abspath(
        os.path.expanduser(os.path.join(base_dir, config.BASELINE_FOLDER))
    )
    target_path = os.path.abspath(
        os.path.expanduser(os.path.join(base_dir, folder_name))
    )

    print(
        f"📥 Loading map imagery (Index #{index}) from:\n"
        f" - Reference Baseline: {default_path}\n"
        f" - Operational Target: {target_path}"
    )

    baseline_files = _list_sorted(default_path)
    target_files = _list_sorted(target_path)

    scenario = config.SCENARIOS.get(canonical_incident(folder_name), {})
    baseline_map = scenario.get("baseline_map") or []

    # Resolve base_image and target_image pair according to active scenario requirements
    if 1 <= index <= len(baseline_map):
        # The scenario names the baseline map behind each of its experiments
        base_name = _pick_by_index(baseline_files, baseline_map[index - 1])
    else:
        base_name = _pick_by_index(baseline_files, index)
    target_name = _pick_by_index(target_files, index)

    print(
        f"🗺️ Scenario '{folder_name}' [Trial #{index}]: reference '{base_name}',"
        f" target '{target_name}'"
    )

    base_image = _open(default_path, base_name)
    target_image = _open(target_path, target_name)

    # Keypoints are found on the baseline and flown on the target, so both maps
    # must share one pixel grid
    if base_image.size != target_image.size:
        base_aspect = base_image.width / base_image.height
        target_aspect = target_image.width / target_image.height
        if abs(base_aspect - target_aspect) <= config.MAP_ASPECT_TOLERANCE * target_aspect:
            print(
                f"↔️ Resizing reference map {base_image.size} to the target map"
                f" size {target_image.size}."
            )
            base_image = base_image.resize(target_image.size, Image.Resampling.LANCZOS)
        else:
            print(
                f"⚠️ Reference map {base_image.size} and target map {target_image.size}"
                " show different frames; keypoints will not line up with the target."
            )

    return base_image, target_image


# ==============================================================================
# Ground Truth and Detection Scoring
# ==============================================================================

def load_ground_truth(
    scenario: str, index: int, path: Optional[str] = None
) -> Optional[List[List[float]]]:
    """Loads annotated incident centres for one experiment, if available.

    Args:
        scenario: Canonical scenario key (e.g. 'fire').
        index: Experiment / map index.
        path: Ground truth JSON file. Defaults to `config.GROUND_TRUTH_FILE`.

    Returns:
        List of [x, y] incident centres (empty when the map has no incident), or
        None when no annotation exists for this experiment.
    """
    path = path or config.GROUND_TRUTH_FILE
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    points = annotations.get(scenario, {}).get(str(index))
    return None if points is None else [list(p) for p in points]


def score_detections(
    detections: Sequence[Sequence[float]],
    truth_points: Optional[Sequence[Sequence[float]]],
    radius: Optional[float] = None,
) -> Dict[str, Any]:
    """Compares detected coordinates with annotated incident centres.

    A detection is correct when it lies within `radius` of any incident; an
    incident is found when any detection lies within `radius` of it.

    Args:
        detections: Detected (x, y) coordinates.
        truth_points: Annotated incident centres, or None when unavailable.
        radius: Match radius in pixels. Defaults to
            `config.GROUND_TRUTH_MATCH_RADIUS_PX`.

    Returns:
        Dictionary with 'true_positives', 'false_positives', 'false_negatives',
        'precision' and 'recall'. All values are None without ground truth;
        precision / recall are None when their denominator is zero.
    """
    if truth_points is None:
        return {
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "precision": None,
            "recall": None,
        }

    radius = config.GROUND_TRUTH_MATCH_RADIUS_PX if radius is None else radius

    def near(a: Sequence[float], b: Sequence[float]) -> bool:
        return math.hypot(a[0] - b[0], a[1] - b[1]) <= radius

    true_positives = sum(
        1 for d in detections if any(near(d, t) for t in truth_points)
    )
    found = sum(1 for t in truth_points if any(near(d, t) for d in detections))

    return {
        "true_positives": true_positives,
        "false_positives": len(detections) - true_positives,
        "false_negatives": len(truth_points) - found,
        "precision": true_positives / len(detections) if detections else None,
        "recall": found / len(truth_points) if truth_points else None,
    }
