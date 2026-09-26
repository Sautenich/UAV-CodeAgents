"""Data loading and terrain imagery ingestion module for UAV-CodeAgents.

Provides tool interfaces to retrieve aerial imagery from remote GitHub repositories
and load calibrated benchmark map pairs (archival reference baseline and real-time
post-incident terrain) for multi-UAV reconnaissance missions.
"""

import io
import os
import re
from typing import Any, List, Tuple

from PIL import Image
import requests
from smolagents import tool


@tool
def fetch_images_from_github(
    repo_owner: str,
    repo_name: str,
    folder_path: str,
    branch: str = "main",
) -> List[Image.Image]:
    """Downloads aerial images from a GitHub repository directory and returns them as PIL Images.

    Args:
        repo_owner: GitHub username or organization identifier (e.g., 'grishakalinin2014-alt').
        repo_name: Target repository name (e.g., 'UAV-Agent').
        folder_path: Path to the target directory inside the repository (e.g., 'images/blank').
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
        and f.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    def extract_number(name: str) -> float:
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else float("inf")

    # Sort files numerically and cap batch size
    image_files.sort(key=lambda x: extract_number(x["name"]))
    image_files = image_files[:10]

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
    base_dir: str = "data",
) -> Tuple[Image.Image, Image.Image]:
    """Loads archival reference base map and corresponding real-time target map for an experiment.

    Args:
        folder_name: Incident subfolder name (e.g., 'fire', 'oil_spill', 'cars_accident').
        index: Map and experiment numerical index (1 to 30). Defaults to 1.
        base_dir: Root dataset directory on the local filesystem. Defaults to 'data'.

    Returns:
        Tuple[Image.Image, Image.Image]:
            - base_image: Archival baseline map used strictly for structure/keypoint extraction.
            - target_image: Real-time ground image used strictly for onboard camera simulation.
    """

    def _load_and_sort_folder(full_path: str) -> List[Tuple[Image.Image, str]]:
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Local directory not found: {full_path}")

        files = [
            f
            for f in os.listdir(full_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not files:
            raise FileNotFoundError(f"No valid images found in directory: {full_path}")

        images = []
        for filename in files:
            filepath = os.path.join(full_path, filename)
            img = Image.open(filepath).convert("RGB")
            images.append((img, filename))

        def extract_number(item: Tuple[Image.Image, str]) -> int:
            match = re.search(r"(\d+)", item[1])
            return int(match.group(1)) if match else 0

        images.sort(key=extract_number)
        return images

    def _pick_image_by_index(
        images_list: List[Tuple[Image.Image, str]], target_num: int
    ) -> Tuple[Image.Image, str]:
        # 1. Exact numeric pattern match in filename (e.g., index 1 in '1_square.jpg' or '1.jpg')
        for img, fname in images_list:
            match = re.search(r"(\d+)", fname)
            if match and int(match.group(1)) == target_num:
                return img, fname

        # 2. Sequential fallback based on 1-based index ordering
        pos = target_num - 1
        if 0 <= pos < len(images_list):
            return images_list[pos]

        # 3. Default fallback to first element in list
        return images_list[0]

    default_path = os.path.abspath(
        os.path.expanduser(os.path.join(base_dir, "default"))
    )
    target_path = os.path.abspath(
        os.path.expanduser(os.path.join(base_dir, folder_name))
    )

    print(
        f"📥 Loading map imagery (Index #{index}) from:\n"
        f" - Reference Baseline: {default_path}\n"
        f" - Operational Target: {target_path}"
    )

    blank_tuple_list = _load_and_sort_folder(default_path)
    target_tuple_list = _load_and_sort_folder(target_path)

    scenario_key = folder_name.lower().strip()

    # Resolve base_image and target_image pair according to active scenario requirements
    if scenario_key == "oil_spill":
        # Fixed maritime reference canvas for oil spill localization
        base_idx = 32 if len(blank_tuple_list) > 32 else 0
        base_image, base_name = blank_tuple_list[base_idx]
        target_image, target_name = _pick_image_by_index(target_tuple_list, index)
        print(
            f"🛢️ Scenario 'oil_spill': reference '{base_name}', target '{target_name}'"
        )
    else:
        base_image, base_name = _pick_image_by_index(blank_tuple_list, index)
        target_image, target_name = _pick_image_by_index(target_tuple_list, index)
        print(
            f"🔥 Scenario '{folder_name}' [Trial #{index}]: reference '{base_name}', target '{target_name}'"
        )

    return base_image, target_image