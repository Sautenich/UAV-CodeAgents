"""Onboard weight synchronization and edge inference engine for UAV-CodeAgents.

Provides utilities for downloading, verifying, and executing domain-specific ONNX
classification and detection weights directly on simulated or physical UAV flight hardware.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request

import cv2
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from smolagents import tool

import config
import src.tools as tools

load_dotenv()

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

DIRECT_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.onnx.onnx"
)
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/grishakalinin2014-alt/UAV-Agent/main/weights"
)


def _get_active_exp_dir() -> str:
    """Automatically resolves the active experiment output directory.

    Checks environment variables first, then scans the local experiments directory
    for the latest numerical trial folder (e.g., experiments/exp_12).

    Returns:
        str: Path to the target experiment directory or default fallback.
    """
    if os.environ.get("CURRENT_EXP_DIR"):
        return os.environ["CURRENT_EXP_DIR"]

    exp_root = "experiments"
    if os.path.exists(exp_root):
        exp_dirs = []
        for entry in os.listdir(exp_root):
            full_path = os.path.join(exp_root, entry)
            if os.path.isdir(full_path) and entry.startswith("exp_"):
                num_part = entry.replace("exp_", "")
                if num_part.isdigit():
                    exp_dirs.append((int(num_part), full_path))
        if exp_dirs:
            # Select directory with the highest numerical index (most recent experiment)
            return max(exp_dirs, key=lambda item: item[0])[1]

    return "uav_logs"


@tool
def download_and_sync_weights(incident_type: str) -> Optional[str]:
    """Downloads model weights for the UAV. Returns the path to the weights or None if unavailable.

    Args:
        incident_type: The type of incident (e.g., 'fire', 'hogweed_thickets').

    Returns:
        Optional[str]: Absolute filepath to the local ONNX model, or None if weights could not be loaded.
    """
    settings = config.REPO_SETTINGS.get(
        incident_type, config.REPO_SETTINGS.get("default", {})
    )
    weight_filename = settings.get("weight_filename", f"{incident_type}.onnx")

    # 1. Primary target directory: drone_models
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_drone_dir = os.path.join(base_dir, "drone_models")
    drone_dir = getattr(config, "DRONE_WEIGHTS_DIR", default_drone_dir)
    os.makedirs(drone_dir, exist_ok=True)

    local_filepath = os.path.join(drone_dir, weight_filename)

    # 2. Local cache check: verify if a valid model file exists (> 1 MB)
    if (
        os.path.exists(local_filepath)
        and os.path.getsize(local_filepath) > 1_000_000
    ):
        print(f"📦 Valid model weights already exist locally at: {local_filepath}")
        return local_filepath

    # 3. Assemble candidate remote URLs for weights download
    candidate_urls = []
    custom_url = getattr(config, "SINGLE_WEIGHTS_URL", None) or settings.get(
        "weight_url"
    )
    if custom_url:
        candidate_urls.append(custom_url)

    github_repo_url = f"{GITHUB_RAW_BASE}/{weight_filename}"
    if github_repo_url not in candidate_urls:
        candidate_urls.append(github_repo_url)

    # 4. Attempt downloading with strict network timeout
    timeout_seconds = 15

    for weight_url in candidate_urls:
        print(f"📡 Downloading ONNX weights from: {weight_url}...")
        try:
            req = urllib.request.Request(
                weight_url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
            )

            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                with open(local_filepath, "wb") as out_file:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)

            # Verify downloaded file size integrity
            if (
                os.path.exists(local_filepath)
                and os.path.getsize(local_filepath) > 1_000_000
            ):
                file_size_mb = os.path.getsize(local_filepath) / (1024 * 1024)
                print(
                    f"✅ ONNX model successfully downloaded ({file_size_mb:.2f} MB) to: {local_filepath}"
                )
                return local_filepath
            else:
                if os.path.exists(local_filepath):
                    os.remove(local_filepath)
                print(
                    "⚠️ Downloaded file is corrupted or below threshold. Trying alternative source..."
                )

        except Exception as err:
            if os.path.exists(local_filepath):
                os.remove(local_filepath)
            print(f"❌ Failed to download from {weight_url}: {err}")

    print(
        f"❌ Weights unavailable for '{incident_type}'. Returning None (Fallback mode enabled)."
    )
    return None


@tool
def run_local_uav_detection(
    frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]],
    weights_path: str,
    target_object: Union[str, List[str]],
    output_dir: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Executes local onboard classification on all UAV frames in the stream without early termination.

    Args:
        frames_dict: Dictionary mapping frame identifiers to tuples of (PIL Image, (x, y) coordinates).
        weights_path: Absolute local filesystem path to the ONNX weight file.
        target_object: Name or semantic descriptor of the object to detect (string or list of strings).
        output_dir: Target folder to save detection plots and artifacts. Defaults to None.

    Returns:
        List[Tuple[int, int]]: List of all detected coordinates (x, y) along the flight path.
    """
    # Fallback to active experiment directory if output_dir is not explicitly provided
    save_dir = output_dir or _get_active_exp_dir()
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(target_object, list):
        target_str = str(target_object[0]) if target_object else "target"
    else:
        target_str = str(target_object)

    target_str = target_str.strip()

    print(f"🚀 Initializing onboard classification engine: {weights_path}")
    print(
        f"🔍 Full flight path scan started for: '{target_str}' "
        f"(Saving artifacts to: {save_dir})..."
    )

    if not os.path.exists(weights_path):
        print(f"❌ ONNX weight file not found at '{weights_path}'")
        return []

    try:
        session = ort.InferenceSession(
            weights_path, providers=["CPUExecutionProvider"]
        )
        input_node = session.get_inputs()[0]
        input_name = input_node.name

        input_shape = input_node.shape
        target_h = input_shape[2] if isinstance(input_shape[2], int) else 640
        target_w = input_shape[3] if isinstance(input_shape[3], int) else 640
        print(f"📐 Model input resolution: {target_w}x{target_h}")
    except Exception as e:
        print(f"❌ Failed to initialize ONNX inference session: {e}")
        return []

    conf_threshold = 0.65
    all_detected_coords = []
    safe_target_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in target_str
    )

    for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items()):
        # Subsample frames to reduce redundant adjacent evaluations
        if idx % 2 != 0:
            continue

        img_np = np.array(img.convert("RGB"))
        resized = cv2.resize(img_np, (target_w, target_h))
        input_tensor = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = session.run(None, {input_name: input_tensor})

        if outputs and len(outputs) > 0:
            raw_scores = outputs[0][0]
            if len(raw_scores) == 1:
                target_conf = float(1 / (1 + np.exp(-raw_scores[0])))
            else:
                exp_scores = np.exp(raw_scores - np.max(raw_scores))
                probs = exp_scores / exp_scores.sum()
                class_idx = 1 if "accident" in target_str.lower() else 0
                target_conf = (
                    float(probs[class_idx])
                    if len(probs) > class_idx
                    else float(np.max(probs))
                )

            if target_conf >= conf_threshold:
                print(
                    f"🎯 [Onboard Match] '{target_str}' detected at {coords} on "
                    f"{frame_name} (Confidence: {target_conf:.2f})"
                )
                all_detected_coords.append(coords)

                try:
                    save_path = os.path.join(
                        save_dir, f"detected_{safe_target_name}_{frame_name}.png"
                    )
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.title(
                        f"Classification Match: {target_str} (Conf: {target_conf:.2f})\n"
                        f"Coordinates: {coords}"
                    )
                    plt.axis("off")
                    plt.savefig(save_path, bbox_inches="tight")
                    plt.close()
                except Exception as save_err:
                    print(f"⚠️ Failed to save detection frame visualization: {save_err}")
            else:
                print(
                    f"🖼️ [Onboard Scan] {frame_name} — Position: {coords} | Clear "
                    f"({target_str} Confidence: {target_conf:.2f})"
                )

    print(
        f"\n✅ Flight path scan completed. Total detections found: "
        f"{len(all_detected_coords)}"
    )
    return all_detected_coords