"""Gemini-based perception and reasoning tools for UAV-CodeAgents.

Provides direct Google Gemini API implementations for high-resolution aerial scene
captioning, pixel-level zero-shot visual grounding, frame-by-frame target verification,
and semantic sub-goal extraction from natural language operator queries.
"""

import base64
import concurrent.futures
from io import BytesIO
import io
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

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
# Scene Captioning Tool
# ==============================================================================

@tool
def describe_satellite_image_Gemini(image: Image.Image) -> str:
    """Analyzes an aerial or satellite image using the Google Gemini multimodal model.

    Args:
        image: A PIL.Image.Image object representing the operational aerial scene.

    Returns:
        str: Comprehensive textual description of detected objects, topology, and spatial layout.
    """
    try:
        # 1. Ensure RGB format and encode image to base64 JPEG
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": config.GOOGLE_API_KEY,
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Describe this image in very detailed manner. "
                                "Name each object you observe and its approximate location."
                            )
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_base64,
                            }
                        },
                    ]
                }
            ]
        }

        # 2. Dispatch request to Gemini endpoint
        response = requests.post(config.GOOGLE_URL, headers=headers, json=data)

        return (
            response.text
            if response and response.text
            else "Error: Empty response received from API."
        )

    except Exception as e:
        print(f"DEBUG: Gemini Authentication/API Error: {e}")
        return f"Error: {e}"


# ==============================================================================
# Pixel-Pointing Grounding Tool
# ==============================================================================

@tool
def pixelpoint_objects_Gemini(image: Image.Image, objects: str) -> List[dict]:
    """Detects target objects on satellite imagery and extracts pixel coordinates via Gemini.

    Args:
        image: A PIL.Image.Image object of the operational aerial terrain.
        objects: Comma-separated list or description of semantic target classes to locate.

    Returns:
        List[dict]: Dictionaries containing 2D pixel coordinates and semantic labels.
    """
    # 1. Standardize image color space
    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = f"""
    This is the input satellite image. You need detect all keypoints of {objects} for drone navigation.
    Return maximum 15 points of '{objects}'.
    Do not confuse with surrounding noise or background textures.
    Provide only the answer in json. Return them STRICTLY in the form:
    ```json
    [
      {{"point_2d": [x1, y1], "label": {objects}}},
      {{"point_2d": [x2, y2], "label": {objects}}}
    ]
    ```
    """

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": img_base64,
                        }
                    },
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    url = f"{config.GOOGLE_URL}?key={config.GOOGLE_API_KEY}"

    # 2. Execute query with exponential retry logic for rate-limit resilience
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=90
            )

            if response.status_code == 200:
                raw_content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                json_str = re.sub(r"```json\s*|\s*```", "", raw_content).strip()

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    objects_found = re.findall(
                        r'\{[^{}]*"point_2d"\s*:\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*"label"\s*:\s*"[^"]+"\s*\}',
                        json_str,
                    )
                    return (
                        json.loads("[\n" + ",\n".join(objects_found) + "\n]")
                        if objects_found
                        else []
                    )

            # Handle transient upstream rate limits (429) or service outages (503)
            print(
                f"⚠️ Attempt {attempt + 1} returned status code {response.status_code}. Waiting..."
            )
            time.sleep(10)

        except Exception as e:
            print(f"DEBUG: Attempt {attempt + 1} failed: {e}")
            time.sleep(10)

    return []


# ==============================================================================
# Target Verification Tool
# ==============================================================================

@tool
def detect_and_display_Gemini(
    frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]],
    target_object: str,
) -> Optional[Tuple[int, int]]:
    """Inspects sub-sampled UAV camera frames sequentially to verify target presence via Gemini.

    Args:
        frames_dict: Mapping of frame identifiers to (PIL Image, (x, y) coordinates) tuples.
        target_object: Name or semantic descriptor of the hazard or object to verify.

    Returns:
        Optional[Tuple[int, int]]: Coordinates (x, y) of the first verified target, or None.
    """
    headers = {"Content-Type": "application/json"}
    url = f"{config.GOOGLE_URL_DETECT}?key={config.GOOGLE_API_KEY}"

    print(f"🔍 Starting {target_object} inspection on every 4th frame...\n")

    request_count = 0

    for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items()):
        # Inspect every 4th frame to manage API quota and throughput
        if idx % 4 != 0:
            continue

        request_count += 1

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"Do you see {target_object} at the picture? Answer ONLY with 'YES' or 'NO'."

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": img_base64,
                            }
                        },
                    ]
                }
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=90)
            if response.status_code != 200:
                print(f"❌ Error processing frame {frame_name}: status code {response.status_code}")
                continue

            resp_json = response.json()
            description = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()

            print(f"🧠 Model response for {frame_name}: {description}")

            if "YES" in description.upper():
                print(f"\n🎯 Target '{target_object}' detected at coordinates {coords}!")

                try:
                    os.makedirs("uav_logs", exist_ok=True)
                    plt.figure()
                    plt.imshow(img)
                    plt.title(f"{target_object} detected in {frame_name}")
                    plt.axis("off")
                    plt.savefig(
                        f"uav_logs/{target_object}_{frame_name}.png",
                        bbox_inches="tight",
                    )
                    plt.close()
                except Exception as visual_err:
                    print(f"Skipping visualization export: {visual_err}")

                return coords

        except Exception as e:
            print(f"DEBUG: Error parsing response for {frame_name}: {e}")
            continue

    print(f"\n✅ Verification complete. No {target_object} detected.")
    return None


# ==============================================================================
# Goal Extraction Tool
# ==============================================================================

@tool
def extract_inspection_targets_Gemini(user_query: str) -> List[str]:
    """Extracts domain-adapted inspection target categories from an operator prompt.

    Args:
        user_query: Natural language mission instruction provided by the human operator.

    Returns:
        List[str]: Formatted list of semantic target categories for subsequent visual grounding.
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

    full_prompt = f'{system_prompt}\n\nUser Request: "{user_query}"'

    url = f"{config.GOOGLE_URL}?key={config.GOOGLE_API_KEY}"
    headers = {"Content-Type": "application/json"}

    data = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=90)
        response.raise_for_status()

        result = response.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

        targets = json.loads(raw_text)
        if isinstance(targets, list):
            return targets
        return [str(targets)]

    except Exception as e:
        print(f"Error during semantic target extraction: {e}")
        return ["target object", "surrounding structure", "access route"]