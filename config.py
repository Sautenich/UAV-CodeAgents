"""Configuration module for the UAV-CodeAgents multi-agent framework.

Defines API credentials, foundation model endpoints, local weights storage paths,
remote benchmark dataset repositories, and default swarm parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ==============================================================================
# Swarm Execution Defaults
# ==============================================================================

# Default number of active UAVs deployed in the mission swarm
COUNTER_DRONES = 3

# ==============================================================================
# API Authentication & Endpoints
# ==============================================================================

# API credentials and endpoint for proxy/gateway service (Fireworks / Starimg / OpenRouter)
FIREWORKS_API_KEY = os.getenv("MY_KEY")
MODEL_ID = "gpt-6-sol"
OPENROUTER_BASE_URL = "https://ai.starimg.ru/v1/chat/completions"

# Google Gemini API credentials and direct service endpoints
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
)
GOOGLE_URL_DETECT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
)


# ==============================================================================
# Model Weights and Local Storage Directories
# ==============================================================================

# Base filesystem directories for caching local edge models and checkpoints
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRONE_WEIGHTS_DIR = os.path.join(BASE_DIR, "drone_models")

# Remote source and filename for onboard lightweight ONNX detection model
SINGLE_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
)
SINGLE_WEIGHTS_FILENAME = "yolov8n.onnx"
DRONE_WEIGHTS_DIR = os.path.abspath("./drone_models")

# ==============================================================================
# Benchmark Dataset Repositories
# ==============================================================================

# GitHub repository configurations for downloading operational aerial terrain imagery
REPO_SETTINGS = {
    "default": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/blank",
    },
    "fire": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/fire_buildings",
    },
    "hogweed_thickets": {
        "owner": "grishakalinin2014-alt",
        "repo": "UAV-Agent",
        "branch": "main",
        "folder": "images/hogweed_thickets",
    },
}