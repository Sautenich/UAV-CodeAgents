"""Configuration module for the UAV-CodeAgents multi-agent framework.

Single place for everything a person reproducing the experiments may want to
tune: API credentials and endpoints, swarm size, experiment batch settings,
scenario definitions, simulation and detection parameters, and storage paths.

Secrets are read from the `.env` file; everything else is a plain constant.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# Swarm Execution Defaults
# ==============================================================================

# Number of active UAVs deployed in the mission swarm
COUNTER_DRONES = 3

# ==============================================================================
# LLM / VLM Gateway (OpenAI-compatible API)
# ==============================================================================

# API key for the OpenAI-compatible gateway. `MY_KEY` is the legacy variable name.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("MY_KEY")
# Gateway root; any OpenAI-compatible provider works (starimg, OpenRouter, Fireworks, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ai.starimg.ru/v1")
LLM_CHAT_COMPLETIONS_URL = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
# Model used both by the agents and by the perception tools
MODEL_ID = os.getenv("LLM_MODEL_ID", "gemini-3.8-flash")

# Agent LLM client behaviour (LiteLLM)
LLM_TIMEOUT_S = 500
LITELLM_REQUEST_TIMEOUT_S = 180.0
LLM_NUM_RETRIES = 5
LLM_RETRY_STRATEGY = "exponential_backoff_retry"

# Sampling temperatures of the perception tools
VLM_TEMPERATURE = 0.5  # scene description, target extraction
GROUNDING_TEMPERATURE = 0.0  # keypoint grounding in `pixelpoint_objects`

# Perception request limits
VLM_DESCRIBE_MAX_TOKENS = 1000
GROUNDING_TIMEOUT_S = 120
GROUNDING_MAX_RETRIES = 3
GROUNDING_RETRY_DELAY_S = 5
TARGET_EXTRACTION_TIMEOUT_S = 180
IMAGE_JPEG_QUALITY = 90  # full maps sent to the VLM

# Ignore HTTP(S)_PROXY / ALL_PROXY variables so local services stay reachable
BYPASS_SYSTEM_PROXY = True
NO_PROXY_HOSTS = "localhost,127.0.0.1,0.0.0.0"

# ==============================================================================
# Tracing (Arize Phoenix)
# ==============================================================================

# Tracing of agent runs in a local Phoenix collector; set PHOENIX_ENABLED=1 in .env to turn it on
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "0") == "1"
PHOENIX_PROJECT_NAME = "uav_agent"
PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"

# ==============================================================================
# Agents (smolagents CodeAgent)
# ==============================================================================

AGENT_MAX_STEPS = 12
AGENT_VERBOSITY_LEVEL = 2
MANAGER_PLANNING_INTERVAL = 3
# Maximum duration of one generated code block (covers the whole swarm dispatch)
AGENT_CODE_TIMEOUT_S = 500

# ==============================================================================
# Experiment Batch
# ==============================================================================

EXPERIMENTS_DIR = "experiments"
# Experiments run for indices EXPERIMENT_START_INDEX..TOTAL_EXPERIMENTS inclusive.
# Set the start index to 1 for a full run; a higher value resumes an interrupted batch.
EXPERIMENT_START_INDEX = 1
TOTAL_EXPERIMENTS = 5
# Seed of the per-experiment base station placement (seed = RANDOM_SEED + index)
RANDOM_SEED = 42
# Base stations are placed on the map border within this coordinate range
HOME_POINT_MIN = 60
HOME_POINT_MAX = 950
PAUSE_BETWEEN_EXPERIMENTS_S = 1

# Scenario used when the operator presses Enter or no keyword is recognised
DEFAULT_SCENARIO = "fire"

# Optional ground truth for precision / recall, as JSON:
#   {"fire": {"1": [[x, y], [x, y]], "2": []}, "oil_spill": {"33": [[x, y]]}}
# keyed by scenario, then by experiment index, with incident centres in pixels of
# the target image. Without the file only "incident detected" is reported.
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground_truth.json")
# A detection counts as correct when it lies within this distance of an incident
GROUND_TRUTH_MATCH_RADIUS_PX = 100

# ==============================================================================
# Scenarios
# ==============================================================================
# One entry per incident type. The key is the canonical `incident_type`.
#   keywords:        whole words / phrases that select the scenario from a request
#   request:         default operator request
#   folder:          sub-folder of DATA_DIR with the target (post-incident) maps
#   target_object:   what the drones look for
#   aliases:         other accepted spellings of the incident type
#   valen_context / valen_question: prompts of the onboard Valen verifier
#   detection_threshold: optional; probability above which a frame is reported
#                    for this scenario (default: DETECTION_CONF_THRESHOLD)
#   required_keywords: optional; the scenario is considered only when the request
#                    contains at least one of them
#   baseline_map:    optional list; experiment k uses baseline map number
#                    baseline_map[k - 1] instead of the map with the same number

SCENARIOS = {
    "fire": {
        "keywords": ["fire", "smoke", "flame", "burn", "burning", "blaze"],
        "request": "I need to check at home; one of them might be on right now fire.",
        "folder": "fire",
        "target_object": "fire",
        "aliases": [],
        "valen_context": "Inspect the structure and ground for fire or smoke hazards.",
        "valen_question": "Is there open flame, active fire, burning structure, or smoke visible?",
        "detection_threshold": 0.65,
    },
    "forest_fire": {
        # A fire request that mentions a forest; counts fire words on top of forest words
        "keywords": [
            "forest", "wildfire", "woodland", "woods", "tree", "taiga",
            "fire", "smoke", "flame", "burn", "burning", "blaze",
        ],
        "required_keywords": ["forest", "wildfire", "woodland", "woods", "tree", "taiga"],
        "request": "A wildfire has been reported. Find where the forest is burning.",
        "folder": "forest_fire",
        "target_object": "forest fire",
        "aliases": ["wildfire", "forest_fires"],
        "valen_context": "Inspect the forest canopy for wildfire.",
        "valen_question": (
            "Is there a forest fire visible: open flames, a burning fire front, a burnt"
            " scorched area, or a smoke plume rising from the trees?"
        ),
        "detection_threshold": 0.65,
        # Forested baseline maps; targets are made by create_forest_fire_dataset.py
        "baseline_map": [1, 10, 26, 27, 29],
    },
    "oil_spill": {
        "keywords": ["oil", "spill", "leak", "slick", "pollution", "pipeline"],
        "request": "Detect any signs of an oil spill in the water bodies or on the ground.",
        "folder": "oil_spill",
        "target_object": "oil spill",
        "aliases": [],
        # The verifier does not separate slicks from clean water when asked about
        # "oil" (its scores for slick frames are below those for open water), so
        # the question describes what a slick looks like from above instead.
        "valen_context": "This is an aerial photo.",
        "valen_question": "Is a large black irregular shape visible in this image?",
        "detection_threshold": 0.74,
        # Coastal baseline maps behind the target maps of experiments 1..5
        "baseline_map": [33, 34, 35, 36, 37],
    },
    "car_accident": {
        "keywords": ["car", "accident", "crash", "collision", "vehicle"],
        "request": "A car accident has occurred. The police are on the scene.",
        "folder": "cars_accident",
        "target_object": "car accident",
        "aliases": ["cars_accident", "car_accidents"],
        # Vehicles are a few pixels wide at this resolution; asking for vehicles on
        # the road separates the scene from background far better than asking for
        # a "collision".
        "valen_context": "This is an aerial photo.",
        "valen_question": "Is a vehicle visible on a road or at a road junction?",
        "detection_threshold": 0.73,
    },
    "hogweed_thickets": {
        "keywords": ["hogweed", "weed", "thicket", "heracleum"],
        "request": "Hogweed thickets are growing in the area.",
        "folder": "hogweed_thickets",
        "target_object": "hogweed thickets",
        "aliases": ["hogweed"],
        "valen_context": "Inspect the terrain and vegetation for invasive plants.",
        "valen_question": (
            "Are there dense invasive hogweed thickets, tall Heracleum plants, or"
            " overgrown weed patches visible?"
        ),
    },
    "building": {
        "keywords": ["illegal", "building", "construction", "unauthorized", "shed"],
        "request": "Illegal buildings are being constructed in the area.",
        "folder": "illegal_buildings",
        "target_object": "illegal building",
        "aliases": ["illegal_building", "illegal_constructions"],
        "valen_context": "Inspect the area for unauthorized construction sites.",
        "valen_question": (
            "Is there an unauthorized building, shed, illegal structure, or ongoing"
            " construction visible?"
        ),
    },
}

# Values returned when a request matches no scenario
UNKNOWN_INCIDENT = "unknown"
UNKNOWN_TARGET_OBJECT = "hazard"

# ==============================================================================
# Datasets
# ==============================================================================

# Root folder with the map imagery; one sub-folder per scenario
DATA_DIR = os.path.join(BASE_DIR, "data")
# Sub-folder with archival (pre-incident) reference maps
BASELINE_FOLDER = "default"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
# A baseline map is resized to its target map when their aspect ratios differ by
# less than this share; keypoints found on the baseline then fit the target
MAP_ASPECT_TOLERANCE = 0.02

# GitHub repository hosting the same imagery (used by `fetch_images_from_github`)
GITHUB_REPO_OWNER = "grishakalinin2014-alt"
GITHUB_REPO_NAME = "UAV-Agent"
GITHUB_BRANCH = "main"
# Maps downloaded per folder at most
GITHUB_MAX_IMAGES = 10

# ==============================================================================
# Visual Grounding (waypoint extraction)
# ==============================================================================

# Side of the normalized square canvas the VLM returns coordinates on
GROUNDING_GRID_SIZE = 1000
MAX_KEYPOINTS = 15
# Large regions (water, forest, fields) are covered with points about this far
# apart (grid units) instead of one point in the middle
GROUNDING_AREA_SPACING = 180
# Area points stay at least this far (grid units) from the image border
GROUNDING_BORDER_MARGIN = 50

# The VLM occasionally answers without looking at the image and returns fabricated
# coordinates. Each grounding therefore asks GROUNDING_VOTES times in parallel and
# uses the complete answer whose points the other answers reproduce best. Points
# are not filtered one by one, so area targets (water, forest) keep all waypoints.
# Set GROUNDING_VOTES = 1 for a single request (cheaper, no cross-check).
GROUNDING_VOTES = 1
# Points of different answers within this distance (grid units) are one object
GROUNDING_MATCH_RADIUS = 20
# Extra voting rounds when no answer is usable
GROUNDING_MAX_ROUNDS = 2
# An answer is discarded as fabricated when this many points share one coordinate
# (within the tolerance, grid units) and are evenly spaced along the other axis
GROUNDING_SYNTHETIC_MIN_POINTS = 5
GROUNDING_SYNTHETIC_AXIS_TOLERANCE = 10
GROUNDING_SYNTHETIC_GAP_VARIATION = 0.15

# ==============================================================================
# Flight Simulation
# ==============================================================================

# Directory for artifacts when no experiment directory is active
DEFAULT_OUTPUT_DIR = "uav_logs"
# Base station used when the caller does not provide one
DEFAULT_HOME_POINT = (50, 50)
# Distance travelled between two consecutive camera frames, in pixels
UAV_SPEED_PX = 6.0
# Side of the square onboard camera footprint, in pixels
CAMERA_CROP_SIZE = 128
# Flight time model: seconds = path_px * GROUND_SAMPLING_DISTANCE_M / CRUISE_SPEED_M_PER_S
# Map scale: metres of ground covered by one pixel
GROUND_SAMPLING_DISTANCE_M = 1.35
# Cruise speed of a UAV in metres per second
CRUISE_SPEED_M_PER_S = 6.0

# Swarm video
VIDEO_FILENAME = "uav_flight_all_drones.mp4"
VIDEO_FPS = 13
VIDEO_CODEC = "mp4v"
SIMULATION_WORKERS = 8
# Video frames rendered and written per chunk (bounds memory use)
VIDEO_CHUNK_FRAMES = 64
DRONE_COLORS = [
    "#FF3333",  # Drone 0: Red
    "#3388FF",  # Drone 1: Blue
    "#33CC33",  # Drone 2: Green
    "#FF9900",  # Drone 3: Orange
    "#9933FF",  # Drone 4: Purple
]
TRAJECTORY_FILENAME = "uav_trajectory.png"
TRAJECTORY_DPI = 300

# ==============================================================================
# Onboard Detection (Valen)
# ==============================================================================

# Probability above which a frame counts as a confirmed incident
DETECTION_CONF_THRESHOLD = 0.6
# The onboard verifier evaluates every N-th frame of the flight
DETECTION_FRAME_STRIDE = 2

# --- Valen onboard verifier ---
VALEN_CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "Valen-Preview-0923")
# "auto" selects CUDA when available; otherwise e.g. "cuda:0" or "cpu"
VALEN_DEVICE = "auto"
VALEN_ATTENTION_IMPLEMENTATION = "sdpa"
VALEN_CALIBRATION_TEMPERATURE = 1.0
# Frames of all drones are merged into shared GPU batches of up to this size.
# Batched bf16 inference shifts scores by up to ~0.02 compared with one frame at
# a time; set to 1 to reproduce single-frame scores exactly (slower).
VALEN_MAX_BATCH_SIZE = 12
# Upper bound on padded tokens per batch (lower it if the GPU runs out of memory)
VALEN_MAX_BATCH_TOKENS = 8192
# How long the batcher waits for frames of other drones before running a batch
VALEN_BATCH_WAIT_S = 0.02
# Frames a single drone submits at a time, so drones share each batch fairly
VALEN_SUBMIT_CHUNK = 4
# Crops are JPEG-encoded before inference, matching the on-disk evaluation pipeline
VALEN_FRAME_JPEG_QUALITY = 90
