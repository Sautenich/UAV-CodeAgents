# UAV-CodeAgents

[![arXiv](https://img.shields.io/badge/arXiv-2505.07236-b31b1b.svg)](https://arxiv.org/abs/2505.07236)

This repository contains the official codebase and experimental framework for **UAV-CodeAgents**, a hierarchical multi-agent framework for scalable UAV mission planning and target reconnaissance using Vision-Language Models (VLMs) and the ReAct paradigm.

The video:

https://github.com/user-attachments/assets/fbf5d73a-65fc-4706-ba0e-b8097f92a239

## 📌 Architecture Overview

The system orchestrates autonomous reconnaissance across large geographic areas via a two-tier agent hierarchy:
1. **AirSpace Manager Agent:** Evaluates wide-area satellite orthomosaics, extracts target region centroids via zero-shot VLM prompting on a normalized coordinate grid, and plans global flight paths.
2. **UAV Agent:** Executes localized flight trajectories, conducts continuous camera frame inspection using onboard ONNX classifiers / VLM verification tools, and flags confirmed target coordinates.

---

## 📁 Repository Structure

```
.
├── config.py                 # API keys, endpoint URLs, and model hyperparameters
├── main.py                   # Main entry point to launch mission runs
├── test.py                   # System and pipeline verification scripts
├── tools.py                  # Core agent inspection, coordinate transformation, and plotting tools
├── requirements.txt          # Python runtime dependencies
├── LICENSE                   # Project license
├── data/                     # Operational satellite datasets across mission scenarios
│   ├── cars_accident/        # Vehicle collision monitoring
│   ├── default/              # Nominal baseline landscape patches
│   ├── fire/                 # Wildfire and thermal anomaly scenes
│   ├── hogweed_thickets/     # Invasive agricultural flora detection
│   ├── illegal_buildings/    # Unauthorized structure and construction monitoring
│   ├── oil_spill/            # Industrial spill mapping
│   ├── unauthorized_festival/# Mass gathering and unauthorized event surveillance
│   └── water_source/         # Water body mapping
├── drone_models/             # detection weight  
├── src/                      # Core framework implementations
│   ├── agent_factory.py      # ReAct agent initialization and system prompts
│   ├── commands.py           # UAV trajectory commands and control actions
│   ├── data_loader.py        # Dataset streaming and map tiling pipeline
│   ├── tools.py              # Primary reconnaissance and detection tools
│   ├── tools_Gemini.py       # Gemini API tool variants
│   ├── weight_tools.py       # Local neural network weight loaders
│   └── assets/               # Simulation sprites and visualization overlays
├── experiments/              # Output experiment directories
└── uav_logs/                 # Intermediate runtime plots and trajectories
```

## ⚙️ Installation

## 1. Clone RepositoryBash
git clone https://github.com/Sautenich/UAV-CodeAgents.git

cd UAV-CodeAgents

## 2. Environment SetupBash
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

## 3. API Credentials
Configure environment variables for model inference in config.py:

FIREWORKS_API_KEY="your_fireworks_api_key"

MODEL_ID="gemini-3.8-flash"

OPENROUTER_BASE_URL="your_URL_api"

## 🚀 Running MissionsRun autonomous mission planning and reconnaissance across scenarios:
## Start code
python3 main.py

## Scenario selection

for example: I need to check at home, one of them might be on fire right now.

<div align="center">

### TABLE I: TOOLS AND FUNCTIONALITIES OF CODEAGENT AND UAV AGENT

<table>
  <thead>
    <tr style="background-color: #e6e6e6;">
      <th align="left">Agent</th>
      <th align="left">Tool Name</th>
      <th align="left">Description and Arguments</th>
    </tr>
  </thead>
  <tbody>
    <!-- Airspace Manager Agent -->
    <tr>
      <td rowspan="5"><b>Airspace Manager Agent</b></td>
      <td><code>read_image</code></td>
      <td>Returns a PIL image object for airspace analysis.<br><i>Arguments:</i> <code>i</code> (integer) - image index</td>
    </tr>
    <tr>
      <td><code>describe_satellite_image</code></td>
      <td>Analyzes image using Qwen2.5-VL via Fireworks API.<br><i>Arguments:</i> <code>image</code> (PIL.Image) - image to process</td>
    </tr>
    <tr>
      <td><code>pixelpoint_objects</code></td>
      <td>Extracts objects with Qwen2.5-VL, auto-repairs JSON.<br><i>Arguments:</i> <code>image</code> (PIL.Image), <code>objects</code> (string)</td>
    </tr>
    <tr>
      <td><code>visualize_keypoints</code></td>
      <td>Renders labeled objects on image with bounding boxes.<br><i>Arguments:</i> <code>image</code> (PIL.Image), <code>keypoints</code> (list)</td>
    </tr>
    <tr>
      <td><code>final_answer</code></td>
      <td>Delivers processed results to user.<br><i>Arguments:</i> <code>answer</code> (any type)</td>
    </tr>
    <!-- UAV Agent -->
    <tr>
      <td rowspan="4"><b>UAV Agent</b></td>
      <td><code>read_image_for_simulation</code></td>
      <td>Provides image for UAV flight simulation.<br><i>Arguments:</i> <code>i</code> (integer) - simulation index</td>
    </tr>
    <tr>
      <td><code>uav_simulation</code></td>
      <td>Simulates flight path and captures frames.<br><i>Arguments:</i> <code>image</code> (PIL.Image), <code>labeled_points</code> (list)</td>
    </tr>
    <tr>
      <td><code>detect_and_display</code></td>
      <td>Identifies fire locations in UAV footage.<br><i>Arguments:</i> <code>frames_dict</code> (dictionary)</td>
    </tr>
    <tr>
      <td><code>final_answer</code></td>
      <td>Returns detection results.<br><i>Arguments:</i> <code>answer</code> (any type)</td>
    </tr>
  </tbody>
</table>

</div>

## 🛠️ Key Agent Tools
fetch_images_from_github: load dataset from GitHub.

load_local_map_images: load dataset from local folder.

download_and_sync_weights: checks for the presence of the weights required for the mission.

extract_inspection_targets: identifies keywords for searching within the satellite image.

pixelpoint_objects: sends a request to generate coordinates for the drone at key locations.

visualize_keypoints_from_image: visualizes the placed points.

uav_simulation: create visualize simulation of drones fly.

detect_and_display: analyzes for fire using an LLM.

run_local_uav_detection: analyzes for fire using an weight.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@misc{sautenkov2025uavcodeagentsscalableuavmission,
      title={UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning}, 
      author={Oleg Sautenkov and Yasheerah Yaqoot and Muhammad Ahsan Mustafa and Faryal Batool and Jeffrin Sam and Artem Lykov and Chih-Yung Wen and Dzmitry Tsetserukou},
      year={2025},
      eprint={2505.07236},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.07236}, 
}
```

