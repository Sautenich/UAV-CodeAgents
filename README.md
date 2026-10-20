# UAV-CodeAgents

[![arXiv](https://img.shields.io/badge/arXiv-2505.07236-b31b1b.svg)](https://arxiv.org/abs/2505.07236)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

UAV-CodeAgents turns one natural-language instruction into a multi-UAV
reconnaissance mission. The operator writes a sentence such as *"I need to
check the houses; one of them might be on fire"*; a ground-station agent plans
the mission by writing and executing Python code, a vision-language model marks
inspection targets on an archival satellite map, the waypoints are split into
balanced routes for N drones, and each drone checks its camera frames with a
small onboard decision model. The operator receives the incident locations with
the supporting frames.

## Demo

<!-- Narrated demo of the multi-UAV version, to be published after 2027-01-13:
[![Demo video: from one sentence to a multi-drone mission](media/demo_preview.jpg)](media/uav_codeagents_demo.mp4)

Two-minute narrated demo of a simulated mission (click the picture to open the
video; captions: [`media/uav_codeagents_demo.srt`](media/uav_codeagents_demo.srt)).
-->

Video of the earlier single-agent version
([arXiv:2505.07236](https://arxiv.org/abs/2505.07236)):

https://github.com/user-attachments/assets/fbf5d73a-65fc-4706-ba0e-b8097f92a239

## How it works

| Step | Component | Where |
|---|---|---|
| 1. Read the request, infer what to inspect | Airspace Management Agent (smolagents `CodeAgent`, ReAct loop) | `src/agent_factory.py` |
| 2. Mark targets on the archival map | `pixelpoint_objects`: VLM grounding on a 1000×1000 grid; objects get one point, areas an even spread | `src/tools.py` |
| 3. Split waypoints into routes | farthest-first allocation balancing the longest route, nearest-neighbour order per drone | `src/agent_factory.py` |
| 4. Fly and collect frames | kinematic simulation over the target map, 128×128 px camera footprint | `src/tools.py` |
| 5. Verify onboard | one UAV agent per drone; Valen-Preview-0923 answers a yes/no question per frame, batched on one GPU | `src/valen_engine.py` |
| 6. Report | coordinates and frames of detections, per-mission CSV and report | `main.py` |

Steps 3–5 run inside one fixed tool, `dispatch_swarm`, so the swarm behaves the
same whichever language model plans the mission.

## Project structure

```
UAV-CodeAgents/
├── main.py                  Experiment runner (CLI): builds the agents, runs a batch of
│                            missions, writes reports, CSV metrics and detection scores
├── config.py                Every setting in one place: gateway and model, swarm size,
│                            scenarios, grounding, flight simulation, onboard verifier
├── requirements.txt         Python dependencies with tested versions
├── .env.example             Template for the API key and gateway URL (copy to .env)
├── LICENSE                  MIT licence of this code
│
├── src/
│   ├── agent_factory.py     Airspace Management Agent and UAV agents, their prompts,
│   │                        route planning (balance_routes, optimize_route) and the
│   │                        dispatch_swarm tool that runs the whole swarm in parallel
│   ├── tools.py             Agent tools: scenario resolution, extract_inspection_targets,
│   │                        pixelpoint_objects (VLM grounding with fabricated-answer
│   │                        filter), keypoint plots, uav_simulation, shared swarm state
│   ├── data_loader.py       Loading reference/target map pairs (local or from GitHub),
│   │                        ground truth and scoring of detections
│   └── valen_engine.py      Onboard verifier: one Valen model shared by all drones with
│                            batched GPU inference, and run_valen_uav_detection, the tool
│                            each UAV agent calls on its camera frames
│
├── valen/                   Valen model code (third party, Apache-2.0, see valen/NOTICE.md)
│
├── data/                    Satellite maps, one folder per scenario
│   ├── default/             Archival reference maps (no incident) used for planning
│   ├── fire/                Target maps with a burning building (30)
│   ├── forest_fire/         Target maps with a synthetic wildfire (5)
│   ├── cars_accident/       Target maps with vehicles stopped at a junction (5)
│   ├── oil_spill/           Coastal target maps with an oil slick (5)
│   ├── hogweed_thickets/    Target maps with hogweed thickets (5)
│   ├── illegal_buildings/   Target maps with new buildings (5)
│   └── ground_truth.json    Incident centres per scenario and map, used for scoring
│
├── tests/                   Unit tests (no API key or GPU needed)
│
├── models/                  Model weights, downloaded during installation (not in git)
└── experiments/             Mission outputs, created when you run (not in git)
```

## Requirements

- Linux (tested on Ubuntu) with Python 3.10 or newer
- NVIDIA GPU with at least 6 GB of memory and a recent driver (tested on an RTX 4050 laptop GPU)
- About 6 GB of disk space for the model weights
- An API key for an OpenAI-compatible gateway that serves the planning model
  (e.g. `gemini-3.8-flash`, `gpt-6.1-sol` or `qwen3.8-max`)

## Installation

**1. Get the code**

```bash
git clone -b 2-zalit-kod-statii https://github.com/Sautenich/UAV-CodeAgents.git
cd UAV-CodeAgents
```

**2. Create a virtual environment and install the dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch does not see your GPU (`python -c "import torch; print(torch.cuda.is_available())"`
prints `False`), install the wheel that matches your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/).

**3. Download the onboard verifier and its base model** (about 5.4 GB in total)

```bash
hf download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
```

**4. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Meaning |
|---|---|
| `LLM_API_KEY` | key of the OpenAI-compatible gateway |
| `LLM_BASE_URL` | gateway root URL, for example `https://ai.starimg.ru/v1` or an OpenRouter endpoint |
| `LLM_MODEL_ID` | default planning model, for example `gemini-3.8-flash` |
| `PHOENIX_ENABLED` | optional, `1` to trace agent runs in Phoenix (see below) |

**5. Check the installation**

```bash
python -m pytest tests
```

All tests should pass; they need neither the API key nor the GPU.

## Running

**One mission.** Inspect structure-fire map 1 with three drones:

```bash
python main.py --scenario fire --drones 3 --start 1 --total 1
```

**A batch.** Run maps 1 to 5 of the oil-spill scenario with another model and
keep the results in their own folder:

```bash
python main.py --scenario oil_spill --model gpt-6.1-sol --drones 3 --start 1 --total 5 --out experiments/oil_gpt
```

**Interactive.** Without `--scenario` the runner asks for a request and picks
the scenario from its wording:

```bash
python main.py
```

| Option | Meaning | Default |
|---|---|---|
| `--scenario` | scenario key (table below) or free request text | asked interactively |
| `--model` | planning model id at the gateway | `LLM_MODEL_ID` from `.env` |
| `--drones` | swarm size | `COUNTER_DRONES` in `config.py` (3) |
| `--start`, `--total` | first and last map index of the batch, inclusive | 1 and 5 |
| `--out` | output folder | `experiments/` |

A mission takes about 2–4 minutes on a laptop GPU. Each one writes to
`<out>/exp_<index>/`:

| File | Content |
|---|---|
| `uav_trajectory.png` | waypoints placed on the reference map |
| `routes.json` | base station and the route of every drone |
| `uav_flight_all_drones.mp4` | simulated flight with the camera footprints |
| `valen_detected_*.png` | reported frames with their score and coordinates |
| `detections.json` | all reported frames per drone |
| `report.txt` | mission summary |

One row per mission is appended to `<out>/experiment_results_<scenario>.csv`
(times, detections and, when the map is in `data/ground_truth.json`, precision
and recall).

## Scenarios

| Scenario key | Map folder | Default request |
|---|---|---|
| `fire` | `data/fire` | I need to check at home; one of them might be on right now fire. |
| `forest_fire` | `data/forest_fire` | A wildfire has been reported. Find where the forest is burning. |
| `oil_spill` | `data/oil_spill` | Detect any signs of an oil spill in the water bodies or on the ground. |
| `car_accident` | `data/cars_accident` | A car accident has occurred. The police are on the scene. |
| `hogweed_thickets` | `data/hogweed_thickets` | Hogweed thickets are growing in the area. |
| `building` | `data/illegal_buildings` | Illegal buildings are being constructed in the area. |

Every mission pairs a reference map, on which the targets are planned, with a
target map of the same area that contains the incident and supplies the camera
frames. Target map `N` of a scenario is paired with `data/default/N_*` unless the
scenario lists its own reference maps in `baseline_map`.

Each scenario in `config.SCENARIOS` defines its keywords, default request, map
folder, the yes/no question asked to the onboard verifier (`valen_question`) and
the probability above which a frame is reported (`detection_threshold`). To add
a scenario, add an entry there and a folder of target maps under `data/`.

## Main settings

| Setting in `config.py` | Meaning | Value |
|---|---|---|
| `COUNTER_DRONES` | default swarm size | 3 |
| `MAX_KEYPOINTS` | most waypoints the VLM may place | 15 |
| `GROUNDING_VOTES` | grounding requests per map; more than 1 cross-checks answers | 1 |
| `CAMERA_CROP_SIZE` | side of the camera footprint, px | 128 |
| `GROUND_SAMPLING_DISTANCE_M` | metres per map pixel | 1.35 |
| `CRUISE_SPEED_M_PER_S` | drone speed, m/s | 6.0 |
| `DETECTION_FRAME_STRIDE` | verify every N-th camera frame | 2 |
| `VALEN_MAX_BATCH_SIZE` | frames per GPU batch of the verifier | 12 |

## Tracing (optional)

Agent runs can be inspected step by step in [Phoenix](https://github.com/Arize-ai/phoenix):

```bash
phoenix serve            # in a separate terminal, UI at http://localhost:6006
```

Then set `PHOENIX_ENABLED=1` in `.env` and run `main.py` as usual.

## Troubleshooting

| Problem | What to do |
|---|---|
| `CUDA out of memory` | lower `VALEN_MAX_BATCH_SIZE` or `VALEN_MAX_BATCH_TOKENS` in `config.py`; close other GPU programs |
| `429 Too Many Requests` from the gateway | the gateway limits parallel requests; run fewer drones or retry later |
| The gateway is reachable only through a proxy | `main.py` ignores proxy variables by default; set `BYPASS_SYSTEM_PROXY = False` in `config.py` and `HTTP_PROXY`/`HTTPS_PROXY` in `.env` |
| `models/... not found` | repeat step 3 of the installation |

## Third-party code

`valen/` is a copy of the `valen` package from
[Liuziyu77/Valen](https://github.com/Liuziyu77/Valen) under the Apache License 2.0;
the model weights come from [Valen-Team](https://huggingface.co/Valen-Team) and
[Qwen](https://huggingface.co/Qwen/Qwen3.5-2B).

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
