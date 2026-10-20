# UAV-CodeAgents

[![arXiv](https://img.shields.io/badge/arXiv-2505.07236-b31b1b.svg)](https://arxiv.org/abs/2505.07236)

UAV-CodeAgents turns one natural-language instruction into a multi-UAV
reconnaissance mission. The operator writes a sentence such as *"I need to
check the houses; one of them might be on fire"*; a ground-station agent plans
the mission by writing and executing Python code, a vision-language model marks
inspection targets on an archival satellite map, the waypoints are split into
balanced routes for N drones, and each drone checks its camera frames with a
small onboard decision model. The operator receives the incident locations with
the supporting frames.

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

## Repository layout

```
main.py              experiment runner (CLI)
config.py            every setting: gateway, models, scenarios, thresholds, simulation
src/
  agent_factory.py   agents, prompts, route planning, dispatch_swarm
  tools.py           scenario resolution, grounding, simulation, visualisation
  data_loader.py     map loading, ground truth, detection scoring
  valen_engine.py    onboard verifier: batched Valen inference shared by all drones
                     and the detection tool the UAV agents call
valen/               Valen model code (third party, Apache-2.0, see valen/NOTICE.md)
data/                reference and target maps per scenario, ground_truth.json
tests/               unit tests (no API key or GPU needed)
```

## Installation

Linux, Python 3.10+, and an NVIDIA GPU with at least 6 GB of memory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the onboard verifier and its base model into `models/`:

```bash
huggingface-cli download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923
huggingface-cli download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
```

Create `.env` from the template and add the key of an OpenAI-compatible gateway:

```bash
cp .env.example .env
```

## Usage

Run five structure-fire missions with three drones:

```bash
python main.py --scenario fire --model gemini-3.8-flash --drones 3 --start 1 --total 5 --out experiments/fire
```

Without `--scenario` the runner asks for a request and picks the scenario from
its wording. Each mission writes to `<out>/exp_<index>/`: the waypoint plot,
`routes.json`, the flight video, the reported frames, `detections.json` and
`report.txt`; one row per mission is appended to `<out>/experiment_results_<scenario>.csv`.

| Scenario key | Map folder | Default request |
|---|---|---|
| `fire` | `data/fire` | I need to check at home; one of them might be on right now fire. |
| `forest_fire` | `data/forest_fire` | A wildfire has been reported. Find where the forest is burning. |
| `oil_spill` | `data/oil_spill` | Detect any signs of an oil spill in the water bodies or on the ground. |
| `car_accident` | `data/cars_accident` | A car accident has occurred. The police are on the scene. |
| `hogweed_thickets` | `data/hogweed_thickets` | Hogweed thickets are growing in the area. |
| `building` | `data/illegal_buildings` | Illegal buildings are being constructed in the area. |

Every mission pairs a reference map (used for planning; `data/default` or the
scenario's `baseline_map` list) with a target map of the same area that contains
the incident. Verifier questions and detection thresholds are set per scenario
in `config.SCENARIOS`; when `data/ground_truth.json` lists the incident, the
runner also scores the detections.

## Tests

```bash
python -m pytest tests
```

## Third-party code

`valen/` is a copy of the `valen` package from
[Liuziyu77/Valen](https://github.com/Liuziyu77/Valen) under the Apache License 2.0.

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
