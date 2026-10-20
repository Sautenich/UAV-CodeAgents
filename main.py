import argparse
import csv
import json
import os
import random
import time
import urllib.request

import config

# Clean up proxy environment variables BEFORE importing litellm, phoenix, httpx, requests
if config.BYPASS_SYSTEM_PROXY:
    for key in [
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "socks_proxy",
        "SOCKS_PROXY",
    ]:
        os.environ.pop(key, None)

    urllib.request.getproxies = lambda: {}
    os.environ["NO_PROXY"] = config.NO_PROXY_HOSTS
    os.environ["no_proxy"] = config.NO_PROXY_HOSTS

# Disable GUI display attempts (prevents Tkinter / Tcl core dumps in headless environments)
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Prevent LiteLLM hangs during model cost map downloads
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
import matplotlib
from smolagents import LiteLLMModel

from src.agent_factory import create_airspace_manager, create_uav_agent
from src.data_loader import load_ground_truth, score_detections
import src.tools as tools
from src.tools import resolve_scenario

litellm.request_timeout = config.LITELLM_REQUEST_TIMEOUT_S
litellm.num_retries = config.LLM_NUM_RETRIES
litellm.retry_strategy = config.LLM_RETRY_STRATEGY

# Register OTLP exporter for Arize Phoenix and attach the smolagents instrumentation hook
tracer_provider = None
if config.PHOENIX_ENABLED:
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    from phoenix.otel import register

    tracer_provider = register(
        project_name=config.PHOENIX_PROJECT_NAME,
        endpoint=config.PHOENIX_ENDPOINT,
    )
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

# Disable Tkinter GUI engine to render figures directly in memory
matplotlib.use("Agg")

CSV_FIELDNAMES = [
    "Experiment Number",
    "Home Point",
    "Folder",
    "Scenario",
    "Model",
    "Drones",
    "Detection Threshold",
    "Request Text",
    "Elapsed Time",
    "Agent Time (s)",
    "Flight Time (s)",
    "Total Mission Time (s)",
    "Status",
    "Incident Detected",
    "Detections Count",
    "Detections Per Drone",
    "Detected Coordinates",
    "True Positives",
    "False Positives",
    "False Negatives",
    "Precision",
    "Recall",
    "Outcome",
]


def prepare_results_csv(csv_file_path: str) -> None:
    """Creates the results CSV, setting aside a file written with other columns."""
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
            header = next(csv.reader(file), None)
        if header == CSV_FIELDNAMES:
            return
        backup_path = f"{csv_file_path}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
        os.replace(csv_file_path, backup_path)
        print(f"ℹ️ Existing results use a different column layout; moved to {backup_path}")

    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_FIELDNAMES).writeheader()


def pick_home_point(experiment_index: int) -> tuple:
    """Places the base station on a random map border, reproducibly per experiment."""
    rng = random.Random(config.RANDOM_SEED + experiment_index)
    low, high = config.HOME_POINT_MIN, config.HOME_POINT_MAX
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return (rng.randint(low, high), low)
    if side == "bottom":
        return (rng.randint(low, high), high)
    if side == "left":
        return (low, rng.randint(low, high))
    return (high, rng.randint(low, high))


def _fmt(value) -> str:
    """Formats an optional metric for CSV / report output."""
    if value is None:
        return ""
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def parse_args() -> argparse.Namespace:
    """Command-line overrides of config.py, for unattended batches of experiments."""
    parser = argparse.ArgumentParser(description="Run UAV-CodeAgents experiments.")
    parser.add_argument("--scenario", help="scenario key or request text; asked interactively when omitted")
    parser.add_argument("--model", help="LLM model id (default: config.MODEL_ID)")
    parser.add_argument("--drones", type=int, help="swarm size (default: config.COUNTER_DRONES)")
    parser.add_argument("--start", type=int, help="first experiment index")
    parser.add_argument("--total", type=int, help="last experiment index")
    parser.add_argument("--out", help="output directory (default: config.EXPERIMENTS_DIR)")
    args = parser.parse_args()

    if args.model:
        config.MODEL_ID = args.model
    if args.drones:
        config.COUNTER_DRONES = args.drones
    if args.start:
        config.EXPERIMENT_START_INDEX = args.start
    if args.total:
        config.TOTAL_EXPERIMENTS = args.total
    if args.out:
        config.EXPERIMENTS_DIR = args.out
    return args


def main():
    args = parse_args()

    # 1. Initialize LLM via OpenAI-compatible proxy gateway
    model = LiteLLMModel(
        model_id=f"openai/{config.MODEL_ID}",
        api_key=config.LLM_API_KEY,
        api_base=config.LLM_BASE_URL,
        timeout=config.LLM_TIMEOUT_S,
        request_timeout=config.LLM_TIMEOUT_S,
    )

    counter_drones = config.COUNTER_DRONES

    # 2. Initialize swarm agents
    uav_agents = [
        create_uav_agent(model, drone_id=i) for i in range(counter_drones)
    ]
    airspace_manager_agent = create_airspace_manager(model, uav_agents)

    # 3. Scenario resolution
    print("list of scenarios:", list(config.SCENARIOS.keys()))
    if args.scenario is not None:
        user_input = args.scenario.strip()
    else:
        user_input = input(
            f"Enter key or query text (Press Enter for default {config.DEFAULT_SCENARIO} scenario): "
        ).strip()

    scenario = resolve_scenario(user_input)

    print(f"\n✅ Scenario: '{scenario['incident_type']}'")
    print(f"📁 Folder: {os.path.join(config.DATA_DIR, scenario['folder'])}")
    print(f"🎯 Target Object: {scenario['target_object']}")
    print(f"🧠 Model: {config.MODEL_ID} | 🚁 Drones: {config.COUNTER_DRONES}")
    print(f"📝 Request: {scenario['request']}\n")

    # 4. Batch experiment setup
    total_experiments = config.TOTAL_EXPERIMENTS
    base_experiments_dir = config.EXPERIMENTS_DIR
    os.makedirs(base_experiments_dir, exist_ok=True)

    csv_file_path = os.path.join(
        base_experiments_dir, f"experiment_results_{scenario['incident_type']}.csv"
    )
    prepare_results_csv(csv_file_path)

    for i in range(config.EXPERIMENT_START_INDEX, total_experiments + 1):
        exp_folder = os.path.join(base_experiments_dir, f"exp_{i}")
        os.makedirs(exp_folder, exist_ok=True)
        os.environ["CURRENT_EXP_DIR"] = exp_folder
        tools.reset_swarm_state()

        home_point = pick_home_point(i)

        print(f"\n=== 🏁 START EXPERIMENT №{i} / {total_experiments} ===")
        print(f"📁 Output Directory: {exp_folder}")
        start_time = time.time()

        user_request_text = f"{scenario['request']} Inspect target map index #{i}."

        if hasattr(airspace_manager_agent, "memory"):
            airspace_manager_agent.memory.reset()
        for drone in uav_agents:
            if hasattr(drone, "memory"):
                drone.memory.reset()

        # Mission inputs are preloaded as variables of the manager's Python environment
        manager_state = airspace_manager_agent.python_executor.state
        manager_state["output_dir"] = exp_folder
        manager_state["user_request"] = scenario["request"]
        manager_state["home_point"] = home_point

        task_prompt = f"""
    You are an AI agent managing a UAV flight inspection system.
    
    Available tools for reference:
        uav_agents:
        tools=[
            run_valen_uav_detection
        ]
    
        airspace_manager:
        tools=[
            get_folder_by_query,
            pixelpoint_objects,
            visualize_keypoints_from_image,
            fetch_images_from_github,
            extract_inspection_targets,
            load_local_map_images,
            dispatch_swarm
        ]

    CRITICAL OPERATIONAL RULES (STRICT COMPLIANCE):
    - Every action must strictly contain Python code enclosed between <code> and </code> tags. 
    - Do not mix natural language thoughts inside the <code> block.
    - Do NOT save intermediate variables to JSON and do NOT use the `open()` function.
    - Coordinate System: ALL coordinates (`keypoints` returned by `pixelpoint_objects`, `home_point`, detections) are PIXEL coordinates of the map images, where (x=0, y=0) is the top-left corner. They are NOT normalized.
    - NEVER rescale, normalize, round, clip, reorder or otherwise modify `keypoints` or `home_point`. Pass them to `visualize_keypoints_from_image` and `dispatch_swarm` exactly as you received them.
    - Keep all variables (keypoints, folder_info, images) in RAM! They are automatically preserved between code execution stages.
    - Do NOT use `open()`, `eval()`, or `exec()` functions.
    - Do NOT use Python introspection functions such as `globals()`, `locals()`, `dir()`, `eval()`, or `exec()`.
    - Use only functions already defined within the created agents.
    - Do NOT run the entire workflow inside a single massive Python code block.
    - Output ONLY 1 short code block per stage, verify the result, and then proceed to the next stage.
    - You MUST perform actions by WRITING AND EXECUTING CORRECT PYTHON CODE.
    - Each stage MUST contain a Python code block in Markdown format:
    ```python

    - Do NOT write text-based plans, overviews, or summaries without code.
    - Always use executable code blocks to call functions.

    TASK CONFIGURATION:
    User request: "{scenario['request']}"
    Experiment index: {i}
    counter_drones: {counter_drones}
    GitHub configuration: Repo_owner="{config.GITHUB_REPO_OWNER}", repo_name="{config.GITHUB_REPO_NAME}", branch="{config.GITHUB_BRANCH}"

    PRELOADED VARIABLES (already defined in your Python environment, use them directly):
    - `user_request` = "{scenario['request']}"
    - `home_point` = {home_point} (pixel coordinates of the base station)
    - `output_dir` = "{exp_folder}"

    CRITICAL WORKFLOW RULES:
    1. IMAGES ROLES:
    - `load_local_map_images()` returns a tuple: `base_image, target_image`.
    - `base_image` is an ARCHIVAL reference map without incidents. Use it ONLY with `pixelpoint_objects` to find infrastructure/waypoints (e.g. houses, buildings, roads).
    - `target_image` represents the REAL-TIME physical ground with the ongoing incident. Pass it ONLY to `dispatch_swarm()`, which generates the drone camera feed.
    - NEVER call `pixelpoint_objects` on `target_image`! That is cheating and strictly forbidden.
    - In `pixelpoint_objects`, do NOT search for the disaster itself (e.g., do NOT search for 'fire'). Search for candidate structures to inspect (e.g., 'residential buildings', 'houses').
    - DO NOT inspect function internals, source code, or dunder attributes (e.g., `__code__`, `__globals__`, `dir()`).
    - Call tools directly with the specified signature without introspecting them.

    2. TASK EXECUTION SEQUENCE:
    Step 1: Extract candidate inspection structures using `inspection_targets = extract_inspection_targets(user_request)`. (These must be structures/places to check, e.g. buildings/forest).
    Step 2: Determine scenario details: `folder_info = get_folder_by_query(user_request)` (a dict with 'folder', 'incident_type' and 'target_object').
    Step 3: Load map images for the experiment:
        `base_image, target_image = load_local_map_images(folder_name=folder_info['folder'], index={i})`
    Step 4: Extract keypoints (pixel coordinates, use them unchanged) from archival base map: `keypoints = pixelpoint_objects(image=base_image, objects=inspection_targets)`
    Step 5: Show the flight path and keypoints on the base map using `visualize_keypoints_from_image(image=base_image, keypoints=keypoints, output_dir=output_dir)`.
    Step 6: Dispatch the swarm. `dispatch_swarm` partitions the keypoints between all drones, simulates each flight, runs reconnaissance scanning on every drone in parallel, and returns the mission summary. Then submit the report:

    ```python
    mission = dispatch_swarm(
        target_image=target_image,
        keypoints=keypoints,
        incident_type=folder_info["incident_type"],
        target_object=folder_info["target_object"],
        home_point=home_point,
        output_dir=output_dir,
    )
    print(mission)

    final_answer(
        f"Multi-UAV reconnaissance successfully executed. Incident detected: {{mission['incident_detected']}}. "
        f"Detections per drone: {{mission['detections']}}"
    )
    ```

    Try to complete the task and detect incidents—after all, you could save lives!
    """

        status = "success"
        try:
            experiment_outcome = airspace_manager_agent.run(task_prompt)
        except Exception as e:
            status = "failed"
            experiment_outcome = f"Execution failed: {str(e)}"
            print(f"❌ Error on experiment #{i}: {e}")

        elapsed_time = time.time() - start_time
        print(f"⏱️ Experiment {i} completed in {elapsed_time:.2f}s.")

        # time mission
        flight_times = dict(tools.SWARM_FLIGHT_TIMES)
        flight_time = max(flight_times.values()) if flight_times else 0.0
        total_time = elapsed_time + flight_time
        agent_time = time.time() - start_time

        print(
            f"⏱️ Experiment {i} completed in {elapsed_time:.2f}s (Agent) + {flight_time:.2f}s (Flight) = {total_time:.2f}s Total."
        )

        # Detection results recorded by the onboard detectors of every drone
        detections = tools.get_swarm_detections()
        detections_per_drone = {
            f"uav_agent_{drone_id}": len(items)
            for drone_id, items in sorted(detections.items())
        }
        detected_coords = [
            list(item["coords"])
            for _, items in sorted(detections.items())
            for item in items
        ]
        incident_detected = bool(detected_coords)
        if status == "success" and not flight_times:
            # The agent finished without ever dispatching the swarm
            status = "no_flight"

        truth_points = load_ground_truth(scenario["incident_type"], i)
        quality = score_detections(detected_coords, truth_points)
        outcome_text = " ".join(str(experiment_outcome).split())

        print(
            f"🎯 Experiment {i}: status={status}, incident detected={incident_detected},"
            f" detections={len(detected_coords)}"
        )

        # Save individual experiment report to its designated folder
        report_file_path = os.path.join(exp_folder, "report.txt")
        with open(report_file_path, mode="w", encoding="utf-8") as rf:
            rf.write(f"Experiment Index: {i}\n")
            rf.write(f"Artifacts Folder: {exp_folder}\n")
            rf.write(f"Scenario: {scenario['incident_type']}\n")
            rf.write(f"Request: {user_request_text}\n")
            rf.write(f"Home Point: {home_point}\n")
            rf.write(f"Elapsed Time: {elapsed_time:.2f} s\n")
            rf.write(f"Agent Time: {agent_time:.2f} s\n")
            rf.write(f"Flight Time: {flight_time:.2f} s\n")
            rf.write(f"Total Mission Time: {total_time:.2f} s\n")
            rf.write(f"Status: {status}\n")
            rf.write(f"Incident Detected: {incident_detected}\n")
            rf.write(f"Detections Count: {len(detected_coords)}\n")
            rf.write(f"Detections Per Drone: {json.dumps(detections_per_drone)}\n")
            rf.write(f"Detected Coordinates: {json.dumps(detected_coords)}\n")
            if truth_points is not None:
                rf.write(f"Ground Truth Incidents: {json.dumps(truth_points)}\n")
                rf.write(f"True Positives: {_fmt(quality['true_positives'])}\n")
                rf.write(f"False Positives: {_fmt(quality['false_positives'])}\n")
                rf.write(f"False Negatives: {_fmt(quality['false_negatives'])}\n")
                rf.write(f"Precision: {_fmt(quality['precision'])}\n")
                rf.write(f"Recall: {_fmt(quality['recall'])}\n")
            rf.write(f"Outcome: {experiment_outcome}\n")

        # Full per-frame detections (frame id, coordinates, probability) per drone
        with open(
            os.path.join(exp_folder, "detections.json"), mode="w", encoding="utf-8"
        ) as df:
            json.dump(
                {f"uav_agent_{drone_id}": items for drone_id, items in detections.items()},
                df,
                indent=2,
            )

        record = {
            "Experiment Number": i,
            "Home Point": str(home_point),
            "Folder": exp_folder,
            "Scenario": scenario["incident_type"],
            "Model": config.MODEL_ID,
            "Drones": counter_drones,
            "Detection Threshold": tools.detection_threshold(scenario["incident_type"]),
            "Request Text": user_request_text,
            "Elapsed Time": f"{elapsed_time:.2f}",
            "Agent Time (s)": f"{agent_time:.2f}",
            "Flight Time (s)": f"{flight_time:.2f}",
            "Total Mission Time (s)": f"{total_time:.2f}",
            "Status": status,
            "Incident Detected": incident_detected,
            "Detections Count": len(detected_coords),
            "Detections Per Drone": json.dumps(detections_per_drone),
            "Detected Coordinates": json.dumps(detected_coords),
            "True Positives": _fmt(quality["true_positives"]),
            "False Positives": _fmt(quality["false_positives"]),
            "False Negatives": _fmt(quality["false_negatives"]),
            "Precision": _fmt(quality["precision"]),
            "Recall": _fmt(quality["recall"]),
            "Outcome": outcome_text,
        }

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
            writer.writerow(record)

        if tracer_provider is not None:
            try:
                tracer_provider.force_flush()
            except Exception:
                pass

        time.sleep(config.PAUSE_BETWEEN_EXPERIMENTS_S)

    print(
        f"\n📊 All {total_experiments} experiments completed! Results logged in {csv_file_path}"
    )


if __name__ == "__main__":
    main()
