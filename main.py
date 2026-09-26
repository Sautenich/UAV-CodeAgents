import csv
import io
import os
import time
import urllib.request
from dotenv import load_dotenv
import httpx
import litellm
import matplotlib
from smolagents import ToolCallingAgent
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from phoenix.otel import register
import requests
from smolagents import CodeAgent, LiteLLMModel
import src.tools as tools
from src.agent_factory import create_airspace_manager, create_uav_agent
import config
import random
from src.data_loader import fetch_images_from_github, load_local_map_images
from src.tools import (
    describe_satellite_image,
    detect_and_display,
    extract_inspection_targets,
    get_folder_by_query,
    pixelpoint_objects,
    uav_simulation,
    visualize_keypoints_from_image,
     SWARM_FLIGHT_TIMES 

)
from src.tools_Gemini import (
    describe_satellite_image_Gemini,
    detect_and_display_Gemini,
    extract_inspection_targets_Gemini,
    pixelpoint_objects_Gemini,
)
from src.weight_tools import download_and_sync_weights, run_local_uav_detection

# 1. Load .env environment variables
load_dotenv(override=True)

# , simulate realistic flight for each drone with GIF generation

# Clean up proxy environment variables BEFORE importing phoenix, httpx, requests, urllib
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

# 1. Disable GUI display attempts (prevents Tkinter / Tcl core dumps in headless environments)
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 2. Prevent LiteLLM hangs during model cost map downloads
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

urllib.request.getproxies = lambda: {}
litellm.request_timeout = 180.0

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

# 2. Register OTLP exporter for Arize Phoenix
tracer_provider = register(
    project_name="uav_agent",
    endpoint="http://localhost:6006/v1/traces",
)

# 3. Attach smolagents instrumentation hook
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
litellm.num_retries = 5
litellm.retry_strategy = "exponential_backoff_retry"

# Disable Tkinter GUI engine to render figures directly in memory
matplotlib.use("Agg")
# litellm._turn_on_debug()  # Enables verbose logging for API calls and network errors

COMMANDS = {
    "fire": {
        "keywords": ["fire", "smoke", "flame", "burn", "burning"],
        "request": (
            "I need to check at home; one of them might be on right now fire."
        ),
        "folder": "fire",
        "target_object": "fire",
        "incident_type": "fire",
    },
    "oil_spill": {
        "keywords": ["oil", "spill", "leak", "slick", "pollution"],
        "request": (
            "Detect any signs of an oil spill in the water bodies or on the"
            " ground."
        ),
        "folder": "oil_spill",
        "target_object": "oil spill",
        "incident_type": "oil_spill",
    },
    "car_accident": {
        "keywords": ["car", "accident", "crash", "collision", "vehicle"],
        "request": "A car accident has occurred. The police are on the scene.",
        "folder": "cars_accident",
        "target_object": "car accident",
        "incident_type": "car_accident",
    },
    "hogweed_thickets": {
        "keywords": ["hogweed", "weed", "thicket", "heracleum"],
        "request": "Hogweed thickets are growing in the area.",
        "folder": "hogweed_thickets",
        "target_object": "hogweed thickets",
        "incident_type": "hogweed_thickets",
    },
    "illegal_buildings": {
        "keywords": [
            "illegal",
            "building",
            "construction",
            "unauthorized",
            "shed",
        ],
        "request": "Illegal buildings are being constructed in the area.",
        "folder": "illegal_buildings",
        "target_object": "illegal building",
        "incident_type": "illegal_buildings",
    },
}


def resolve_scenario(user_input: str, commands: dict) -> dict:
    raw_text = user_input.strip().lower()

    # 1. Default to 'fire' if input is empty (user pressed Enter)
    if not raw_text:
        return commands["fire"].copy()

    # 2. Direct match with dictionary keys (e.g. 'fire' or 'oil_spill')
    if raw_text in commands:
        return commands[raw_text].copy()

    # 3. Match keywords inside the entered sentence
    for key, data in commands.items():
        for kw in data.get("keywords", []):
            # Check keyword presence as a standalone marker
            if kw in raw_text:
                matched_scenario = data.copy()
                # Preserve the extended sentence as the specific user prompt
                if len(user_input.strip().split()) > 1:
                    matched_scenario["request"] = user_input.strip()
                return matched_scenario

    # 4. Fallback: default to 'fire' if no scenario matched
    print(
        f"⚠️ Keywords not recognized in '{user_input}'. Defaulting to 'fire'"
        " scenario."
    )
    default_scenario = commands["fire"].copy()
    if len(user_input.strip().split()) > 1:
        default_scenario["request"] = user_input.strip()
    return default_scenario


def main():
    # """
    # Debug workflow to validate individual tools and pipeline components
    # within a single script without invoking full multi-agent orchestration.
    # """

    # selected_key = input("Enter scenario: ").strip()

    # target_object = extract_inspection_targets(selected_key)
    # print(target_object)

    # # 1. Scenario selection
    # # print("Available scenarios:", list(COMMANDS.keys()))
    # # selected_key = input("Enter scenario key (or press Enter for 'fire'): ").strip()
    # if not selected_key or selected_key not in COMMANDS:
    #     selected_key = "fire"
    # user_request = COMMANDS[selected_key]
    # print(f"\n📝 Selected prompt: '{user_request}'")

    # base_path = "data"

    # scenario = COMMANDS[selected_key]
    # user_request = scenario["request"]
    # folder_name = scenario["folder"]
    # # target_object = scenario["target_object"]
    # incident_type = scenario["incident_type"]

    # print(f"\n📝 Selected prompt: '{user_request}'")
    # print(f"📊 Parameters: Folder='{folder_name}', Target='{target_object}', Incident='{incident_type}'")

    # # 2. Download / Synchronize model weights
    # print(f"\n--- STEP 1: Syncing weights for incident '{incident_type}' ---")
    # weights_path = download_and_sync_weights(incident_type)
    # print(f"📍 Onboard weights path: {weights_path}")

    # # 3. Load regional map images
    # print(f"\n--- STEP 2: Loading terrain imagery ---")

    # base_data_path = f"data/{folder_name}"

    # if os.path.exists("data/default") and os.path.exists(base_data_path):
    #     print("📥 Loading maps from local directories...")
    #     blank_tuple_list = load_local_images("data/default")
    #     target_tuple_list = load_local_images(base_data_path)

    #     base_image = blank_tuple_list[0][0]
    #     target_image = target_tuple_list[0][0]
    # else:
    #     print("🌐 Fetching maps from GitHub repository...")
    #     settings_blank = config.REPO_SETTINGS.get("default", {})
    #     settings_target = config.REPO_SETTINGS.get(folder_name, config.REPO_SETTINGS.get("default", {}))

    #     blank_images = fetch_images_from_github(
    #         repo_owner=settings_blank.get("owner", "grishakalinin2014-alt"),
    #         repo_name=settings_blank.get("repo", "UAV-Agent"),
    #         folder_path=settings_blank.get("folder", "images/blank"),
    #         branch=settings_blank.get("branch", "main")
    #     )

    #     target_images = fetch_images_from_github(
    #         repo_owner=settings_target.get("owner", "grishakalinin2014-alt"),
    #         repo_name=settings_target.get("repo", "UAV-Agent"),
    #         folder_path=settings_target.get("folder", f"images/{folder_name}"),
    #         branch=settings_target.get("branch", "main")
    #     )

    #     base_image = blank_images[0]
    #     target_image = target_images[0]

    # # 4. Detect waypoints for flight trajectory
    # print(f"\n--- STEP 3: Detecting reference structures on baseline map ---")
    # # objects_to_find = resolve_objects_from_query(user_request)
    # print(f"Target structures: {target_object}")

    # keypoints = pixelpoint_objects_Gemini(image=base_image, objects=target_object) # objects=objects_to_find
    # print(f"Detected keypoints count: {len(keypoints)}")
    # print(f"Keypoint coordinates: {keypoints}")

    # if not keypoints:
    #     print("⚠️ No keypoints detected. Falling back to test coordinates.")
    #     keypoints = [
    #         {'point_2d': [180, 220], 'label': 'CheckPoint_1'},
    #         {'point_2d': [450, 150], 'label': 'CheckPoint_2'},
    #         {'point_2d': [820, 280], 'label': 'CheckPoint_3'},
    #         {'point_2d': [290, 480], 'label': 'CheckPoint_4'},
    #         {'point_2d': [610, 420], 'label': 'CheckPoint_5'},
    #         {'point_2d': [880, 560], 'label': 'CheckPoint_6'},
    #         {'point_2d': [150, 750], 'label': 'CheckPoint_7'},
    #         {'point_2d': [480, 710], 'label': 'CheckPoint_8'},
    #         {'point_2d': [730, 830], 'label': 'CheckPoint_9'},
    #         {'point_2d': [340, 910], 'label': 'CheckPoint_10'}
    #     ]

    # # 5. Flight path visualization
    # print(f"\n--- STEP 4: Visualizing UAV flight trajectory ---")
    # visualize_keypoints_from_image(image=base_image, keypoints=keypoints)

    # # 6. Flight simulation and frame generation
    # print(f"\n--- STEP 5: Simulating flight and cropping inspection frames ---")
    # frames_dict = uav_simulation(image=target_image, labeled_points=keypoints)
    # print(f"Generated inspection frames: {len(frames_dict)}")

    # # 7. Onboard local model inference
    # print(f"\n--- STEP 6: Executing local onboard model inference ---")
    # detected_coords = run_local_uav_detection(
    #     frames_dict=frames_dict,
    #     weights_path=weights_path,
    #     target_object=incident_type
    # )

    # # 8. Summary
    # print("\n" + "=" * 50)
    # if detected_coords:
    #     print(f"🎯 MISSION SUCCESS: Target '{incident_type}' detected at coordinates {detected_coords}")
    # else:
    #     print(f"✅ MISSION COMPLETE: No occurrences of '{incident_type}' found across frames.")
    # print("=" * 50)

    # """
    # Core multi-agent execution pipeline orchestrated by Gemini and smolagents:
    # - Dynamically loads maps and ground footage.
    # - Instructs Airspace Manager to parse incoming queries and plan flight paths.
    # - Partitions and delegates inspection sectors across subordinate UAV agents.
    # - Gathers and exports detection metrics to CSV.
    # """

    # litellm.suppress_debug_info = True

    # Register model without 'gemini/' prefix as smolagents expects 'gemini-3.7-flash'
    # litellm.register_model({
    #     "gemini-3.7-flash": {
    #         "max_tokens": 8192,
    #         "input_cost_per_token": 0.000000075,
    #         "output_cost_per_token": 0.0000003,
    #         "litellm_provider": "gemini",
    #         "mode": "chat",
    #     },
    #     "gemini/gemini-3.7-flash": {
    #         "max_tokens": 8192,
    #         "input_cost_per_token": 0.000000075,
    #         "output_cost_per_token": 0.0000003,
    #         "litellm_provider": "gemini",
    #         "mode": "chat",
    #     }
    # })

    litellm.num_retries = 5
    litellm.retry_strategy = "exponential_backoff_retry"

    # 1. Initialize LLM via OpenAI-compatible proxy gateway
    model = LiteLLMModel(
        model_id=f"openai/{config.MODEL_ID}",
        api_key=config.FIREWORKS_API_KEY,
        api_base="https://ai.starimg.ru/v1",
        timeout=500,
        request_timeout=500,
    )

    counter_drones = config.COUNTER_DRONES

    # 2. Initialize swarm agents
    uav_agents = [
        create_uav_agent(model, drone_id=i) for i in range(counter_drones)
    ]
    airspace_manager_agent = create_airspace_manager(model, uav_agents)
    drone_names = [agent.name for agent in uav_agents]
    drones_code_repr = f"[{', '.join(drone_names)}]"

    # 3. Scenario resolution
    print("list of scenarios:", list(COMMANDS.keys()))
    user_input = input(
        "Enter key or query text (Press Enter for default fire scenario): "
    ).strip()

    # Automatically resolve scenario by keywords
    scenario = resolve_scenario(user_input, COMMANDS)

    print(f"\n✅ Scenario: '{scenario['incident_type']}'")
    print(f"📁 Folder: data/{scenario['folder']}")
    print(f"🎯 Target Object: {scenario['target_object']}")
    print(f"📝 Request: {scenario['request']}\n")

    repo_owner = "grishakalinin2014-alt"
    repo_name = "UAV-Agent"
    branch = "main"

    # 4. Batch experiment setup
    total_experiments = 30
    base_experiments_dir = "experiments"
    os.makedirs(base_experiments_dir, exist_ok=True)

    csv_file_path = os.path.join(
        base_experiments_dir, f"experiment_results_{scenario['incident_type']}.csv"
    )
    fieldnames = [
        "Experiment Number",
        "Home Point",
        "Folder",
        "Scenario",
        "Request Text",
        "Elapsed Time",
        "Agent Time (s)", 
        "Flight Time (s)",  
        "Total Mission Time (s)",
        # "Outcome",
    ]

    # Initialize CSV file with headers if it does not exist
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    # 4. Run experiment iteration
    for i in range(20, total_experiments + 1):
        exp_folder = os.path.join(base_experiments_dir, f"exp_{i}")
        os.makedirs(exp_folder, exist_ok=True)
        os.environ["CURRENT_EXP_DIR"] = exp_folder
        tools._SWARM_TRAJECTORIES.clear()
        tools.SWARM_FLIGHT_TIMES.clear()

        rng = random.Random(42 + i)
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            home_point = (rng.randint(60, 950), 60)
        elif side == "bottom":
            home_point = (rng.randint(60, 950), 950)
        elif side == "left":
            home_point = (60, rng.randint(60, 950))
        else:  
            home_point = (950, rng.randint(60, 950))

        print(f"\n=== 🏁 START EXPERIMENT №{i} / {total_experiments} ===")
        print(f"📁 Output Directory: {exp_folder}")
        start_time = time.time()

        user_request_text = f"{scenario['request']} Inspect target map index #{i}."

        # Reset memory state to prevent token context accumulation across runs
        if hasattr(airspace_manager_agent, "memory"):
            airspace_manager_agent.memory.reset()
        for drone in uav_agents:
            if hasattr(drone, "memory"):
                drone.memory.reset()

        # Invalidate cached executor state for the current run
        airspace_manager_agent.python_executor.state["output_dir"] = exp_folder

        task_prompt = f"""
    You are an AI agent managing a UAV flight inspection system.
    
    Available tools for reference:
        uav_agents:
        tools=[
            run_local_uav_detection,
            detect_and_display
        ]
    
        airspace_manager:
        tools=[
            get_folder_by_query,
            pixelpoint_objects,
            visualize_keypoints_from_image,
            fetch_images_from_github,
            download_and_sync_weights,
            extract_inspection_targets,
            load_local_map_images
        ]

    CRITICAL OPERATIONAL RULES (STRICT COMPLIANCE):
    - Every action must strictly contain Python code enclosed between <code> and </code> tags. 
    - Do not mix natural language thoughts inside the <code> block.
    - Do NOT save intermediate variables to JSON and do NOT use the `open()` function.
    - Coordinate System: Use a strict normalized integer grid from 0 to 1000, where (x=0, y=0) is the absolute top-left corner and (x=1000, y=1000) is the bottom-right corner.
    - Keep all variables (keypoints, weights_path, target_object, images) in RAM! They are automatically preserved between code execution stages.
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
    User request: "{user_input}"
    Experiment index: {i}
    counter_drones: {counter_drones}
    GitHub configuration: Repo_owner="{repo_owner}", repo_name="{repo_name}", branch="{branch}"

    CRITICAL WORKFLOW RULES:
    1. IMAGES ROLES:
    - `load_local_map_images()` returns a tuple: `base_image, target_image`.
    - `base_image` is an ARCHIVAL reference map without incidents. Use it ONLY with `pixelpoint_objects` to find infrastructure/waypoints (e.g. houses, buildings, roads).
    - `target_image` represents the REAL-TIME physical ground with the ongoing incident. Pass it ONLY to `uav_simulation()` to generate the drone camera feed.
    - NEVER call `pixelpoint_objects` on `target_image`! That is cheating and strictly forbidden.
    - In `pixelpoint_objects`, do NOT search for the disaster itself (e.g., do NOT search for 'fire'). Search for candidate structures to inspect (e.g., 'residential buildings', 'houses').
    - DO NOT inspect function internals, source code, or dunder attributes (e.g., `__code__`, `__globals__`, `dir()`).
    - Call tools directly with the specified signature without introspecting them.

    2. TASK EXECUTION SEQUENCE:
    Step 1: Extract candidate inspection structures using `inspection_targets = extract_inspection_targets(user_request)`. (These must be structures/places to check, e.g. buildings/forest).
    Step 2: Determine scenario details:
        `folder_info = get_folder_by_query(user_request)`
        `target_object = extract_inspection_targets(user_request)`
    Step 3: Load map images and weights for the experiment:
        `base_image, target_image = load_local_map_images(folder_name=folder_info['folder'], index={i})`
        `weights_path = download_and_sync_weights(folder_info['incident_type'])`
    Step 4: Extract keypoints in coordinat from archival base map: `keypoints = pixelpoint_objects(image=base_image, objects=inspection_targets)`
    Step 5: Show the flight path and keypoints on the base map using `visualize_keypoints_from_image(image=base_image, keypoints=keypoints, output_dir="{exp_folder}")`.
    Step 6: Partition inspection keypoints, simulate realistic flight for each drone with GIF generation, execute parallel reconnaissance scanning, and submit report:

    ```python
    import math
    from concurrent.futures import ThreadPoolExecutor

    drones = {drones_code_repr}
    num_drones = len(drones)
    home_point = {home_point}
    output_dir = "{exp_folder}"

    # 1. Coordinate normalization
    def get_xy(p):
        if isinstance(p, dict):
            if "point_2d" in p:
                return p["point_2d"][0], p["point_2d"][1]
            return p.get("x", 0), p.get("y", 0)
        return p[0], p[1]

    # 2. Greedy Nearest Neighbor route optimization (TSP heuristic)
    def optimize_route(home, points):
        if not points:
            return []

        unvisited = list(points)
        current_pos = home
        optimized_path = []

        while unvisited:
            nearest_idx = min(
                range(len(unvisited)),
                key=lambda idx: (get_xy(unvisited[idx])[0] - current_pos[0]) ** 2
                + (get_xy(unvisited[idx])[1] - current_pos[1]) ** 2,
            )
            next_point = unvisited.pop(nearest_idx)
            optimized_path.append(next_point)
            current_pos = get_xy(next_point)

        return optimized_path

    # 3. Balance points
    def balance_routes_by_time(home, points, num_drones):
      if not points:
        return [[] for _ in range(num_drones)]

      assigned_routes = [[] for _ in range(num_drones)]

      def get_route_dist(route):
        if not route:
          return 0.0
        pts = [home] + [get_xy(p) for p in route] + [home]
        return sum(
            math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
            for k in range(1, len(pts))
        )

      sorted_pts = sorted(
          points,
          key=lambda p: math.hypot(
              get_xy(p)[0] - home[0], get_xy(p)[1] - home[1]
          ),
          reverse=True,
      )

      for pt in sorted_pts:
        best_drone = min(
            range(num_drones),
            key=lambda i: get_route_dist(assigned_routes[i] + [pt]),
        )
        assigned_routes[best_drone].append(pt)

      return assigned_routes

    drone_point_groups = balance_routes_by_time(
        home_point, keypoints, len(drones)
    )
    drone_tasks = []

    # 4. Local TSP-optim
    for idx, drone in enumerate(drones):
      assigned_points = drone_point_groups[idx]
      optimized_points = optimize_route(home_point, assigned_points)

      if optimized_points:
        drone_frames = uav_simulation(
            image=target_image,
            labeled_points=optimized_points,
            drone_id=idx,
            total_drones=num_drones,
            home_point=home_point,
            output_dir=output_dir,
        )
        drone_tasks.append((drone, drone_frames, idx))

    # 5. Parallel reconnaissance worker
    def execute_drone_sector(drone, frames, drone_idx):
        drone.python_executor.state["frames_dict"] = frames
        drone.python_executor.state["target_object"] = target_object
        drone.python_executor.state["weights_path"] = weights_path

        if weights_path:
            task_command = (
                "Call run_local_uav_detection(frames_dict=frames_dict, "
                "weights_path=weights_path, target_object=target_object) EXACTLY ONCE. "
                "Return the result via final_answer(...) as code. Do not output raw JSON."
            )
        else:
            task_command = (
                "Call detect_and_display(frames_dict=frames_dict, "
                "target_object=target_object) EXACTLY ONCE. "
                "Return the result via final_answer(...) as code. Do not output raw JSON."
            )

        drone_name = getattr(drone, "name", f"uav_agent_{{drone_idx}}")
        return drone_name, drone(task_command)

    futures = []
    with ThreadPoolExecutor(max_workers=len(drone_tasks)) as executor:
        for drone, frames, idx in drone_tasks:
            futures.append(executor.submit(execute_drone_sector, drone, frames, idx))

    all_drone_results = dict(f.result() for f in futures)
    print("Reconnaissance results from drones:", all_drone_results)

    final_answer(
        f"Multi-UAV reconnaissance successfully executed. Drone outcomes: {{all_drone_results}}"
    )
    ```

    Try to complete the task and detect incidents—after all, you could save lives!
    """
        
        try:
            experiment_outcome = airspace_manager_agent.run(task_prompt)
        except Exception as e:
            experiment_outcome = f"Execution failed: {str(e)}"
            print(f"❌ Error on experiment #{i}: {e}")

        elapsed_time = time.time() - start_time
        print(f"⏱️ Experiment {i} completed in {elapsed_time:.2f}s.")


        # time mission
        flight_time = max(SWARM_FLIGHT_TIMES.values()) if SWARM_FLIGHT_TIMES else 0.0
        total_time = elapsed_time + flight_time

        agent_time = time.time() - start_time

        print(
            f"⏱️ Experiment {i} completed in {elapsed_time:.2f}s (Agent) + {flight_time:.2f}s (Flight) = {total_time:.2f}s Total."
        )

        # clear dictionary
        SWARM_FLIGHT_TIMES.clear()

        # Save individual experiment report to its designated folder
        report_file_path = os.path.join(exp_folder, "report.txt")
        with open(report_file_path, mode="w", encoding="utf-8") as rf:
            rf.write(f"Experiment Index: {i}\n")
            rf.write(f"Artifacts Folder: {exp_folder}\n")
            rf.write(f"Scenario: {scenario['incident_type']}\n")
            rf.write(f"Request: {user_request_text}\n")
            rf.write(f"Elapsed Time: {elapsed_time:.2f} s\n")
            rf.write(f"Agent Time: {agent_time:.2f} s\n")
            rf.write(f"Flight Time: {flight_time:.2f} s\n")
            rf.write(f"Total Mission Time: {total_time:.2f} s\n")
            # rf.write(f"Outcome:\n{experiment_outcome}\n")

        # Append row to the master CSV log
        record = {
            "Experiment Number": i,
            "Home Point": str(home_point),
            "Folder": exp_folder,
            "Scenario": scenario["incident_type"],
            "Request Text": user_request_text,
            "Elapsed Time": f"{elapsed_time:.2f}",
            "Agent Time (s)": f"{agent_time:.2f}",
            "Flight Time (s)": f"{flight_time:.2f}",
            "Total Mission Time (s)": f"{total_time:.2f}",
            # "Outcome": str(experiment_outcome),
        }

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow(record)

        # Flush telemetry exporter
        try:
            tracer_provider.force_flush()
        except Exception:
            pass

        time.sleep(1)

    print(
        f"\n📊 All {total_experiments} experiments completed! Results logged in {csv_file_path}"
    )

if __name__ == "__main__":
    main()