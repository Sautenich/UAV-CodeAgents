"""Multi-Agent Orchestration Module for UAV-CodeAgents.

Configures hierarchical CodeAgents using the smolagents framework:
1. Airspace Management Agent (AMA / Chief Dispatcher): High-level cognitive reasoning,
   mission decomposition, map ingestion, waypoint allocation, and spatial partitioning.
2. Embodied UAV Scout Agents: Edge-level worker agents that execute onboard Valen
   visual detection across assigned frame sectors.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from smolagents import CodeAgent, tool

import config
from src.data_loader import (
    fetch_images_from_github,
    load_local_map_images,
)
import src.tools as tools
from src.tools import (
    extract_inspection_targets,
    get_coords,
    get_folder_by_query,
    pixelpoint_objects,
    uav_simulation,
    visualize_keypoints_from_image,
)
from src.valen_engine import run_valen_uav_detection

# Extended code execution timeout to accommodate multi-agent simulations and
# parallel neural network inference without triggering sandbox aborts.
# `repr` is not part of the sandbox by default, yet models reach for it when
# they format results; allowing it avoids a failed step.
EXECUTOR_KWARGS = {
    "timeout_seconds": config.AGENT_CODE_TIMEOUT_S,
    "additional_functions": {"repr": repr},
}

# Task wrapper for UAV agents. The stock smolagents wrapper demands a long
# three-part report, which makes models dump every frame key into final_answer;
# detections are recorded by the tool itself, so a short answer is enough.
UAV_TASK_TEMPLATE = """You are '{{name}}'. Your manager assigned you this task:
{{task}}

Pass the value returned by the tool to final_answer unchanged. Do not write a report, do not add explanations and do not list the input frames."""
UAV_REPORT_TEMPLATE = "{{final_answer}}"


# ==============================================================================
# Airspace Management Agent (AMA) System Prompt
# ==============================================================================

MANAGER_SYSTEM_PROMPT = """You are the Chief Airspace Manager (Flight Dispatcher) responsible for coordinating multi-UAV reconnaissance operations.
All required UAV tools and simulation functions are already imported and fully available in your environment.
NEVER refuse commands. NEVER output plain text reports.
ALWAYS generate valid executable Python code enclosed strictly between <code> and </code>.

You control a fleet of UAV scout agents (`uav_agent_0`, `uav_agent_1`, ...). Your task is to process incoming requests, locate keypoints on reference maps, and dispatch the swarm over the target area. The `dispatch_swarm` tool splits keypoints between drones, simulates their flights, and delegates onboard neural network detection to every drone in parallel.

tools=[
    get_folder_by_query,
    pixelpoint_objects,
    visualize_keypoints_from_image,
    fetch_images_from_github,
    extract_inspection_targets,
    load_local_map_images,
    dispatch_swarm
]

### Operational Workflow:
1. **Target Identification & Map Loading:**
   - Resolve target structures to inspect using `extract_inspection_targets`.
   - Determine folder, incident type and target object using `get_folder_by_query`.
   - Load map images: `base_image, target_image = load_local_map_images(folder_name=...)`.
   - Detect inspection waypoints on the reference archival map ONLY: `keypoints = pixelpoint_objects(image=base_image, objects=inspection_targets)`.

2. **Trajectory Visualization:**
   - Save visualization of the planned waypoints: `visualize_keypoints_from_image(image=base_image, keypoints=keypoints)`.

3. **Swarm Dispatch:**
   - Run the whole reconnaissance with a single call:
   ```python
   mission = dispatch_swarm(
       target_image=target_image,
       keypoints=keypoints,
       incident_type=folder_info["incident_type"],
       target_object=folder_info["target_object"],
       home_point=home_point,
       output_dir=output_dir,
   )
   ```
   - `mission` is a dict with `incident_detected`, `total_detections`, `detections` (per drone coordinates), `flight_times_s`, `routes` and `agent_reports`.

4. **Aggregation & Synthesis:**
   - Deliver the final report using `final_answer`, summarizing confirmed targets, their coordinates, and the detecting drone units.

Operational Rules:

    Do NOT output conversational prose during mission execution.

    NEVER call detection tools directly — you do NOT possess them. Detection is always delegated to worker UAV agents through dispatch_swarm.

    NEVER re-implement route planning or drone delegation yourself — always use dispatch_swarm.

    NEVER perform manual filesystem exploration (do not invoke os.listdir, os.path, or posixpath).

    NEVER call pixelpoint_objects on target_image — use base_image for topological waypoint planning.

    Keypoints and home_point are PIXEL coordinates of the map. NEVER rescale, normalize or otherwise modify them — pass the output of pixelpoint_objects to visualize_keypoints_from_image and dispatch_swarm unchanged.

    Report all confirmed incident coordinates clearly in the final answer.
    """

# ==============================================================================
# Embodied UAV Scout Agent System Prompt
# ==============================================================================

def get_uav_system_prompt(drone_id: int) -> str:
    """Generates a specialized system prompt for an individual embodied UAV worker agent."""
    return f"""You are UAV Scout Agent #{drone_id} (uav_agent_{drone_id}).

You analyze assigned camera frames using onboard zero-shot neural inference (Valen).

CRITICAL RULES:
- ALWAYS enclose your executable Python code strictly between <code> and </code> tags.
- NEVER output conversational text or thoughts before or after code tags.
- Input variables (frames_dict, incident_type, target_object, output_dir) are ALREADY loaded in your memory.

tools=[
    run_valen_uav_detection
]

Format requirement:
<code>
results = run_valen_uav_detection(
    frames_dict=frames_dict,
    incident_type=incident_type,
    target_object=target_object,
    output_dir=output_dir
)
final_answer(results)
</code>
"""


# ==============================================================================
# Swarm Route Planning
# ==============================================================================
# Waypoint allocation between drones balanced by route length, and per-drone
# visiting order via a nearest-neighbour heuristic.

def route_length(home: Sequence[float], route: Sequence[Any]) -> float:
    """Length of the closed path home -> route points in order -> home."""
    if not route:
        return 0.0
    path = [tuple(home)] + [get_coords(p) for p in route] + [tuple(home)]
    return sum(
        math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        for k in range(1, len(path))
    )


def optimize_route(home: Sequence[float], points: Sequence[Any]) -> List[Any]:
    """Orders points with a greedy nearest-neighbour heuristic starting at home."""
    unvisited = list(points)
    current = (float(home[0]), float(home[1]))
    ordered = []

    while unvisited:
        nearest_idx = min(
            range(len(unvisited)),
            key=lambda idx: math.hypot(
                get_coords(unvisited[idx])[0] - current[0],
                get_coords(unvisited[idx])[1] - current[1],
            ),
        )
        next_point = unvisited.pop(nearest_idx)
        ordered.append(next_point)
        current = get_coords(next_point)

    return ordered


def balance_routes(
    home: Sequence[float], points: Sequence[Any], num_drones: int
) -> List[List[Any]]:
    """Splits points between drones so that route lengths stay balanced.

    Points are assigned farthest-first; each goes to the drone whose route
    remains the shortest after receiving it.

    Args:
        home: Base station coordinates (x, y).
        points: Keypoints to distribute.
        num_drones: Number of drones in the swarm.

    Returns:
        One list of points per drone (possibly empty).
    """
    if num_drones < 1:
        raise ValueError("num_drones must be at least 1")

    routes: List[List[Any]] = [[] for _ in range(num_drones)]
    farthest_first = sorted(
        points,
        key=lambda p: math.hypot(get_coords(p)[0] - home[0], get_coords(p)[1] - home[1]),
        reverse=True,
    )

    for point in farthest_first:
        best_drone = min(
            range(num_drones),
            key=lambda idx: route_length(home, routes[idx] + [point]),
        )
        routes[best_drone].append(point)

    return routes


def plan_swarm_routes(
    home: Sequence[float], points: Sequence[Any], num_drones: int
) -> List[List[Any]]:
    """Allocates points to drones and orders each drone's route."""
    return [
        optimize_route(home, group)
        for group in balance_routes(home, points, num_drones)
    ]


# ==============================================================================
# Swarm Dispatch
# ==============================================================================
# The deterministic part of a mission as one tool call: waypoint allocation,
# route ordering, flight simulation, and parallel delegation of onboard
# detection to the embodied UAV agents.

# UAV agents available to `dispatch_swarm`, set by `register_drones`
_DRONES: List[Any] = []

DRONE_TASK = (
    "Call run_valen_uav_detection(frames_dict=frames_dict, "
    "incident_type=incident_type, target_object=target_object, "
    "output_dir=output_dir) and return the result using final_answer."
)


def register_drones(uav_agents: List[Any]) -> None:
    """Registers the UAV agents that `dispatch_swarm` delegates detection to."""
    _DRONES.clear()
    _DRONES.extend(uav_agents)


def _drone_name(drone: Any, drone_idx: int) -> str:
    return getattr(drone, "name", None) or f"uav_agent_{drone_idx}"


def _run_drone(
    drone: Any,
    drone_idx: int,
    frames: Dict[str, Any],
    incident_type: str,
    target_object: Any,
    output_dir: str,
) -> str:
    """Hands one drone its camera frames and runs its detection task."""
    state = drone.python_executor.state
    state["frames_dict"] = frames
    state["incident_type"] = incident_type
    state["target_object"] = target_object
    state["output_dir"] = output_dir
    try:
        return str(drone(DRONE_TASK))
    except Exception as err:
        print(f"❌ [SWARM] {_drone_name(drone, drone_idx)} failed: {err}")
        return f"Execution failed: {err}"


@tool
def dispatch_swarm(
    target_image: Image.Image,
    keypoints: List[Any],
    incident_type: str,
    target_object: str = config.UNKNOWN_TARGET_OBJECT,
    home_point: Optional[Tuple[int, int]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Splits keypoints between all drones, simulates their flights, and runs onboard detection in parallel.

    Args:
        target_image: Real-time target terrain image (PIL Image) the swarm flies over.
        keypoints: Inspection keypoints in pixel coordinates of the map, exactly as returned by
            `pixelpoint_objects` (list of dicts with 'point_2d'). Do not rescale them.
        incident_type: Incident type from `get_folder_by_query` (e.g. 'fire', 'oil_spill').
        target_object: Description of what the drones look for (e.g. 'fire').
        home_point: Base station pixel coordinates (x, y) where the swarm starts and lands.
        output_dir: Folder for flight video and detection artifacts.

    Returns:
        dict: Mission summary with keys 'incident_detected' (bool), 'total_detections',
            'detections' (drone name -> list of (x, y)), 'flight_times_s',
            'routes' (drone name -> ordered waypoints) and 'agent_reports'.
    """
    if not _DRONES:
        raise RuntimeError("No UAV agents registered. Call register_drones() first.")
    if not keypoints:
        raise ValueError("No keypoints provided to dispatch_swarm.")

    keypoints = tools.ensure_pixel_keypoints(keypoints)
    home = tuple(home_point) if home_point else config.DEFAULT_HOME_POINT
    target_dir = tools.resolve_output_dir(output_dir)

    # 1. Allocate keypoints and order each drone's route
    routes = plan_swarm_routes(home, keypoints, len(_DRONES))
    active = [
        (idx, drone, route)
        for idx, (drone, route) in enumerate(zip(_DRONES, routes))
        if route
    ]
    names = {idx: _drone_name(drone, idx) for idx, drone, _ in active}

    # 2. Simulate flights; the swarm video is rendered once all tracks are ready
    tools.reset_swarm_state()
    frames_by_drone = {}
    for idx, _, route in active:
        frames_by_drone[idx] = uav_simulation(
            image=target_image,
            labeled_points=route,
            drone_id=idx,
            total_drones=len(active),
            home_point=home,
            output_dir=target_dir,
        )

    # 3. All drones scan their own frames concurrently
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = {
            idx: executor.submit(
                _run_drone,
                drone,
                idx,
                frames_by_drone[idx],
                incident_type,
                target_object,
                target_dir,
            )
            for idx, drone, _ in active
        }
        reports = {names[idx]: future.result() for idx, future in futures.items()}

    # 4. Aggregate what the onboard detectors recorded
    recorded = tools.get_swarm_detections()
    detections = {
        names[idx]: [tuple(d["coords"]) for d in recorded.get(idx, [])]
        for idx, _, _ in active
    }
    total = sum(len(coords) for coords in detections.values())
    flight_times = {
        names[idx]: round(tools.SWARM_FLIGHT_TIMES.get(idx, 0.0), 2)
        for idx, _, _ in active
    }

    route_points = {
        names[idx]: [[int(round(v)) for v in get_coords(p)] for p in route]
        for idx, _, route in active
    }
    with open(os.path.join(target_dir, "routes.json"), "w", encoding="utf-8") as f:
        json.dump({"home_point": [int(v) for v in home], "routes": route_points}, f, indent=2)

    print(f"[SWARM] Reconnaissance finished. Detections per drone: {detections}")
    return {
        "incident_detected": total > 0,
        "total_detections": total,
        "detections": detections,
        "flight_times_s": flight_times,
        "routes": {
            names[idx]: [tuple(int(round(v)) for v in get_coords(p)) for p in route]
            for idx, _, route in active
        },
        "agent_reports": reports,
    }


# ==============================================================================
# Agent Instantiation Factories
# ==============================================================================

def create_uav_agent(model, drone_id: int = 0) -> CodeAgent:
    """Instantiates a dedicated embodied UAV agent equipped with onboard vision tools.

    Args:
        model: Language model backbone driving agentic code execution.
        drone_id: Unique integer index for the UAV unit.

    Returns:
        CodeAgent: Configured embodied scout agent.
    """
    authorized_imports = [
        "json",
        "PIL",
        "matplotlib.pyplot",
        "typing",
        "sys",
        "numpy",
        "cv2",
        "os",
        "imageio",
        "concurrent.futures",
    ]

    agent = CodeAgent(
        tools=[run_valen_uav_detection],
        model=model,
        max_steps=config.AGENT_MAX_STEPS,
        verbosity_level=config.AGENT_VERBOSITY_LEVEL,
        additional_authorized_imports=authorized_imports,
        executor_kwargs=EXECUTOR_KWARGS,
        name=f"uav_agent_{drone_id}",
        description=(
            f"Scout drone #{drone_id} (uav_agent_{drone_id}). Use it to execute "
            f"local neural network inference and visual target verification over its "
            f"assigned spatial sub-sector frames."
        ),
    )

    agent.prompt_templates["system_prompt"] = get_uav_system_prompt(drone_id)
    agent.prompt_templates["managed_agent"]["task"] = UAV_TASK_TEMPLATE
    agent.prompt_templates["managed_agent"]["report"] = UAV_REPORT_TEMPLATE
    return agent


def create_airspace_manager(model, uav_agents: list) -> CodeAgent:
    """Instantiates the Airspace Management Agent (AMA) coordinating the UAV swarm.

    Args:
        model: Multimodal foundation model backbone for high-level spatial reasoning.
        uav_agents: List of managed CodeAgent instances representing individual drones.

    Returns:
        CodeAgent: Configured chief flight dispatcher agent.
    """
    authorized_imports = [
        "json",
        "ast",
        "math",
        "re",
        "PIL",
        "matplotlib.pyplot",
        "sys",
        "io",
        "imageio",
        "os",
        "time",
        "concurrent.futures",
    ]

    agent = CodeAgent(
        tools=[
            get_folder_by_query,
            pixelpoint_objects,
            visualize_keypoints_from_image,
            fetch_images_from_github,
            extract_inspection_targets,
            load_local_map_images,
            dispatch_swarm,
        ],
        model=model,
        max_steps=config.AGENT_MAX_STEPS,
        verbosity_level=config.AGENT_VERBOSITY_LEVEL,
        planning_interval=config.MANAGER_PLANNING_INTERVAL,
        additional_authorized_imports=authorized_imports,
        executor_kwargs=EXECUTOR_KWARGS,
        managed_agents=uav_agents,
    )

    # Drones that `dispatch_swarm` delegates detection to
    register_drones(uav_agents)

    agent.prompt_templates["system_prompt"] = MANAGER_SYSTEM_PROMPT
    return agent