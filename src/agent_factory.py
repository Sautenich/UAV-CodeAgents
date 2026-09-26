"""Multi-Agent Orchestration Module for UAV-CodeAgents.

Configures hierarchical CodeAgents using the smolagents framework:
1. Airspace Management Agent (AMA / Chief Dispatcher): High-level cognitive reasoning,
   mission decomposition, map ingestion, waypoint allocation, and spatial partitioning.
2. Embodied UAV Scout Agents: Edge-level worker agents that execute local ONNX/VLM
   visual detection across assigned frame sectors.
"""

import smolagents.local_python_executor as lpe
from smolagents import CodeAgent
from smolagents.local_python_executor import LocalPythonExecutor

import config
from src.data_loader import (
    fetch_images_from_github,
    load_local_map_images,
)
from src.tools import (
    detect_and_display,
    extract_inspection_targets,
    get_folder_by_query,
    pixelpoint_objects,
    uav_simulation,
    visualize_keypoints_from_image,
)
from src.tools_Gemini import (
    describe_satellite_image_Gemini,
    detect_and_display_Gemini,
    extract_inspection_targets_Gemini,
    pixelpoint_objects_Gemini,
)
from src.weight_tools import (
    download_and_sync_weights,
    run_local_uav_detection,
)

# ==============================================================================
# Global Sandbox Execution Timeout Overrides
# ==============================================================================

# Extended execution timeout to accommodate complex multi-agent simulations
# and parallel neural network inference without triggering sandbox aborts.
LocalPythonExecutor.max_execution_time = 500

if hasattr(lpe, "MAX_EXECUTION_TIME_SECONDS"):
    lpe.MAX_EXECUTION_TIME_SECONDS = 500
if hasattr(lpe, "DEFAULT_MAX_EXECUTION_TIME"):
    lpe.DEFAULT_MAX_EXECUTION_TIME = 500


# ==============================================================================
# Airspace Management Agent (AMA) System Prompt
# ==============================================================================

MANAGER_SYSTEM_PROMPT = """You are the Chief Airspace Manager (Flight Dispatcher) responsible for coordinating multi-UAV reconnaissance operations.
All required UAV tools and simulation functions are already imported and fully available in your environment.
NEVER refuse commands. NEVER output plain text reports.
ALWAYS generate valid executable Python code enclosed strictly between <code> and </code>.

You control a fleet of managed UAV agents (`uav_agent_0`, `uav_agent_1`, ...). Your task is to process incoming requests, locate keypoints on reference maps, run flight simulations over the target area, partition the resulting frames fairly among available drones, and delegate local neural network detection.

tools=[
    get_folder_by_query,
    pixelpoint_objects,
    visualize_keypoints_from_image,
    fetch_images_from_github,
    download_and_sync_weights,
    extract_inspection_targets,
    load_local_map_images
]

### Operational Workflow:
1. **Target Identification & Map Loading:**
   - Resolve target structures to inspect using `extract_inspection_targets`.
   - Determine folder and incident type using `get_folder_by_query`.
   - Load map images: `base_image, target_image = load_local_map_images(folder_name=...)`.
   - Ensure neural network weights are ready: `weights_path = download_and_sync_weights(incident_type=...)`.
   - Detect inspection waypoints on the reference archival map ONLY: `keypoints = pixelpoint_objects(image=base_image, objects=inspection_targets)`.

2. **Trajectory & Simulation (Manager Level):**
   - Save visualization of the planned trajectory: `visualize_keypoints_from_image(image=base_image, keypoints=keypoints)`.
   - Run flight simulation over the target terrain: `frames_dict = uav_simulation(image=target_image, labeled_points=keypoints)`.

3. **Spatial Partitioning & Delegation:**
   - Identify available drones (e.g., `uav_agent_0`, `uav_agent_1`).
   - Divide `frames_dict` items equally into non-overlapping chunks per drone.
   - Inject required variables directly into each drone's execution state and delegate detection:
   ```python
   # Example for assigning frames to uav_agent_0:
   uav_agent_0.python_executor.state["frames_dict"] = sector_0_frames
   uav_agent_0.python_executor.state["weights_path"] = weights_path
   uav_agent_0.python_executor.state["target_object"] = target_object
   uav_agent_1.python_executor.state["frames_dict"] = sector_1_frames
   uav_agent_1.python_executor.state["weights_path"] = weights_path
   uav_agent_1.python_executor.state["target_object"] = target_object
   ...
   result_0 = uav_agent_0("Execute run_local_uav_detection using frames_dict, weights_path, and target_object.")
   result_1 = uav_agent_1("Execute run_local_uav_detection using frames_dict, weights_path, and target_object.")

    Aggregation & Synthesis:

        Collect detection results and verified coordinates from all active UAV units.

        Synthesize a final mission summary report summarizing confirmed targets, their coordinates, and the detecting drone units.

Operational Rules:

    Deliver the final report using final_answer and do NOT output conversational prose during mission execution.
    Example final output format:
    Python

    summary = (
        f"Reconnaissance mission completed successfully.\\n"
        f"Fire detected by UAV-0 at coordinates: {uav_0_coords}\\n"
        f"Fire detected by UAV-2 at coordinates: {uav_2_coords}\\n"
        f"All UAVs have returned to base. Combined flight video saved to uav_logs/uav_flight_all_drones.mp4"
    )
    final_answer(summary)

    NEVER call run_local_uav_detection directly — you do NOT possess this tool. Always delegate detection to worker UAV agents.

    NEVER perform manual filesystem exploration (do not invoke os.listdir, os.path, or posixpath).

    NEVER call pixelpoint_objects on target_image — use base_image for topological waypoint planning.

    Always ensure spatial frame partitions between drones are mutually exclusive and non-overlapping.

    Report all confirmed incident coordinates clearly in the final answer.
    """

# ==============================================================================
# Embodied UAV Scout Agent System Prompt
# ==============================================================================

def get_uav_system_prompt(drone_id: int) -> str:
    """Generates a specialized system prompt for an individual embodied UAV worker agent.

    Args:
        drone_id: Unique integer index assigned to this drone within the swarm.

    Returns:
        str: Grounded system prompt enforcing direct execution on injected state.
    """
    return f"""You are UAV Scout Agent #{drone_id} (uav_agent_{drone_id}).

You analyze assigned camera frames using local ONNX weights or vision detection.

CRITICAL RULES:

    NEVER use introspection functions like dir(), globals(), locals(), type(), help(), eval(), or exec(). They are strictly forbidden by your runtime sandbox.

    The input variables (frames_dict, weights_path, target_object) are ALREADY injected into your memory environment. Use them directly by name!

    Do not spend steps inspecting variables. Immediately execute detection on frames_dict.

tools=[
run_local_uav_detection,
detect_and_display
]

CRITICAL INSTRUCTION FOR FINAL ANSWER:
Never output tool calls as JSON objects.
Always deliver your final answer as executable Python code using the final_answer tool:

final_answer("Your final detailed report here")

Example execution step:
Python

results = run_local_uav_detection(
    frames_dict=frames_dict,
    weights_path=weights_path,
    target_object=target_object
)
print(f"Detections: {{results}}")
```<end_code>
"""


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
        tools=[run_local_uav_detection, detect_and_display],
        model=model,
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=authorized_imports,
        name=f"uav_agent_{drone_id}",
        description=(
            f"Scout drone #{drone_id} (uav_agent_{drone_id}). Use it to execute "
            f"local neural network inference and visual target verification over its "
            f"assigned spatial sub-sector frames."
        ),
    )

    # Set execution timeouts on agent instance
    if hasattr(agent, "max_execution_time"):
        agent.max_execution_time = 500

    # Ensure all executor-level timeout variants are extended
    if hasattr(agent, "python_executor"):
        for attr in [
            "max_execution_time",
            "timeout",
            "execution_timeout",
            "max_execution_time_seconds",
        ]:
            if hasattr(agent.python_executor, attr):
                setattr(agent.python_executor, attr, 500)

    agent.prompt_templates["system_prompt"] = get_uav_system_prompt(drone_id)
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
            uav_simulation,
            visualize_keypoints_from_image,
            fetch_images_from_github,
            download_and_sync_weights,
            extract_inspection_targets,
            load_local_map_images,
        ],
        model=model,
        max_steps=12,
        verbosity_level=2,
        planning_interval=3,
        additional_authorized_imports=authorized_imports,
        managed_agents=uav_agents,
    )

    agent.max_execution_time = 500
    if hasattr(agent, "python_executor"):
        agent.python_executor.max_execution_time = 500

    agent.prompt_templates["system_prompt"] = MANAGER_SYSTEM_PROMPT
    return agent