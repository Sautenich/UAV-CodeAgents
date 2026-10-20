"""Onboard incident verification for UAV-CodeAgents with the Valen decision model.

`ValenEngine` provides local zero-shot verification without autoregressive token
generation. Frames submitted concurrently by several drones are merged into shared
GPU batches, so the swarm is verified in parallel on a single model instance.
`run_valen_uav_detection` is the tool each UAV agent calls on its camera frames;
one model covers every incident type.
"""

from concurrent.futures import Future
import io
import os
from pathlib import Path
import json
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from smolagents import tool
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from valen.data.compilers.qwen import Compiler
from valen.evaluation.inference import (
    answer,
    build_compiler,
    build_model,
    load_checkpoint,
    normalize_model_config,
    predict,
    with_attention_implementation,
)

import config
import src.tools as tools
from src.tools import canonical_incident

QUESTION_ID = "incident_detected"


def get_domain_prompt(incident_type: str, target_name: str = "hazard") -> Dict[str, str]:
    """Returns the Valen context and question for an incident type.

    Args:
        incident_type: Scenario key or alias from `config.SCENARIOS`.
        target_name: Description of the target, used for unknown incident types.

    Returns:
        Dictionary with 'context' and 'question' prompt strings.
    """
    scenario = config.SCENARIOS.get(canonical_incident(incident_type))
    if scenario and scenario.get("valen_question"):
        return {
            "context": scenario["valen_context"],
            "question": scenario["valen_question"],
        }
    return {
        "context": f"Inspect the area for {target_name}.",
        "question": f"Is there any {target_name} or related damage visible?",
    }


class FrameCompiler(Compiler):
    """Valen compiler that accepts in-memory PIL frames instead of media files."""

    def _messages(self, state):
        media = []
        for message in state["messages"]:
            for item in message["content"]:
                if item["type"] == "text":
                    # Prevent prompt text from injecting Qwen control tokens.
                    if any(t in item["text"] for t in self.tokenizer.all_special_tokens):
                        raise ValueError("Reserved tokenizer control token in state")
                elif item["type"] == "image":
                    media.append({"path": None, "sha256": None, "type": "image_url"})
                else:
                    raise ValueError(f"Unsupported content type: {item['type']}")
        return state["messages"], media


class ValenEngine:
    _instance: Optional["ValenEngine"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        checkpoint_dir: str = config.VALEN_CHECKPOINT_DIR,
        device: str = config.VALEN_DEVICE,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        print(
            f"📦 [ValenEngine] Loading model on {self.device} with"
            f" {config.VALEN_ATTENTION_IMPLEMENTATION.upper()}..."
        )
        config_file = Path(checkpoint_dir) / "config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        model_config = normalize_model_config(raw_config)
        model_config = with_attention_implementation(
            model_config, config.VALEN_ATTENTION_IMPLEMENTATION
        )

        self.model = build_model(model_config).to(self.device)
        load_checkpoint(checkpoint_dir, self.model)
        self.model.eval()

        base = build_compiler(raw_config)
        self.compiler = FrameCompiler(
            base.processor,
            base.media_root,
            base.max_length,
            base.media_kwargs,
            execution=base.execution,
        )

        # Requests of all drones go through one queue and one batching worker,
        # which is the only thread that touches the compiler and the model.
        self._requests: "queue.Queue[Tuple[Image.Image, Dict[str, str], Future]]" = (
            queue.Queue()
        )
        self._worker = threading.Thread(
            target=self._batch_loop, name="valen-batcher", daemon=True
        )
        self._worker.start()
        print("✅ [ValenEngine] Ready for multi-UAV inference.")

    @classmethod
    def get_instance(cls) -> "ValenEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def evaluate_frames(
        self,
        images: Sequence[Image.Image],
        incident_type: str,
        target_name: str = "hazard",
    ) -> List[float]:
        """Evaluates camera crops and returns incident probabilities (0.0 to 1.0).

        Thread-safe: calls from different drones are batched together on the GPU.
        """
        domain = get_domain_prompt(incident_type, target_name)
        futures = []
        for img in images:
            future: Future = Future()
            self._requests.put((img, domain, future))
            futures.append(future)
        return [future.result() for future in futures]

    def evaluate_frame(
        self,
        img: Image.Image,
        incident_type: str,
        target_name: str = "hazard",
    ) -> float:
        """Evaluates a single crop image and returns the probability of incident presence."""
        return self.evaluate_frames([img], incident_type, target_name)[0]

    # --------------------------------------------------------------------------
    # Batching worker
    # --------------------------------------------------------------------------

    def _batch_loop(self) -> None:
        while True:
            batch = [self._requests.get()]
            deadline = time.monotonic() + config.VALEN_BATCH_WAIT_S
            while len(batch) < config.VALEN_MAX_BATCH_SIZE:
                try:
                    timeout = max(0.0, deadline - time.monotonic())
                    batch.append(self._requests.get(timeout=timeout))
                except queue.Empty:
                    break

            try:
                scores = self._infer([(img, domain) for img, domain, _ in batch])
                for (_, _, future), score in zip(batch, scores):
                    future.set_result(score)
            except Exception as err:  # Report the failure to every waiting drone
                for _, _, future in batch:
                    if not future.done():
                        future.set_exception(err)

    def _compile(self, img: Image.Image, domain: Dict[str, str]) -> Any:
        # JPEG round-trip in memory keeps inputs identical to file-based evaluation
        buffer = io.BytesIO()
        img.convert("RGB").save(
            buffer, format="JPEG", quality=config.VALEN_FRAME_JPEG_QUALITY
        )
        frame = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")

        record = {
            "group_id": "uav_agent",
            "id": "probe",
            "request": {
                "state": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": frame},
                                {"type": "text", "text": domain["context"]},
                            ],
                        }
                    ]
                },
                "questions": {
                    QUESTION_ID: {
                        "type": "noul",
                        "instructions": domain["question"],
                    }
                },
            },
            "targets": {},
            "assets": [],
        }
        return self.compiler.compile(record)

    @torch.no_grad()
    def _infer(self, items: List[Tuple[Image.Image, Dict[str, str]]]) -> List[float]:
        temperature = config.VALEN_CALIBRATION_TEMPERATURE
        compiled = [self._compile(img, domain) for img, domain in items]

        if any(state.inputs is not None for state in compiled):
            # Shared-state checkpoints: evaluate one frame at a time
            return [
                float(predict(self.model, state, temperature)["answers"][QUESTION_ID]["noul"])
                for state in compiled
            ]

        questions = [state.questions[0] for state in compiled]
        logits = self.model.forward_batch(
            questions, max_tokens=config.VALEN_MAX_BATCH_TOKENS
        )
        return [
            float(answer(question, row, temperature)["noul"])
            for question, row in zip(questions, logits)
        ]


# ==============================================================================
# Onboard detection tool of the UAV agents
# ==============================================================================

def _save_detection_plot(img: Image.Image, title: str, save_path: str) -> None:
    """Saves an annotated detection frame without pyplot, so drones can plot in parallel."""
    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")


def _drone_id_from_frames(frames_dict: Dict[str, Any]) -> int:
    """Reads the drone index from frame keys such as 'drone2_frame_0010'."""
    for frame_name in frames_dict:
        match = re.match(r"drone(\d+)_", str(frame_name))
        if match:
            return int(match.group(1))
    return 0


def _get_active_exp_dir() -> str:
    """Automatically resolves the active experiment output directory.

    Checks environment variables first, then scans the local experiments directory
    for the latest numerical trial folder (e.g., experiments/exp_12).

    Returns:
        str: Path to the target experiment directory or default fallback.
    """
    if os.environ.get("CURRENT_EXP_DIR"):
        return os.environ["CURRENT_EXP_DIR"]

    exp_root = config.EXPERIMENTS_DIR
    if os.path.exists(exp_root):
        exp_dirs = []
        for entry in os.listdir(exp_root):
            full_path = os.path.join(exp_root, entry)
            if os.path.isdir(full_path) and entry.startswith("exp_"):
                num_part = entry.replace("exp_", "")
                if num_part.isdigit():
                    exp_dirs.append((int(num_part), full_path))
        if exp_dirs:
            # Select directory with the highest numerical index (most recent experiment)
            return max(exp_dirs, key=lambda item: item[0])[1]

    return config.DEFAULT_OUTPUT_DIR


@tool
def run_valen_uav_detection(
    frames_dict: Dict[str, Tuple[Image.Image, Tuple[int, int]]],
    incident_type: str = "fire",
    target_object: Union[str, List[str]] = "hazard",
    output_dir: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Executes onboard zero-shot vision-language verification on UAV frames using Valen.

    Args:
        frames_dict: Dictionary mapping frame identifiers to tuples of (PIL Image crop, (x, y) coordinates).
        incident_type: Benchmark domain key such as 'fire', 'oil_spill', 'car_accident', 'hogweed_thickets', or 'illegal_buildings'.
            Unknown types are verified with a generic question about target_object.
        target_object: Target semantic descriptor or label to detect.
        output_dir: Optional target folder to save detection plots and visual artifacts.

    Returns:
        List[Tuple[int, int]]: List of confirmed incident coordinates (x, y).
    """
    save_dir = output_dir or _get_active_exp_dir()
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(target_object, (list, tuple)):
        target_str = ", ".join(str(t) for t in target_object)
    else:
        target_str = str(target_object or "")
    target_str = target_str.strip() or config.UNKNOWN_TARGET_OBJECT

    # Known scenarios use their own Valen prompt; anything else is described by
    # the target object instead of being silently treated as another incident.
    hazard_type = canonical_incident(incident_type)
    label = hazard_type if hazard_type != config.UNKNOWN_INCIDENT else target_str

    drone_id = _drone_id_from_frames(frames_dict)
    print(f"🦅 [Valen UAV Tool] Drone #{drone_id} scanning frames for INCIDENT: '{label}'...")
    engine = ValenEngine.get_instance()

    conf_threshold = tools.detection_threshold(hazard_type)
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)

    selected = [
        (frame_name, img, coords)
        for idx, (frame_name, (img, coords)) in enumerate(frames_dict.items())
        if idx % config.DETECTION_FRAME_STRIDE == 0
    ]

    detected_coords = []
    detections = []
    chunk = max(1, config.VALEN_SUBMIT_CHUNK)

    # Frames are submitted in small chunks; the engine merges the chunks of all
    # drones into shared GPU batches, so the swarm is scanned in parallel.
    for start in range(0, len(selected), chunk):
        part = selected[start : start + chunk]
        probs = engine.evaluate_frames(
            [img for _, img, _ in part],
            incident_type=hazard_type,
            target_name=target_str,
        )

        for (frame_name, img, coords), prob in zip(part, probs):
            prob_pct = prob * 100.0

            if prob >= conf_threshold:
                print(f"🔥 [Valen MATCH] {label.upper()} detected at {coords} | Prob: {prob_pct:.2f}%")
                detected_coords.append(coords)
                detections.append(
                    {"frame": frame_name, "coords": tuple(coords), "probability": prob}
                )

                try:
                    save_path = os.path.join(
                        save_dir, f"valen_detected_{safe_name}_{frame_name}.png"
                    )
                    _save_detection_plot(
                        img,
                        f"Valen Match: {label.upper()} ({prob_pct:.1f}%)\nCoords: {coords}",
                        save_path,
                    )
                except Exception as e:
                    print(f"Warning: {e}")
            else:
                print(f"🖼️ [Valen Scan] {frame_name} — No {label} ({prob_pct:.1f}%)")

    tools.record_detections(drone_id, detections)
    print(f"✅ [Valen UAV Tool] Completed. Confirmed '{label}' locations: {len(detected_coords)}")
    return detected_coords
