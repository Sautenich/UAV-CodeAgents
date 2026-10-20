"""Tests for the swarm dispatch tool with stand-in UAV agents."""

import os
import threading

from PIL import Image

import src.agent_factory as swarm
import src.tools as tools


class FakeExecutor:
    def __init__(self):
        self.state = {}


class FakeDrone:
    """Reports a detection on the first frame it receives."""

    def __init__(self, drone_id, barrier):
        self.name = f"uav_agent_{drone_id}"
        self.drone_id = drone_id
        self.python_executor = FakeExecutor()
        self.barrier = barrier

    def __call__(self, task):
        # Passes only if all drones are inside their task at the same time
        self.barrier.wait(timeout=10)
        frames = self.python_executor.state["frames_dict"]
        frame_name, (_, coords) = next(iter(frames.items()))
        tools.record_detections(
            self.drone_id,
            [{"frame": frame_name, "coords": coords, "probability": 0.9}],
        )
        return "done"


def test_dispatch_swarm_runs_all_drones_in_parallel(tmp_path, monkeypatch):
    monkeypatch.delenv("CURRENT_EXP_DIR", raising=False)
    barrier = threading.Barrier(3)
    swarm.register_drones([FakeDrone(i, barrier) for i in range(3)])

    image = Image.new("RGB", (400, 400), (40, 90, 40))
    keypoints = [
        {"point_2d": [x, y], "label": "house"}
        for x, y in [(50, 60), (350, 40), (200, 200), (60, 340), (330, 330), (120, 250)]
    ]

    mission = swarm.dispatch_swarm(
        target_image=image,
        keypoints=keypoints,
        incident_type="fire",
        target_object="fire",
        home_point=(10, 10),
        output_dir=str(tmp_path),
    )

    assert mission["incident_detected"] is True
    assert mission["total_detections"] == 3
    assert set(mission["detections"]) == {"uav_agent_0", "uav_agent_1", "uav_agent_2"}
    assert sum(len(route) for route in mission["routes"].values()) == len(keypoints)
    assert all(t > 0 for t in mission["flight_times_s"].values())
    assert os.path.exists(tmp_path / "uav_flight_all_drones.mp4")


def test_dispatch_swarm_with_fewer_points_than_drones(tmp_path, monkeypatch):
    monkeypatch.delenv("CURRENT_EXP_DIR", raising=False)
    barrier = threading.Barrier(1)
    swarm.register_drones([FakeDrone(i, barrier) for i in range(3)])

    mission = swarm.dispatch_swarm(
        target_image=Image.new("RGB", (300, 300)),
        keypoints=[{"point_2d": [150, 150], "label": "house"}],
        incident_type="fire",
        home_point=(10, 10),
        output_dir=str(tmp_path),
    )

    assert len(mission["routes"]) == 1
    assert os.path.exists(tmp_path / "uav_flight_all_drones.mp4")
