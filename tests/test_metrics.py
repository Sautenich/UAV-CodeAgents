"""Tests for detection quality metrics."""

import json

from src.data_loader import load_ground_truth, score_detections


def test_no_ground_truth_gives_empty_metrics():
    quality = score_detections([(1, 1)], None)
    assert quality["precision"] is None and quality["recall"] is None


def test_precision_and_recall():
    truth = [(100, 100), (500, 500)]
    detections = [(110, 95), (90, 105), (900, 900)]
    quality = score_detections(detections, truth, radius=50)
    assert quality["true_positives"] == 2
    assert quality["false_positives"] == 1
    assert quality["false_negatives"] == 1
    assert abs(quality["precision"] - 2 / 3) < 1e-9
    assert quality["recall"] == 0.5


def test_no_detections():
    quality = score_detections([], [(1, 1)], radius=10)
    assert quality["precision"] is None
    assert quality["recall"] == 0.0


def test_load_ground_truth(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"fire": {"3": [[10, 20]], "4": []}}))
    assert load_ground_truth("fire", 3, str(path)) == [[10, 20]]
    assert load_ground_truth("fire", 4, str(path)) == []
    assert load_ground_truth("fire", 5, str(path)) is None
    assert load_ground_truth("fire", 3, str(tmp_path / "missing.json")) is None
