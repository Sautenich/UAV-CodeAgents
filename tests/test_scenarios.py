"""Tests for scenario naming and keyword resolution."""

import config
from src.tools import canonical_incident, get_folder_by_query, match_scenario, resolve_scenario


def test_aliases_resolve_to_canonical_key():
    assert canonical_incident("cars_accident") == "car_accident"
    assert canonical_incident("Car Accident") == "car_accident"
    by_folder = {v["folder"]: k for k, v in config.SCENARIOS.items()}
    assert canonical_incident("illegal_buildings") == by_folder["illegal_buildings"]
    assert canonical_incident("something else") == config.UNKNOWN_INCIDENT


def test_keywords_match_whole_words_only():
    assert match_scenario("Be careful with the scarf") is None
    assert match_scenario("Water the plants near the house") is None
    assert match_scenario("Two cars collided on the highway") == "car_accident"
    assert match_scenario("Smoke is rising over the warehouse") == "fire"


def test_default_requests_resolve_to_their_own_scenario():
    for key, scenario in config.SCENARIOS.items():
        assert match_scenario(scenario["request"]) == key


def test_resolve_scenario_from_console_input():
    assert resolve_scenario("")["incident_type"] == config.DEFAULT_SCENARIO
    assert resolve_scenario("cars_accident")["folder"] == "cars_accident"
    custom = resolve_scenario("There is an oil leak near the pier")
    assert custom["incident_type"] == "oil_spill"
    assert custom["request"] == "There is an oil leak near the pier"


def test_get_folder_by_query_uses_one_naming_scheme():
    info = get_folder_by_query("A car accident has occurred.")
    assert info == {
        "folder": "cars_accident",
        "incident_type": "car_accident",
        "target_object": "car accident",
    }
    unknown = get_folder_by_query("Count the sheep")
    assert unknown["incident_type"] == config.UNKNOWN_INCIDENT
    assert unknown["folder"] == config.BASELINE_FOLDER


def test_forest_fire_is_selected_only_when_a_forest_is_mentioned():
    assert match_scenario("A forest fire is burning near the village") == "forest_fire"
    assert match_scenario("Smoke and flames are rising over the woodland") == "forest_fire"
    assert match_scenario("A house is on fire, I see smoke and flames") == "fire"
    assert match_scenario(config.SCENARIOS["fire"]["request"]) == "fire"
    assert canonical_incident("wildfire") == "forest_fire"


def test_forest_fire_pairs_targets_with_their_baseline_maps():
    import contextlib
    import io
    import re

    from src.data_loader import load_local_map_images

    for k, baseline_number in enumerate(config.SCENARIOS["forest_fire"]["baseline_map"], start=1):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            base, target = load_local_map_images(folder_name="forest_fire", index=k)
        names = re.search(r"reference '([^']+)',\s+target '([^']+)'", log.getvalue())
        assert names.group(1).startswith(f"{baseline_number}_square")
        assert names.group(2) == f"{k}_square.jpg"
        assert base.size == target.size


def test_oil_spill_pairs_every_target_with_its_own_baseline():
    import contextlib
    import io
    import re

    from src.data_loader import load_local_map_images

    for k, number in enumerate(config.SCENARIOS["oil_spill"]["baseline_map"], start=1):
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            load_local_map_images(folder_name="oil_spill", index=k)
        names = re.search(r"reference '([^']+)',\s+target '([^']+)'", log.getvalue())
        assert names.group(1).startswith(f"{number}_square")
        assert names.group(2).startswith(f"{number}_square")


def test_baseline_is_resized_to_a_target_of_the_same_aspect(tmp_path):
    from PIL import Image

    from src.data_loader import load_local_map_images

    (tmp_path / "default").mkdir()
    (tmp_path / "fire").mkdir()
    Image.new("RGB", (1500, 900)).save(tmp_path / "default" / "1_square.jpg")
    Image.new("RGB", (1000, 600)).save(tmp_path / "fire" / "1_square.jpg")
    base, target = load_local_map_images(folder_name="fire", index=1, base_dir=str(tmp_path))
    assert base.size == target.size == (1000, 600)


def test_detection_threshold_falls_back_to_the_default():
    from src.tools import detection_threshold

    assert detection_threshold("something else") == config.DETECTION_CONF_THRESHOLD
    for key, scenario in config.SCENARIOS.items():
        expected = scenario.get("detection_threshold", config.DETECTION_CONF_THRESHOLD)
        assert detection_threshold(key) == expected
