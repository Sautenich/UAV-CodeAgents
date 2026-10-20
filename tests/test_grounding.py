"""Tests for the keypoint grounding robustness filters."""

from src.tools import looks_synthetic, normalize_points, select_best_answer


def _pts(coords, label="house"):
    return [{"point_2d": list(c), "label": label} for c in coords]


# Answer actually returned by the VLM for a map whose buildings are elsewhere
FABRICATED = [(420, 22), (420, 110), (420, 198), (419, 288), (419, 377), (420, 467),
              (413, 556), (416, 647), (418, 757), (182, 778), (182, 45), (65, 47)]
# Buildings along a diagonal village street
REAL = [(897, 391), (817, 506), (793, 532), (818, 546), (761, 574), (742, 601),
        (743, 613), (692, 683), (702, 725), (727, 701), (677, 767), (653, 750)]


def test_fabricated_column_is_detected():
    assert looks_synthetic(_pts(FABRICATED)) is True
    assert looks_synthetic(_pts([(y, x) for x, y in FABRICATED])) is True


def test_real_layouts_are_accepted():
    assert looks_synthetic(_pts(REAL)) is False
    # Houses along an exactly vertical street are irregularly spaced
    street = [(500, 100), (503, 160), (498, 300), (505, 330), (500, 520), (497, 700)]
    assert looks_synthetic(_pts(street)) is False
    assert looks_synthetic([]) is False


def test_best_answer_is_the_one_others_confirm():
    scattered = _pts([(50, 900), (300, 120), (640, 410), (120, 480), (880, 60), (510, 950)])
    shifted = _pts([(x + 4, y - 3) for x, y in REAL])
    best, confirmed = select_best_answer([scattered, _pts(REAL), shifted], radius=20)
    assert [tuple(p["point_2d"]) for p in best] == REAL
    assert confirmed == len(REAL)


def test_area_targets_keep_a_complete_answer():
    # Open water: every answer marks different, equally valid spots
    water_a = _pts([(700, 300), (850, 500), (600, 700), (300, 800), (900, 150)], "open water")
    water_b = _pts([(650, 450), (800, 250), (450, 750), (950, 600)], "open water")
    water_c = _pts([(750, 650), (550, 350), (350, 600), (880, 820)], "open water")
    best, confirmed = select_best_answer([water_a, water_b, water_c], radius=20)
    assert confirmed == 0
    assert len(best) == 5


def test_confirmed_points_come_first_and_the_cap_applies():
    run_a = _pts([(900, 900), (100, 100), (500, 500)])
    run_b = _pts([(104, 97), (498, 505)])
    best, confirmed = select_best_answer([run_a, run_b], radius=20, max_points=2)
    assert confirmed == 2
    assert [tuple(p["point_2d"]) for p in best] == [(100, 100), (500, 500)]


def test_nothing_is_returned_when_most_answers_are_empty():
    assert select_best_answer([[], [], _pts(REAL)], radius=20) == ([], 0)
    assert select_best_answer([[], []], radius=20) == ([], 0)
    assert len(select_best_answer([_pts(REAL)], radius=20)[0]) == len(REAL)


def test_normalize_points_drops_malformed_and_out_of_grid():
    raw = [{"point_2d": [10, 20], "label": "a"}, {"point_2d": [5000, 1]}, {"x": 1}, "junk", [30, 40]]
    points = normalize_points(raw, "target")
    assert [tuple(p["point_2d"]) for p in points] == [(10, 20), (30, 40)]
    assert [p["label"] for p in points] == ["a", "target"]
    assert normalize_points("not a list", "target") == []


# Pixel keypoints returned for a 1068x1068 map and the same points after an
# agent rescaled them to the 0..1000 grid (taken from a real run)
PIXELS = [(419, 670), (708, 673), (672, 752), (557, 530), (499, 541), (1034, 778)]
RESCALED = [(392, 628), (663, 630), (629, 704), (522, 496), (467, 507), (969, 729)]


def test_rescaled_keypoints_are_restored():
    from src.tools import undo_normalization

    reference = _pts(PIXELS)
    restored, changed = undo_normalization(_pts(RESCALED, "x"), reference, (1068, 1068), grid=1000)
    assert changed is True
    assert [tuple(p["point_2d"]) for p in restored] == PIXELS
    assert restored[0]["label"] == "x"


def test_unchanged_or_unrelated_keypoints_are_kept():
    from src.tools import undo_normalization

    reference = _pts(PIXELS)
    same, changed = undo_normalization(_pts(PIXELS[:3]), reference, (1068, 1068), grid=1000)
    assert changed is False and [tuple(p["point_2d"]) for p in same] == PIXELS[:3]
    other = _pts([(10, 10), (500, 900), (333, 444)])
    assert undo_normalization(other, reference, (1068, 1068), grid=1000) == (other, False)
    # Non-square map: each axis has its own scale
    wide = _pts([(1920 * 0.5, 1080 * 0.5), (1920 * 0.9, 1080 * 0.2), (300, 900)])
    scaled = _pts([(500, 500), (900, 200), (156, 833)])
    restored, changed = undo_normalization(scaled, wide, (1920, 1080), grid=1000)
    assert changed is True and tuple(restored[1]["point_2d"]) == (1728.0, 216.0)


def test_even_spread_over_an_area_is_not_treated_as_fabricated():
    grid = [(x, y) for y in (300, 480, 660) for x in (150, 330, 510, 690, 870)]
    as_area = normalize_points(
        [{"point_2d": list(c), "label": "open water", "kind": "area"} for c in grid], "t"
    )
    assert all(p["kind"] == "area" for p in as_area)
    assert looks_synthetic(as_area) is False
    # The same pattern claimed for compact objects is still rejected
    as_objects = normalize_points([{"point_2d": list(c), "label": "house"} for c in grid], "t")
    assert all(p["kind"] == "object" for p in as_objects)
    assert looks_synthetic(as_objects) is True
