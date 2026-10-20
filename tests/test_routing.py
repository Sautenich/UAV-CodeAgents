"""Tests for swarm route planning."""

from src.agent_factory import balance_routes, optimize_route, plan_swarm_routes, route_length

HOME = (0, 0)


def _points(coords):
    return [{"point_2d": list(c), "label": f"p{i}"} for i, c in enumerate(coords)]


def test_every_point_is_assigned_exactly_once():
    points = _points([(10, 10), (900, 50), (500, 500), (50, 900), (800, 800), (300, 100)])
    routes = plan_swarm_routes(HOME, points, 3)
    assert len(routes) == 3
    visited = [p["label"] for route in routes for p in route]
    assert sorted(visited) == sorted(p["label"] for p in points)


def test_nearest_neighbour_order():
    points = [(300, 0), (100, 0), (200, 0)]
    assert optimize_route(HOME, points) == [(100, 0), (200, 0), (300, 0)]


def test_balancing_spreads_distant_clusters():
    points = [(1000, 0), (990, 10), (0, 1000), (10, 990)]
    routes = balance_routes(HOME, points, 2)
    assert sorted(len(r) for r in routes) == [2, 2]
    lengths = [route_length(HOME, optimize_route(HOME, r)) for r in routes]
    assert abs(lengths[0] - lengths[1]) < 50


def test_fewer_points_than_drones_leaves_empty_routes():
    routes = plan_swarm_routes(HOME, [(5, 5)], 3)
    assert sorted(len(r) for r in routes) == [0, 0, 1]
    assert route_length(HOME, []) == 0.0
