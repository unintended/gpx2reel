import numpy as np

from gpx2reel import world as W
from tests.conftest import EXAG


def test_grid_covers_route_and_uv_in_range(setup):
    _, _, grid, path = setup
    assert grid.x[0] < path.xyz[:, 0].min() and grid.x[-1] > path.xyz[:, 0].max()
    assert grid.y[0] < path.xyz[:, 1].min() and grid.y[-1] > path.xyz[:, 1].max()
    assert 0 <= grid.uv.min() and grid.uv.max() <= 1


def test_grid_heights_exaggerated(setup):
    _, _, grid, _ = setup
    # synthetic cone peaks at 2650 m
    assert 2650 * EXAG * 0.8 < grid.z.max() < 2650 * EXAG * 1.05


def test_path_lies_on_terrain(setup):
    _, _, grid, path = setup
    ground = grid.height_at(path.xyz[:, 0], path.xyz[:, 1])
    assert np.allclose(path.xyz[:, 2] - ground, W.RIBBON_LIFT_M * EXAG)


def test_path_km_and_spacing(setup):
    route, _, _, path = setup
    assert np.all(np.diff(path.km) >= -1e-9)
    assert abs(path.km[-1] - route["totals"]["distance_km"]) < 0.05
    step = np.linalg.norm(np.diff(path.xyz[:, :2], axis=0), axis=1)
    same = np.diff(path.piece) == 0
    assert np.median(step[same]) < 25


def test_gap_arc_spans_the_gap(setup):
    route, _, _, path = setup
    arcs = W.gap_arcs(route, path, {})
    assert len(arcs) == 1
    pts = arcs[0]["points"]
    gap = route["gaps"][0]
    assert abs(np.linalg.norm(pts[-1, :2] - pts[0, :2]) / 1000 - gap["distance_km"]) < 1.0
    assert pts[:, 2].max() > max(pts[0, 2], pts[-1, 2]) + 300
    assert W.gap_arcs(route, path, {2: "skip"}) == []


def test_heading_is_unit(setup):
    _, _, _, path = setup
    assert np.allclose(np.linalg.norm(path.heading(), axis=1), 1, atol=1e-6)


def test_join_bridges_gap_on_the_ground(setup):
    route, _, grid, path = setup
    gap = route["gaps"][0]
    joined = W.join_gaps(route, path, grid, EXAG, {gap["after_day"]: "join"})
    assert len(np.unique(joined.piece)) == 1
    assert np.all(np.diff(joined.s) >= 0) and np.all(np.diff(joined.km) >= -1e-9)
    # s grows by the bridged distance, km by the ridden one
    assert abs((joined.s[-1] - joined.km[-1]) - gap["distance_km"]) < 0.1
    ground = grid.height_at(joined.xyz[:, 0], joined.xyz[:, 1])
    assert np.allclose(joined.xyz[:, 2] - ground, W.RIBBON_LIFT_M * EXAG)
    assert W.gap_arcs(route, joined, {gap["after_day"]: "join"}) == []


def test_straight_gap_keeps_boundary(setup):
    route, _, grid, path = setup
    kept = W.join_gaps(route, path, grid, EXAG, {})     # fixture gap is a transfer → straight
    assert len(np.unique(kept.piece)) == 2
    assert len(W.gap_arcs(route, kept, {})) == 1


def test_road_gap_follows_the_road_and_is_marked(setup):
    route, _, grid, path = setup
    gap = route["gaps"][0]
    i = np.flatnonzero(np.diff(path.piece))[0]
    a, b = path.xyz[i, :2], path.xyz[i + 1, :2]
    detour = np.array([a + (b - a) * 0.5 + [0, 20_000]])          # a road bending 20 km aside
    joined = W.join_gaps(route, path, grid, EXAG, {gap["after_day"]: "road"}, roads={gap["after_day"]: detour})
    assert len(np.unique(joined.piece)) == 1
    tr = joined.xyz[joined.transport]
    assert tr[:, 1].max() > max(a[1], b[1]) + 15_000              # went through the detour
    assert joined.transport.sum() > 0 and not joined.transport[0]


def test_grid_has_sea_mask(setup):
    _, _, grid, _ = setup
    assert grid.sea is not None and grid.sea.shape == grid.z.shape and grid.sea.dtype == bool
    assert not grid.sea[grid.z > 1].any()                  # land above the sea is never sea
