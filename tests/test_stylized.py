import numpy as np

from gpx2reel import landcover as LC
from gpx2reel import stylize as S
from gpx2reel.blender.stylized import TREE_SPACING_M, forest_points


def test_tiles_for_bbox():
    assert LC.tiles_for([31.5, 130.7, 33.6, 132.0]) == [(30, 129), (33, 129)]
    assert LC.tile_name(30, 129) == "N30E129" and LC.tile_name(-3, -60) == "S03W060"


def _lc(setup):
    route = setup[0]
    b = route["bbox"]
    bbox = [b[0] - 0.3, b[1] - 0.5, b[2] + 0.3, b[3] + 0.5]
    h, w = 200, 300
    cls = np.full((h, w), LC.TREE, np.uint8)
    cls[:, : w // 3] = LC.CROP                 # west third: fields
    return LC.LandCover(cls, bbox)


def test_landcover_sample_orientation(setup):
    lc = _lc(setup)
    b = lc.bbox
    lat = (b[0] + b[2]) / 2
    assert lc.sample(lat, b[1] + 0.01) == LC.CROP and lc.sample(lat, b[3] - 0.01) == LC.TREE


def test_style_texture_palette_and_water(setup):
    _, proj, grid, _ = setup
    tex = S.style_texture(proj, grid, _lc(setup), 128)
    assert tex.shape[1] == 128 and tex.shape[2] == 4
    forest = S._rgb(S.PALETTE[LC.TREE]).astype(int)
    land = tex[..., 3] == 0
    # most land texels are forest-coloured (up to the elevation tint)
    close = np.abs(tex[land][:, :3].astype(int) - forest).sum(1) < 60
    assert close.mean() > 0.3


def test_forest_points_on_trees_and_route_distance(setup):
    _, proj, grid, path = setup
    pts = forest_points(proj, grid, _lc(setup), path.xyz[:, :2])
    assert pts.shape[1] == 5 and len(pts) > 100
    lat, lon = proj.to_latlon(pts[:, 0], pts[:, 1])
    assert np.all(_lc(setup).sample(np.asarray(lat), np.asarray(lon)) == LC.TREE)
    near = np.min(np.linalg.norm(path.xyz[None, ::50, :2] - pts[:200, None, :2], axis=2), axis=1)
    assert np.all(pts[:200, 4] <= near + 1e-6)          # KD distance ≤ distance to a subsample
    assert np.median(np.diff(np.sort(np.unique(np.round(pts[:, 0] / TREE_SPACING_M))))) == 1


def test_tree_selection_is_static_with_soft_corridor():
    from gpx2reel.blender.stylized import CORRIDOR_KM, ROUTE_CLEAR_M, TREE_KEEP, select_trees

    rng = np.random.default_rng(0)
    n = 200_000
    rd = rng.uniform(0, 40_000, n)
    pts = np.column_stack([rng.random((n, 4)), rd])
    a, b = select_trees(pts), select_trees(pts)
    assert np.array_equal(a, b)                          # same seed → same trees every build
    assert a[:, 4].min() > ROUTE_CLEAR_M
    near = ((rd > ROUTE_CLEAR_M) & (rd < CORRIDOR_KM[0] * 1000)).sum()
    kept_near = (a[:, 4] < CORRIDOR_KM[0] * 1000).sum()
    assert abs(kept_near / near - TREE_KEEP) < 0.02
    mid = (a[:, 4] > (CORRIDOR_KM[0] + 4) * 1000) & (a[:, 4] < (CORRIDOR_KM[1] - 4) * 1000)
    assert 0 < mid.sum() < kept_near                     # thinning out, not a hard edge
    assert a[:, 4].max() < CORRIDOR_KM[1] * 1000


def test_shore_gradient_lightens_water_near_land():
    from gpx2reel.geo import LocalProjection
    from gpx2reel.world import TerrainGrid

    proj = LocalProjection(33.0, 131.0)
    x = np.arange(0, 40_000, 500.0)
    y = np.arange(0, 20_000, 500.0)
    z = np.zeros((len(y), len(x)), np.float32)
    z[:, : len(x) // 4] = 50.0                              # land on the west quarter, sea elsewhere
    grid = TerrainGrid(x, y, z, np.zeros((len(y), len(x), 2)))
    lat0, lon0 = proj.to_latlon(x[0], y[0])
    lat1, lon1 = proj.to_latlon(x[-1], y[-1])
    lc = LC.LandCover(np.full((20, 40), LC.GRASS, np.uint8), [float(lat0), float(lon0), float(lat1), float(lon1)])
    tex = S.style_texture(proj, grid, lc, 160)
    row = tex[tex.shape[0] // 2]
    sea = np.flatnonzero(row[:, 3] == 255)
    near, far = row[sea[2], :3].astype(int), row[sea[-1], :3].astype(int)
    assert near.sum() > far.sum() + 30                      # shallow rim is paler than the open sea
