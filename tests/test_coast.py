"""Beach set placement (coast.py): sea side, shore fade, palms on the backshore, the camera's clear lane."""
import numpy as np

from gpx2reel import coast as C


class Flat:
    """Land at 3 m for y > 0, sea (0) below — the coastline runs along the x axis."""

    def height_at(self, x, y):
        return np.where(np.asarray(y) > 0, 3.0, 0.0)


# OSM keeps the land on the left: walking +x with land at y > 0
COAST = [np.array([[-2000.0, 0.0], [2000.0, 0.0]])]


def test_sea_side_is_right_of_the_coastline():
    xy = np.array([[0, 50.0], [0, -50.0], [500, 5.0], [500, -5.0]])
    assert C.sea_side(COAST, xy).tolist() == [False, True, False, True]


def test_shore_alpha_fades_out_near_the_coast():
    a = C.shore_alpha(COAST, np.array([[0, -10.0], [0, -130.0], [0, -1000.0]]))
    assert a[0] == 0 and 0 < a[1] < 1 and a[2] == 1


def test_shore_alpha_ignores_small_islets():
    rock = np.array([[0, -500.0], [30, -500.0], [30, -470.0], [0, -470.0], [0, -500.0]])
    a = C.shore_alpha(COAST + [rock], np.array([[15, -485.0]]))
    assert a[0] == 1                                   # open water around a rock, no hole


def test_palms_stand_inland_on_the_backshore():
    p = C.palm_points(COAST, Flat(), np.array([0.0, 0.0]), exaggeration=1.0, radius=800)
    assert len(p) > 20
    assert (p[:, 1] > 0).all() and (p[:, 1] < C.PALM_INLAND_M[1] + 10).all()
    assert (np.abs(p[:, 0]) < 800).all()


def test_flatten_sea_lays_shallows_at_sea_level():
    from gpx2reel.world import TerrainGrid

    x = y = np.arange(-100.0, 101.0, 10.0)
    z = np.full((len(y), len(x)), 1.0, np.float32)     # 1 m everywhere: noise in the shallows, a low shore
    g = TerrainGrid(x, y, z, np.zeros((len(y), len(x), 2), np.float32))
    C.flatten_sea(g, COAST, exaggeration=1.0)
    assert (g.z[y < 0] == 0).all() and (g.z[y > 0] == 1).all()


def test_clear_lane_drops_houses_in_view():
    start, d = np.array([0.0, 100.0]), np.array([0.0, -1.0])            # looking at the sea (−y)
    boxes = np.array([[0, 150, 3, 8, 8, 6, 0], [100, 150, 3, 8, 8, 6, 0], [5, 60, 3, 8, 8, 6, 0]], float)
    kept = C.clear_lane(boxes, start, d, back_m=110)
    assert kept[:, 0].tolist() == [100]


def test_grove_frames_the_view_and_leaves_the_middle_open():
    start, d = np.array([0.0, 300.0]), np.array([0.0, -1.0])
    g = C.grove_points(start, d, 110, Flat(), exaggeration=1.0)
    assert len(g) > 10
    assert (np.abs(g[:, 0]) >= 8).all()                 # a lane for the sun in the middle
