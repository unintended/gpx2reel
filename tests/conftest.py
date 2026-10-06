import json

import numpy as np
import pytest

from gpx2reel import terrain as T
from gpx2reel import world as W
from gpx2reel.ingest import ingest
from tests.fixtures.make_gpx import ele_at, make_trip
from tests.test_terrain import fake_fetch_factory

EXAG = 1.5


@pytest.fixture(scope="session")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("world")
    route = ingest(sorted(make_trip(tmp / "tracks").glob("*.gpx")))
    mp = pytest.MonkeyPatch()
    mp.setattr(T, "CACHE", tmp / "cache")
    cone = np.vectorize(lambda lat, lon: ele_at(lat, lon))
    T.download(route, tmp / "terrain", buffer_km=3, dem_zoom=10, ortho_zoom=10,
               fetch=fake_fetch_factory(cone), log=lambda *_: None)
    mp.undo()
    meta = json.loads((tmp / "terrain" / "terrain.json").read_text())
    proj = W.projection_of(route)
    grid = W.terrain_grid(proj, T.Dem.load(tmp / "terrain"), meta, EXAG, step_m=200)
    path = W.route_path(route, proj, grid, EXAG)
    return route, proj, grid, path


