import re

import numpy as np
import pytest

from gpx2reel.geo import rdp_mask
from gpx2reel.ingest import ingest
from gpx2reel.sanitize import sanitize
from tests.fixtures.make_gpx import make_trip


def _seg_dist(p, a, b):
    ab = b - a
    u = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(p - a - u * ab)


def test_rdp_straight_line_keeps_ends():
    pts = np.column_stack([np.linspace(0, 1000, 50), np.zeros(50)])
    keep = rdp_mask(pts, 1.0)
    assert keep.sum() == 2 and keep[0] and keep[-1]


def test_rdp_within_tolerance():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5000, 2000)
    pts = np.column_stack([x, 40 * np.sin(x / 300) + rng.normal(0, 2, len(x))])
    eps = 10.0
    keep = rdp_mask(pts, eps)
    assert keep.sum() < len(pts) / 5
    kept = np.flatnonzero(keep)
    for i, j in zip(kept, kept[1:]):           # every dropped point is within eps of the segment that replaced it
        for k in range(i + 1, j):
            assert _seg_dist(pts[k], pts[i], pts[j]) <= eps + 1e-9


def test_rdp_keeps_out_and_back_turn():
    # out 1 km and back along the same line: the turn lies on the end points' line, but far from the segment
    x = np.concatenate([np.linspace(0, 1000, 11), np.linspace(900, 0, 10)])
    pts = np.column_stack([x, np.zeros(len(x))])
    keep = rdp_mask(pts, 5.0)
    assert keep[10]


def test_rdp_3d_keeps_a_climb():
    x = np.linspace(0, 2000, 201)
    z = np.where(np.abs(x - 1000) < 100, 50.0, 0.0)              # a 50 m bump on a straight road
    keep2 = rdp_mask(np.column_stack([x, np.zeros_like(x)]), 10.0)
    keep3 = rdp_mask(np.column_stack([x, np.zeros_like(x), z]), 10.0)
    assert keep2.sum() == 2 and keep3.sum() > 2


@pytest.fixture(scope="module")
def trip(tmp_path_factory):
    tracks = make_trip(tmp_path_factory.mktemp("trip") / "tracks")
    # a second recording of day 2 (another device) with sensor data and a device name
    text = (tracks / "day2.gpx").read_text()
    text = text.replace('creator="test"', 'creator="Garmin Edge 1040 serial 3412345678"')
    text = text.replace("</ele>", '</ele><extensions><gpxtpx:TrackPointExtension xmlns:gpxtpx="x"><gpxtpx:hr>131'
                                  "</gpxtpx:hr><gpxtpx:cad>88</gpxtpx:cad></gpxtpx:TrackPointExtension></extensions>")
    (tracks / "day2-edge.gpx").write_text(text)
    return tracks


@pytest.fixture(scope="module")
def route(trip):
    return ingest(sorted(trip.glob("*.gpx")))


def test_sanitized_reingests_as_the_same_days(route, tmp_path):
    out = tmp_path / "public"
    files = sanitize(route, out)
    assert [f.name for f in files] == ["day01-2026-07-10.gpx", "day02-2026-07-11.gpx", "day03-2026-07-12.gpx"]
    again = ingest(files)
    assert [d["date"] for d in again["days"]] == [d["date"] for d in route["days"]]
    assert [len(d["files"]) for d in again["days"]] == [1, 1, 1]
    for a, b in zip(route["days"], again["days"]):
        assert abs(a["distance_km"] - b["distance_km"]) / a["distance_km"] < 0.02
    n_in = sum(len(s["points"]) for d in route["days"] for s in d["segments"])
    n_out = sum(len(s["points"]) for d in again["days"] for s in d["segments"])
    assert n_out < n_in / 3


def test_sanitized_has_only_position_and_time(route, tmp_path):
    for f in sanitize(route, tmp_path):
        text = f.read_text()
        assert 'creator="gpx2reel"' in text
        for word in ("extensions", "hr>", "cad", "Garmin", "serial", "Edge"):
            assert word not in text
        assert "<ele>" in text and "<time>" in text


def test_trim_cuts_the_ends_of_every_day(route, tmp_path):
    full = ingest(sanitize(route, tmp_path / "a", tolerance_m=0))
    cut = ingest(sanitize(route, tmp_path / "b", tolerance_m=0, trim_m=500))
    for a, b in zip(full["days"], cut["days"]):
        assert a["distance_km"] - b["distance_km"] == pytest.approx(1.0, abs=0.05)
        assert b["start_time"] > a["start_time"] and b["end_time"] < a["end_time"]


def test_sanitize_keeps_tracks_without_elevation(tmp_path):
    src = make_trip(tmp_path / "tracks")
    for f in src.glob("*.gpx"):
        f.write_text(re.sub(r"<ele>[^<]*</ele>", "", f.read_text()))
    files = sanitize(ingest(sorted(src.glob("*.gpx"))), tmp_path / "out")
    assert len(files) == 3 and "<ele>" not in files[0].read_text()
