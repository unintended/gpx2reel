from pathlib import Path

import numpy as np
import pytest

from gpx2reel.ingest import IngestConfig, ingest, summary_text
from tests.fixtures.make_gpx import make_trip


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    folder = make_trip(tmp_path_factory.mktemp("trip") / "tracks")
    return ingest(sorted(folder.glob("*.gpx")), IngestConfig())


def test_days_by_local_date(route):
    assert route["timezone"] == "Asia/Kamchatka"
    assert route["totals"]["days"] == 3
    assert [d["date"] for d in route["days"]] == ["2026-07-10", "2026-07-11", "2026-07-12"]


def test_distances_close_to_generated(route):
    km = [d["distance_km"] for d in route["days"]]
    # generated 40 / 55 / 35 km; spike + jitter must not inflate distance
    for got, want in zip(km, [40, 55, 35]):
        assert abs(got - want) / want < 0.03, (got, want)


def test_spike_removed(route):
    assert any("spikes" in w for w in route["warnings"])


def test_gap_detected_as_transfer(route):
    gaps = [g for g in route["gaps"] if not g["within_day"]]
    assert len(gaps) == 1
    g = gaps[0]
    assert g["after_day"] == 2 and g["before_day"] == 3
    assert g["kind"] == "transfer"
    assert 30 < g["distance_km"] < 80


def test_km_monotonic_and_gap_excluded(route):
    last = -1
    for d in route["days"]:
        for s in d["segments"]:
            for p in s["points"]:
                assert p[4] >= last - 1e-9
                last = p[4]
    assert abs(last - route["totals"]["distance_km"]) < 0.05


def test_ascent_positive(route):
    assert route["totals"]["ascent_m"] > 0


def test_summary_renders(route):
    s = summary_text(route)
    assert "transfer" in s and "Day" in s


def test_file_mode(tmp_path):
    folder = make_trip(tmp_path / "tracks")
    r = ingest(sorted(folder.glob("*.gpx")), IngestConfig(day_mode="file"))
    assert [d["files"] for d in r["days"]] == [["day1.gpx"], ["day2.gpx"], ["day3.gpx"]]


def test_gap_kind_by_distance_after_night(tmp_path):
    from datetime import datetime, timezone

    from tests.fixtures.make_gpx import _day, write_gpx

    utc = timezone.utc
    d1 = _day(53.02, 158.65, 0, 10, datetime(2026, 7, 9, 21, 0, tzinfo=utc), seed=1)
    # next morning 2 km away (hotel off the track), then 4.5 km away (bus drop-off)
    d2 = _day(d1[-1][0] + 0.018, d1[-1][1], 0, 10, datetime(2026, 7, 10, 21, 0, tzinfo=utc), seed=2)
    d3 = _day(d2[-1][0] + 0.040, d2[-1][1], 0, 10, datetime(2026, 7, 11, 21, 0, tzinfo=utc), seed=3)
    for i, d in enumerate((d1, d2, d3), 1):
        write_gpx(tmp_path / f"day{i}.gpx", [d])
    r = ingest(sorted(tmp_path.glob("*.gpx")), IngestConfig())
    assert [g["kind"] for g in r["gaps"]] == ["overnight", "missing"]


def test_same_ride_from_two_devices_keeps_the_longer(tmp_path):
    from datetime import datetime, timezone

    from tests.fixtures.make_gpx import _day, write_gpx

    full = _day(53.02, 158.65, 20, 30, datetime(2026, 7, 9, 21, 0, tzinfo=timezone.utc), seed=1)
    write_gpx(tmp_path / "bike_computer.gpx", [full])
    write_gpx(tmp_path / "watch.gpx", [full[len(full) // 3:]])        # switched on later
    r = ingest(sorted(tmp_path.glob("*.gpx")), IngestConfig())
    assert r["totals"]["days"] == 1 and r["days"][0]["files"] == ["bike_computer.gpx"]
    assert abs(r["totals"]["distance_km"] - 30) < 1
    assert any("watch.gpx" in w for w in r["warnings"])


def _seg(name, t0, t1, source="gpx", asc=None, maker=None, climb=200.0):
    """A ride along a meridian: position and altitude are functions of time, so recordings overlap exactly."""
    from gpx2reel.ingest import Segment

    t = np.arange(t0, t1, 10.0)
    return Segment(name, 53.0 + t * 1e-6, np.full(len(t), 158.6), 100 + climb * t / 10_000, t, source=source,
                   rec_ascent=asc, rec_descent=0 if asc is not None else None, maker=maker)


def test_device_recording_beats_app_export():
    from gpx2reel.ingest import drop_duplicates

    keep, _ = drop_duplicates([_seg("komoot.gpx", 0, 10_000), _seg("watch.fit", 300, 10_000, "fit")])
    assert [s.file for s in keep] == ["watch.fit"]


def test_late_watch_gets_the_start_from_the_bike_computer():
    from gpx2reel.ingest import drop_duplicates

    keep, notes = drop_duplicates([_seg("magene.fit", 0, 10_000, "fit", asc=150),
                                   _seg("garmin.fit", 3_000, 10_000, "fit", asc=170, maker="garmin")])
    assert len(keep) == 1 and keep[0].file == "garmin.fit" and keep[0].also == ["magene.fit"]
    assert keep[0].t[0] < 10 and np.all(np.diff(keep[0].t) > 0)       # the whole ride, in order
    assert 170 + 55 < keep[0].rec_ascent < 170 + 65                    # + the climb of the stitched 30 %
    assert any("start" in n for n in notes)


def test_day_climb_comes_from_the_device_totals(monkeypatch, tmp_path):
    import gpx2reel.ingest as I

    f = tmp_path / "ride.fit"
    f.write_bytes(b"")
    monkeypatch.setattr(I, "parse_track", lambda p: [_seg(p.name, 1.78e9, 1.78e9 + 7200, "fit", asc=1723)])
    r = I.ingest([f])
    assert r["days"][0]["ascent_m"] == 1723 and r["days"][0]["ascent_source"] == "device"
