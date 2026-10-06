"""Synthetic multi-day trip for tests: 3 days, a transfer gap, GPS spike, jitter at a stop."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

VOLCANO = (53.255, 158.83)  # synthetic cone centre


def ele_at(lat: float, lon: float) -> float:
    dy = (lat - VOLCANO[0]) * 111_000
    dx = (lon - VOLCANO[1]) * 111_000 * math.cos(math.radians(lat))
    r = math.hypot(dx, dy)
    return 50 + 2600 * math.exp(-((r / 6000) ** 2))


def _day(lat0, lon0, heading_deg, km, start: datetime, speed_kmh=18, step_s=5, wiggle=0.002, seed=0):
    rng = np.random.default_rng(seed)
    n = int(km / (speed_kmh / 3600 * step_s))
    pts = []
    lat, lon = lat0, lon0
    h = math.radians(heading_deg)
    step_m = speed_kmh / 3.6 * step_s
    for i in range(n):
        hh = h + 0.6 * math.sin(i / 60) + rng.normal(0, 0.02)
        lat += step_m * math.cos(hh) / 111_000
        lon += step_m * math.sin(hh) / (111_000 * math.cos(math.radians(lat)))
        pts.append((lat, lon, ele_at(lat, lon), start + timedelta(seconds=i * step_s)))
    return pts


def write_gpx(path: Path, segments: list[list[tuple]]):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1"><trk><name>t</name>']
    for seg in segments:
        lines.append("<trkseg>")
        for lat, lon, ele, t in seg:
            lines.append(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"><ele>{ele:.1f}</ele>'
                         f'<time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>')
        lines.append("</trkseg>")
    lines.append("</trk></gpx>")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_trip(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    utc = timezone.utc
    # Kamchatka is UTC+12: 21:00Z = 09:00 local next day
    d1 = _day(53.02, 158.65, 20, 40, datetime(2026, 7, 9, 21, 0, tzinfo=utc), seed=1)
    # stationary jitter (lunch stop) in the middle of day 1
    mid = len(d1) // 2
    la, lo, el, t = d1[mid]
    rng = np.random.default_rng(5)
    stop = [(la + rng.normal(0, 1e-5), lo + rng.normal(0, 1e-5), el, t + timedelta(seconds=5 * k)) for k in range(1, 200)]
    shift = timedelta(seconds=5 * 200)
    d1 = d1[: mid + 1] + stop + [(a, b, c, tt + shift) for a, b, c, tt in d1[mid + 1 :]]
    # GPS spike in day 1
    k = len(d1) // 4
    a, b, c, tt = d1[k]
    d1[k] = (a + 0.05, b + 0.05, c, tt)

    last = d1[-1]
    d2 = _day(last[0], last[1], 60, 55, datetime(2026, 7, 10, 20, 30, tzinfo=utc), seed=2)
    last = d2[-1]
    # transfer ~50 km (bus) before day 3
    d3 = _day(last[0] + 0.3, last[1] + 0.5, 150, 35, datetime(2026, 7, 11, 22, 0, tzinfo=utc), seed=3)

    write_gpx(folder / "day1.gpx", [d1])
    write_gpx(folder / "day2.gpx", [d2])
    write_gpx(folder / "day3.gpx", [d3])
    return folder


if __name__ == "__main__":
    import sys

    make_trip(Path(sys.argv[1] if len(sys.argv) > 1 else "sample_trip/tracks"))
