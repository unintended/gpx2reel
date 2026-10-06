"""Road geometry for gaps travelled by bus / car (storyboard gap mode `road`), from the public OSRM server."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

OSRM = "https://router.project-osrm.org/route/v1/{profile}/{a_lon},{a_lat};{b_lon},{b_lat}?overview=full&geometries=geojson"
UA = "gpx2reel/0.1 (personal project)"


def fetch_road(a: tuple[float, float], b: tuple[float, float], profile: str = "driving") -> list[list[float]]:
    """(lat, lon) → (lat, lon): road polyline as [[lat, lon], …]."""
    url = OSRM.format(profile=profile, a_lat=a[0], a_lon=a[1], b_lat=b[0], b_lon=b[1])
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM: no route {a} → {b}: {data.get('code')}")
    return [[lat, lon] for lon, lat in data["routes"][0]["geometry"]["coordinates"]]


def road_gaps(route: dict, modes: dict[int, str], cache: Path, log=print) -> dict[int, list[list[float]]]:
    """after_day → road polyline for every gap in mode 'road'; cached in `cache` (JSON)."""
    stored = json.loads(cache.read_text()) if cache.exists() else {}
    out = {}
    for g in route["gaps"]:
        d = g["after_day"]
        if g["within_day"] or modes.get(d) != "road":
            continue
        key = f"{d}:{g['from'][0]:.5f},{g['from'][1]:.5f}:{g['to'][0]:.5f},{g['to'][1]:.5f}"
        if key not in stored:
            log(f"Road for the gap after day {d} (OSRM)…")
            stored[key] = fetch_road(tuple(g["from"][:2]), tuple(g["to"][:2]))
        out[d] = stored[key]
    cache.write_text(json.dumps(stored))
    return out
