"""Points of interest along the route from OpenStreetMap (Overpass): peaks, volcanoes, towns.

`gpx2reel poi <trip>` writes build/poi.json with ranked candidates; Claude picks the ones worth a label
and writes them into track.yaml `labels:`. Nothing here decides what goes into the reel.
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .geo import rdp_mask
from .trip import TripPaths

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
UA = "gpx2reel/0.1 (personal project)"
PEAK_RADIUS_M = 10_000
TOWN_RADIUS_M = 4_000


def route_line(route: dict, max_points: int = 150) -> np.ndarray:
    """Simplified (lat, lon) polyline of the whole route for Overpass `around`."""
    pts = np.array([[q[0], q[1]] for d in route["days"] for s in d["segments"] for q in s["points"]])
    eps = 1e-4
    keep = rdp_mask(pts, eps)
    while keep.sum() > max_points:
        eps *= 1.6
        keep = rdp_mask(pts, eps)
    return pts[keep]


def build_query(line: np.ndarray) -> str:
    coords = ",".join(f"{a:.5f},{b:.5f}" for a, b in line)
    return (
        "[out:json][timeout:90];("
        f'node["natural"~"^(peak|volcano)$"]["name"](around:{PEAK_RADIUS_M},{coords});'
        f'node["place"~"^(city|town)$"]["name"](around:{TOWN_RADIUS_M},{coords});'
        + "".join(f'node["place"~"^(city|town|village)$"]["name"](around:1500,{a:.5f},{b:.5f});'
                  for a, b in (line[0], line[-1]))                      # start / finish may be a village
        + ");out body;"
    )


def fetch(query: str, log=print, rounds: int = 5) -> dict:
    """POST to the Overpass mirrors in turn; busy servers answer 429/504 — retry with a growing pause."""
    import time

    data = urllib.parse.urlencode({"data": query}).encode()
    last = None
    for r in range(rounds):
        for url in ENDPOINTS:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=200) as resp:
                    return json.loads(resp.read())
            except Exception as e:  # noqa: BLE001 — busy mirrors answer with HTML, 429/504 or time out
                log(f"  {url}: {str(e)[:80]}")
                last = e
        time.sleep(15 * (r + 1))
    raise RuntimeError(f"Overpass unavailable: {last}")


def _num(v) -> float | None:
    try:
        return float(str(v).split()[0].replace(",", "."))
    except (TypeError, ValueError):
        return None


def name_of(tags: dict, lang: str = "en") -> str:
    return tags.get(f"name:{lang}") or tags.get("name:en") or tags.get("name")


# per kind: max distance from the track (km) and how the score is built
MAX_DIST_KM = {"peak": 10, "town": 4, "lake": 5, "wetland": 5, "waterfall": 3, "viewpoint": 2, "sight": 2}
KIND_TITLES = {"peak": "Peaks", "town": "Towns", "lake": "Lakes", "wetland": "Wetlands", "waterfall": "Waterfalls",
               "viewpoint": "Viewpoints", "sight": "Sights"}


def candidates(route: dict, osm: dict, lang: str = "en") -> list[dict]:
    """Rank OSM places against the route (elements: [{lat, lon, tags, area_km2?}]).
    Peaks by height, towns by population, lakes by area, the rest by being well known (wikidata) and close."""
    from .osm import _poi_kind

    pts = np.array([[q[0], q[1], q[4]] for d in route["days"] for s in d["segments"] for q in s["points"]])
    out = []
    for el in osm.get("elements", []):
        tags = el.get("tags", {})
        kind = _poi_kind(tags)
        if kind is None:
            continue
        lat, lon = el["lat"], el["lon"]
        d = _dist_to_points(lat, lon, pts)
        i = int(np.argmin(d))
        dist = float(d[i]) / 1000
        if dist > MAX_DIST_KM[kind]:
            continue
        ele = _num(tags.get("ele"))
        pop = _num(tags.get("population"))
        wiki = 1.4 if tags.get("wikidata") else 1.0
        c = {
            "name": name_of(tags, lang), "name_local": tags.get("name"), "kind": kind,
            "volcano": tags.get("natural") == "volcano", "lat": lat, "lon": lon,
            "ele": ele, "population": pop, "place": tags.get("place"), "area_km2": el.get("area_km2"),
            "dist_km": round(dist, 2), "at_km": round(float(pts[i, 2]), 1), "wikidata": tags.get("wikidata"),
        }
        if kind == "peak":
            if ele is None:
                continue
            score = ele / 1000 * (1.3 if c["volcano"] else 1.0) * (1.2 if c["wikidata"] else 1.0) / (1 + dist / 5)
        elif kind == "town":
            score = math.log10(max(pop or 1000, 1000)) / (1 + dist / 3)
        elif kind in ("lake", "wetland"):
            score = (1 + math.log10(1 + 20 * (c["area_km2"] or 0.01))) * wiki / (1 + dist / 3)
        else:
            score = wiki / (1 + dist / 1.5) * (1.3 if kind == "waterfall" else 1.0)
        c["score"] = round(score, 3)
        out.append(c)
    start, end = pts[0], pts[-1]
    towns = [c for c in out if c["kind"] == "town"]
    for label, p in (("start", start), ("finish", end)):
        if towns:
            best = min(towns, key=lambda c: _dist_to_points(c["lat"], c["lon"], p[None])[0])
            best.setdefault("role", []).append(label)
    # the same place is often mapped several times; keep the highest-scored per name
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: -c["score"]):
        key = (c["kind"], c["name"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _dist_to_points(lat: float, lon: float, pts: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.minimum(a, 1)))


def run(p: TripPaths, refresh: bool = False, lang: str = "en", log=print) -> list[dict]:
    """Candidates from build/osm/poi_osm.json (Geofabrik extract, `gpx2reel osm`) or else Overpass."""
    from .ingest import load_route

    build = p.build
    route = load_route(p.route)
    local = build / "osm" / "poi_osm.json"
    raw = build / "poi_raw.json"
    if local.exists():
        osm = {"elements": json.loads(local.read_text(encoding="utf-8"))}
    elif raw.exists() and not refresh:
        osm = json.loads(raw.read_text(encoding="utf-8"))
    else:
        osm = fetch(build_query(route_line(route)), log=log)
        raw.write_text(json.dumps(osm, ensure_ascii=False), encoding="utf-8")
    cands = candidates(route, osm, lang)
    (build / "poi.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    return cands


def table(cands: list[dict], top: int = 25) -> str:
    rows = []
    for kind, title in KIND_TITLES.items():
        items = [c for c in cands if c["kind"] == kind][:top]
        if not items:
            continue
        rows.append(f"{title}:")
        for c in items:
            if kind == "peak":
                extra = f"{c['ele']:.0f} m" + (" volcano" if c["volcano"] else "")
            elif kind == "town":
                extra = f"{c['place']}, pop. {c['population']:.0f}" if c["population"] else c["place"]
            elif kind in ("lake", "wetland"):
                extra = f"{c['area_km2']:.2f} km²" if c["area_km2"] else ""
            else:
                extra = "wikidata" if c["wikidata"] else ""
            role = f" [{', '.join(c['role'])}]" if c.get("role") else ""
            rows.append(f"  {c['score']:6.2f}  {c['name']:<28} {extra:<22} {c['dist_km']:5.1f} km from track, "
                        f"at {c['at_km']:.0f} km  ({c['lat']:.4f}, {c['lon']:.4f}){role}")
        rows.append("")
    return "\n".join(rows)
