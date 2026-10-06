"""OSM data for the stylized scene from a Geofabrik extract (PBF), parsed locally with pyosmium.

Overpass is often overloaded; a regional extract is one fast download and works offline afterwards.
The smallest Geofabrik region covering the terrain area is picked from their index. One pass gives:
  vectors.json — major roads and rivers as lat/lon polylines (drawn as ribbons in blender/stylized.py);
  poi_osm.json — named peaks, volcanoes, waterfalls, lakes, viewpoints, attractions, shrines/temples
                 (poi.py ranks them against the route).
"""
from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

import numpy as np
from .trip import TripPaths

INDEX_URL = "https://download.geofabrik.de/index-v1.json"
CACHE = Path.home() / ".cache" / "gpx2reel" / "osm"
UA = "gpx2reel/0.1 (personal project)"

ROADS = {"motorway", "trunk", "primary", "secondary"}
ROAD_RANK = {"motorway": 4, "trunk": 3, "primary": 2, "secondary": 1}


def _get(url: str, dest: Path, log=print) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def pick_region(bbox, log=print) -> dict:
    """Smallest Geofabrik region whose polygon contains the whole bbox."""
    from shapely.geometry import box, shape

    idx = json.loads(_get(INDEX_URL, CACHE / "index-v1.json", log).read_text())
    area = box(bbox[1], bbox[0], bbox[3], bbox[2])
    best = None
    for f in idx["features"]:
        p = f["properties"]
        if "pbf" not in p.get("urls", {}) or f.get("geometry") is None:
            continue
        g = shape(f["geometry"])
        if g.contains(area) and (best is None or g.area < best[0]):
            best = (g.area, p)
    if best is None:
        raise RuntimeError("No Geofabrik region covers the area")
    return best[1]


def _poi_kind(tags) -> str | None:
    n = tags.get("natural")
    if n in ("peak", "volcano"):
        return "peak"
    if tags.get("waterway") == "waterfall" or n == "waterfall":
        return "waterfall"
    if tags.get("tourism") == "viewpoint":
        return "viewpoint"
    if n == "wetland":
        return "wetland"
    if n == "water" and tags.get("water") in (None, "lake", "reservoir", "pond", "lagoon"):
        return "lake"
    if tags.get("tourism") == "attraction" or (tags.get("amenity") == "place_of_worship" and tags.get("wikidata")):
        return "sight"
    if tags.get("place") in ("city", "town", "village"):
        return "town"
    return None


def extract(pbf: Path, bbox, log=print) -> tuple[dict, list[dict]]:
    import osmium

    la0, lo0, la1, lo1 = bbox

    def inside(lat, lon):
        return la0 <= lat <= la1 and lo0 <= lon <= lo1

    vectors = {"roads": [], "rivers": []}
    pois: list[dict] = []

    def poi(tags, lat, lon, osm_id, area=None):
        kind = _poi_kind(tags)
        name = tags.get("name")
        if kind is None or not name or not inside(lat, lon):
            return
        pois.append({"id": osm_id, "kind": kind, "lat": lat, "lon": lon, "area_km2": area,
                     "tags": {t.k: t.v for t in tags if t.k.startswith("name") or t.k in (
                         "natural", "waterway", "tourism", "amenity", "religion", "place", "population",
                         "ele", "wikidata", "height", "water")}})

    keys = ("highway", "waterway", "natural", "tourism", "amenity", "place")
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.KeyFilter(*keys)))          # C++ side: untagged nodes never reach Python
    for obj in fp:
        tags = obj.tags
        if obj.is_node():
            if len(tags) and obj.location.valid():
                poi(tags, obj.location.lat, obj.location.lon, f"n{obj.id}")
            continue
        hw, ww = tags.get("highway"), tags.get("waterway")
        layer = "roads" if hw in ROADS else ("rivers" if ww == "river" else None)
        is_lake = tags.get("natural") in ("water", "wetland") and tags.get("name")
        if layer is None and not is_lake:
            continue
        try:
            coords = [(n.location.lat, n.location.lon) for n in obj.nodes if n.location.valid()]
        except osmium.InvalidLocationError:
            continue
        if len(coords) < 2:
            continue
        c = np.asarray(coords)
        if not ((c[:, 0] >= la0) & (c[:, 0] <= la1) & (c[:, 1] >= lo0) & (c[:, 1] <= lo1)).any():
            continue
        if layer:
            vectors[layer].append({"id": obj.id, "tags": {"highway": hw} if hw else {"waterway": ww},
                                   "coords": c.round(6).tolist()})
        if is_lake and obj.is_closed():
            lat_m = math.radians(float(c[:, 0].mean()))
            x = (c[:, 1] - c[:, 1].mean()) * 111.32 * math.cos(lat_m)
            y = (c[:, 0] - c[:, 0].mean()) * 110.57
            area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
            poi(tags, float(c[:, 0].mean()), float(c[:, 1].mean()), f"w{obj.id}", round(area, 3))
    log(f"OSM: {len(vectors['roads'])} roads, {len(vectors['rivers'])} rivers, {len(pois)} places")
    return vectors, pois


def download(p: TripPaths, log=print) -> Path:
    tdir = p.build / "terrain"
    bbox = json.loads((tdir / "terrain.json").read_text())["bbox"]
    region = pick_region(bbox, log)
    url = region["urls"]["pbf"]
    log(f"Geofabrik region: {region['name']}")
    pbf = _get(url, CACHE / url.rsplit("/", 1)[-1], log)
    vectors, pois = extract(pbf, bbox, log)
    out = p.build / "osm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vectors.json").write_text(json.dumps(vectors, ensure_ascii=False), encoding="utf-8")
    (out / "poi_osm.json").write_text(json.dumps(pois, ensure_ascii=False), encoding="utf-8")
    return out / "vectors.json"


def polylines_xy(ways: list[dict], proj, step_m: float = 60.0) -> list[tuple[np.ndarray, dict]]:
    """Ways → local-projection polylines resampled every step_m (for draping on the terrain)."""
    out = []
    for w in ways:
        ll = np.asarray(w["coords"], float)
        if len(ll) < 2:
            continue
        x, y = proj.to_xy(ll[:, 0], ll[:, 1])
        xy = np.column_stack([x, y])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        if s[-1] < step_m:
            continue
        k = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1] / step_m)) + 1))
        out.append((np.column_stack([np.interp(k, s, xy[:, 0]), np.interp(k, s, xy[:, 1])]), w["tags"]))
    return out
