"""Beach set for a sunrise opening on the coast (`days[n].opening_set: beach`): what stands around the start.

Data (fetched by `gpx2reel terrain` into build/terrain/opening<day>/ next to its high-res patch):
coastline and buildings from OSM, built-up cells from ESA WorldCover. Placement is numpy here, the meshes are
in blender/coast.py. Palms line the coast a little inland; houses stand on OSM footprints and, where OSM has
none (rural Japan is sparsely mapped), on WorldCover built-up cells.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SET_RADIUS_M = 1500.0         # palms / houses within this distance of the start
PATCH_RADIUS_KM = 3.0         # high-res terrain around the start
PALM_STEP_M = 9.0             # along the coast
PALM_INLAND_M = (12.0, 55.0)  # from the coastline
PALM_KEEP = 0.55              # share of candidate spots that get a palm (gaps read as groups)
PALM_MAX_Z = 25.0             # m above sea (unexaggerated): palms stay on the backshore, not up the cliffs
HOUSE_CELL_M = 14.0           # one house per built-up cell of this size
HOUSE_KEEP = 0.7


def day_start(route: dict, day: int) -> tuple[float, float]:
    d = next(x for x in route["days"] if x["day"] == day)
    p = d["segments"][0]["points"][0]
    return float(p[0]), float(p[1])


def sunrise_dir(route: dict, day: int) -> np.ndarray:
    """Unit (x, y) towards where the sun comes up on the morning of `day` at its start (grid ≈ true north)."""
    from .sun import solar_position, sunrise_before

    d = next(x for x in route["days"] if x["day"] == day)
    lat, lon = day_start(route, day)
    t = sunrise_before(lat, lon, float(d["segments"][0]["points"][0][3]))
    az, _ = solar_position(np.array([lat]), np.array([lon]), np.array([t]))
    a = math.radians(float(az[0]))
    return np.array([math.sin(a), math.cos(a)])


def grove_points(start_xy: np.ndarray, d: np.ndarray, back_m: float, height, exaggeration: float,
                 seed: int = 21) -> np.ndarray:
    """Palms around the sunrise camera's path (from back_m to 0.6·back_m behind the start, looking along d): big
    silhouettes at the frame edges, a clear lane in the middle for the sun. (n, 4) like palm_points."""
    rng = np.random.default_rng(seed)
    side = np.array([-d[1], d[0]])
    cam_end = start_xy[:2] - d * 0.6 * back_m
    pts = []
    for _ in range(26):                                     # the grove between the camera and the beach
        a = rng.uniform(20, 0.6 * back_m + 70)
        lat = rng.choice([-1, 1]) * rng.uniform(9, 55)
        pts.append(cam_end + d * a + side * lat)
    for sgn in (-1, 1):                                     # close framing trunks
        for _ in range(2):
            pts.append(cam_end + d * rng.uniform(8, 22) + side * sgn * rng.uniform(10, 18))
    q = np.asarray(pts)
    z = height.height_at(q[:, 0], q[:, 1])
    keep = z > 0.3 * exaggeration
    return np.column_stack([q[keep], z[keep], rng.random(int(keep.sum()))])


def clear_lane(boxes: np.ndarray, start_xy: np.ndarray, d: np.ndarray, back_m: float,
               half_width: float = 30.0) -> np.ndarray:
    """Drop houses standing in the camera's view lane (from behind the camera to past the start)."""
    rel = boxes[:, :2] - start_xy[None, :2]
    along = rel @ d
    lat = rel @ np.array([-d[1], d[0]])
    inside = (along > -back_m - 20) & (along < 150) & (np.abs(lat) < half_width)
    return boxes[~inside]


def beach_days(sb) -> list[int]:
    """Days whose opening is on the beach and that this reel (or day story) actually opens."""
    return [d.day for d in sb.days if d.opening == "sunrise_beach" and sb.focus_day in (None, d.day)]


def day_end(route: dict, day: int) -> tuple[float, float]:
    d = next(x for x in route["days"] if x["day"] == day)
    p = d["segments"][-1]["points"][-1]
    return float(p[0]), float(p[1])


def coast_sets(sb) -> list[tuple[int, str, str, str, tuple | None]]:
    """(day, kind, terrain subfolder, detail tag, (lat, lon) or None) of the coast sets this reel (or day story)
    shows. Kinds: "beach" at the day's start (sunrise_beach opening); "sunset" at its finish (a harbour: houses
    and water); "sunset_beach" at `closing_at`, a sunset spot away from the finish (palms and surf too)."""
    out = [(day, "beach", f"opening{day}", f"o{day}", None) for day in beach_days(sb)]
    for d in sb.days:
        if d.closing == "sunset" and sb.focus_day in (None, d.day):
            if d.closing_at:
                out.append((d.day, "sunset_beach", f"closing{d.day}s", f"cs{d.day}", tuple(d.closing_at)))
            else:
                out.append((d.day, "sunset", f"closing{d.day}", f"c{d.day}", None))
    return out


def set_anchor(route: dict, day: int, kind: str, at: tuple | None = None) -> tuple[float, float]:
    if at is not None:
        return float(at[0]), float(at[1])
    return day_start(route, day) if kind == "beach" else day_end(route, day)


def sunset_dir(route: dict, day: int) -> np.ndarray:
    """Unit (x, y) towards where the sun goes down on the evening of `day` (seen from its finish)."""
    from .sun import solar_position, sun_descends_to

    d = next(x for x in route["days"] if x["day"] == day)
    lat, lon = day_end(route, day)
    t = sun_descends_to(lat, lon, float(d["segments"][-1]["points"][-1][3]), 0.0)
    if t is None:
        return np.array([-1.0, 0.0])
    az, _ = solar_position(np.array([lat]), np.array([lon]), np.array([t]))
    a = math.radians(float(az[0]))
    return np.array([math.sin(a), math.cos(a)])


COAST_RADIUS_M = 7000.0       # coastline for the water's shore fade: as far as the low camera sees the shore
SHORE_FADE_M = (40.0, 220.0)   # water: transparent near the coastline (satellite shallows, surf) → solid;
                              # small coves are ~1 km across
ISLET_M = 800.0               # closed coastlines shorter than this (rocks; Aoshima island is 1.2 km) don't fade the water


def overpass_query(lat: float, lon: float, radius_m: float = SET_RADIUS_M * 1.4) -> str:
    r = f"around:{radius_m:.0f},{lat:.6f},{lon:.6f}"
    rc = f"around:{COAST_RADIUS_M:.0f},{lat:.6f},{lon:.6f}"
    return f"""[out:json][timeout:120];
(way["natural"="coastline"]({rc}); way["building"]({r}););
out geom tags;"""


def download(lat: float, lon: float, out_dir: Path, log=print) -> Path:
    """OSM coastline + buildings and WorldCover classes around the start → out_dir/coast.json, landcover.npz."""
    from . import landcover as LC
    from .poi import fetch

    out_dir.mkdir(parents=True, exist_ok=True)
    osm = fetch(overpass_query(lat, lon), log=log)
    coast, buildings = [], []
    for e in osm.get("elements", []):
        g = e.get("geometry")
        if not g:
            continue
        pts = [[round(p["lat"], 7), round(p["lon"], 7)] for p in g]
        t = e.get("tags", {})
        if t.get("natural") == "coastline":
            coast.append(pts)
        elif "building" in t:
            lv = t.get("building:levels")
            buildings.append({"ring": pts, "levels": float(lv) if lv and lv.replace(".", "").isdigit() else None})
    (out_dir / "coast.json").write_text(json.dumps({"center": [lat, lon], "coastline": coast,
                                                    "buildings": buildings}))
    d = SET_RADIUS_M * 1.4 / 111_000
    dl = d / math.cos(math.radians(lat))
    LC.fetch([lat - d, lon - dl, lat + d, lon + dl], 10.0, log=log).save(out_dir / "landcover.npz")
    log(f"Coast: {len(coast)} lines, {len(buildings)} OSM buildings → {out_dir}")
    return out_dir


def _resample(line: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` m along a polyline and the unit direction there."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return np.zeros((0, 2)), np.zeros((0, 2))
    t = np.arange(0, s[-1], step)
    p = np.column_stack([np.interp(t, s, line[:, 0]), np.interp(t, s, line[:, 1])])
    d = np.gradient(p, axis=0)
    return p, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def coast_lines_xy(coast: dict, proj) -> list[np.ndarray]:
    out = []
    for pts in coast["coastline"]:
        a = np.asarray(pts, float)
        x, y = proj.to_xy(a[:, 0], a[:, 1])
        out.append(np.column_stack([x, y]))
    return out


def palm_points(lines: list[np.ndarray], height, center: np.ndarray, exaggeration: float,
                radius: float = SET_RADIUS_M, seed: int = 5) -> np.ndarray:
    """(n, 4): x, y, z, rand. OSM coastlines keep the land on the left of the way's direction."""
    rng = np.random.default_rng(seed)
    out = []
    for line in lines:
        p, d = _resample(line, PALM_STEP_M)
        if not len(p):
            continue
        left = np.column_stack([-d[:, 1], d[:, 0]])
        off = rng.uniform(*PALM_INLAND_M, len(p))
        q = p + left * off[:, None] + rng.normal(0, 2.5, (len(p), 2))
        out.append(q)
    if not out:
        return np.zeros((0, 4))
    q = np.concatenate(out)
    q = q[np.linalg.norm(q - center[None, :2], axis=1) < radius]
    z = height.height_at(q[:, 0], q[:, 1])
    keep = (z > 0.4 * exaggeration) & (z < PALM_MAX_Z * exaggeration) & (rng.random(len(q)) < PALM_KEEP)
    q, z = q[keep], z[keep]
    return np.column_stack([q, z, rng.random(len(q))])


def house_boxes(coast: dict, lc, proj, height, center: np.ndarray, exaggeration: float,
                radius: float = SET_RADIUS_M, seed: int = 9) -> np.ndarray:
    """(n, 7): x, y, z (ground), width, depth, height, rotation — OSM footprints as their oriented boxes, plus
    WorldCover built-up cells that no OSM building covers."""
    from . import landcover as LC

    rng = np.random.default_rng(seed)
    boxes = []
    for b in coast["buildings"]:
        a = np.asarray(b["ring"], float)
        x, y = proj.to_xy(a[:, 0], a[:, 1])
        xy = np.column_stack([x, y])
        c = xy.mean(axis=0)
        if np.linalg.norm(c - center[:2]) > radius:
            continue
        e = np.diff(xy, axis=0)
        k = int(np.argmax(np.linalg.norm(e, axis=1)))
        rot = math.atan2(e[k, 1], e[k, 0])
        rr = np.array([[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]])
        loc = (xy - c) @ rr.T
        w, dpt = np.ptp(loc[:, 0]), np.ptp(loc[:, 1])
        levels = b["levels"] or (2 if w * dpt < 250 else 3)
        boxes.append([c[0], c[1], w, dpt, levels * 3.0, rot])
    # built-up cells
    xs = np.arange(center[0] - radius, center[0] + radius, HOUSE_CELL_M)
    gx, gy = np.meshgrid(xs, np.arange(center[1] - radius, center[1] + radius, HOUSE_CELL_M))
    x = gx.ravel() + rng.uniform(-0.2, 0.2, gx.size) * HOUSE_CELL_M
    y = gy.ravel() + rng.uniform(-0.2, 0.2, gx.size) * HOUSE_CELL_M
    lat, lon = proj.to_latlon(x, y)
    built = (lc.sample(np.asarray(lat), np.asarray(lon)) == LC.BUILT)
    near = np.hypot(x - center[0], y - center[1]) < radius
    keep = built & near & (rng.random(len(x)) < HOUSE_KEEP)
    if boxes:
        osm = np.asarray(boxes)
        dmin = np.min(np.hypot(x[:, None] - osm[None, :, 0], y[:, None] - osm[None, :, 1]), axis=1)
        keep &= dmin > 12.0
    x, y = x[keep], y[keep]
    n = len(x)
    rot0 = rng.choice([0.0, math.pi / 2], n) + rng.normal(0, 0.12, n)
    w = rng.uniform(7, 11, n)
    dpt = w * rng.uniform(0.7, 1.0, n)
    h = rng.choice([4.5, 6.0, 6.5, 8.5], n, p=[0.3, 0.35, 0.2, 0.15])
    auto = np.column_stack([x, y, w, dpt, h, rot0])
    allb = np.concatenate([np.asarray(boxes).reshape(-1, 6), auto])
    z = height.height_at(allb[:, 0], allb[:, 1])
    ok = z > 0.3 * exaggeration
    allb, z = allb[ok], z[ok]
    return np.column_stack([allb[:, :2], z, allb[:, 2:4], allb[:, 4], allb[:, 5]])


def shore_alpha(lines: list[np.ndarray], xy: np.ndarray) -> np.ndarray:
    """Water opacity at points xy: 0 within SHORE_FADE_M[0] of a coastline, 1 beyond SHORE_FADE_M[1]."""
    from scipy.spatial import cKDTree

    # rocks and islets (small closed rings) would punch holes of satellite sea into the open water
    big = [l for l in lines if not (np.allclose(l[0], l[-1]) and np.sum(np.linalg.norm(np.diff(l, axis=0), axis=1)) < ISLET_M)]
    pts = [p for p in (_resample(l, 10.0)[0] for l in big) if len(p)]
    if not pts:
        return np.ones(len(xy))
    dist = cKDTree(np.concatenate(pts)).query(xy)[0]
    u = np.clip((dist - SHORE_FADE_M[0]) / (SHORE_FADE_M[1] - SHORE_FADE_M[0]), 0, 1)
    return u * u * (3 - 2 * u)


def sea_side(lines: list[np.ndarray], xy: np.ndarray) -> np.ndarray:
    """True where a point lies on the sea side of its nearest coastline (OSM keeps the sea on the right)."""
    from scipy.spatial import cKDTree

    ps, ds = zip(*[_resample(l, 5.0) for l in lines if len(l) > 1]) if lines else ((), ())
    ps = [p for p in ps if len(p)]
    if not ps:
        return np.zeros(len(xy), bool)
    p, d = np.concatenate(ps), np.concatenate([d for d in ds if len(d)])
    k = cKDTree(p).query(xy)[1]
    rel = xy - p[k]
    return d[k, 0] * rel[:, 1] - d[k, 1] * rel[:, 0] < 0


def flatten_sea(grid, lines: list[np.ndarray], exaggeration: float, below_m: float = 2.0) -> None:
    """DEM noise in the shallows (0.5–1.5 m) would poke through the water plane: lay it at sea level."""
    gx, gy = np.meshgrid(grid.x, grid.y)
    sea = sea_side(lines, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    grid.z[sea & (grid.z < below_m * exaggeration)] = 0.0
    # the glossy-sea mask follows the coastline too: the DEM's coarse bathymetry draws it as a staircase
    grid.sea = sea & (grid.z <= 0)


def download_coastline(bbox, out: Path, log=print) -> Path:
    """Every OSM coastline in the terrain area → terrain/coastline.json: the sea / land mask of the whole trip
    (the DEM is noisy at the shore: 0–2 m over the sea left imagery seams and stepped coasts)."""
    from .poi import fetch

    osm = fetch(f'[out:json][timeout:600];way["natural"="coastline"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out geom;',
                log=log)
    lines = [[[round(p["lat"], 6), round(p["lon"], 6)] for p in e["geometry"]] for e in osm.get("elements", [])
             if e.get("geometry")]
    out.write_text(json.dumps({"bbox": list(bbox), "coastline": lines}))
    log(f"Coastline: {len(lines)} lines → {out}")
    return out


def sea_mask(lines: list[np.ndarray], grid, step_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(water, distance to the coastline in m) per vertex of grid: the sea side of the nearest coastline."""
    from scipy.spatial import cKDTree

    step = step_m or max(10.0, float(grid.x[1] - grid.x[0]) / 4)
    ps, ds = [], []
    for l in lines:
        if len(l) > 1:
            p, d = _resample(l, step)
            if len(p):
                ps.append(p), ds.append(d)
    gx, gy = np.meshgrid(grid.x, grid.y)
    if not ps:
        return np.zeros(gx.shape, bool), np.full(gx.shape, 1e9)
    p, d = np.concatenate(ps), np.concatenate(ds)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    dist, k = cKDTree(p).query(xy)
    rel = xy - p[k]
    water = d[k, 0] * rel[:, 1] - d[k, 1] * rel[:, 0] < 0
    return water.reshape(gx.shape), dist.reshape(gx.shape)
