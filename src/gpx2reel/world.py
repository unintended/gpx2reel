"""Scene geometry without Blender: terrain grid, route path, gap arcs in local metric coordinates.

Everything is in the route's LocalProjection (UTM, origin = route centre), metres, Z up.
Heights are DEM heights times the storyboard exaggeration. The same arrays feed the Blender
scene and the timeline, so both agree on where the route lies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geo import LocalProjection, rdp_mask
from .terrain import Dem, lonlat_to_px

RIBBON_LIFT_M = 3.0      # route ribbon above the terrain surface (before exaggeration)
SEA_BELOW_M = -0.5       # DEM below this is sea (Terrarium carries bathymetry); coastal flats at 0 stay land


def route_center(route: dict) -> tuple[float, float]:
    b = route["bbox"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def projection_of(route: dict) -> LocalProjection:
    return LocalProjection(*route_center(route))


@dataclass
class TerrainGrid:
    x: np.ndarray            # (nx,) local metres
    y: np.ndarray            # (ny,)
    z: np.ndarray            # (ny, nx) scene heights (exaggerated, sea clamped to 0)
    uv: np.ndarray           # (ny, nx, 2) ortho texture coordinates, v up
    sea: np.ndarray | None = None   # (ny, nx) True where the DEM is below sea level (bathymetry → glossy water)
    depth: np.ndarray | None = None  # (ny, nx) metres below sea level (0 on land) — open water hides imagery seams

    def height_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Bilinear height on the grid (matches the mesh surface up to the quad diagonal)."""
        fx = np.clip((px - self.x[0]) / (self.x[1] - self.x[0]), 0, len(self.x) - 1.001)
        fy = np.clip((py - self.y[0]) / (self.y[1] - self.y[0]), 0, len(self.y) - 1.001)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - i, fy - j
        Z = self.z
        return (Z[j, i] * (1 - tx) * (1 - ty) + Z[j, i + 1] * tx * (1 - ty)
                + Z[j + 1, i] * (1 - tx) * ty + Z[j + 1, i + 1] * tx * ty)


class Surface:
    """Base terrain grid + high-res detail patches: heights come from the finest grid covering a point."""

    def __init__(self, base: TerrainGrid, details: list[TerrainGrid] | None = None, inset_cells: float = 2):
        self.base, self.details, self.inset = base, details or [], inset_cells

    @staticmethod
    def rect(g: TerrainGrid, inset_cells: float = 0) -> tuple[float, float, float, float]:
        dx, dy = g.x[1] - g.x[0], g.y[1] - g.y[0]
        return g.x[0] + inset_cells * dx, g.x[-1] - inset_cells * dx, g.y[0] + inset_cells * dy, g.y[-1] - inset_cells * dy

    def height_at(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        px, py = np.asarray(px, float), np.asarray(py, float)
        h = self.base.height_at(px, py)
        for g in self.details:
            x0, x1, y0, y1 = self.rect(g, self.inset)
            m = (px > x0) & (px < x1) & (py > y0) & (py < y1)
            if m.any():
                h[m] = g.height_at(px[m], py[m])
        return h


def terrain_grid(proj: LocalProjection, dem: Dem, terrain_meta: dict, exaggeration: float,
                 step_m: float = 100.0) -> TerrainGrid:
    """Regular grid inside the downloaded area. Bathymetry is clamped to sea level."""
    b = terrain_meta["bbox"]
    # the lat/lon box is slightly rotated in UTM: keep the inner axis-aligned rectangle
    cx, cy = proj.to_xy(np.array([b[0], b[0], b[2], b[2]]), np.array([b[1], b[3], b[1], b[3]]))
    x0, x1 = max(cx[0], cx[2]), min(cx[1], cx[3])
    y0, y1 = max(cy[0], cy[1]), min(cy[2], cy[3])
    x = np.arange(x0, x1, step_m)
    y = np.arange(y0, y1, step_m)
    gx, gy = np.meshgrid(x, y)
    lat, lon = proj.to_latlon(gx.ravel(), gy.ravel())
    raw = dem.sample(lat, lon).reshape(gx.shape)
    h = np.maximum(raw, 0.0)
    sea = raw < SEA_BELOW_M
    o = terrain_meta.get("ortho")
    if o:
        px, py = lonlat_to_px(lon, lat, o["z"])
        u = (px - o["px0"]) / o["width"]
        v = 1.0 - (py - o["py0"]) / o["height"]
        uv = np.stack([u, v], -1).reshape(*gx.shape, 2)
    else:
        uv = np.zeros((*gx.shape, 2))
    return TerrainGrid(x, y, (h * exaggeration).astype(np.float32), uv.astype(np.float32), sea,
                       np.maximum(-raw, 0.0).astype(np.float32))


@dataclass
class Path3D:
    xyz: np.ndarray          # (n, 3)
    km: np.ndarray           # (n,) route km (gaps excluded, as in route.json)
    day: np.ndarray          # (n,) day number
    piece: np.ndarray        # (n,) continuous piece id: a new piece starts after every gap not joined
    s: np.ndarray | None = None   # (n,) km along the drawn line: km + joined gaps; animation runs on s
    transport: np.ndarray | None = None   # (n,) True on bridges travelled by bus / car (gap mode `road`)

    def __post_init__(self):
        if self.s is None:
            self.s = self.km.copy()
        if self.transport is None:
            self.transport = np.zeros(len(self.km), bool)

    def heading(self) -> np.ndarray:
        """Unit direction of travel per point (xy), from a central difference."""
        d = np.gradient(self.xyz[:, :2], axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-9)


def route_path(route: dict, proj: LocalProjection, grid: TerrainGrid, exaggeration: float,
               step_m: float = 20.0, simplify_m: float = 3.0) -> Path3D:
    """Track resampled every step_m, lying on the terrain surface (not GPS heights)."""
    xs, kms, days, pieces = [], [], [], []
    piece = 0
    for d in route["days"]:
        for s in d["segments"]:
            p = np.array([[q[0], q[1], q[4]] for q in s["points"]], float)
            x, y = proj.to_xy(p[:, 0], p[:, 1])
            xy = np.column_stack([x, y])
            keep = rdp_mask(xy, simplify_m)
            xy, km = xy[keep], p[keep, 2]
            n = max(2, int((km[-1] - km[0]) * 1000 / step_m) + 1)
            k = np.linspace(km[0], km[-1], n)
            xs.append(np.column_stack([np.interp(k, km, xy[:, 0]), np.interp(k, km, xy[:, 1])]))
            kms.append(k)
            days.append(np.full(n, d["day"]))
            pieces.append(np.full(n, piece))
            piece += 1
    xy = np.concatenate(xs)
    z = grid.height_at(xy[:, 0], xy[:, 1]) + RIBBON_LIFT_M * exaggeration
    return Path3D(np.column_stack([xy, z]), np.concatenate(kms), np.concatenate(days), np.concatenate(pieces))


def gap_mode(g: dict, modes: dict[int, str]) -> str:
    """Storyboard mode of a route gap; defaults match storyboard.skeleton."""
    default = "straight" if g["kind"] == "transfer" else "join"
    return modes.get(f"in{g['after_day']}", default) if g["within_day"] else modes.get(g["after_day"], default)


def _boundaries(path: Path3D, route: dict) -> list[tuple[int, dict | None]]:
    """(last index before it, gap) for every piece boundary; km does not grow across a gap.
    gap is None where pieces meet without a recorded gap (files that continue each other)."""
    out = []
    for i in np.flatnonzero(np.diff(path.piece)):
        g = min(route["gaps"], key=lambda g: abs(g["at_km"] - path.km[i]), default=None)
        out.append((int(i), g if g is not None and abs(g["at_km"] - path.km[i]) < 1e-3 else None))
    return out


def _resample(poly: np.ndarray, step_m: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], max(2, int(s[-1] / step_m) + 1))
    return np.column_stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])])


def join_gaps(route: dict, path: Path3D, grid: TerrainGrid, exaggeration: float, modes: dict[int, str],
              step_m: float = 20.0, roads: dict[int, np.ndarray] | None = None) -> Path3D:
    """Bridge 'join' gaps with a line draped on the terrain ('road' gaps: along roads[after_day], local xy);
    s grows across them, km does not."""
    parts, start, piece = [], 0, 0
    ids = np.zeros(len(path.km), int)
    joins: dict[int, np.ndarray] = {}
    by_road: set[int] = set()
    for i, g in _boundaries(path, route):
        ids[start:i + 1] = piece
        mode = "join" if g is None else gap_mode(g, modes)
        a, b = path.xyz[i, :2], path.xyz[i + 1, :2]
        if mode == "road" and roads and g["after_day"] in roads:
            poly = np.vstack([a, roads[g["after_day"]], b])
            joins[i] = _resample(poly, step_m)[1:-1]
            by_road.add(i)
        elif mode in ("join", "road", "ferry"):
            n = max(2, int(np.linalg.norm(b - a) / step_m) + 1)
            joins[i] = a + (b - a) * np.linspace(0, 1, n)[1:-1, None]
            if mode == "ferry":                       # drawn like a bus leg: travelled, not ridden
                by_road.add(i)
        else:
            piece += 1
        start = i + 1
    ids[start:] = piece
    xyz, km, day, pc, tr = [], [], [], [], []
    prev = 0
    for i in sorted(joins) + [len(path.km) - 1]:
        sl = slice(prev, i + 1)
        xyz.append(path.xyz[sl]); km.append(path.km[sl]); day.append(path.day[sl]); pc.append(ids[sl])
        tr.append(path.transport[sl])
        if i in joins:
            c = joins[i]
            z = grid.height_at(c[:, 0], c[:, 1]) + RIBBON_LIFT_M * exaggeration
            xyz.append(np.column_stack([c, z])); km.append(np.full(len(c), path.km[i]))
            day.append(np.full(len(c), path.day[i])); pc.append(np.full(len(c), ids[i]))
            tr.append(np.full(len(c), i in by_road))
        prev = i + 1
    xyz, pc = np.concatenate(xyz), np.concatenate(pc)
    step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / 1000
    step[np.diff(pc) != 0] = 0.0
    return Path3D(xyz, np.concatenate(km), np.concatenate(day), pc, np.concatenate([[0.0], np.cumsum(step)]),
                  np.concatenate(tr))


def gap_arc(a: np.ndarray, b: np.ndarray, n: int = 64, rise: float = 0.25, min_rise_m: float = 300.0) -> np.ndarray:
    """Parabolic arc from a to b (scene xyz), apex rise ∝ horizontal distance."""
    t = np.linspace(0, 1, n)[:, None]
    p = a + (b - a) * t
    h = max(min_rise_m, rise * float(np.linalg.norm((b - a)[:2])))
    p[:, 2] += 4 * h * t[:, 0] * (1 - t[:, 0])
    return p


def gap_arcs(route: dict, path: Path3D, modes: dict[int, str]) -> list[dict]:
    """Arcs for the gaps drawn 'straight' (call after join_gaps: joined gaps are no boundaries)."""
    out = []
    for i, g in _boundaries(path, route):
        if g is None or gap_mode(g, modes) != "straight":
            continue
        out.append({"after_day": g["after_day"], "at_km": g["at_km"], "points": gap_arc(path.xyz[i], path.xyz[i + 1])})
    return out


def save_world(path: Path, grid: TerrainGrid, route_p: Path3D, arcs: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, grid_x=grid.x, grid_y=grid.y, grid_z=grid.z,
        path_xyz=route_p.xyz, path_km=route_p.km, path_day=route_p.day, path_piece=route_p.piece, path_s=route_p.s, path_transport=route_p.transport,
        arc_km=np.array([a["at_km"] for a in arcs]), arc_after_day=np.array([a["after_day"] for a in arcs], int), arc_points=np.array([a["points"] for a in arcs]).reshape(-1, 64, 3),
        meta=json.dumps(meta),
    )


def load_world(path: Path) -> tuple[TerrainGrid, Path3D, dict[int, np.ndarray], dict]:
    """Counterpart of save_world; arcs keyed by the day they follow (uv is not stored)."""
    w = np.load(path)
    grid = TerrainGrid(w["grid_x"], w["grid_y"], w["grid_z"], np.zeros((0, 0, 2)))
    p = Path3D(w["path_xyz"], w["path_km"], w["path_day"], w["path_piece"], w["path_s"],
               w["path_transport"] if "path_transport" in w else None)
    arcs = {int(d): a for d, a in zip(w["arc_after_day"], w["arc_points"])}
    return grid, p, arcs, json.loads(str(w["meta"]))
