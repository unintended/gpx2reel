"""Small geodesy helpers: distances, local projection, simplification."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from pyproj import CRS, Transformer

EARTH_R = 6_371_008.8  # mean Earth radius, m


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(min(1.0, math.sqrt(a)))


def haversine_np(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distances between consecutive points (len n-1), metres."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dp = np.diff(lat_r)
    dl = np.diff(lon_r)
    a = np.sin(dp / 2) ** 2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def utm_crs_for(lat: float, lon: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


class LocalProjection:
    """WGS84 <-> metric plane centred on the route (UTM, origin shifted to centre)."""

    def __init__(self, lat0: float, lon0: float):
        self.crs = utm_crs_for(lat0, lon0)
        self._fwd = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self.x0, self.y0 = self._fwd.transform(lon0, lat0)

    def to_xy(self, lat, lon):
        x, y = self._fwd.transform(lon, lat)
        return np.asarray(x) - self.x0, np.asarray(y) - self.y0

    def to_latlon(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x) + self.x0, np.asarray(y) + self.y0)
        return lat, lon

    def to_dict(self) -> dict:
        return {"epsg": self.crs.to_epsg(), "x0": self.x0, "y0": self.y0}


def rdp_mask(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker over points of any dimension (xy, or xyz to keep the profile too); returns a mask of
    the points to keep. Distances are to the segment, not its line: an out-and-back's turn point stays."""
    pts = np.asarray(pts, float)
    pts = pts[:, None] if pts.ndim == 1 else pts
    n = len(pts)
    keep = np.zeros(n, dtype=bool)
    if n <= 2:
        keep[:] = True
        return keep
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = pts[i], pts[j] - pts[i]
        rel = pts[i + 1:j] - a
        den = float(ab @ ab)
        u = np.clip(rel @ ab / den, 0.0, 1.0) if den > 0 else np.zeros(len(rel))
        d = np.linalg.norm(rel - u[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return keep


def moving_average(v: Sequence[float], window: int) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if window <= 1 or len(v) < window:
        return v.copy()
    pad = window // 2
    vp = np.pad(v, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(vp, kernel, mode="valid")[: len(v)]
