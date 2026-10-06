"""Terrain + imagery download (slippy-map tiles) with a disk cache.

DEM: AWS Terrain Tiles, Terrarium encoding (free, no key), ~38 m/px at z12.
Imagery: Esri World Imagery (no key; attribution "Esri, Maxar, Earthstar Geographics" in the video).

Output (in the trip's .work/track/build/terrain/):
  dem.npy        float32 heights, metres, Web-Mercator pixel grid
  ortho.jpg      satellite mosaic covering the same bounds
  terrain.json   bounds + zooms + attribution
Runs on the user's computer: these hosts are not reachable from the cloud sandbox.
"""
from __future__ import annotations

import io
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

PROVIDERS = {
    "terrarium": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    "esri_imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}
ATTRIBUTION = {
    "terrarium": "Terrain: Mapzen/AWS Terrain Tiles (SRTM, GMTED, ETOPO1 and others)",
    "esri_imagery": "Imagery: Esri, Maxar, Earthstar Geographics",
}
CACHE = Path.home() / ".cache" / "gpx2reel" / "tiles"
TILE = 256
BUFFER_KM = 30.0   # the drone camera flies ~20 km away and sees ~25 km past the route
UA = "gpx2reel/0.1 (personal project)"


# --------------------------------------------------------------------------- tile math

def lonlat_to_px(lon, lat, z):
    """Global Web-Mercator pixel coordinates at zoom z (float)."""
    n = TILE * 2**z
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    lat_r = np.radians(np.asarray(lat))
    y = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def px_to_lonlat(x, y, z):
    n = TILE * 2**z
    lon = np.asarray(x) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * np.asarray(y) / n))))
    return lon, lat


def expand_bbox(bbox, buffer_km: float):
    """bbox = [min_lat, min_lon, max_lat, max_lon]."""
    dlat = buffer_km / 111.0
    mid = math.radians((bbox[0] + bbox[2]) / 2)
    dlon = buffer_km / (111.0 * max(0.2, math.cos(mid)))
    return [bbox[0] - dlat, bbox[1] - dlon, bbox[2] + dlat, bbox[3] + dlon]


def tile_range(bbox, z):
    x0, y1 = lonlat_to_px(bbox[1], bbox[0], z)
    x1, y0 = lonlat_to_px(bbox[3], bbox[2], z)
    return int(x0 // TILE), int(y0 // TILE), int(x1 // TILE), int(y1 // TILE)


# --------------------------------------------------------------------------- fetching

def http_fetch(url: str, retries: int = 3) -> bytes:
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"Failed to download {url}: {last}")


def get_tile(provider: str, z: int, x: int, y: int, fetch: Callable[[str], bytes] = http_fetch) -> Image.Image:
    path = CACHE / provider / str(z) / str(x) / f"{y}.img"
    if path.exists():
        data = path.read_bytes()
    else:
        data = fetch(PROVIDERS[provider].format(z=z, x=x, y=y))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


def mosaic(provider: str, bbox, z: int, fetch=http_fetch, workers: int = 8, progress=None) -> tuple[np.ndarray, dict]:
    tx0, ty0, tx1, ty1 = tile_range(bbox, z)
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    W, H = (tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE
    canvas = np.zeros((H, W, 3), np.uint8)
    done = 0

    def job(xy):
        return xy, np.asarray(get_tile(provider, z, xy[0], xy[1], fetch))

    with ThreadPoolExecutor(workers) as ex:
        for (x, y), arr in ex.map(job, tiles):
            canvas[(y - ty0) * TILE : (y - ty0 + 1) * TILE, (x - tx0) * TILE : (x - tx0 + 1) * TILE] = arr
            done += 1
            if progress:
                progress(done, len(tiles))
    meta = {"z": z, "px0": tx0 * TILE, "py0": ty0 * TILE, "width": W, "height": H, "tiles": len(tiles)}
    return canvas, meta


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return r * 256.0 + g + b / 256.0 - 32768.0


# --------------------------------------------------------------------------- DEM sampling

@dataclass
class Dem:
    heights: np.ndarray
    z: int
    px0: int
    py0: int

    @classmethod
    def load(cls, folder: Path) -> "Dem":
        meta = json.loads((folder / "terrain.json").read_text())["dem"]
        return cls(np.load(folder / "dem.npy"), meta["z"], meta["px0"], meta["py0"])

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear height at lat/lon (arrays ok)."""
        x, y = lonlat_to_px(lon, lat, self.z)
        x = np.asarray(x) - self.px0 - 0.5
        y = np.asarray(y) - self.py0 - 0.5
        h, w = self.heights.shape
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        H = self.heights
        return (
            H[y0, x0] * (1 - fx) * (1 - fy)
            + H[y0, x0 + 1] * fx * (1 - fy)
            + H[y0 + 1, x0] * (1 - fx) * fy
            + H[y0 + 1, x0 + 1] * fx * fy
        )


# --------------------------------------------------------------------------- main entry

def estimate(bbox, dem_zoom: int, ortho_zoom: int) -> dict:
    def n(z):
        a, b, c, d = tile_range(bbox, z)
        return (c - a + 1) * (d - b + 1)

    return {"dem_tiles": n(dem_zoom), "ortho_tiles": n(ortho_zoom)}


def download(route: dict, out_dir: Path, buffer_km: float = BUFFER_KM, dem_zoom: int = 12, ortho_zoom: int = 13,
             imagery: bool = True, fetch=http_fetch, log=print) -> dict:
    bbox = expand_bbox(route["bbox"], buffer_km)
    est = estimate(bbox, dem_zoom, ortho_zoom)
    log(f"Area {bbox[2]-bbox[0]:.2f}° × {bbox[3]-bbox[1]:.2f}°: DEM {est['dem_tiles']} tiles (z{dem_zoom}), "
        f"imagery {est['ortho_tiles']} tiles (z{ortho_zoom})")
    out_dir.mkdir(parents=True, exist_ok=True)

    def prog(name):
        def p(i, n):
            if i == n or i % 50 == 0:
                log(f"  {name}: {i}/{n}")
        return p

    rgb, dem_meta = mosaic("terrarium", bbox, dem_zoom, fetch=fetch, progress=prog("DEM"))
    heights = decode_terrarium(rgb)
    np.save(out_dir / "dem.npy", heights.astype(np.float32))
    meta = {
        "bbox": bbox,
        "dem": dem_meta | {"min_m": float(heights.min()), "max_m": float(heights.max())},
        "attribution": [ATTRIBUTION["terrarium"]],
    }
    if imagery:
        img, ortho_meta = mosaic("esri_imagery", bbox, ortho_zoom, fetch=fetch, progress=prog("imagery"))
        Image.fromarray(img).save(out_dir / "ortho.jpg", quality=90)
        meta["ortho"] = ortho_meta
        meta["attribution"].append(ATTRIBUTION["esri_imagery"])
    (out_dir / "terrain.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Elevation in the area: {heights.min():.0f}–{heights.max():.0f} m → {out_dir}")
    return meta


# --------------------------------------------------------------------------- intro zoom context

def context_zooms(lat: float, extent_km: float, px_across: int = 3072) -> tuple[int, int]:
    """(dem_zoom, ortho_zoom) giving ~px_across imagery pixels over extent_km."""
    tile_km = extent_km / (px_across / TILE)
    z = math.ceil(math.log2(40075.0 * math.cos(math.radians(lat)) / tile_km))
    return max(z - 1, 1), max(z, 2)


CONTEXT_MARGIN = 1.6   # the frame is extent_km tall; the tilted camera sees past its top edge


def context_bbox(center: tuple[float, float], extent_km: float) -> list[float]:
    """Square around center (lat, lon), CONTEXT_MARGIN × extent_km wide."""
    return expand_bbox([center[0], center[1], center[0], center[1]], CONTEXT_MARGIN * extent_km / 2)


def download_context(center: tuple[float, float], extent_km: float, out_dir: Path, fetch=http_fetch, log=print) -> dict:
    dz, oz = context_zooms(center[0], CONTEXT_MARGIN * extent_km)
    return download({"bbox": context_bbox(center, extent_km)}, out_dir, 0.0, dz, oz, fetch=fetch, log=log)


# --------------------------------------------------------------------------- highlight detail

DETAIL_RADIUS_KM = 5.0


def download_detail(center: tuple[float, float], out_dir: Path, radius_km: float = DETAIL_RADIUS_KM,
                    dem_zoom: int = 14, ortho_zoom: int = 16, fetch=http_fetch, log=print) -> dict:
    """High-res patch around a highlight (detail: high_res_dem): z14 DEM ≈ 8 m/px, z16 imagery ≈ 2 m/px."""
    return download({"bbox": [center[0], center[1], center[0], center[1]]}, out_dir, radius_km,
                    dem_zoom, ortho_zoom, fetch=fetch, log=log)
