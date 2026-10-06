"""Land cover from ESA WorldCover 10 m v200 (2021): a class raster for the terrain area.

The tiles are public Cloud-Optimised GeoTIFFs (3°×3°, named by the lower-left corner); rasterio reads just
the window we need over HTTP, from the overview level closest to the requested resolution. Missing tiles
(open ocean) come back as water. Output: build/landcover/<name>.npz with the class grid in lat/lon.
Classes: https://esa-worldcover.org/en/data-access (10 tree cover … 100 moss and lichen).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .trip import TripPaths

URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
TILE_DEG = 3
NATIVE_DEG = 1 / 12000                      # 10 m pixels: 36000 per 3°
CACHE = Path.home() / ".cache" / "gpx2reel" / "worldcover"

TREE, SHRUB, GRASS, CROP, BUILT, BARE, SNOW, WATER, WETLAND, MANGROVE, MOSS = 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100
CLASSES = {
    TREE: "forest", SHRUB: "shrubland", GRASS: "grassland", CROP: "cropland", BUILT: "built-up", BARE: "bare ground",
    SNOW: "snow and ice", WATER: "water", WETLAND: "wetland", MANGROVE: "mangroves", MOSS: "moss",
}


def tile_name(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tiles_for(bbox) -> list[tuple[int, int]]:
    """Lower-left corners of the 3° tiles intersecting bbox = [min_lat, min_lon, max_lat, max_lon]."""
    la0 = math.floor(bbox[0] / TILE_DEG) * TILE_DEG
    lo0 = math.floor(bbox[1] / TILE_DEG) * TILE_DEG
    return [(la, lo) for la in range(la0, math.ceil(bbox[2] / TILE_DEG) * TILE_DEG, TILE_DEG)
            for lo in range(lo0, math.ceil(bbox[3] / TILE_DEG) * TILE_DEG, TILE_DEG)]


def read_window(bbox, px_deg: float, log=print) -> np.ndarray:
    """Class grid covering bbox, rows north → south, pixel size px_deg (≥ NATIVE_DEG)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    h = int(round((bbox[2] - bbox[0]) / px_deg))
    w = int(round((bbox[3] - bbox[1]) / px_deg))
    out = np.full((h, w), WATER, np.uint8)            # missing tiles are open sea
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
           "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}
    with rasterio.Env(**env):
        for la, lo in tiles_for(bbox):
            # part of bbox inside this tile
            b = [max(bbox[0], la), max(bbox[1], lo), min(bbox[2], la + TILE_DEG), min(bbox[3], lo + TILE_DEG)]
            if b[0] >= b[2] or b[1] >= b[3]:
                continue
            r0 = int(round((bbox[2] - b[2]) / px_deg))
            r1 = int(round((bbox[2] - b[0]) / px_deg))
            c0 = int(round((b[1] - bbox[1]) / px_deg))
            c1 = int(round((b[3] - bbox[1]) / px_deg))
            if r1 <= r0 or c1 <= c0:
                continue
            url = "/vsicurl/" + URL.format(tile=tile_name(la, lo))
            try:
                with rasterio.open(url) as ds:
                    win = from_bounds(b[1], b[0], b[3], b[2], ds.transform)
                    data = ds.read(1, window=win, out_shape=(r1 - r0, c1 - c0), resampling=Resampling.mode)
            except rasterio.errors.RasterioIOError:
                log(f"  tile {tile_name(la, lo)} missing — treating as sea")
                continue
            data[data == 0] = WATER                   # nodata inside tiles is sea as well
            out[r0:r1, c0:c1] = data
            log(f"  tile {tile_name(la, lo)}: {c1 - c0}×{r1 - r0}")
    return out


class LandCover:
    """Class grid in lat/lon with nearest-pixel sampling."""

    def __init__(self, classes: np.ndarray, bbox):
        self.classes, self.bbox = classes, list(bbox)

    @classmethod
    def load(cls, path: Path) -> "LandCover":
        z = np.load(path)
        return cls(z["classes"], z["bbox"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, classes=self.classes, bbox=np.array(self.bbox))

    def sample(self, lat, lon) -> np.ndarray:
        h, w = self.classes.shape
        b = self.bbox
        r = ((b[2] - np.asarray(lat)) / (b[2] - b[0]) * h).astype(int)
        c = ((np.asarray(lon) - b[1]) / (b[3] - b[1]) * w).astype(int)
        return self.classes[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)]

    def fractions(self) -> dict[str, float]:
        vals, counts = np.unique(self.classes, return_counts=True)
        return {CLASSES.get(int(v), str(v)): round(float(c) / self.classes.size, 3) for v, c in zip(vals, counts)}


def fetch(bbox, px_m: float, log=print) -> LandCover:
    px_deg = max(NATIVE_DEG, px_m / 111_000)
    return LandCover(read_window(bbox, px_deg, log), bbox)


def download(p: TripPaths, px_m: float = 30.0, ctx_px_m: float = 600.0, log=print) -> list[Path]:
    """Main terrain area at px_m, intro-zoom context layers (except the widest) at ctx_px_m."""
    tdir = p.build / "terrain"
    out_dir = p.build / "landcover"
    written = []
    jobs = [("main", tdir / "terrain.json", px_m)]
    ctx = sorted(tdir.glob("ctx*/terrain.json"))
    jobs += [(p.parent.name, p, ctx_px_m) for p in ctx[1:]]       # ctx0 (a whole country) stays satellite
    for name, meta_path, px in jobs:
        bbox = json.loads(meta_path.read_text())["bbox"]
        log(f"WorldCover {name}: {bbox[2] - bbox[0]:.2f}° × {bbox[3] - bbox[1]:.2f}°, pixel {px:.0f} m")
        lc = fetch(bbox, px, log)
        p = out_dir / f"{name}.npz"
        lc.save(p)
        log(f"  → {p}  {lc.fractions()}")
        written.append(p)
    return written
