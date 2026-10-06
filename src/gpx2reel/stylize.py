"""Stylized ground (no Blender): land-cover palette baked into a texture aligned with a terrain grid.

The texture covers the grid rectangle in the local projection (u = x, v = y, north up), so a second UV
map (StyleUV) on the terrain mesh addresses it directly. RGB = palette colour with a soft elevation tint;
A = water mask (the material makes water glossy).
"""
from __future__ import annotations

import numpy as np

from . import landcover as LC
from .geo import LocalProjection
from .world import TerrainGrid

# pastel diorama palette (sRGB)
PALETTE = {
    LC.TREE: "#5B8A5C",
    LC.SHRUB: "#8CA76C",
    LC.GRASS: "#AFC391",
    LC.CROP: "#D9D1A4",
    LC.BUILT: "#D2C8BC",
    LC.BARE: "#CFC4A6",
    LC.SNOW: "#F3F5F7",
    LC.WATER: "#74AFD1",
    LC.WETLAND: "#93BFA2",
    LC.MANGROVE: "#649470",
    LC.MOSS: "#B8C39E",
}
DEEP_WATER = "#4A8AB6"
SHALLOW_WATER = "#93CBDD"      # rim along the coast, fading into DEEP_WATER over SHORE_M
SHORE_M = 2500.0
HIGH_TINT = "#E9E4D4"          # summits fade towards a pale rock colour


def _rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


def lut() -> np.ndarray:
    t = np.tile(_rgb(PALETTE[LC.GRASS]), (256, 1))
    for k, v in PALETTE.items():
        t[k] = _rgb(v)
    return t


def texel_latlon(proj: LocalProjection, grid: TerrainGrid, width: int, height: int):
    """lat/lon of texel centres over the grid rectangle, rows north → south."""
    x = grid.x[0] + (np.arange(width) + 0.5) / width * (grid.x[-1] - grid.x[0])
    y = grid.y[-1] - (np.arange(height) + 0.5) / height * (grid.y[-1] - grid.y[0])
    gx, gy = np.meshgrid(x, y)
    lat, lon = proj.to_latlon(gx.ravel(), gy.ravel())
    return np.asarray(lat).reshape(height, width), np.asarray(lon).reshape(height, width), gx, gy


def style_texture(proj: LocalProjection, grid: TerrainGrid, lc: LC.LandCover, width: int,
                  exaggeration: float = 1.0) -> np.ndarray:
    """RGBA uint8 (height, width, 4) texture for grid; height keeps the grid aspect."""
    aspect = (grid.y[-1] - grid.y[0]) / (grid.x[-1] - grid.x[0])
    height = int(round(width * aspect))
    lat, lon, gx, gy = texel_latlon(proj, grid, width, height)
    cls = lc.sample(lat, lon)
    z = grid.height_at(gx.ravel(), gy.ravel()).reshape(height, width) / max(exaggeration, 1e-6)
    land_sea = z <= 0.5
    cls = np.where(land_sea & (cls != LC.BUILT), LC.WATER, cls)       # coast: DEM at sea level wins
    rgb = lut()[cls]
    # soft elevation tint: lighter and paler with height (above ~1000 m), deeper blue off the coast
    t = np.clip((z - 1000.0) / 1400.0, 0, 1)[..., None] * 0.35
    rgb = rgb * (1 - t) + _rgb(HIGH_TINT) * t
    # low-frequency brightness noise (±7 %) breaks up large single-class areas such as forest
    rgb = rgb * (1 + 0.07 * _noise(gx, gy, 6000.0))[..., None]
    water = cls == LC.WATER
    sea = water & land_sea
    if sea.any():
        from scipy.ndimage import distance_transform_edt

        texel_m = (grid.x[-1] - grid.x[0]) / width
        d = distance_transform_edt(sea) * texel_m                    # metres from the nearest land texel
        k = np.exp(-d / SHORE_M)[..., None]
        rgb[sea] = (_rgb(DEEP_WATER) + (_rgb(SHALLOW_WATER) - _rgb(DEEP_WATER)) * k)[sea]
    a = np.where(water, 255, 0).astype(np.uint8)
    return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), a])


def _noise(x: np.ndarray, y: np.ndarray, scale_m: float, octaves: int = 3, seed: int = 3) -> np.ndarray:
    """Cheap smooth value noise in [-1, 1] over world coordinates (sum of random sines)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x, dtype=np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        k = 2 ** o / scale_m
        for _ in range(3):
            a = rng.uniform(0, 2 * np.pi)
            ph = rng.uniform(0, 2 * np.pi)
            out += amp * np.sin(k * (x * np.cos(a) + y * np.sin(a)) * 2 * np.pi + ph).astype(np.float32)
            total += amp
        amp *= 0.5
    return out / total * 1.7
