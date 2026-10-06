"""Terrain download logic with a fake tile server (no network)."""
import io

import numpy as np
from PIL import Image

from gpx2reel import terrain as T


def fake_fetch_factory(height_fn):
    def fetch(url: str) -> bytes:
        parts = url.rstrip("/").split("/")
        if "terrarium" in url:
            z, x, y = int(parts[-3]), int(parts[-2]), int(parts[-1].split(".")[0])
            px = np.arange(256) + 0.5
            gx, gy = np.meshgrid(x * 256 + px, y * 256 + px)
            lon, lat = T.px_to_lonlat(gx, gy, z)
            h = height_fn(lat, lon) + 32768.0
            r = np.floor(h / 256)
            g = np.floor(h - r * 256)
            b = np.round((h - r * 256 - g) * 256)
            rgb = np.stack([r, g, np.minimum(b, 255)], -1).astype(np.uint8)
        else:
            rgb = np.full((256, 256, 3), 90, np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "PNG")
        return buf.getvalue()

    return fetch


def test_download_and_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CACHE", tmp_path / "cache")
    cone = lambda lat, lon: 100 + 1500 * np.exp(-(((lat - 32.9) * 111) ** 2 + ((lon - 131.1) * 93) ** 2) / 25)
    route = {"bbox": [32.85, 131.0, 32.95, 131.2]}
    meta = T.download(route, tmp_path / "terrain", buffer_km=2, dem_zoom=11, ortho_zoom=12,
                      fetch=fake_fetch_factory(cone), log=lambda *_: None)
    assert (tmp_path / "terrain" / "dem.npy").exists()
    assert (tmp_path / "terrain" / "ortho.jpg").exists()
    dem = T.Dem.load(tmp_path / "terrain")
    lat = np.array([32.9, 32.87, 32.93])
    lon = np.array([131.1, 131.05, 131.15])
    got = dem.sample(lat, lon)
    want = cone(lat, lon)
    assert np.all(np.abs(got - want) < 40), (got, want)   # z11 ≈ 65 m/px, cone is smooth
    assert meta["dem"]["max_m"] > 1500


def test_cache_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "CACHE", tmp_path / "cache")
    calls = []
    base = fake_fetch_factory(lambda lat, lon: np.zeros_like(lat))

    def counting(url):
        calls.append(url)
        return base(url)

    T.mosaic("terrarium", [32.85, 131.0, 32.9, 131.05], 10, fetch=counting)
    n = len(calls)
    T.mosaic("terrarium", [32.85, 131.0, 32.9, 131.05], 10, fetch=counting)
    assert len(calls) == n
