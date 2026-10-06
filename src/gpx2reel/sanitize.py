"""Shareable tracks: one plain GPX per ride day from the ingested route (duplicates dropped, recordings stitched).

Only lat / lon / ele / time survive — no heart rate, cadence, power, device names or extensions. Points are
thinned with Ramer–Douglas–Peucker over (x, y, ele) in metres, so the profile keeps its climbs, and the first /
last `trim_m` of every day can be cut off (a privacy zone around where the nights were spent). The output
re-ingests as a trip with the same days.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .geo import haversine_np, rdp_mask
from .world import projection_of


def simplify(pts: np.ndarray, proj, tolerance_m: float) -> np.ndarray:
    """pts: rows [lat, lon, ele, t] (NaN where missing) → the rows RDP keeps."""
    if len(pts) <= 2 or tolerance_m <= 0:
        return pts
    x, y = proj.to_xy(pts[:, 0], pts[:, 1])
    ele = pts[:, 2]
    ok = np.isfinite(ele)
    if ok.any():
        idx = np.arange(len(ele))
        z = np.interp(idx, idx[ok], ele[ok])
    else:
        z = np.zeros(len(ele))
    return pts[rdp_mask(np.column_stack([x, y, z]), tolerance_m)]


def trim(segments: list[np.ndarray], trim_m: float) -> list[np.ndarray]:
    """Drop the first and last trim_m metres of a day (over all its segments); empty segments go."""
    if trim_m <= 0:
        return segments
    steps = [np.concatenate([[0.0], haversine_np(s[:, 0], s[:, 1])]) for s in segments]
    offset, cum = 0.0, []
    for st in steps:
        cum.append(offset + np.cumsum(st))
        offset = cum[-1][-1]
    out = []
    for s, c in zip(segments, cum):
        m = (c >= trim_m) & (c <= offset - trim_m)
        if m.sum() >= 2:
            out.append(s[m])
    return out


def _time(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpx_text(name: str, segments: list[np.ndarray]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="gpx2reel" xmlns="http://www.topografix.com/GPX/1/1">',
             f"  <trk><name>{escape(name)}</name>"]
    for s in segments:
        lines.append("    <trkseg>")
        for lat, lon, ele, t in s:
            inner = (f"<ele>{ele:.1f}</ele>" if np.isfinite(ele) else "") + (f"<time>{_time(t)}</time>"
                                                                              if np.isfinite(t) else "")
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">{inner}</trkpt>')
        lines.append("    </trkseg>")
    lines += ["  </trk>", "</gpx>", ""]
    return "\n".join(lines)


def sanitize(route: dict, out: Path, tolerance_m: float = 5.0, trim_m: float = 0.0) -> list[Path]:
    """route: ingest() output. Writes day<NN>-<date>.gpx into out; returns the paths."""
    proj = projection_of(route)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for d in route["days"]:
        segs = [np.array([[np.nan if v is None else v for v in p[:4]] for p in s["points"]], float)
                for s in d["segments"]]
        segs = [simplify(s, proj, tolerance_m) for s in trim(segs, trim_m)]
        if not segs:
            continue
        name = f"day{d['day']:02d}" + (f"-{d['date']}" if d["date"] else "")
        path = out / f"{name}.gpx"
        path.write_text(gpx_text(f"Day {d['day']}", segs), encoding="utf-8")
        written.append(path)
    return written
