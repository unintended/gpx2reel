"""Track ingest (GPX, FIT): parse files, clean, group into days, find gaps, compute stats -> route.json."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import gpxpy
import numpy as np

from .geo import LocalProjection, haversine_m, haversine_np, moving_average


@dataclass
class IngestConfig:
    gap_threshold_m: float = 500.0      # end->start further than this = visible gap (dashed line)
    min_step_m: float = 1.5             # decimate stationary GPS jitter
    max_speed_kmh: float = 150.0        # faster than this between two fixes = spike or teleport
    ele_smooth_window: int = 7          # points, for ascent calculation
    ascent_hysteresis_m: float = 3.0
    moving_speed_kmh: float = 2.0       # below = standing
    moving_max_dt_s: float = 300.0      # longer pauses between fixes are not moving time
    transfer_speed_kmh: float = 30.0    # implied speed across a gap above this -> "transfer"
    overnight_max_m: float = 3000.0     # day change with a shorter gap -> "overnight" (hotel off the track)
    day_mode: str = "auto"              # auto | date | file


@dataclass
class Segment:
    file: str
    lat: np.ndarray
    lon: np.ndarray
    ele: np.ndarray                     # NaN where missing
    t: np.ndarray                       # float epoch seconds, NaN where missing
    order: tuple = (0, 0, 0)
    notes: list[str] = field(default_factory=list)
    source: str = "gpx"                 # "fit" = a device recording (barometric altitude, own totals)
    rec_ascent: float | None = None     # the device's own session totals for the whole recording (what
    rec_descent: float | None = None    # Garmin Connect shows); every piece of a split recording carries them
    maker: str | None = None            # FIT file_id manufacturer, e.g. "garmin"
    also: list[str] = field(default_factory=list)   # other files stitched into this recording

    @property
    def has_time(self) -> bool:
        return bool(np.isfinite(self.t).all()) and len(self.t) > 0

    @property
    def start_t(self) -> float:
        return float(self.t[0]) if self.has_time else math.nan

    def __len__(self) -> int:
        return len(self.lat)


# --------------------------------------------------------------------------- parsing

def _to_epoch(dt: datetime | None) -> float:
    if dt is None:
        return math.nan
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_gpx(path: Path) -> list[Segment]:
    with open(path, encoding="utf-8") as fh:
        g = gpxpy.parse(fh)
    raw: list[list] = []
    for trk in g.tracks:
        for seg in trk.segments:
            raw.append(seg.points)
    if not raw:  # route-only files (planned routes)
        for rte in g.routes:
            raw.append(rte.points)
    out = []
    for si, pts in enumerate(raw):
        if len(pts) < 2:
            continue
        lat = np.array([p.latitude for p in pts], float)
        lon = np.array([p.longitude for p in pts], float)
        ele = np.array([p.elevation if p.elevation is not None else np.nan for p in pts], float)
        t = np.array([_to_epoch(getattr(p, "time", None)) for p in pts], float)
        out.append(Segment(path.name, lat, lon, ele, t, order=(0, si, 0)))
    return out


FIT_SEMICIRCLE = 180 / 2**31


def parse_fit(path: Path, data: bytes | None = None, name: str | None = None) -> list[Segment]:
    """Bike computer / watch FIT activity: one segment of the records that have a position,
    plus the device's session totals (climb / descent)."""
    import io

    import fitdecode

    lat, lon, ele, t = [], [], [], []
    asc = desc = maker = None
    with fitdecode.FitReader(io.BytesIO(data) if data is not None else str(path)) as fr:
        for m in fr:
            if isinstance(m, fitdecode.FitDataMessage) and m.name == "file_id" and m.has_field("manufacturer"):
                maker = str(m.get_value("manufacturer")).lower()
            if isinstance(m, fitdecode.FitDataMessage) and m.name == "session":
                a = m.get_value("total_ascent") if m.has_field("total_ascent") else None
                d = m.get_value("total_descent") if m.has_field("total_descent") else None
                asc = (asc or 0) + a if a is not None else asc
                desc = (desc or 0) + d if d is not None else desc
            if not isinstance(m, fitdecode.FitDataMessage) or m.name != "record":
                continue
            if not m.has_field("position_lat") or m.get_value("position_lat") is None:
                continue
            alt = None
            for k in ("enhanced_altitude", "altitude"):
                if m.has_field(k) and m.get_value(k) is not None:
                    alt = m.get_value(k)
                    break
            lat.append(m.get_value("position_lat") * FIT_SEMICIRCLE)
            lon.append(m.get_value("position_long") * FIT_SEMICIRCLE)
            ele.append(np.nan if alt is None else float(alt))
            t.append(_to_epoch(m.get_value("timestamp")))
    if len(lat) < 2:
        return []
    return [Segment(name or path.name, np.array(lat), np.array(lon), np.array(ele), np.array(t), order=(0, 0, 0),
                    source="fit", rec_ascent=asc, rec_descent=desc, maker=maker)]


def parse_track(path: Path) -> list[Segment]:
    """GPX, FIT, or a Garmin Connect «export original» ZIP with FIT inside."""
    suffix = path.suffix.lower()
    if suffix == ".fit":
        return parse_fit(path)
    if suffix == ".zip":
        import zipfile

        out = []
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".fit"):
                    out += parse_fit(path, zf.read(n), name=path.name)
        return out
    return parse_gpx(path)


STITCH_MAX_M = 500.0   # a missing head / tail from another device is stitched on if it ends this close


def _priority(s: Segment) -> int:
    """Whose numbers count: the Garmin (what Garmin Connect shows) > another device > an app export."""
    return 2 if s.maker == "garmin" else 1 if s.source == "fit" else 0


def _cut(s: Segment, m: np.ndarray) -> Segment:
    return Segment(s.file, s.lat[m], s.lon[m], s.ele[m], s.t[m], source=s.source, maker=s.maker)


def _stitch(primary: Segment, others: list[Segment], notes: list[str]) -> Segment:
    """Prepend / append what another recording of the same ride has before the primary started / after it
    stopped (a watch switched on late). The climb of that piece is added to the primary's device total."""
    t0, t1 = primary.t[0], primary.t[-1]
    head = min((o for o in others if o.t[0] < t0 - 5), key=lambda o: o.t[0], default=None)
    tail = max((o for o in others if o.t[-1] > t1 + 5), key=lambda o: o.t[-1], default=None)
    parts, extra_up, extra_down, also = [primary], 0.0, 0.0, list(primary.also)
    for piece, where in ((head, "head"), (tail, "tail")):
        if piece is None:
            continue
        m = piece.t < t0 - 5 if where == "head" else piece.t > t1 + 5
        if m.sum() < 2:
            continue
        cut = _cut(piece, m)
        a, b = (cut, primary) if where == "head" else (primary, cut)
        if haversine_m(a.lat[-1], a.lon[-1], b.lat[0], b.lon[0]) > STITCH_MAX_M:
            continue
        e = moving_average(cut.ele, 5) if np.isfinite(cut.ele).any() else cut.ele
        up, down = ascent_descent(e, 1.0 if cut.source == "fit" else 3.0)
        extra_up, extra_down = extra_up + up, extra_down + down
        parts = [cut] + parts if where == "head" else parts + [cut]
        also.append(piece.file)
        km = float(haversine_np(cut.lat, cut.lon).sum() / 1000)
        notes.append(f"{primary.file}: {'start' if where == 'head' else 'end'} ({km:.1f} km, +{up:.0f} m) taken from {piece.file}")
    if len(parts) == 1:
        return primary
    out = Segment(primary.file, *(np.concatenate([getattr(p, k) for p in parts]) for k in ("lat", "lon", "ele", "t")),
                  order=primary.order, source=primary.source, maker=primary.maker, also=sorted(set(also)),
                  rec_ascent=None if primary.rec_ascent is None else primary.rec_ascent + extra_up,
                  rec_descent=None if primary.rec_descent is None else primary.rec_descent + extra_down)
    return out


def drop_duplicates(segments: list[Segment], min_overlap: float = 0.5) -> tuple[list[Segment], list[str]]:
    """The same ride recorded / exported several times (watch, bike computer, Komoot): group the recordings,
    keep the one whose numbers count (_priority, then the longest) and stitch on what it missed at the start / end."""
    timed = sorted((s for s in segments if s.has_time), key=lambda s: (-_priority(s), -(s.t[-1] - s.t[0])))
    groups: list[list[Segment]] = []
    for s in timed:
        dur = max(s.t[-1] - s.t[0], 1.0)
        g = next((g for g in groups if min(s.t[-1], g[0].t[-1]) - max(s.t[0], g[0].t[0])
                  > min_overlap * min(dur, max(g[0].t[-1] - g[0].t[0], 1.0))), None)
        if g is None:
            groups.append([s])
        else:
            g.append(s)
    keep, notes = [], []
    for g in groups:
        primary, others = g[0], g[1:]
        for o in others:
            notes.append(f"{o.file}: same ride as {primary.file} (by time) — using {primary.file}")
        keep.append(_stitch(primary, others, notes) if others else primary)
    return keep + [s for s in segments if not s.has_time], notes


# --------------------------------------------------------------------------- cleaning

def clean_segment(seg: Segment, cfg: IngestConfig) -> list[Segment]:
    """Drop invalid points, spikes and jitter; split where the track teleports."""
    ok = np.isfinite(seg.lat) & np.isfinite(seg.lon)
    lat, lon, ele, t = seg.lat[ok], seg.lon[ok], seg.ele[ok], seg.t[ok]
    if len(lat) < 2:
        return []
    has_time = bool(np.isfinite(t).all())

    pieces: list[list[int]] = [[0]]
    spikes = 0
    jitter = 0
    n = len(lat)
    i = 1
    while i < n:
        last = pieces[-1][-1]
        d = haversine_m(lat[last], lon[last], lat[i], lon[i])
        if d < cfg.min_step_m and i != n - 1:
            # drop only stationary jitter; slow-but-moving points keep their distance
            dt_j = (t[i] - t[last]) if has_time else 0.0
            if not has_time or dt_j <= 0 or d / dt_j * 3.6 < cfg.moving_speed_kmh or d < 1.0:
                jitter += 1
                i += 1
                continue
        too_fast = False
        if has_time:
            dt = t[i] - t[last]
            if dt > 0 and d / dt * 3.6 > cfg.max_speed_kmh:
                too_fast = True
            elif dt <= 0 and d > 50:
                too_fast = True
        if has_time:
            teleport = d > cfg.gap_threshold_m and (too_fast or (t[i] - t[last]) > 600)
        else:  # no timestamps: only split on really big jumps (planned routes have sparse points)
            teleport = d > 5 * cfg.gap_threshold_m
        if too_fast or teleport:
            # spike (next point comes back) or real jump (next point continues from here)?
            nxt = i + 1
            if nxt < n and haversine_m(lat[nxt], lon[nxt], lat[i], lon[i]) > haversine_m(
                lat[nxt], lon[nxt], lat[last], lon[last]
            ):
                spikes += 1
                i += 1
                continue
            if d > cfg.gap_threshold_m:
                pieces.append([i])  # split: GPS off / transfer inside one file
                i += 1
                continue
            spikes += 1
            i += 1
            continue
        pieces[-1].append(i)
        i += 1

    out = []
    for k, idx in enumerate(pieces):
        if len(idx) < 2:
            continue
        idx = np.array(idx)
        s = Segment(seg.file, lat[idx], lon[idx], ele[idx], t[idx], order=(seg.order[0], seg.order[1], k),
                    source=seg.source, rec_ascent=seg.rec_ascent, rec_descent=seg.rec_descent, maker=seg.maker,
                    also=list(seg.also))
        out.append(s)
    if out:
        if spikes:
            out[0].notes.append(f"{seg.file}: GPS spikes removed: {spikes}")
        if len(out) > 1:
            out[0].notes.append(f"{seg.file}: track split into {len(out)} parts (coordinate jump)")
    return out


# --------------------------------------------------------------------------- metrics

def ascent_descent(ele: np.ndarray, hysteresis: float) -> tuple[float, float]:
    e = ele[np.isfinite(ele)]
    if len(e) < 2:
        return 0.0, 0.0
    up = down = 0.0
    ref = e[0]
    for v in e[1:]:
        if v - ref >= hysteresis:
            up += v - ref
            ref = v
        elif ref - v >= hysteresis:
            down += ref - v
            ref = v
    return up, down


def moving_time(seg: Segment, cfg: IngestConfig) -> float:
    if not seg.has_time:
        return 0.0
    d = haversine_np(seg.lat, seg.lon)
    dt = np.diff(seg.t)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(dt > 0, d / dt * 3.6, 0)
    mask = (v >= cfg.moving_speed_kmh) & (dt <= cfg.moving_max_dt_s)
    return float(dt[mask].sum())


# --------------------------------------------------------------------------- main

def _local_tz(lat: float, lon: float) -> str:
    try:
        from timezonefinder import TimezoneFinder

        return TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"
    except Exception:
        return "UTC"


def ingest(files: list[Path], cfg: IngestConfig | None = None) -> dict:
    cfg = cfg or IngestConfig()
    files = sorted(files, key=lambda p: p.name)
    warnings: list[str] = []

    raw: list[Segment] = []
    for fi, f in enumerate(files):
        for s in parse_track(f):
            s.order = (fi, s.order[1], 0)
            raw.append(s)
    raw, notes = drop_duplicates(raw)
    warnings.extend(notes)
    used = {f for s in raw for f in [s.file, *s.also]}
    files = [f for f in files if f.name in used]
    segments: list[Segment] = []
    for s in raw:
        for c in clean_segment(s, cfg):
            warnings.extend(c.notes)
            segments.append(c)
    if not segments:
        raise ValueError("No track with points found in the files")

    all_have_time = all(s.has_time for s in segments)
    if all_have_time:
        segments.sort(key=lambda s: s.start_t)
    else:
        segments.sort(key=lambda s: s.order)
        warnings.append("Not all points have timestamps — order and days taken from file names")

    lat0 = float(np.mean(np.concatenate([s.lat for s in segments])))
    lon0 = float(np.mean(np.concatenate([s.lon for s in segments])))
    tz_name = _local_tz(segments[0].lat[0], segments[0].lon[0])
    tz = ZoneInfo(tz_name)

    # ---- day assignment
    mode = cfg.day_mode
    if mode == "auto":
        mode = "date" if all_have_time else "file"
    if mode == "date" and not all_have_time:
        warnings.append("day_mode=date needs timestamps, using file")
        mode = "file"
    keys = []
    for s in segments:
        if mode == "date":
            keys.append(datetime.fromtimestamp(s.start_t, tz).date().isoformat())
        else:
            keys.append(s.file)
    day_keys: list[str] = []
    for k in keys:
        if k not in day_keys:
            day_keys.append(k)
    day_of = [day_keys.index(k) + 1 for k in keys]

    # ---- cumulative km, gaps
    km = 0.0
    gaps = []
    seg_out: list[dict] = []
    prev: Segment | None = None
    prev_day = None
    for s, day in zip(segments, day_of):
        if prev is not None:
            d = haversine_m(prev.lat[-1], prev.lon[-1], s.lat[0], s.lon[0])
            dt_h = (s.start_t - prev.t[-1]) / 3600 if (s.has_time and prev.has_time) else None
            if d > cfg.gap_threshold_m:
                speed = (d / 1000) / dt_h if dt_h and dt_h > 0 else None
                if day != prev_day and d < cfg.overnight_max_m:
                    kind = "overnight"      # slept roughly where the day ended
                elif (speed is not None and speed > cfg.transfer_speed_kmh) or d > 20_000:
                    kind = "transfer"
                else:
                    kind = "missing"
                gaps.append(
                    {
                        "after_day": prev_day,
                        "before_day": day,
                        "within_day": day == prev_day,
                        "from": [float(prev.lat[-1]), float(prev.lon[-1]), _f(prev.ele[-1])],
                        "to": [float(s.lat[0]), float(s.lon[0]), _f(s.ele[0])],
                        "distance_km": round(d / 1000, 2),
                        "time_gap_h": round(dt_h, 2) if dt_h is not None else None,
                        "implied_speed_kmh": round(speed, 1) if speed is not None else None,
                        "kind": kind,
                        "at_km": round(km, 3),
                    }
                )
        steps = np.concatenate([[0.0], haversine_np(s.lat, s.lon)])
        cum = km + np.cumsum(steps) / 1000
        km = float(cum[-1])
        seg_out.append({"segment": s, "day": day, "km": cum})
        prev, prev_day = s, day

    # ---- per-day aggregation
    days = []
    for di, key in enumerate(day_keys, start=1):
        items = [x for x in seg_out if x["day"] == di]
        dist = sum(x["km"][-1] - x["km"][0] for x in items)
        up = down = 0.0
        mt = 0.0
        pts_segments = []
        for x in items:
            s: Segment = x["segment"]
            e = moving_average(s.ele, cfg.ele_smooth_window) if np.isfinite(s.ele).any() else s.ele
            u, dn = ascent_descent(e, 1.0 if s.source == "fit" else cfg.ascent_hysteresis_m)  # barometer: less noise
            up += u
            down += dn
            mt += moving_time(s, cfg)
            pts = [
                [round(float(a), 6), round(float(b), 6), _f(c), _f(tt), round(float(k), 4)]
                for a, b, c, tt, k in zip(s.lat, s.lon, s.ele, s.t, x["km"])
            ]
            pts_segments.append({"file": s.file, "points": pts})
        # the device's own totals (what Garmin Connect shows) when every recording of the day has them
        recs = {x["segment"].file: x["segment"] for x in items}
        dev_up = sum(r.rec_ascent for r in recs.values()) if all(r.rec_ascent is not None for r in recs.values()) else None
        dev_down = (sum(r.rec_descent for r in recs.values())
                    if all(r.rec_descent is not None for r in recs.values()) else None)
        first, last = items[0]["segment"], items[-1]["segment"]
        days.append(
            {
                "day": di,
                "key": key,
                "date": datetime.fromtimestamp(first.start_t, tz).date().isoformat() if first.has_time else None,
                "files": sorted({f for x in items for f in [x["segment"].file, *x["segment"].also]}),
                "distance_km": round(dist, 2),
                "ascent_m": round(dev_up if dev_up is not None else up),
                "descent_m": round(dev_down if dev_down is not None else down),
                "ascent_source": "device" if dev_up is not None else "track",
                "moving_time_s": round(mt),
                "start_time": _iso(first.t[0], tz) if first.has_time else None,
                "end_time": _iso(last.t[-1], tz) if last.has_time else None,
                "km_start": round(float(items[0]["km"][0]), 3),
                "km_end": round(float(items[-1]["km"][-1]), 3),
                "segments": pts_segments,
            }
        )
        if len({x["segment"].file for x in items}) > 1 and mode == "date":   # stitched pieces don't count
            warnings.append(f"Day {di}: several files ({', '.join(days[-1]['files'])}) — merged")

    all_lat = np.concatenate([s.lat for s in segments])
    all_lon = np.concatenate([s.lon for s in segments])
    all_ele = np.concatenate([s.ele for s in segments])
    proj = LocalProjection(lat0, lon0)
    fin = all_ele[np.isfinite(all_ele)]
    if len(fin) == 0:
        warnings.append("Tracks have no elevation — taking it from the terrain (DEM)")

    return {
        "version": 1,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": [f.name for f in files],
        "timezone": tz_name,
        "day_mode": mode,
        "projection": proj.to_dict(),
        "bbox": [float(all_lat.min()), float(all_lon.min()), float(all_lat.max()), float(all_lon.max())],
        "totals": {
            "days": len(days),
            "distance_km": round(sum(d["distance_km"] for d in days), 2),
            "ascent_m": sum(d["ascent_m"] for d in days),
            "descent_m": sum(d["descent_m"] for d in days),
            "moving_time_s": sum(d["moving_time_s"] for d in days),
            "min_ele_m": round(float(fin.min())) if len(fin) else None,
            "max_ele_m": round(float(fin.max())) if len(fin) else None,
            "gaps": len(gaps),
        },
        "days": days,
        "gaps": gaps,
        "warnings": warnings,
    }


def _f(v) -> float | None:
    v = float(v)
    return None if not math.isfinite(v) else round(v, 1)


def _iso(epoch: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(epoch, tz).isoformat(timespec="minutes")


def save_route(route: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(route, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load_route(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_duration(sec: float) -> str:
    h, m = divmod(int(round(sec / 60)), 60)
    return f"{h} h {m:02d} min"


def summary_text(route: dict) -> str:
    t = route["totals"]
    lines = [
        f"Trip: {t['days']} days, {t['distance_km']:.1f} km, {t['ascent_m']} m climbed, "
        f"{fmt_duration(t['moving_time_s'])} moving  (time zone {route['timezone']})",
    ]
    if t["min_ele_m"] is not None:
        lines.append(f"Elevation: {t['min_ele_m']}–{t['max_ele_m']} m")
    lines.append("")
    lines.append(" Day | Date       |     Km | Climb |     Moving | Files")
    for d in route["days"]:
        lines.append(
            f"{d['day']:>4} | {d['date'] or '—':<10} | {d['distance_km']:>6.1f} | {d['ascent_m']:>5} | "
            f"{fmt_duration(d['moving_time_s']):>10} | {', '.join(d['files'])}"
        )
    if route["gaps"]:
        lines.append("")
        lines.append("Gaps:")
        names = {"transfer": "transfer", "missing": "no track", "overnight": "overnight"}
        for g in route["gaps"]:
            where = f"within day {g['after_day']}" if g["within_day"] else f"between days {g['after_day']} and {g['before_day']}"
            extra = f", {g['time_gap_h']} h" if g["time_gap_h"] is not None else ""
            lines.append(f"  • {where}: {g['distance_km']} km{extra} — {names[g['kind']]} (at {g['at_km']:.1f} km)")
    if route["warnings"]:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  • {w}" for w in route["warnings"])
    return "\n".join(lines)
