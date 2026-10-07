"""Per-frame animation state from the storyboard: marker, trail, camera (numpy, no Blender).

Shots follow the storyboard order: intro (zoom stages → route overview) → per day (ride, split by
highlights) → joined gap / gap arc → … → outro. Seconds per day come from storyboard.plan_timing.
Motion runs on the path's draw coordinate s (km + joined gaps); km is kept for the overlays.
With ~9 km of route per second the camera flies high: its distance follows the marker's on-screen
speed, and the marker size and the trail width follow the camera distance (constant on screen).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .storyboard import Storyboard, plan_timing
from .world import RIBBON_LIFT_M, Path3D, TerrainGrid

BIKE_LEN_M = 1.75             # procedural Grizl, wheel to wheel with tyres
IMPLEMENTED_CAMERAS = {"overview", "chase", "drone", "orbit", "sunrise", "sunrise_beach", "sunset", "zoom_out"}
SUNRISE_LEAD_S = 20 * 60        # a sunrise opening starts this long before the sun comes up


@dataclass
class CameraRig:
    follow_s: float = 2.5          # camera distance = on-screen speed × follow_s
    d_min: float = 1_500.0
    d_max: float = 40_000.0
    chase_pitch_deg: float = 38.0
    chase_side_deg: float = 35.0   # off the direction of travel: the trail reads better at an angle
    drone_pitch_deg: float = 48.0
    drone_side_deg: float = 70.0
    drift_deg: float = 22.0        # the side angle swings ± this over drift_period_s: parallax, never a jump
    drift_period_s: float = 18.0
    pitch_drift_deg: float = 5.0
    overview_fill: float = 0.8     # the route overview fills this share of the half-frame
    outro_turn_deg: float = 10.0   # the outro overview slowly orbits the route instead of standing still
    overview_pitch_deg: float = 60.0
    zoom_pitch_deg: float = 80.0   # widest intro level looks almost straight down
    orbit_pitch_deg: float = 28.0
    orbit_dist: float = 4_000.0
    sunrise_back_m: float = 700.0  # sunrise opening: camera this far behind the start point (towards land)
    sunrise_height_m: float = 160.0
    sunrise_look_m: float = 9_000.0    # around a highlight's point of interest (lat/lon in the storyboard)
    beach_back_m: float = 110.0    # sunrise_beach: low behind the palms lining the coast, looking level at the sun
    beach_height_m: float = 14.0
    sunset_back_m: float = 520.0   # sunset closing: over the water east of the finish, looking at the sinking sun
    sunset_height_m: float = 35.0
    sunset_look_m: float = 1400.0  # also the camera distance the haze scales with: ridges fade in layers
    sunset_pitch_deg: float = 7.0  # the horizon in the lower third, the sun above it
    sunset_spot_back_m: float = 150.0      # closing_at (a beach away from the finish): low behind its palms,
    sunset_spot_height_m: float = 16.0     # like the beach sunrise
    sunset_high_deg: float = 10.0  # the shore lies this far below the camera's level → a viewpoint on a hill
    sunset_dip: float = 0.55       # … which tilts down by this share of that angle
    sunset_above_ridge_deg: float = 2.6    # the clock stops with the sun this far above the terrain's skyline
    sunset_above_sea_deg: float = 1.3      # … or with its lower edge on a sea horizon
    sunset_end_deg: float = -5.0   # the shot ends in the dusk, with the sun this far below the horizon
    sunset_span_deg: float = 6.0   # … after sinking this much during the shot
    beach_crane_s: float = 1.3     # the end of the beach opening rises and tilts down onto the start
    beach_crane_m: float = 150.0   # high enough that the 2 m/px imagery below is not a blur
    beach_crane_back_m: float = 100.0
    orbit_turn: float = 0.35       # fraction of a full circle per highlight
    lens_mm: float = 30.0
    blend_s: float = 1.2           # overview ↔ ride camera blend
    finale_trail_k: float = 0.55     # … the trail ends at this share of its usual width (full width reads too bold)
    finale_marker_k: float = 0.7
    finale_hold: tuple = (0.2, 0.3)  # zoom_out outro: share held on the route first and on the widest level last
    finale_dots: tuple = (1.3, 3.0)  # … the day dots shrink to nothing between these × the hold distance
    zoom_hold: float = 0.15        # share of the intro spent on the widest level before zooming
    smooth_s: float = 0.25         # camera path smoothing
    clearance: float = 0.08        # min height above terrain, × camera distance
    marker_screen: float = 0.009   # marker radius / camera distance
    checkpoint_screen: float = 0.0035
    trail_screen: float = 0.004    # trail width / camera distance
    trail_min_m: float = 6.0
    pop_s: float = 0.3             # checkpoint pop-in
    bike_screen: float = 0.06      # procedural bike avatar: length / camera distance
    wheel_rev_s: float = 2.0       # visual spin while moving (real speed would alias)
    crank_rev_s: float = 1.4


@dataclass
class Shot:
    kind: str                      # zoom | overview | opening | ride | highlight | night | join | gap
    camera: str
    t0: float
    t1: float
    s0: float = 0.0
    s1: float = 0.0
    day: int | None = None
    label: str = ""
    arc: np.ndarray | None = field(default=None, repr=False)
    target: np.ndarray | None = field(default=None, repr=False)    # highlight point of interest (scene xyz)
    orbit_m: float | None = None                                    # highlight orbit radius (Highlight.orbit_km)
    orbit_pitch: float | None = None                                # degrees (Highlight.orbit_pitch_deg)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


# --------------------------------------------------------------------------- shots

def _day_ranges(path: Path3D) -> dict[int, tuple[float, float, float]]:
    """day → (s at start, s where riding ends, s at the last point incl. a joined bridge)."""
    out = {}
    for d in np.unique(path.day):
        idx = np.flatnonzero(path.day == d)
        km = path.km[idx]
        ride_end = idx[int(np.searchsorted(km, km[-1]))]          # first point at the day's last km
        out[int(d)] = (float(path.s[idx[0]]), float(path.s[ride_end]), float(path.s[idx[-1]]))
    return out


def s_at_km(path: Path3D, km: float) -> float:
    return float(path.s[min(int(np.searchsorted(path.km, km)), len(path.s) - 1)])


def plan_shots(sb: Storyboard, route: dict, path: Path3D, arcs: dict[int, np.ndarray],
               pois: dict[str, list] | None = None) -> tuple[list[Shot], list[str]]:
    """arcs: after_day → arc points (gaps drawn 'straight'); pois: highlight name → scene xyz."""
    warns: list[str] = []
    timing = plan_timing(sb, route)
    plans = {d.day: d for d in sb.days}
    gap_plans = {g.after_day: g for g in sb.gaps}
    ranges = _day_ranges(path)
    shots: list[Shot] = []
    t = 0.0

    def add(kind, camera, dur, **kw):
        nonlocal t
        if dur <= 0:
            return
        if camera not in IMPLEMENTED_CAMERAS:
            warns.append(f"camera '{camera}' is not implemented yet — shooting as chase")
            camera = "chase"
        shots.append(Shot(kind, camera, t, t + dur, **kw))
        t += dur

    focus = sb.focus_day                             # day story: only this day is ridden
    s_pre = ranges[focus][0] if focus in ranges else 0.0   # earlier days are already drawn
    s_from = s_pre
    if sb.story_chain and focus is not None and focus - 1 in ranges:
        s_from = ranges[focus - 1][1]                # chained: start where the previous story ended
    add("zoom" if sb.intro.zoom and sb.intro.play_zoom else "overview", "overview", sb.intro.duration_s, s0=s_from,
        s1=s_pre, day=focus,
        label="intro")
    for d in route["days"]:
        n = d["day"]
        if focus is not None and n != focus:
            continue
        s_start, s_ride_end, s_last = ranges[n]
        cam = plans[n].camera if n in plans else "chase"
        hs = sorted((h for h in sb.highlights if d["km_start"] <= h.at_km <= d["km_end"]), key=lambda h: h.at_km)
        if n in plans and plans[n].opening in ("sunrise", "sunrise_beach"):
            add("opening", plans[n].opening, plans[n].opening_s, s0=s_start, s1=s_start, day=n,
                label=plans[n].opening_caption or "")
        stops = [s_start] + [s_at_km(path, h.at_km) for h in hs] + [s_ride_end]
        ride_s = timing["per_day_s"][n]
        span = max(s_ride_end - s_start, 1e-6)
        for i in range(len(stops) - 1):
            add("ride", cam, ride_s * (stops[i + 1] - stops[i]) / span, s0=stops[i], s1=stops[i + 1], day=n,
                label=f"day {n}")
            if i < len(hs):
                tgt = (pois or {}).get(hs[i].poi)
                add("highlight", hs[i].camera, hs[i].duration_s, s0=stops[i + 1], s1=stops[i + 1], day=n,
                    label=hs[i].poi, target=None if tgt is None else np.asarray(tgt, float),
                    orbit_m=hs[i].orbit_km * 1000 if hs[i].orbit_km else None, orbit_pitch=hs[i].orbit_pitch_deg)
        if n in plans and plans[n].closing == "sunset":
            spot = (pois or {}).get(f"closing{n}")         # closing_at: a sunset spot away from the finish
            add("closing", "sunset", plans[n].closing_s, s0=s_ride_end, s1=s_ride_end, day=n, label="sunset",
                target=None if spot is None else np.asarray(spot, float))
        if sb.style.day_night and focus is None and n != route["days"][-1]["day"]:
            add("night", cam, sb.style.night_s, s0=s_ride_end, s1=s_ride_end, day=n, label="night")
        g = gap_plans.get(n)
        if g is None or focus is not None:          # a day story ends with its day
            continue
        label = g.caption or "transfer"
        if g.mode in ("join", "road") and s_last > s_ride_end:
            add("join", cam, g.duration_s, s0=s_ride_end, s1=s_last, day=n, label=label)
        elif g.mode == "straight" and n in arcs:
            add("gap", "chase", g.duration_s, s0=s_last, s1=s_last, day=n, arc=arcs[n], label=label)
    s_end = ranges[focus][1] if focus in ranges else float(path.s[-1])
    add("overview", "zoom_out" if sb.outro.zoom_out and sb.intro.zoom else sb.outro.camera, sb.outro.duration_s,
        s0=s_end, s1=s_end, label="outro")
    return shots, warns


# --------------------------------------------------------------------------- helpers

def _ease(u: np.ndarray, a: float = 0.5) -> np.ndarray:
    """Linear blended with smoothstep: keeps some speed at the ends."""
    return u + a * (u * u * (3 - 2 * u) - u)


def _smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _gauss(v: np.ndarray, sigma_frames: float) -> np.ndarray:
    if sigma_frames < 0.5 or len(v) < 3:
        return v.copy()
    r = int(3 * sigma_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    k /= k.sum()
    pad = np.pad(v, [(r, r)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    if v.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(v.shape[1])], 1)


def _pose(target: np.ndarray, dist, pitch_rad) -> np.ndarray:
    """Camera south of target at dist, pitch above the horizon."""
    return target + np.column_stack([np.zeros_like(dist), -dist * np.cos(pitch_rad), dist * np.sin(pitch_rad)])


def blend_pose(ta, ca, tb, cb, u):
    """A → B per frame: distance geometric, target moves with the zoom, view direction lerped."""
    u = np.asarray(u, float)[:, None]
    da = np.linalg.norm(ca - ta, axis=-1, keepdims=True)
    db = np.linalg.norm(cb - tb, axis=-1, keepdims=True)
    d = da ** (1 - u) * db ** u
    diff = da - db
    w = np.where(np.abs(diff) > 1e-6, (da - d) / np.where(np.abs(diff) > 1e-6, diff, 1.0), u)
    target = ta + (tb - ta) * w
    o = (ca - ta) / da + ((cb - tb) / db - (ca - ta) / da) * u
    o /= np.linalg.norm(o, axis=-1, keepdims=True)
    return target, target + o * d


class _Track:
    """Ground position by s; the path is lifted for the trail, the marker stands on the ground."""

    def __init__(self, path: Path3D, lift_m: float):
        self.path, self.lift = path, lift_m

    def at(self, s: np.ndarray) -> np.ndarray:
        xyz = np.column_stack([np.interp(s, self.path.s, self.path.xyz[:, j]) for j in range(3)])
        xyz[:, 2] -= self.lift
        return xyz

    def direction(self, s: np.ndarray, half_window_km: np.ndarray) -> np.ndarray:
        lo, hi = self.path.s[0], self.path.s[-1]
        d = (self.at(np.clip(s + half_window_km, lo, hi)) - self.at(np.clip(s - half_window_km, lo, hi)))[:, :2]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-9)

    def day(self, s: np.ndarray) -> np.ndarray:
        return self.path.day[np.clip(np.searchsorted(self.path.s, s), 0, len(self.path.s) - 1)]

    def km(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.path.s, self.path.km)


def overview_pose(path: Path3D, aspect: float, rig: CameraRig) -> tuple[np.ndarray, np.ndarray]:
    """Camera from the south framing the whole route; returns (camera, target)."""
    lo, hi = path.xyz[:, :2].min(0), path.xyz[:, :2].max(0)
    c = (lo + hi) / 2
    ex, ey = hi - lo
    pitch = math.radians(rig.overview_pitch_deg)
    tan_v = 12.0 / rig.lens_mm                 # sensor 24 mm, fit vertical
    d = 1.12 * max(ey * math.sin(pitch) / 2 / tan_v, ex / 2 / (tan_v * aspect))
    target = np.array([c[0], c[1], float(np.median(path.xyz[:, 2]))])
    # perspective: the near edge of the route looks bigger than the bbox estimate — fit the projected points
    # (also after the outro's turn) into rig.overview_fill of the frame
    pts = path.xyz[:: max(1, len(path.xyz) // 400)]
    for _ in range(6):
        worst = 0.0
        for turn in (0.0, math.radians(rig.outro_turn_deg)):
            off = _pose(np.zeros((1, 3)), np.array([d]), pitch)[0]
            ca, sa = math.cos(turn), math.sin(turn)
            cam = target + np.array([off[0] * ca - off[1] * sa, off[0] * sa + off[1] * ca, off[2]])
            fwd = (target - cam) / np.linalg.norm(target - cam)
            right = np.cross(fwd, [0.0, 0.0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            rel = pts - cam
            z = rel @ fwd
            worst = max(worst, float(np.max(np.abs(rel @ right) / z / (tan_v * aspect))),
                        float(np.max(np.abs(rel @ up) / z / tan_v)))
        if abs(worst - rig.overview_fill) < 0.01:
            break
        d *= worst / rig.overview_fill
    return _pose(target[None], np.array([d]), pitch)[0], target


def outro_end_pose(path: Path3D, aspect: float, rig: CameraRig) -> tuple[np.ndarray, np.ndarray]:
    """(camera, target) of the last outro frame: the overview pushed out and turned (see build_frames)."""
    cam, tgt = overview_pose(path, aspect, rig)
    off = (cam - tgt) * 1.06
    a = math.radians(rig.outro_turn_deg)
    ca, sa = math.cos(a), math.sin(a)
    return tgt + np.array([off[0] * ca - off[1] * sa, off[0] * sa + off[1] * ca, off[2]]), tgt


def zoom_poses(zoom: list[dict], ov_cam, ov_target, rig: CameraRig) -> list[tuple[np.ndarray, np.ndarray]]:
    """(camera, target) per zoom level (extent = visible frame height), ending on the route overview."""
    tan_v = 12.0 / rig.lens_mm
    out = []
    p0, p1 = math.radians(rig.zoom_pitch_deg), math.radians(rig.overview_pitch_deg)
    for i, z in enumerate(zoom):
        pitch = p0 + (p1 - p0) * i / max(len(zoom), 1)
        tgt = np.array([z["center_xy"][0], z["center_xy"][1], 0.0])
        out.append((_pose(tgt[None], np.array([z["extent_m"] / 2 / tan_v]), pitch)[0], tgt))
    out.append((ov_cam, ov_target))
    return out


# --------------------------------------------------------------------------- frames

def day_clock_of(route: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """day → (km, epoch) of its track points that have a time (for frame_times)."""
    out = {}
    for d in route["days"]:
        pts = np.array([[q[4], q[3]] for s in d["segments"] for q in s["points"] if q[3] is not None], float)
        if len(pts):
            out[d["day"]] = (pts[:, 0], pts[:, 1])
    return out


def _finale_dots(cam_dist: np.ndarray, rig: CameraRig) -> np.ndarray:
    """Size factor of the day dots from the finale's hold frame on (cam_dist[0] = the hold distance): from far
    off they are white specks on a small route, so they shrink away as the camera backs out."""
    a, b = (math.log(k) for k in rig.finale_dots)
    return 1.0 - _smoothstep((np.log(cam_dist / cam_dist[0]) - a) / (b - a))


def frame_times(shots: list[Shot], t: np.ndarray, shot_of: np.ndarray, km: np.ndarray, day_clock: dict,
                openings: dict[int, float] | None = None, chain: dict | None = None,
                latlon: tuple[float, float] | None = None,
                closings: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    """Real time (epoch s) per frame. day_clock: day → (km array, epoch array) of its track points.
    Night shots run the clock from the day's end to the next day's start (or to its opening: day → epoch the
    opening begins); an opening runs from there to the ride start, lingering at the beginning (sunrise)."""
    from .sun import one_night

    openings = openings or {}
    between = (lambda a, b, u: one_night(latlon[0], latlon[1], a, b, u)) if latlon else (lambda a, b, u: a + (b - a) * u)
    days = sorted(day_clock)
    start = {d: float(day_clock[d][1][0]) for d in days}
    ride_end = {d: float(day_clock[d][1][-1]) for d in days}
    closings = closings or {}
    end = {d: closings[d][1] if d in closings else ride_end[d] for d in days}   # the evening goes on after a closing
    nxt = {d: days[min(i + 1, len(days) - 1)] for i, d in enumerate(days)}
    out = np.zeros(len(t))
    for si, sh in enumerate(shots):
        m = shot_of == si
        if not m.any():
            continue
        u = (t[m] - sh.t0) / max(sh.duration, 1e-9)
        if sh.kind in ("ride", "highlight"):
            k, e = day_clock[sh.day]
            out[m] = np.interp(km[m], k, e)
        elif sh.kind == "night":
            nd = nxt[sh.day]
            out[m] = between(end[sh.day], openings.get(nd, start[nd]), u)
        elif sh.kind == "opening":
            tb = openings.get(sh.day, start[sh.day])
            out[m] = tb + (start[sh.day] - tb) * u ** 3.2              # linger around the moment of sunrise
        elif sh.kind == "closing":
            ts, te = closings.get(sh.day, (ride_end[sh.day], ride_end[sh.day]))
            u0 = min(0.8 / max(sh.duration, 1e-9), 0.5)            # time-lapse into the evening, then linger
            x = np.clip((u - u0) / (1 - u0), 0, 1)
            out[m] = np.where(u < u0, ride_end[sh.day] + (ts - ride_end[sh.day]) * _smoothstep(u / u0),
                              ts + (te - ts) * x)                  # steadily down, behind the horizon, into dusk
        elif sh.kind in ("join", "gap"):
            out[m] = start[nxt[sh.day]]
        elif chain and si == 0 and "intro" in chain:                # chained story: night → dawn
            a, b = chain["intro"]
            out[m] = between(a, b, u)
        elif chain and si == len(shots) - 1 and "outro" in chain:   # chained story: dusk → night
            a, b = chain["outro"]
            out[m] = a + (b - a) * u
        else:
            first = sh.day if sh.day in start else days[0]           # a day story's intro is its morning
            out[m] = end[days[-1]] if si == len(shots) - 1 else start[first]
    return out


def build_frames(shots: list[Shot], path: Path3D, grid: TerrainGrid, lift_m: float, fps: int, aspect: float,
                 zoom: list[dict] | None = None, checkpoints: list[dict] | None = None,
                 day_rgba: dict[int, tuple] | None = None, rig: CameraRig | None = None,
                 day_clock: dict | None = None, latlon: tuple[float, float] | None = None,
                 chain: bool = False, climb_weight: float = 0.0, sun_disc: str = "natural") -> dict:
    """chain: day stories that flow into each other (storyboard.story_chain): the outro ends in the night on a
    pose the next story starts from."""
    rig = rig or CameraRig()
    track = _Track(path, lift_m)
    n = int(round(shots[-1].t1 * fps))
    t = np.arange(n) / fps
    shot_of = np.clip(np.searchsorted([s.t1 for s in shots], t, side="right"), 0, len(shots) - 1)

    # ---- marker position along the path; a ride spends its time by effort, not by distance: climbs take longer
    effort = _effort(path, lift_m, climb_weight)
    s = np.zeros(n)
    marker = np.zeros((n, 3))
    for si, sh in enumerate(shots):
        m = shot_of == si
        if not m.any():
            continue
        u = (t[m] - sh.t0) / max(sh.duration, 1e-9)
        if sh.kind == "gap":
            a = sh.arc
            seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
            q = _ease(u) * seg[-1]
            marker[m] = np.column_stack([np.interp(q, seg, a[:, j]) for j in range(3)])
            marker[m, 2] -= lift_m
            s[m] = sh.s0
            continue
        if sh.kind == "ride" and climb_weight > 0:
            e0, e1 = np.interp([sh.s0, sh.s1], path.s, effort)
            s[m] = np.interp(e0 + (e1 - e0) * _ease(u), effort, path.s)
        else:
            s[m] = sh.s0 + (sh.s1 - sh.s0) * (u if sh.kind == "join" else _ease(u))
        marker[m] = track.at(s[m])

    # ---- camera distance from on-screen speed
    speed = _gauss(np.linalg.norm(np.gradient(marker, axis=0), axis=1) * fps, 0.5 * fps)
    dist = np.clip(speed * rig.follow_s, rig.d_min, rig.d_max)
    for si, sh in enumerate(shots):                    # the marker rests at night: keep the camera where it was
        m = np.flatnonzero(shot_of == si)
        if sh.kind == "night" and len(m) and m[0] > 0:
            dist[m] = dist[m[0] - 1]
    dist = _gauss(dist, 0.3 * fps)

    # ---- real time and sun (needed by the sunrise camera)
    light = {}
    if day_clock and latlon:
        from .sun import (SEA_MIRROR, SUNRISE_DISKS, SUNSET_AMBIENT, SUNSET_DISK, night_after, sky_look, solar_position, sun_descends_to,
                          sun_vector, sunrise_before, sunset_look)

        openings = {sh.day: sunrise_before(latlon[0], latlon[1], float(day_clock[sh.day][1][0])) - SUNRISE_LEAD_S
                    for sh in shots if sh.kind == "opening" and sh.day in day_clock}
        closings = {}
        for sh in shots:
            if sh.kind == "closing" and sh.day in day_clock:
                span = _sunset_span(sh, track, grid, rig, latlon, float(day_clock[sh.day][1][-1]))
                if span:
                    closings[sh.day] = span
        chain_clock = {}
        focus = shots[0].day
        if chain and focus in day_clock:
            end_f = closings[focus][1] if focus in closings else float(day_clock[focus][1][-1])
            chain_clock["outro"] = (end_f, night_after(latlon[0], latlon[1], end_f))
            if focus - 1 in day_clock:
                end_p = float(day_clock[focus - 1][1][-1])
                chain_clock["intro"] = (night_after(latlon[0], latlon[1], end_p),
                                        openings.get(focus, float(day_clock[focus][1][0])))
        epoch = frame_times(shots, t, shot_of, track.km(s), day_clock, openings, chain_clock, latlon, closings)
        az, el = solar_position(latlon[0], latlon[1], epoch)
        warm = np.zeros(n)                              # sunset closings: the sky, haze and sun turn orange
        for sh in shots:
            if sh.kind == "closing":
                warm = np.maximum(warm, _smoothstep((t - sh.t0) / 0.8) * (1 - _smoothstep((t - sh.t1) / 1.0)))
        rise = SUNRISE_DISKS[sun_disc]
        disk = rise[None] + (SUNSET_DISK - rise)[None] * (warm > 0.01)[:, None]
        # the disc normally shrinks to nothing near the horizon (it showed through the terrain's soft edges at
        # night); in a sunset it keeps its size and goes down behind the sea or the ridge, then is gone
        k = np.clip((el + 1.5) / 2.0, 0, 1)
        disk_k = np.where(warm > 0.5, (el > -2.0).astype(float), k * k * (3 - 2 * k))
        # the disc only exists in the sunrise / sunset shots: elsewhere (an overview at dawn) EEVEE lit the land
        # with it as a big red glow from below the horizon
        near = np.zeros(n)
        for sh in shots:
            if sh.kind in ("opening", "closing"):
                near = np.maximum(near, ((t > sh.t0 - 0.5) & (t < sh.t1 + 0.5)).astype(float))
        disk_k = disk_k * near
        light = {"epoch": epoch, "sun_el": el, "sun_vec": sun_vector(az, el),
                 **sunset_look(sky_look(el), warm, el),
                 "sun_disk_rgba": disk, "sun_disk_k": disk_k, "sunset": warm,
                 "ambient": 1 - (1 - SUNSET_AMBIENT) * warm,
                 "mirror": SEA_MIRROR + (1 - SEA_MIRROR) * warm}

    cam = np.zeros((n, 3))
    target = np.zeros((n, 3))
    heading = track.direction(s, np.maximum(0.3, dist / 2000))
    ov_cam, ov_target = overview_pose(path, aspect, rig)
    chain_pose = None                                   # the previous story's last camera pose
    if chain and shots[0].day is not None:
        prev = shots[0].day - 1
        ranges = _day_ranges(path)
        if prev in ranges:
            keep = path.s <= ranges[prev][1] + 1e-9
            sub = Path3D(path.xyz[keep], path.km[keep], path.day[keep], path.piece[keep], path.s[keep])
            chain_pose = outro_end_pose(sub, aspect, rig)
    # slow, continuous swing of the ride camera's side angle and pitch (parallax without cuts)
    phase = 2 * math.pi * t / max(rig.drift_period_s, 1e-6)
    side_drift = np.radians(rig.drift_deg * np.sin(phase))
    pitch_drift = np.radians(rig.pitch_drift_deg * np.sin(phase * 0.5 + 1.0))
    for si, sh in enumerate(shots):
        m = np.flatnonzero(shot_of == si)
        if len(m) == 0:
            continue
        u = (t[m] - sh.t0) / max(sh.duration, 1e-9)
        if sh.kind == "zoom":
            poses = zoom_poses(zoom or [], ov_cam, ov_target, rig)
            k = len(poses) - 1
            v = np.clip((u - rig.zoom_hold) / (1 - rig.zoom_hold), 0, 1) * k
            j = np.minimum(v.astype(int), k - 1)
            target[m], cam[m] = blend_pose(np.array([poses[i][1] for i in j]), np.array([poses[i][0] for i in j]),
                                           np.array([poses[i + 1][1] for i in j]), np.array([poses[i + 1][0] for i in j]),
                                           _smoothstep(v - j))
            continue
        if sh.kind == "overview" and si == 0 and chain_pose is not None:
            cam[m], target[m] = chain_pose[0], chain_pose[1]
            continue
        if sh.kind == "overview" and sh.camera == "zoom_out" and zoom:
            # the finale: hold on the route, back out through the intro's zoom stages, hold on the widest
            poses = _keep_route_in_frame(zoom_poses(zoom, ov_cam, ov_target, rig), path, aspect, rig)[::-1]
            k = len(poses) - 1
            v = np.clip((u - rig.finale_hold[0]) / (1 - sum(rig.finale_hold)), 0, 1) * k
            j = np.minimum(v.astype(int), k - 1)
            target[m], cam[m] = blend_pose(np.array([poses[i][1] for i in j]), np.array([poses[i][0] for i in j]),
                                           np.array([poses[i + 1][1] for i in j]), np.array([poses[i + 1][0] for i in j]),
                                           _smoothstep(v - j))
            continue
        if sh.kind == "overview":
            if chain and si == len(shots) - 1:            # chained: the very last frame lands on the handoff pose
                u = np.clip((t[m] - sh.t0) / max(sh.duration - 1.0 / fps, 1e-9), 0, 1)
            push = 1.06 - 0.06 * u if si == 0 else 1.0 + 0.06 * u
            off = (ov_cam - ov_target)[None] * push[:, None]
            if si == len(shots) - 1:                              # outro: orbit a little around the route
                a = np.radians(rig.outro_turn_deg) * _ease(u)
                ca, sa = np.cos(a), np.sin(a)
                off = np.column_stack([off[:, 0] * ca - off[:, 1] * sa, off[:, 0] * sa + off[:, 1] * ca, off[:, 2]])
            cam[m] = ov_target + off
            target[m] = ov_target
            continue
        if sh.kind == "opening":
            # low over the start point, looking at the sun where it comes up; a slow push towards it
            sv = light["sun_vec"][m] if light else np.tile([1.0, 0.0, 0.0], (len(m), 1))
            k = int(np.argmin(np.abs(light["sun_el"][m]))) if light else 0
            d = sv[k, :2] / max(np.linalg.norm(sv[k, :2]), 1e-9)
            base = marker[m[0]]
            beach = sh.camera == "sunrise_beach"
            h = rig.beach_height_m if beach else rig.sunrise_height_m
            back = (rig.beach_back_m if beach else rig.sunrise_back_m) * (1 - 0.4 * _smoothstep(u))
            xy = base[:2][None] - d[None] * back[:, None]
            cam[m] = np.column_stack([xy, grid.height_at(xy[:, 0], xy[:, 1]) + h])
            tgt = base[:2] + d * rig.sunrise_look_m
            # beach: level, the horizon mid-frame between the sky and the palms; high: looking down a little
            target[m] = np.array([tgt[0], tgt[1], cam[m, 2].mean() if beach else h * 0.35])
            if beach:                  # crane up and tilt onto the start: the ride camera takes over from above
                cam[m], target[m] = _crane(cam[m], target[m], base, d, t[m], sh.t1, rig)
            continue
        if sh.kind == "closing":
            # over the water east of the finish, looking at the sun going down behind the shore; a slow push in
            sv = light["sun_vec"][m] if light else np.tile([-1.0, 0.0, 0.0], (len(m), 1))
            k = int(np.argmin(np.abs(light["sun_el"][m]))) if light else -1    # where the sun meets the horizon
            d = sv[k, :2] / max(np.linalg.norm(sv[k, :2]), 1e-9)
            base = marker[m[0]] if sh.target is None else sh.target
            back0, h = ((rig.sunset_back_m, rig.sunset_height_m) if sh.target is None
                        else (rig.sunset_spot_back_m, rig.sunset_spot_height_m))
            back = back0 * (1 - 0.3 * _smoothstep(u))
            xy = base[:2][None] - d[None] * back[:, None]
            cam[m] = np.column_stack([xy, grid.height_at(xy[:, 0], xy[:, 1]) + h])
            # from a hill above the shore the camera looks down into the bay (the horizon in the upper third);
            # from low over the water it looks up a little
            dep = math.degrees(math.atan2(float(cam[m, 2].mean() - base[2]), float(back.mean())))
            high = dep > rig.sunset_high_deg
            pitch = rig.sunset_pitch_deg - (rig.sunset_dip * dep if high else 0.0)
            target[m] = cam[m] + np.array([*(d * rig.sunset_look_m),
                                           math.tan(math.radians(pitch)) * rig.sunset_look_m])
            continue                   # no crane: the camera stays on the shore while the dusk comes down
        hd = heading[m]
        if sh.kind == "gap":
            d2 = np.gradient(marker[m, :2], axis=0) if len(m) > 1 else (sh.arc[-1:, :2] - sh.arc[:1, :2])
            hd = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
            heading[m] = hd
        D = dist[m]
        drone = sh.camera == "drone"
        side = math.radians(rig.drone_side_deg if drone else rig.chase_side_deg) + side_drift[m]
        if sh.camera == "orbit":
            centre = marker[m] if sh.target is None else np.repeat(sh.target[None], len(m), 0)
            D = np.full(len(m), rig.d_min if sh.target is None else (sh.orbit_m or rig.orbit_dist))
            # start behind the direction of travel, i.e. where the chase camera already is
            a0 = math.atan2(hd[0, 1], hd[0, 0]) + math.pi + side[0]
            ang = a0 + 2 * math.pi * rig.orbit_turn * u
            p = math.radians(sh.orbit_pitch or rig.orbit_pitch_deg)
            off = np.column_stack([np.cos(ang) * math.cos(p), np.sin(ang) * math.cos(p), np.full(len(m), math.sin(p))])
            target[m] = centre
        else:
            p = math.radians(rig.drone_pitch_deg if drone else rig.chase_pitch_deg) + pitch_drift[m]
            c, sn = np.cos(side), np.sin(side)
            back = np.column_stack([-(hd[:, 0] * c - hd[:, 1] * sn), -(hd[:, 0] * sn + hd[:, 1] * c)])
            off = np.column_stack([back * np.cos(p)[:, None], np.sin(p)])
            target[m] = marker[m] + np.column_stack([hd * D[:, None] * 0.15, np.zeros(len(m))])
        cam[m] = target[m] + off * D[:, None]

    raw_cam, raw_target = cam.copy(), target.copy()
    cam, target = _gauss(cam, rig.smooth_s * fps), _gauss(target, rig.smooth_s * fps)
    if chain:                                         # the outro is smooth already; keep its handoff pose exact
        last = shot_of == len(shots) - 1
        cam[last], target[last] = raw_cam[last], raw_target[last]
    # the beach opening is smooth already, and smoothing across its cut would lift the camera off the beach
    # before it has tilted down (the ride camera starts kilometres up)
    beach_op = np.isin(shot_of, [i for i, sh in enumerate(shots) if sh.camera in ("sunrise_beach", "sunset")])
    cam[beach_op], target[beach_op] = raw_cam[beach_op], raw_target[beach_op]
    # wide (zoom / overview) ↔ ride ↔ highlight: fly between the poses over the first blend_s of the next shot
    nb = max(int(rig.blend_s * fps), 1)
    mode = ["wide" if sh.kind in ("zoom", "overview") else "focus" if sh.kind in ("highlight", "opening", "closing") else "ride"
            for sh in shots]
    for si in range(len(shots) - 1):
        if mode[si] == mode[si + 1]:
            continue
        f = int(round(shots[si].t1 * fps))
        b = np.arange(f, min(f + nb, n))
        if len(b) == 0 or f == 0:
            continue
        u = _smoothstep((b - (f - 1)) / nb)
        target[b], cam[b] = blend_pose(np.repeat(target[f - 1][None], len(b), 0), np.repeat(cam[f - 1][None], len(b), 0),
                                       target[b], cam[b], u)
    cam_dist = np.linalg.norm(cam - target, axis=1)
    ground = grid.height_at(cam[:, 0], cam[:, 1])
    low = np.isin(shot_of, [i for i, sh in enumerate(shots) if sh.kind in ("opening", "closing")])
    low_h = {"sunrise_beach": rig.beach_height_m * 0.6, "sunset": rig.sunset_spot_height_m * 0.6}
    low_clear = np.array([low_h.get(shots[i].camera, rig.sunrise_height_m * 0.5) for i in shot_of])
    clear = np.where(low, low_clear, rig.clearance * cam_dist)                     # sunrise / sunset cameras fly low
    cam[:, 2] = np.maximum(cam[:, 2], ground + clear)

    day = track.day(s)
    rgba = np.array([day_rgba[int(d)] for d in day], float) if day_rgba else np.ones((n, 4))
    cps = checkpoints or []
    cp_scale = np.zeros((n, len(cps)))
    for k, cp in enumerate(cps):
        hit = s >= cp["s"] - 1e-6
        reached = (-np.inf if hit[0] else t[np.argmax(hit)]) if hit.any() else np.inf
        d_cp = np.linalg.norm(cam - np.asarray(cp["xyz"], float), axis=1) if "xyz" in cp else cam_dist
        cp_scale[:, k] = _smoothstep((t - reached) / rig.pop_s) * d_cp * rig.checkpoint_screen

    # chained day story: today's trail greys into «the road so far» before the cut (the next story opens on it)
    past_mix = np.zeros(n)
    fumarole_show = np.zeros(n)       # fumaroles exist only around their highlight (EEVEE/Metal now and then
    for sh in shots:                  # failed a volume shader: a magenta blob far off in a ride shot)
        if sh.kind == "highlight":
            fumarole_show = np.maximum(fumarole_show, ((t > sh.t0 - 1.5) & (t < sh.t1 + 1.0)).astype(float))
    finale = np.zeros(n)              # the zoom_out finale thins the trail and the marker as it backs out
    for sh in shots:
        if sh.camera == "zoom_out":
            finale = np.maximum(finale, _smoothstep((t - sh.t0) / max(sh.duration * 0.6, 1e-6)))
            hold = int(np.argmin(np.abs(t - (sh.t0 + rig.finale_hold[0] * sh.duration))))
            cp_scale[hold:] *= _finale_dots(cam_dist[hold:], rig)[:, None]
    beach_show = np.zeros(n)          # the coast set's water and surf: in its shot, fading out into the next
    trail_hide = np.zeros(n)          # the beach opening also hides the trail (see trail_width)
    for sh in shots:
        if sh.camera in ("sunrise_beach", "sunset"):
            show = _smoothstep((t - sh.t0) / 0.5) * (1 - _smoothstep((t - sh.t1) / 1.5))
            beach_show = np.maximum(beach_show, show)
            if sh.camera == "sunrise_beach" or sh.target is not None:   # low among the palms: the trail of the
                trail_hide = np.maximum(trail_hide, show)               # ride along that shore fills the foreground
    if chain and shots[-1].kind == "overview":
        t1 = shots[-1].t1
        past_mix = _smoothstep((t - (t1 - 1.6)) / 1.1)

    ang = np.unwrap(np.arctan2(-heading[:, 0], heading[:, 1]))      # procedural bike: +Y → heading
    moving = _gauss((speed > 1.0).astype(float), 0.3 * fps)
    return {
        "t": t, "shot": shot_of, "s": s, "km": track.km(s), "day": day,
        # the marker keeps its size on screen by its own distance (the sunrise camera stands right next to it)
        "marker": marker, "marker_radius": np.linalg.norm(cam - marker, axis=1) * rig.marker_screen
                                           * (1 - (1 - rig.finale_marker_k) * finale),
        "marker_rgba": _gauss(rgba, 0.15 * fps),
        "cp_scale": cp_scale,
        "cam": cam, "target": target, "cam_dist": cam_dist,
        # the beach opening hides the trail: the day before ended here and its line would cross the foreground
        "trail_width": np.maximum(rig.trail_min_m, np.linalg.norm(cam - marker, axis=1) * rig.trail_screen)
                       * (1 - trail_hide) * (1 - (1 - rig.finale_trail_k) * finale),
        "bike_rot_z": _gauss(ang, 0.15 * fps), "bike_scale": cam_dist * rig.bike_screen / BIKE_LEN_M,
        "wheel": -np.cumsum(moving) * 2 * math.pi * rig.wheel_rev_s / fps,
        "crank": -np.cumsum(moving) * 2 * math.pi * rig.crank_rev_s / fps,
        "lens_mm": rig.lens_mm,
        "past_mix": past_mix,
        "beach_show": beach_show,
        "fumarole_show": fumarole_show,
        **light,
    }


def _effort(path: Path3D, lift_m: float, weight: float) -> np.ndarray:
    """Cumulative effort along path.s: distance × (1 + weight·uphill grade), the grade from the terrain the
    trail lies on (heights un-exaggerated by the lift ratio), smoothed over ~300 m against DEM noise."""
    ds = np.diff(path.s) * 1000
    if weight <= 0 or not len(ds):
        return np.asarray(path.s, float).copy()
    exag = lift_m / RIBBON_LIFT_M if lift_m else 1.0          # lift = RIBBON_LIFT_M × exaggeration
    z = path.xyz[:, 2] / max(exag, 1e-6)
    k = max(1, int(300 / max(np.median(ds[ds > 0]) if (ds > 0).any() else 20, 1)))
    z = np.convolve(np.pad(z, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    grade = np.where(ds > 0, np.diff(z) / np.maximum(ds, 1e-6), 0.0)
    w = 1 + weight * np.clip(grade, 0, 0.25)
    return np.concatenate([[0.0], np.cumsum(np.diff(path.s) * w)])


def _keep_route_in_frame(poses, path: Path3D, aspect: float, rig: CameraRig, margin: float = 0.08):
    """Shift the wide zoom levels sideways so the route stays in a portrait frame (the intro's Japan is centred
    on Japan, and Kyushu fell off the left edge of the finale)."""
    lo, hi = path.xyz[:, :2].min(axis=0), path.xyz[:, :2].max(axis=0)
    tan_v = 12.0 / rig.lens_mm
    out = []
    for cam, tgt in poses[:-1]:
        half_w = np.linalg.norm(cam - tgt) * tan_v * aspect * (1 - 2 * margin)
        half_h = np.linalg.norm(cam - tgt) * tan_v * (1 - 2 * margin)
        shift = np.zeros(3)
        for ax, half in ((0, half_w), (1, half_h)):
            if hi[ax] - lo[ax] < 2 * half:            # move just enough to bring the route's box inside
                shift[ax] = np.clip(tgt[ax], hi[ax] - half, lo[ax] + half) - tgt[ax]
        # …and into the upper half: the finale's summary card fills the lower part of the frame
        if hi[1] - lo[1] < half_h:             # (not all the way: the frame's bottom would leave the widest layer)
            shift[1] = min(shift[1], lo[1] - tgt[1] + 0.15 * half_h)
        out.append((cam + shift, tgt + shift))
    return out + [poses[-1]]


def _crane(cam: np.ndarray, target: np.ndarray, base: np.ndarray, d: np.ndarray, t: np.ndarray, t1: float,
           rig: CameraRig) -> tuple[np.ndarray, np.ndarray]:
    """The last beach_crane_s of a low shot: rise, pull back and tilt down onto `base`, so the next camera
    (kilometres up) takes over from above instead of from a level look at the horizon."""
    w = _smoothstep((t - (t1 - rig.beach_crane_s)) / rig.beach_crane_s)[:, None]
    cam = cam.copy()
    cam[:, 2] += w[:, 0] * rig.beach_crane_m
    cam[:, :2] -= d[None] * w * rig.beach_crane_back_m
    look = target - cam
    down = base[None] - cam
    far = np.linalg.norm(look, axis=1, keepdims=True)
    dist_m = np.linalg.norm(down, axis=1, keepdims=True)
    v = look / far + (down / dist_m - look / far) * w
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return cam, cam + v * (far + (dist_m - far) * w)


def _sunset_span(sh: Shot, track: "_Track", grid, rig: CameraRig, latlon, ride_end: float):
    """(start, end) epochs of a sunset closing: the sun sinks sunset_span_deg and stops sunset_above_ridge_deg
    over the skyline the camera sees towards it (hills hide the sun long before the sea horizon would)."""
    from .sun import solar_position, sun_descends_to

    t0 = sun_descends_to(latlon[0], latlon[1], ride_end, 0.0)
    if t0 is None:
        return None
    az, _ = solar_position(np.array([latlon[0]]), np.array([latlon[1]]), np.array([t0]))
    a = math.radians(float(az[0]))
    d = np.array([math.sin(a), math.cos(a)])
    base = track.at(np.array([sh.s0]))[0] if sh.target is None else sh.target
    back, h = ((rig.sunset_back_m, rig.sunset_height_m) if sh.target is None
               else (rig.sunset_spot_back_m, rig.sunset_spot_height_m))
    cam_xy = base[:2] - d * back
    cam_z = float(grid.height_at(cam_xy[:1], cam_xy[1:])[0]) + h
    r = np.arange(200.0, 60_000.0, 100.0)
    h = grid.height_at(cam_xy[0] + d[0] * r, cam_xy[1] + d[1] * r)
    ridge = max(float(np.degrees(np.arctan2(h - cam_z, r)).max()), 0.0)
    # over hills the whole disc stays above the ridge; over the open sea it comes down to touch the horizon
    above = rig.sunset_above_ridge_deg if ridge > 0.5 else rig.sunset_above_sea_deg
    # the shot starts sunset_span_deg above the moment the sun would rest on the skyline, and runs on until it
    # is well below the horizon: it sets and the dusk comes down on the same shore
    te = sun_descends_to(latlon[0], latlon[1], ride_end, rig.sunset_end_deg)
    ts = sun_descends_to(latlon[0], latlon[1], ride_end, ridge + above + rig.sunset_span_deg)
    if te is None:
        return None
    return (ts if ts is not None else ride_end), te


def shot_table(shots: list[Shot]) -> str:
    rows = []
    for s in shots:
        where = f"s {s.s0:.1f}–{s.s1:.1f}" if s.kind in ("ride", "join") else ""
        rows.append(f"{s.t0:5.1f}–{s.t1:5.1f} s  {s.kind:<9} {s.camera:<8} {s.label:<16} {where}")
    return "\n".join(rows)
