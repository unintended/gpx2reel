"""track.yaml schema (the storyboard) + semantic validation against route.json.

The storyboard is the single contract between the Claude session and the render scripts:
Claude writes it, the user edits it, the scripts execute it literally.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import presets as P
from .blender.bike import PALETTE

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
MIN_DAY_S = 3.0
# warm ramp yellow → violet: reads as time passing and stays visible on green/blue satellite imagery
DAY_COLORS = ["#FFD23F", "#FF9F1C", "#FF6B3D", "#F2385A", "#C2307A", "#8E3A9D", "#5B4BB7", "#3D6CD1"]
ROUTE_COLOR = "#FF6B3D"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Style(_M):
    """The look: terrain, route colours, trail, light and grade, overlay language."""
    terrain: str = Field("satellite", description="Ground look, presets.TERRAIN_STYLES; stylized / hybrid need "
                                                  "`landcover` and `osm` before `scene`")
    exaggeration: float = Field(1.6, ge=1.0, le=4.0, description="Vertical exaggeration of the terrain")
    text_style: str = Field("bold_minimal", description="Overlay text look, presets.TEXT_STYLES")
    color_mode: Literal["single", "per_day"] = Field("single", description="One route colour, or one per day")
    color: str = Field(ROUTE_COLOR, description="Route colour #RRGGBB (color_mode: single)")
    day_colors: list[str] = Field(DAY_COLORS[:4], description="Colours of days 1, 2, … (color_mode: per_day); "
                                                              "repeat when there are more days")
    trail: str = Field("band", description="Trail shape, presets.TRAIL_STYLES")
    ahead: str = Field("hidden", description="Route ahead of the marker, presets.AHEAD_MODES")
    day_night: bool = Field(True, description="Real sun by date, time and place; dusk → night → dawn between days")
    night_s: float = Field(1.2, ge=0, le=5, description="Length of each night between days, s")
    haze: float = Field(0.6, ge=0, le=1, description="Aerial perspective on distant terrain")
    bloom: float = Field(0.8, ge=0, le=2, description="Glow around the trail and marker (emission only)")
    contrast: float = Field(0.5, ge=0, le=1, description="S-curve + a little saturation (0 = raw render)")
    vignette: float = Field(0.4, ge=0, le=1, description="Corner darkening")
    climb_weight: float = Field(20.0, ge=0, description="Ride time by effort: × (1 + weight·grade), climbs take "
                                                        "longer (0 = even speed)")
    clouds: float = Field(0.45, ge=0, le=1, description="Cumulus coverage with shadows (0 = none); cleared around "
                                                        "highlights")
    language: str = Field("en", description="Overlay text language: en | ru")
    sun_disc: Literal["natural", "hinomaru"] = Field("natural", description="Sunrise / sunset disc: a golden sun, "
                                                                            "or the red disc of the Japanese flag")
    credits: bool = Field(True, description="Data attribution (imagery, terrain, OSM) under the outro and on covers")

    @field_validator("day_colors")
    @classmethod
    def _hex(cls, v):
        bad = [c for c in v if not HEX.match(c)]
        if bad:
            raise ValueError(f"not hex colors: {bad}")
        return v

    @field_validator("color")
    @classmethod
    def _hex1(cls, v):
        if not HEX.match(v):
            raise ValueError(f"not a hex color: {v}")
        return v

    def colors_for(self, n_days: int) -> list[str]:
        """Colour of each day, 1-based order."""
        if self.color_mode == "single":
            return [self.color] * n_days
        return [self.day_colors[i % len(self.day_colors)] for i in range(n_days)]


class Avatar(_M):
    """The marker riding the route."""
    model: str = Field("puck", description="presets.AVATARS")
    color: str = Field("periwinkle", description="Bike colour: palette name (blender/bike.py) or #RRGGBB albedo")
    path: Optional[str] = Field(None, description="Your own .glb (model: file)")
    scale: float = Field(1.0, gt=0, description="Size multiplier")


class Shot(_M):
    duration_s: float = Field(ge=0, description="Length, s")
    camera: str = Field("overview", description="presets.CAMERAS")
    caption: Optional[str] = Field(None, description="Text shown over the shot")


class ZoomStage(_M):
    """One step of the intro zoom: a view extent_km wide centred on center."""
    extent_km: float = Field(gt=0, description="Width of the view, km")
    center: Optional[tuple[float, float]] = Field(None, description="[lat, lon]; default: the route centre")
    label: Optional[str] = Field(None, description="Overlay text, e.g. \"Japan\"")


class Intro(Shot):
    """The opening: zooms in from the widest stage to the route overview."""
    zoom: list[ZoomStage] = Field([], description="Zoom stages, widest first; ends on the route overview")
    play_zoom: bool = Field(True, description="False: keep the zoom layers as far terrain, skip the zoom shot")


class Outro(Shot):
    """The finale over the whole route."""
    show_totals: bool = Field(True, description="Trip totals: km, climb, days")
    zoom_out: bool = Field(False, description="Back out through intro.zoom (route → region → country) and hold")
    route_points: list[str] = Field([], description="The finale's summary line (zoom_out): \"Beppu → Aso → …\"")


class DayPlan(_M):
    """One ride day; days not listed get the defaults."""
    day: int = Field(ge=1, description="Day number as `ingest` prints it")
    duration_s: Optional[float] = Field(None, gt=0, description="Riding time, s; default: split by `pacing`")
    caption: Optional[str] = Field(None, description="Pin label; default \"Day N\" in the overlay language")
    camera: str = Field("chase", description="presets.CAMERAS")
    start: Optional[str] = Field(None, description="Place name on the day card «start → finish» (1–2 words)")
    finish: Optional[str] = Field(None, description="Where the night after this day was spent")
    note: Optional[str] = Field(None, description="A line under the finish in the outro, e.g. \"Rest day ahead\"")
    opening: Optional[str] = Field(None, description="Shot before the ride, presets.OPENINGS")
    opening_s: float = Field(4.0, gt=0, le=10, description="Length of the opening, s")
    opening_caption: Optional[str] = Field(None, description="Opening caption, e.g. \"Land of the Rising Sun\"")
    opening_note: Optional[str] = Field(None, description="Big line above the caption, e.g. \"日出ずる国\" "
                                                          "(kanji from the vendored font subset)")
    closing: Optional[str] = Field(None, description="Shot after the ride, presets.CLOSINGS")
    closing_s: float = Field(5.0, gt=0, le=10, description="Length of the closing, s")
    closing_at: Optional[tuple[float, float]] = Field(None, description="[lat, lon] of a better sunset spot than "
                                                                        "the finish: the camera flies there, palms "
                                                                        "and surf stand on that shore")


class GapPlan(_M):
    """How a gap in the tracks is shown (gaps: `ingest` lists them)."""
    after_day: int = Field(ge=1, description="The day the gap follows (or lies in, with within_day)")
    within_day: bool = Field(False, description="A gap inside that day (e.g. a ferry), not after it")
    mode: str = Field("straight", description="presets.GAP_MODES")
    duration_s: float = Field(1.5, ge=0, description="Screen time of the crossing, s")
    caption: Optional[str] = Field(None, description="e.g. \"Train transfer · 240 km\"")


class Highlight(_M):
    """A stop on the ride: the camera leaves the marker to show a place."""
    poi: str = Field(description="Name shown in the callout")
    at_km: float = Field(ge=0, description="Route km where the ride pauses for it (`gpx2reel poi` prints it)")
    duration_s: float = Field(gt=0, le=15, description="Length, s")
    camera: str = Field("orbit", description="presets.CAMERAS")
    caption: Optional[str] = Field(None, description="Second callout line, e.g. \"Active volcano\"")
    material_preset: str = Field("default", description="presets.MATERIAL_PRESETS")
    detail: str = Field("normal", description="presets.DETAIL_LEVELS; high_res_dem needs lat / lon")
    effects: list[str] = Field([], description="presets.EFFECTS, placed at lat / lon")
    lat: Optional[float] = Field(None, description="The place itself (orbit centre, detail patch, effects)")
    lon: Optional[float] = None
    orbit_km: Optional[float] = Field(None, gt=0, description="Orbit radius, km; default 4")
    orbit_pitch_deg: Optional[float] = Field(None, gt=0, lt=90, description="Camera pitch, degrees; default 28. A "
                                                                            "close orbit among hills looks down "
                                                                            "steeper")


class Label(_M):
    """A named place shown when the camera passes by (candidates: `gpx2reel poi`)."""
    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    kind: str = Field("peak", description="presets.LABEL_KINDS")
    ele: Optional[float] = Field(None, description="Elevation shown for peaks: \"Aso · 1592 m\"")


class Storyboard(_M):
    """track.yaml: one film. Concepts: docs/configuration.md; preset names: `gpx2reel presets`."""
    version: Literal[1] = 1
    title: str = Field(description="Title over the intro (\"\" for none)")
    platform: str = Field("instagram_story", description="Caps duration_s, presets.PLATFORMS")
    pacing: Literal["km", "mixed", "even"] = Field("km", description="Seconds per day ∝ km / √km / equal "
                                                                     "(long trips: mixed)")
    focus_day: Optional[int] = Field(None, ge=1, description="Day story: only this day is ridden, earlier ones "
                                                             "pre-drawn (set by `day-story`)")
    story_chain: bool = Field(False, description="Day stories flow into each other: N ends on the frame N+1 "
                                                 "starts (set by `day-story`)")
    clock_offset_s: float = Field(0.0, ge=0, description="Story time before this one, so clouds and steam keep "
                                                         "drifting across a chain (set by `day-story`)")
    duration_s: float = Field(45, ge=5, le=180, description="Video length, s")
    fps: Literal[24, 30, 60] = 30
    resolution: tuple[int, int] = Field((1080, 1920), description="Width, height in px")
    style: Style = Style()
    avatar: Avatar = Avatar()
    music: Optional[str] = Field(None, description="Audio file, relative to the trip folder; mixed in with fades")
    intro: Intro = Intro(duration_s=2.5, camera="overview")
    days: list[DayPlan] = []
    gaps: list[GapPlan] = []
    highlights: list[Highlight] = []
    labels: list[Label] = Field([], description="Earlier entries win when labels overlap")
    outro: Outro = Outro(duration_s=4.5, camera="overview")

    def gap_modes(self) -> dict:
        """after_day → mode for gaps between days, "in<day>" → mode for gaps inside a day (world.gap_mode)."""
        return {(f"in{g.after_day}" if g.within_day else g.after_day): g.mode for g in self.gaps}


# --------------------------------------------------------------------------- IO

def load(path: Path) -> Storyboard:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Storyboard.model_validate(data)


def dump(sb: Storyboard, path: Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(sb.model_dump(mode="json", exclude_none=True), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- timing

def _weight(km: float, pacing: str) -> float:
    return {"km": km, "mixed": km ** 0.5, "even": 1.0}[pacing]


def plan_timing(sb: Storyboard, route: dict, min_day_s: float = MIN_DAY_S) -> dict:
    """Seconds per day: explicit where given, the rest split by `pacing` (∝ km, √km or equal).
    A day story (focus_day) rides only that day: no nights, gaps or highlights of other days."""
    route_days = {d["day"]: d for d in route["days"]}
    if sb.focus_day is not None:
        route_days = {k: v for k, v in route_days.items() if k == sb.focus_day}
    plans = {d.day: d for d in sb.days if d.day in route_days}
    fixed = sb.intro.duration_s + sb.outro.duration_s
    if sb.style.day_night and sb.focus_day is None:
        fixed += sb.style.night_s * max(len(route_days) - 1, 0)
    fixed += sum(d.opening_s for d in plans.values() if d.opening)
    fixed += sum(d.closing_s for d in plans.values() if d.closing)
    if sb.focus_day is None:
        fixed += sum(g.duration_s for g in sb.gaps)
    fixed += sum(h.duration_s for h in sb.highlights if _day_of_km(route, h.at_km) in route_days)
    explicit = {k: p.duration_s for k, p in plans.items() if p.duration_s}
    fixed += sum(explicit.values())
    free = sb.duration_s - fixed
    auto_days = [k for k in route_days if k not in explicit]
    w = {k: _weight(route_days[k]["distance_km"], sb.pacing) for k in auto_days}
    total_w = sum(w.values()) or 1.0
    out = dict(explicit)
    for k in auto_days:
        out[k] = max(min_day_s, free * w[k] / total_w) if free > 0 else 0.0
    return {"per_day_s": out, "fixed_s": fixed, "free_s": free}


def _day_of_km(route: dict, km: float) -> int | None:
    for d in route["days"]:
        if d["km_start"] - 1e-6 <= km <= d["km_end"] + 1e-6:
            return d["day"]
    return None


# --------------------------------------------------------------------------- validation

MAX_PLACE_CHARS = 20          # «Tadewara Wetlands» fits; «Sun Royal Hotel, Kagoshima» does not


def validate(sb: Storyboard, route: dict | None = None) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block rendering."""
    errors: list[str] = []
    warns: list[str] = []

    def check(value: str, registry: dict, what: str):
        if value not in registry:
            errors.append(f"{what}: unknown value '{value}'. Available: {', '.join(registry)}")

    check(sb.style.terrain, P.TERRAIN_STYLES, "style.terrain")
    check(sb.style.text_style, P.TEXT_STYLES, "style.text_style")
    check(sb.platform, P.PLATFORMS, "platform")
    limit = P.PLATFORM_MAX_S.get(sb.platform)
    if limit is not None and sb.duration_s > limit:
        errors.append(f"duration_s={sb.duration_s:.0f} s exceeds the {sb.platform} platform limit ({limit:.0f} s)")
    check(sb.style.trail, P.TRAIL_STYLES, "style.trail")
    check(sb.style.ahead, P.AHEAD_MODES, "style.ahead")
    check(sb.intro.camera, P.CAMERAS, "intro.camera")
    check(sb.outro.camera, P.CAMERAS, "outro.camera")
    avatar_kind = "file" if sb.avatar.model.endswith(".glb") else sb.avatar.model
    check(avatar_kind, P.AVATARS, "avatar.model")
    if avatar_kind == "file" and sb.avatar.path is None and not sb.avatar.model.endswith(".glb"):
        errors.append("avatar: model=file needs avatar.path")
    if not HEX.match(sb.avatar.color) and sb.avatar.color not in PALETTE:
        errors.append(f"avatar.color: '{sb.avatar.color}' is neither a hex color nor a palette name. Palette: {', '.join(PALETTE)}")
    ext = [z.extent_km for z in sb.intro.zoom]
    if ext != sorted(ext, reverse=True):
        errors.append("intro.zoom: levels must go from wide to close (extent_km decreasing)")
    for i, z in enumerate(sb.intro.zoom):
        if z.center and not (-90 <= z.center[0] <= 90 and -180 <= z.center[1] <= 180):
            errors.append(f"intro.zoom[{i}].center: expected [lat, lon], got {list(z.center)}")
    for lb in sb.labels:
        check(lb.kind, P.LABEL_KINDS, f"labels '{lb.name}'.kind")
    for d in sb.days:
        check(d.camera, P.CAMERAS, f"days[{d.day}].camera")
        if d.opening:
            check(d.opening, P.OPENINGS, f"days[{d.day}].opening")
        if d.closing:
            check(d.closing, P.CLOSINGS, f"days[{d.day}].closing")
    for g in sb.gaps:
        check(g.mode, P.GAP_MODES, f"gaps[after_day={g.after_day}].mode")
    for i, h in enumerate(sb.highlights):
        tag = f"highlights[{i}] '{h.poi}'"
        check(h.camera, P.CAMERAS, f"{tag}.camera")
        check(h.material_preset, P.MATERIAL_PRESETS, f"{tag}.material_preset")
        check(h.detail, P.DETAIL_LEVELS, f"{tag}.detail")
        for e in h.effects:
            check(e, P.EFFECTS, f"{tag}.effects")

    hs = sorted(sb.highlights, key=lambda h: h.at_km)
    for a, b in zip(hs, hs[1:]):
        if b.at_km - a.at_km < 1.0:
            warns.append(f"highlights '{a.poi}' and '{b.poi}' are within 1 km of each other — the scenes will overlap")

    if route is not None:
        n_days = route["totals"]["days"]
        total_km = route["totals"]["distance_km"]
        for d in sb.days:
            if d.day > n_days:
                errors.append(f"days: day {d.day} is not in the route ({n_days} days)")
        if sb.focus_day is not None and sb.focus_day > n_days:
            errors.append(f"focus_day: day {sb.focus_day} is not in the route ({n_days} days)")
        gap_days = {(g["after_day"], g["within_day"]) for g in route["gaps"]}
        for g in sb.gaps:
            if (g.after_day, g.within_day) not in gap_days:
                where = "within" if g.within_day else "after"
                warns.append(f"gaps: the route has no gap {where} day {g.after_day} — entry ignored")
        for h in sb.highlights:
            if h.at_km > total_km + 0.01:
                errors.append(f"highlight '{h.poi}': at_km={h.at_km} is beyond the route length ({total_km:.1f} km)")
        if sb.style.color_mode == "per_day" and len(sb.style.day_colors) < n_days:
            warns.append(f"style.day_colors: {len(sb.style.day_colors)} colors for {n_days} days — colors will repeat")

        t = plan_timing(sb, route)
        explicit = {d.day for d in sb.days if d.duration_s and d.day in t["per_day_s"]}
        n_auto = sum(1 for d in t["per_day_s"] if d not in explicit)
        need = t["fixed_s"] + MIN_DAY_S * n_auto
        if need > sb.duration_s + 0.01:
            errors.append(
                f"Not enough time: scenes plus at least {MIN_DAY_S:.0f} s per day need "
                f"{need:.1f} s, but the video is {sb.duration_s:.0f} s"
            )
        else:
            for day in explicit:
                if t["per_day_s"][day] < MIN_DAY_S:
                    warns.append(f"Day {day} gets only {t['per_day_s'][day]:.1f} s — it will flash by")
            total = t["fixed_s"] + sum(v for k, v in t["per_day_s"].items() if k not in explicit)
            if total > sb.duration_s + 0.5:
                warns.append(
                    f"With the {MIN_DAY_S:.0f} s per-day minimum the video will be ~{total:.1f} s instead of {sb.duration_s:.0f} s"
                )
    for d in sb.days:                                 # the day card line is «start → finish»: short names fit
        for role, name in (("start", d.start), ("finish", d.finish)):
            if name and len(name) > MAX_PLACE_CHARS:
                warns.append(f"days[{d.day}].{role} '{name}': longer than {MAX_PLACE_CHARS} characters — "
                             f"prefer a town or landmark in 1–2 words (no hotels or campsites)")
    for lb in sb.labels:
        if len(lb.name) > MAX_PLACE_CHARS:
            warns.append(f"labels '{lb.name}': longer than {MAX_PLACE_CHARS} characters — the label won't fit next to the dot")
    return errors, warns


def validate_file(path: Path, route: dict | None) -> tuple[list[str], list[str]]:
    try:
        sb = load(path)
    except ValidationError as e:
        return [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()], []
    except yaml.YAMLError as e:
        return [f"YAML: {e}"], []
    return validate(sb, route)


def skeleton(route: dict, title: str = "My trip", duration_s: float = 45) -> Storyboard:
    """Minimal valid storyboard for a route — starting point for Claude."""
    sb = Storyboard(title=title, duration_s=duration_s)
    sb.days = [DayPlan(day=d["day"]) for d in route["days"]]
    n = len(route["days"])
    sb.style.day_colors = [DAY_COLORS[i % len(DAY_COLORS)] for i in range(max(4, n))]
    # overnight: the dot just continues; no track: drive through it; transfer: arc over the terrain
    plan = {"overnight": ("join", 0.0), "missing": ("join", 1.0), "transfer": ("straight", 1.5)}
    sb.gaps = [
        GapPlan(after_day=g["after_day"], mode=plan[g["kind"]][0], duration_s=plan[g["kind"]][1])
        for g in route["gaps"]
        if not g["within_day"]
    ]
    return sb
