"""2D overlays drawn with Pillow over the rendered frames: km counters, checkpoint labels, zoom labels,
gap captions, title / totals. One transparent PNG per frame (build/overlay/),
composited by ffmpeg in render.py. Positions of 3D things come from projecting them with the timeline camera.

Layout keeps Instagram-story safe zones free: nothing important above SAFE_TOP or below SAFE_BOTTOM.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .trip import TripPaths

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path(__file__).parent / "assets" / "fonts"
BASE_H = 1920                  # sizes below are for a 1080×1920 frame and scale with the height
SAFE_TOP, SAFE_BOTTOM = 250, 1600
PROFILE_TOP, PROFILE_BOTTOM = 300, 400        # elevation profile band at the top (far terrain / sky behind it)
LABELS_TOP, LABELS_BOTTOM = 430, 1340         # 3D labels: below the profile, above the counters
TEXT = {
    "ru": {"day": "День", "total": "Всего", "km": "км", "m": "м", "finish": "Финиш", "night": "Ночь", "climb": "Подъём",
           "totals": "{days} · набор {climb} м", "hours": "{h} ч в седле", "days": ("день", "дня", "дней"), "time": "{h} ч {m:02d} мин",
           "months": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
                      "октября", "ноября", "декабря"], "date": "{d} {month}"},
    "en": {"day": "Day", "total": "Total", "km": "km", "m": "m", "finish": "Finish", "night": "Night", "climb": "Climb",
           "totals": "{days} · {climb} m climbed", "hours": "{h} h riding", "days": ("day", "days", "days"), "time": "{h} h {m:02d} min",
           "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "date": "{month} {d}"},
}
SENSOR_H_MM = 24.0             # Blender default; cameras use sensor_fit VERTICAL


# --------------------------------------------------------------------------- camera projection

def project(points: np.ndarray, cam: np.ndarray, target: np.ndarray, lens_mm: float,
            size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """World points (n, 3) → pixel xy (n, 2) and a mask of points in front of the camera.
    Matches a Blender camera with a TRACK_TO constraint (-Z to target, +Y up)."""
    w, h = size
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = points - cam
    z = rel @ fwd
    tan_v = SENSOR_H_MM / 2 / lens_mm
    tan_h = tan_v * w / h
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (rel @ right) / z / tan_h
        y = (rel @ up) / z / tan_v
    px = np.column_stack([(0.5 + x / 2) * w, (0.5 - y / 2) * h])
    return px, z > 0


# --------------------------------------------------------------------------- styles

@dataclass(frozen=True)
class TextStyle:
    number_font: str = "InterDisplay-Black.otf"
    label_font: str = "Inter-SemiBold.otf"
    small_font: str = "Inter-Medium.otf"
    color: tuple = (255, 255, 255, 255)
    shadow: bool = True                    # soft dark glow behind text (readable on bright terrain)
    plate: tuple | None = None             # rounded translucent plate behind text blocks
    label_tracking: float = 0.12           # letter spacing of small caps labels, × font size


STYLES = {
    "bold_minimal": TextStyle(),
    "badge": TextStyle(shadow=False, plate=(15, 18, 24, 150)),
    "retro_map": TextStyle(shadow=False, plate=(242, 232, 210, 225), color=(40, 34, 28, 255)),
}


JP_FONT = "NotoSansJP-Bold-subset.otf"    # kana + selected kanji (see assets/fonts/OFL-NotoSansJP.txt)


def _is_cjk(text: str) -> bool:
    return any("\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef" for ch in text)


@lru_cache(maxsize=64)
def font(name: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), px)


def font_for(text: str, name: str, px: int) -> ImageFont.FreeTypeFont:
    """Inter for Latin / Cyrillic; the Japanese subset when the text has kana / kanji."""
    return font(JP_FONT if _is_cjk(text) else name, px)


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def window(t: float, t0: float, t1: float, fade: float = 0.35) -> float:
    """1 inside [t0, t1], fading in/out over `fade` seconds at the ends."""
    return float(_smooth((t - t0) / fade) * _smooth((t1 - t) / fade))


def fmt_km(v: float) -> str:
    return f"{v:.1f}".replace(".", ",") if v < 100 else f"{v:.0f}"


def fmt_int(v: float) -> str:
    return f"{int(round(v)):,}".replace(",", " ")


# --------------------------------------------------------------------------- drawing

class Canvas:
    """RGBA layer with styled text blocks; `u` scales design pixels (1080×1920) to the frame."""

    def __init__(self, size: tuple[int, int], style: TextStyle):
        self.size, self.style = size, style
        self.u = size[1] / BASE_H
        self.text = Image.new("RGBA", size, (0, 0, 0, 0))
        self.back = Image.new("RGBA", size, (0, 0, 0, 0))         # plates / shadows
        self.dt = ImageDraw.Draw(self.text)
        self.db = ImageDraw.Draw(self.back)
        self.taken: list[tuple] = []          # boxes of text blocks drawn so far; place labels avoid them

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.u)))

    def _rgba(self, alpha: float, rgb=None):
        c = rgb or self.style.color[:3]
        return (*c[:3], int(round(self.style.color[3] * alpha)))

    def label_text(self, s: str) -> str:
        return s.upper()

    def spans(self, x: float, y: float, parts: list[tuple[str, str, int]], alpha: float,
              anchor: str = "ls", rgb=None, draw: bool = True) -> tuple[int, int, int, int]:
        """Draw text runs on one baseline: parts = (text, font file, design px). Returns bbox.
        draw=False only measures."""
        if alpha <= 0.003:
            return (0, 0, 0, 0)
        widths = []
        for text, f, size in parts:
            fnt = font_for(text, f, self.px(size))
            widths.append(self.dt.textlength(text, font=fnt))
        total = sum(widths)
        x0 = x - (total if anchor[0] == "r" else total / 2 if anchor[0] == "m" else 0)
        top, bottom = y, y
        cx = x0
        for (text, f, size), wdt in zip(parts, widths):
            fnt = font_for(text, f, self.px(size))
            if draw:
                self.dt.text((cx, y), text, font=fnt, fill=self._rgba(alpha, rgb), anchor="ls")
            b = self.dt.textbbox((cx, y), text, font=fnt, anchor="ls")
            top, bottom = min(top, b[1]), max(bottom, b[3])
            cx += wdt
        return (int(x0), int(top), int(x0 + total), int(bottom))

    def tracked(self, x: float, y: float, text: str, f: str, size: int, alpha: float, anchor="ls", rgb=None):
        """Small caps label with letter spacing."""
        if alpha <= 0.003:
            return (0, 0, 0, 0)
        fnt = font_for(text, f, self.px(size))
        track = self.style.label_tracking * self.px(size)
        chars = list(self.label_text(text))
        widths = [self.dt.textlength(c, font=fnt) for c in chars]
        total = sum(widths) + track * (len(chars) - 1)
        cx = x - (total if anchor[0] == "r" else total / 2 if anchor[0] == "m" else 0)
        b = self.dt.textbbox((cx, y), "".join(chars), font=fnt, anchor="ls")
        for c, wdt in zip(chars, widths):
            self.dt.text((cx, y), c, font=fnt, fill=self._rgba(alpha, rgb), anchor="ls")
            cx += wdt + track
        return (int(cx - total - track), b[1], int(cx - track), b[3])

    def plate(self, box, alpha: float, pad: float = 22):
        """Plate behind a text block (if the style has one); also marks the block as occupied."""
        if alpha > 0.003 and box[2] > box[0]:
            self.taken.append(box)
        if self.style.plate is None or alpha <= 0.003 or box[2] <= box[0]:
            return
        p = self.px(pad)
        c = self.style.plate
        self.db.rounded_rectangle((box[0] - p, box[1] - p, box[2] + p, box[3] + p), radius=self.px(26),
                                  fill=(*c[:3], int(c[3] * alpha)))

    def dot(self, x: float, y: float, r: float, rgb, alpha: float):
        r = self.px(r)
        self.dt.ellipse((x - r, y - r, x + r, y + r), fill=(*rgb[:3], int(255 * alpha)))

    def icon(self, kind: str, x: float, y: float, alpha: float):
        """Small glyph for a place kind (drawn with shapes: the font has no pictograms)."""
        a = int(255 * alpha)
        white, blue, dark = (255, 255, 255, a), (74, 160, 214, a), (30, 30, 30, a)
        r = self.px(11)
        if kind == "waterfall":                      # drop
            self.dt.polygon([(x, y - 1.7 * r), (x - r, y), (x + r, y)], fill=blue, outline=white)
            self.dt.ellipse((x - r, y - r, x + r, y + r), fill=blue, outline=white, width=self.px(2))
        elif kind == "lake":
            self.dt.ellipse((x - 1.4 * r, y - 0.9 * r, x + 1.4 * r, y + 0.9 * r), fill=blue, outline=white,
                            width=self.px(2))
        elif kind == "wetland":                      # a green tussock with reeds and cattails
            self.dt.ellipse((x - 1.5 * r, y - 0.15 * r, x + 1.5 * r, y + 0.95 * r), fill=(98, 140, 82, a),
                            outline=white, width=self.px(2))
            for dx, h, head in ((-0.6, 1.25, False), (0.0, 1.75, True), (0.6, 1.45, True)):
                top = (x + dx * r + 0.15 * r * dx, y - h * r)
                for col, wd in ((white, 3.5), ((52, 66, 36, a), 2)):
                    self.dt.line([(x + dx * r, y + 0.4 * r), top], fill=col, width=self.px(wd))
                if head:
                    hx, hy = top[0], top[1] + 0.35 * r
                    self.dt.ellipse((hx - 0.2 * r, hy - 0.4 * r, hx + 0.2 * r, hy + 0.4 * r), fill=(122, 82, 46, a),
                                    outline=white)
        elif kind == "viewpoint":
            self.dt.polygon([(x, y - 1.2 * r), (x - 1.1 * r, y + 0.8 * r), (x + 1.1 * r, y + 0.8 * r)],
                            fill=white, outline=dark)
        elif kind == "sight":
            pts = [(x + (1.25 if k % 2 == 0 else 0.5) * r * math.sin(k * math.pi / 5),
                    y - (1.25 if k % 2 == 0 else 0.5) * r * math.cos(k * math.pi / 5)) for k in range(10)]
            self.dt.polygon(pts, fill=(255, 214, 90, a), outline=dark)
        else:                                        # town
            self.dot(x, y, 9, (255, 255, 255), alpha)
            self.dot(x, y, 5, (30, 30, 30), alpha)

    def line(self, a, b, width: float, alpha: float):
        self.dt.line([a, b], fill=self._rgba(alpha), width=self.px(width))

    def compose(self) -> Image.Image:
        out = self.back
        if self.style.shadow:
            a = self.text.getchannel("A")
            glow = Image.new("RGBA", self.size, (0, 0, 0, 0))
            glow.putalpha(a.filter(ImageFilter.GaussianBlur(self.px(11))).point(lambda v: min(255, int(v * 1.5))))
            out = Image.alpha_composite(out, glow)
        return Image.alpha_composite(out, self.text)


# --------------------------------------------------------------------------- the overlay plan

@dataclass
class OverlayData:
    """Everything the per-frame drawing needs, loaded once."""
    size: tuple[int, int]
    fps: int
    lens_mm: float
    style: TextStyle
    title: str
    shots: list[dict]
    days: list[dict]                 # route days (km_start, distance_km, …)
    totals: dict
    checkpoints: list[dict]          # {label, s, xyz, day}
    zoom: list[dict]                 # {extent_m, label}
    captions: dict[int, str]         # gap caption by day it follows
    show_totals: bool
    route_rgb: dict[int, tuple]      # day → sRGB 0..255
    places: list[dict]               # storyboard labels with scene xyz: {name, kind, ele, xyz}
    profile: dict                    # km / ele / day along the route (downsampled) for the profile strip
    frames: dict                     # timeline.npz arrays
    lang: str = "en"                 # key of TEXT (storyboard style.language)
    day_places: dict = field(default_factory=dict)       # day → (start, finish) names
    highlight_notes: dict = field(default_factory=dict)  # highlight poi → caption
    openings: dict = field(default_factory=dict)         # day → (caption, note) of its opening shot
    chain: bool = False                                  # chained day stories (timeline.json "chain")
    day_notes: dict = field(default_factory=dict)        # day → note shown under the finish in the outro
    callouts: dict = field(default_factory=dict)         # highlight label → (side, smoothed screen xy per frame)
    place_layout: list | None = None  # per frame, per place: see plan_places
    place_alpha: np.ndarray | None = None
    climbs: list = field(default_factory=list)            # (km0, km1, ele0, ele1) of the big climbs
    route_points: list = field(default_factory=list)      # the finale's route line (storyboard.outro.route_points)
    credits: list = field(default_factory=list)           # attribution lines (style.credits), see credit_lines

    def tr(self, key: str):
        return TEXT.get(self.lang, TEXT["en"])[key]

    def n_days(self, n: int) -> str:
        """«1 day / 3 days»; Russian picks one of three plural forms."""
        one, few, many = self.tr("days")
        if self.lang == "ru":
            k = one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many
        else:
            k = one if n == 1 else many
        return f"{n} {k}"

    @property
    def n(self) -> int:
        return len(self.frames["t"])


def load_overlay_data(p: TripPaths) -> OverlayData:
    from .ingest import load_route

    build = p.build
    route = load_route(p.route)
    sb = p.storyboard(route)
    tl = json.loads((build / "timeline.json").read_text(encoding="utf-8"))
    frames = dict(np.load(build / "timeline.npz"))
    meta = json.loads(str(np.load(build / "world.npz")["meta"]))
    plans = {d.day: d for d in sb.days}
    days = {d["day"]: d for d in route["days"]}
    cps = []
    for c in meta["checkpoints"]:
        d = c["day"]
        lang = sb.style.language if sb.style.language in TEXT else "en"
        if c["label"] in ("Finish", TEXT["ru"]["finish"]):      # world.npz from older builds says it in Russian
            label = TEXT[lang]["finish"]
        else:
            label = plans[d].caption if d in plans and plans[d].caption else f"{TEXT[lang]['day']} {d}"
        cps.append({**c, "label": label})
    rgb = {i: tuple(int(h[k:k + 2], 16) for k in (1, 3, 5)) for i, h in enumerate(meta["day_colors"], start=1)}
    places = []
    if sb.labels:
        from .world import load_world, projection_of

        grid = load_world(build / "world.npz")[0]
        x, y = projection_of(route).to_xy(np.array([lb.lat for lb in sb.labels]), np.array([lb.lon for lb in sb.labels]))
        z = grid.height_at(np.asarray(x), np.asarray(y))
        places = [{"name": lb.name, "kind": lb.kind, "ele": lb.ele, "xyz": np.array([a, b, c])}
                  for lb, a, b, c in zip(sb.labels, x, y, z)]
    data = OverlayData(
        size=tuple(tl["resolution"]), fps=tl["fps"], lens_mm=tl["lens_mm"],
        style=STYLES.get(sb.style.text_style, STYLES["bold_minimal"]), title=sb.title, shots=tl["shots"],
        days=route["days"], totals=route["totals"], checkpoints=cps, zoom=meta.get("zoom", []),
        captions={g.after_day: g.caption for g in sb.gaps if g.caption}, show_totals=sb.outro.show_totals,
        route_rgb=rgb, places=places, profile=_profile(route), frames=frames,
        lang=sb.style.language if sb.style.language in TEXT else "en",
        day_places=_day_places(sb, route), highlight_notes={h.poi: h.caption for h in sb.highlights if h.caption},
        openings={d.day: (d.opening_caption, d.opening_note) for d in sb.days if d.opening},
        chain=bool(tl.get("chain")),
        day_notes={d.day: d.note for d in sb.days if d.note},
        credits=credit_lines(sb.style.terrain) if sb.style.credits else [],
    )
    plan_places(data)
    plan_callouts(data)
    data.climbs = climbs(data.profile)
    data.route_points = list(sb.outro.route_points) if sb.outro.zoom_out else []
    return data


def credit_lines(terrain: str) -> list[str]:
    """Who the map data comes from: imagery, elevation, OSM (coastline, roads, places), WorldCover when used."""
    second = "© OpenStreetMap contributors" + (" · Land cover © ESA WorldCover" if terrain != "satellite" else "")
    return ["Imagery © Esri · Terrain © Mapzen, AWS", second]


CREDITS_PX = 20               # design px: small, under the outro totals


def _draw_credits(cv: Canvas, data: OverlayData, alpha: float) -> None:
    W, H = data.size
    y = SAFE_BOTTOM / BASE_H * H + 80 * cv.u      # clear of the finale numbers, which end at SAFE_BOTTOM
    for line in data.credits:
        cv.spans(W / 2, y, [(line, data.style.label_font, CREDITS_PX)], 0.75 * alpha, anchor="ms")
        y += 28 * cv.u


def _day_places(sb, route: dict) -> dict[int, tuple[str | None, str | None]]:
    """day → (start, finish): a day starts where the previous one finished unless the storyboard says otherwise."""
    plans = {d.day: d for d in sb.days}
    out, prev = {}, None
    for d in route["days"]:
        p = plans.get(d["day"])
        start = (p.start if p and p.start else None) or prev
        finish = p.finish if p and p.finish else None
        out[d["day"]] = (start, finish)
        prev = finish
    return out


def _profile(route: dict, n: int = 900) -> dict:
    pts = np.array([[q[4], q[2], d["day"]] for d in route["days"] for s in d["segments"] for q in s["points"]
                    if q[2] is not None], float)
    idx = np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)
    ele = np.convolve(np.pad(pts[:, 1], 15, mode="edge"), np.ones(31) / 31, mode="valid")[idx]
    return {"km": pts[idx, 0], "ele": ele, "day": pts[idx, 2].astype(int)}


CLIMB_MIN_M = 300.0           # a climb worth a counter: this much up…
CLIMB_DIP_M = 40.0            # …without dropping more than this on the way
CLIMB_FOOT_M = 25.0           # the climb starts where it leaves its low for good (not at the far end of a flat)


def climbs(profile: dict) -> list[tuple[float, float, float, float]]:
    """(km0, km1, ele0, ele1) of the big climbs in the (smoothed) elevation profile, per day."""
    out = []
    def climb(k, e, i0, top):
        i0 = i0 + int(np.flatnonzero(e[i0:top + 1] <= e[i0] + CLIMB_FOOT_M)[-1])
        return float(k[i0]), float(k[top]), float(e[i0]), float(e[top])

    for d in np.unique(profile["day"]):
        m = np.flatnonzero(profile["day"] == d)
        k, e = profile["km"][m], profile["ele"][m]
        i0 = 0
        top = 0
        for i in range(1, len(e)):
            if e[i] <= e[i0]:                                  # a new low (or flat): the climb starts later
                if e[top] - e[i0] >= CLIMB_MIN_M and top > i0:
                    out.append(climb(k, e, i0, top))
                i0 = top = i
            elif e[i] >= e[top] or top < i0:
                top = i
            elif e[top] - e[i] > CLIMB_DIP_M:                 # it went down for real: close the climb
                if e[top] - e[i0] >= CLIMB_MIN_M:
                    out.append(climb(k, e, i0, top))
                i0 = top = i
        if top > i0 and e[top] - e[i0] >= CLIMB_MIN_M:
            out.append(climb(k, e, i0, top))
    return out


def _draw_climb(cv: Canvas, data: OverlayData, km: float, alpha: float) -> None:
    """On a big climb: «CLIMB» and the metres gained so far, counting up as the marker goes up (no grade: it says little)."""
    W, _ = data.size
    u = cv.u
    for k0, k1, e0, e1 in data.climbs:
        if not (k0 - 0.05 <= km <= k1 + 0.6):
            continue
        a = alpha * min(1.0, (km - k0 + 0.05) / 0.4) * min(1.0, (k1 + 0.6 - km) / 0.5)
        if a <= 0.01:
            return
        gained = float(np.interp(min(km, k1), data.profile["km"], data.profile["ele"])) - e0
        y = PROFILE_BOTTOM * u + 200 * u                       # under the profile: the marker rides lower down
        b1 = cv.tracked(W / 2, y - 120 * u, data.tr("climb"), data.style.label_font, 34, a, anchor="ms")
        b2 = cv.spans(W / 2, y, [(f"+{fmt_int(max(gained, 0))}", data.style.number_font, 112),
                                 (f" {data.tr('m')}", data.style.label_font, 40)], a, anchor="ms")
        cv.plate((min(b1[0], b2[0]), b1[1], max(b1[2], b2[2]), b2[3]), a)
        return


def _draw_profile(cv: Canvas, data: OverlayData, km: float, alpha: float) -> None:
    """Elevation strip over the whole route: ridden part bright, the rest faint, a dot with the altitude."""
    W, H = data.size
    u = cv.u
    pr = data.profile
    x0, x1 = 70 * u, W - 70 * u
    y0, y1 = PROFILE_TOP * u, PROFILE_BOTTOM * u
    k, e = pr["km"], pr["ele"]
    lo, hi = float(e.min()), float(e.max())
    xs = x0 + (x1 - x0) * (k - k[0]) / max(k[-1] - k[0], 1e-6)
    ys = y1 - (y1 - y0) * (e - lo) / max(hi - lo, 1.0)
    for d in np.unique(pr["day"]):
        m = np.flatnonzero(pr["day"] == d)
        rgb = data.route_rgb.get(int(d), (255, 255, 255))
        # ridden part: soft fill in the route colour + bright line; ahead: a faint white line only
        for part, fill_a, line_a, line_rgb in ((m[k[m] <= km], 0.32, 1.0, rgb), (m[k[m] > km], 0.0, 0.35, (255, 255, 255))):
            if len(part) < 2:
                continue
            if fill_a > 0:
                poly = [(xs[j], ys[j]) for j in part] + [(xs[part[-1]], y1), (xs[part[0]], y1)]
                cv.db.polygon(poly, fill=(*rgb, int(255 * fill_a * alpha)))
            cv.dt.line([(xs[j], ys[j]) for j in part], fill=(*line_rgb, int(255 * line_a * alpha)), width=cv.px(3))
    cv.db.line([(x0, y1), (x1, y1)], fill=(255, 255, 255, int(60 * alpha)), width=cv.px(2))
    cx = float(np.interp(km, k, xs))
    cy = float(np.interp(km, k, ys))
    ele_now = float(np.interp(km, k, e))
    cv.dot(cx, cy, 9, (255, 255, 255), alpha)
    side = "rs" if cx > 0.8 * W else ("ls" if cx < 0.2 * W else "ms")
    cv.spans(cx, cy - 20 * u, [(f"{fmt_int(ele_now)} {data.tr('m')}", data.style.label_font, 34)], alpha, anchor=side)
    cv.taken.append((int(x0), int(y0 - 60 * u), int(x1), int(y1)))


PULSE_S = 1.6                  # marker pulse ring period


def _draw_pulse(cv: Canvas, data: OverlayData, i: int, alpha: float) -> None:
    """A ring expanding from the marker every PULSE_S (the "you are here" pulse of map UIs)."""
    f = data.frames
    xy, front = project(f["marker"][i][None], f["cam"][i], f["target"][i], data.lens_mm, data.size)
    if not front[0]:
        return
    x, y = xy[0]
    W, H = data.size
    if not (0 <= x <= W and 0 <= y <= H):
        return
    r0 = float(f["marker_radius"][i] / f["cam_dist"][i] / (SENSOR_H_MM / 2 / data.lens_mm) * H / 2)   # marker px
    rgb = data.route_rgb.get(int(f["day"][i]), (255, 255, 255))
    for k in (0.0, 0.5):                                  # two rings, half a period apart
        p = ((float(f["t"][i]) / PULSE_S) + k) % 1.0
        r = r0 * (1.3 + 2.4 * p)
        a = alpha * 0.55 * (1 - p) ** 1.5
        if a > 0.01:
            cv.dt.ellipse((x - r, y - r, x + r, y + r), outline=(*rgb, int(255 * a)), width=cv.px(3))


DAY_CARD_T = (0.5, 4.5)        # seconds after the day's first ride shot starts: fade in … fade out
DAY_CARD_FADE = 0.3
DAY_CARD_MIN_S = 2.0           # a card shorter than this is not worth showing
DAY_CARD_Y = 560               # design px, baseline of the first line


def day_card_windows(shots: list[dict]) -> list[tuple[int, float, float]]:
    """(day, t0, t1) of every day card: shown after the swoop settles at the day's first ride shot. A highlight
    in that window owns the screen with its callout: the card leaves as it starts, or — with less than
    DAY_CARD_MIN_S before it — waits until it is over, if the day still rides that long. Cards never overlap."""
    out = []
    for sh in shots:
        if sh["kind"] == "ride" and sh["day"] and not any(
                o["kind"] == "ride" and o["day"] == sh["day"] and o["t0"] < sh["t0"] for o in shots):
            t0 = sh["t0"] + DAY_CARD_T[0]
            t1 = t0 + DAY_CARD_T[1] - DAY_CARD_T[0]
            for o in shots:
                if o["kind"] == "highlight" and o["day"] == sh["day"] and t0 < o["t1"] and o["t0"] < t1:
                    if o["t0"] - t0 >= DAY_CARD_MIN_S:
                        t1 = o["t0"]
                    else:
                        t0 = o["t1"] + 0.3
                        day_end = max(r["t1"] for r in shots if r["kind"] == "ride" and r["day"] == sh["day"])
                        t1 = min(t0 + DAY_CARD_T[1] - DAY_CARD_T[0], day_end)
            if t1 - t0 >= DAY_CARD_MIN_S:
                out.append((int(sh["day"]), t0, t1))
    starts = [t0 for _, t0, _ in out[1:]] + [math.inf]
    return [(day, t0, min(t1, nxt)) for (day, t0, t1), nxt in zip(out, starts)]


def day_card_alpha(shots: list[dict], t: float) -> float:
    """How visible a day card is at t — what shares its spot (the climb counter) fades out by as much."""
    return max((window(t, t0, t1, fade=DAY_CARD_FADE) for _, t0, t1 in day_card_windows(shots)), default=0.0)


def day_card_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Screen box the day card occupies (generous: place labels keep out of it while it is up)."""
    W, H = size
    u = H / BASE_H
    return (int(W / 2 - 340 * u), int((DAY_CARD_Y - 60) * u), int(W / 2 + 340 * u), int((DAY_CARD_Y + 300) * u))


def _stack(cv: Canvas, lines: list[tuple], alpha: float, top: float | None = None, bottom: float | None = None,
           gap_px: float = 22) -> list[tuple]:
    """Centered lines laid out by their real glyph boxes with one gap: ("tracked", text, font, size) or
    ("spans", parts). Anchored at the top edge or at the bottom edge. Returns the drawn boxes."""
    W = cv.size[0]
    u = cv.u
    gap = gap_px * u
    ext = []
    for line in lines:
        if line[0] == "tracked":
            fnt = font_for(line[1], line[2], cv.px(line[3]))
            ext.append(cv.dt.textbbox((0, 0), cv.label_text(line[1]), font=fnt, anchor="ls")[1::2])
        else:
            b = cv.spans(W / 2, 0, line[1], 1.0, anchor="ms", draw=False)
            ext.append((b[1], b[3]))
    if top is None:
        top = bottom - sum(b - a for a, b in ext) - gap * (len(lines) - 1)
    boxes = []
    for line, (a, b) in zip(lines, ext):
        baseline = top - a
        if line[0] == "tracked":
            boxes.append(cv.tracked(W / 2, baseline, line[1], line[2], line[3], alpha, anchor="ms"))
        else:
            boxes.append(cv.spans(W / 2, baseline, line[1], alpha, anchor="ms"))
        top = baseline + b + gap
    return boxes


def _draw_day_card(cv: Canvas, data: OverlayData, day: int, alpha: float) -> None:
    """«DAY 2 · SEP 22», «Tadewara Wetlands → Aso foothills», big day distance, climb and moving time."""
    W, H = data.size
    u = cv.u
    d = next(x for x in data.days if x["day"] == day)
    date = ""
    if d.get("date"):
        y, mo, dd = map(int, d["date"].split("-"))
        date = " · " + data.tr("date").format(d=dd, month=data.tr("months")[mo - 1])
    h, mnt = divmod(int(round(d.get("moving_time_s", 0) / 60)), 60)
    start, finish = data.day_places.get(day, (None, None))
    lines = [("tracked", f"{data.tr('day')} {day}{date}", data.style.label_font, 34)]
    if start or finish:
        lines.append(("spans", [(f"{start or '…'}  →  {finish or '…'}", data.style.label_font, 40)]))
    lines.append(("spans", [(fmt_km(d["distance_km"]), data.style.number_font, 120),
                            (f" {data.tr('km')}", data.style.label_font, 44)]))
    lines.append(("spans", [(f"+{fmt_int(d.get('ascent_m', 0))} {data.tr('m')} · "
                             + data.tr("time").format(h=h, m=mnt), data.style.label_font, 36)]))
    # stacked by the real glyph boxes with one gap: the big number sits evenly between its neighbours
    boxes = _stack(cv, lines, alpha, top=DAY_CARD_Y * u - 30 * u, gap_px=36)
    cv.plate((min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3]), alpha)


def _draw_totals(cv: Canvas, data: OverlayData, km: float, days: int, climb: float, finish: str | None,
                 alpha: float, note: str | None = None, start: str | None = None) -> None:
    """Big distance, «N days · climb», and «start → finish» (+ a note under it) above them."""
    W, H = data.size
    u = cv.u
    lines = []
    if finish:
        route = f"{start}  →  {finish}" if start else f"→ {finish}"
        lines.append(("spans", [(route, data.style.label_font, 44 if len(route) <= 34 else 36)]))
    if note:
        lines.append(("tracked", note, data.style.label_font, 30))
    lines.append(("spans", [(fmt_km(km), data.style.number_font, 150), (f" {data.tr('km')}", data.style.label_font, 52)]))
    lines.append(("tracked", data.tr("totals").format(days=data.n_days(days), climb=fmt_int(climb)),
                  data.style.label_font, 34))
    boxes = _stack(cv, lines, alpha, bottom=SAFE_BOTTOM / BASE_H * H - 50 * u, gap_px=44)
    cv.plate((min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3]), alpha)


FINALE_POINTS_PER_LINE = 3    # the route line: three places a line (even lines)
FINALE_ROUTE_GAP_PX = 70      # between the route lines and the distance (more air there)
FINALE_CARD_IN_S = 3.9        # into the finale: the camera has left the route overview (white on white before)


def _route_lines(points: list[str]) -> list[str]:
    """"Beppu → Aso → Takachiho →" three places a line; a line that continues ends with its arrow."""
    n = FINALE_POINTS_PER_LINE
    rows = [points[i:i + n] for i in range(0, len(points), n)]
    return [" → ".join(r) + (" →" if k < len(rows) - 1 else "") for k, r in enumerate(rows)]


def _draw_finale(cv: Canvas, data: OverlayData, km: float, days: int, climb: float, moving_s, alpha: float) -> None:
    """The last story's summary: the route through its key places, the total distance, days · climb · hours."""
    W, H = data.size
    u = cv.u
    stats = data.tr("totals").format(days=data.n_days(days), climb=fmt_int(climb))
    if moving_s:
        stats += " · " + data.tr("hours").format(h=int(round(moving_s / 3600)))
    numbers = _stack(cv, [("spans", [(fmt_km(km), data.style.number_font, 170),
                                     (f" {data.tr('km')}", data.style.label_font, 56)]),
                          ("tracked", stats, data.style.label_font, 32)],
                     alpha, bottom=SAFE_BOTTOM / BASE_H * H, gap_px=40)     # the numbers sit lower…
    route = _stack(cv, [("spans", [(ln, data.style.label_font, 44)]) for ln in _route_lines(data.route_points)],
                   alpha, bottom=numbers[0][1] - FINALE_ROUTE_GAP_PX * u, gap_px=40)   # some air above the km
    boxes = route + numbers
    cv.plate((min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3]), alpha)


# seconds from the start of the highlight shot; the camera flies into / out of the orbit during its first
# ~1.2 s and last ~0.6 s (the point slides across the screen), so the callout lives in between
CALLOUT = dict(ring=(1.15, 1.45), leader=(1.3, 1.7), rule=(1.6, 1.95), title=(1.75, 2.45), note=(2.2, 3.1),
               fade=0.35, end_before=0.6)


def _typed(text: str, t: float, span: tuple[float, float]) -> tuple[str, bool]:
    """Typewriter: the part of `text` shown at t, and whether it is still typing."""
    u = (t - span[0]) / max(span[1] - span[0], 1e-6)
    n = int(np.clip(u, 0, 1) * len(text) + 1e-9)
    return text[:n], 0 < u < 1


def plan_callouts(data: OverlayData) -> None:
    """One side per highlight (where the point spends most of the shot) and a smoothed screen track of the point:
    deciding per frame made the label jump sides as the orbit swept the point across the middle."""
    f = data.frames
    W, H = data.size
    for sh in data.shots:
        if sh["kind"] != "highlight" or sh.get("target") is None:
            continue
        m = np.flatnonzero((f["t"] >= sh["t0"] - 0.2) & (f["t"] <= sh["t1"] + 0.2))
        if not len(m):
            continue
        pts = np.array([project(np.array([sh["target"]]), f["cam"][i], f["target"][i], data.lens_mm, data.size)[0][0]
                        for i in m])
        pts = np.nan_to_num(pts, nan=0.0)
        k = max(1, int(0.25 * data.fps))                    # ~quarter-second moving average
        ker = np.ones(k) / k
        sm = np.column_stack([np.convolve(np.pad(pts[:, j], (k // 2, k - 1 - k // 2), mode="edge"), ker, "valid")
                              for j in range(2)])
        side = -1 if np.median(sm[:, 0]) > 0.5 * W else 1
        data.callouts[sh["label"]] = (side, dict(zip(m.tolist(), map(tuple, sm))))


def _draw_callout(cv: Canvas, data: OverlayData, i: int, shot: dict, t: float) -> None:
    """A leader line from the highlight's point to a label whose text types itself in; follows the point on screen."""
    W, H = data.size
    u = cv.u
    side_fixed, track = data.callouts.get(shot["label"], (None, {}))
    x, y = track.get(i, (W / 2, 0.45 * H))
    x = float(np.clip(x, 0.12 * W, 0.88 * W))
    y = float(np.clip(y, (LABELS_TOP + 260) / BASE_H * H, LABELS_BOTTOM / BASE_H * H))
    lt = t - shot["t0"]
    fade = _smooth((shot["t1"] - CALLOUT["end_before"] - t) / CALLOUT["fade"])
    ease = lambda span: float(_smooth((lt - span[0]) / (span[1] - span[0])))      # noqa: E731
    rise, run = 190 * u, 110 * u
    title = shot["label"]
    note = data.highlight_notes.get(title, "")
    fnt_title, fnt_note = data.style.number_font, data.style.label_font
    tw = cv.dt.textlength(title, font=font_for(title, fnt_title, cv.px(64)))
    nw = cv.dt.textlength(note, font=font_for(note, fnt_note, cv.px(30))) if note else 0
    width = max(tw, nw) + 20 * u
    margin = 40 * u                                            # one side for the whole highlight (plan_callouts)
    side = side_fixed if side_fixed is not None else (-1 if x > 0.5 * W else 1)
    ex, ey = x + side * run, y - rise                          # elbow
    tx = ex + side * 18 * u                                    # text anchor after the elbow
    if side > 0 and tx + width > W - margin:                   # still too wide: slide the text in
        tx = W - margin - width
    if side < 0 and tx - width < margin:
        tx = margin + width
    # ring around the point
    a = ease(CALLOUT["ring"]) * fade
    if a > 0:
        r = (14 + 10 * (1 - ease(CALLOUT["ring"]))) * u
        cv.dt.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 255, int(255 * a)), width=cv.px(3))
        cv.dot(x, y, 5, (255, 255, 255), a)
    # leader: up-diagonal to the elbow
    k = ease(CALLOUT["leader"])
    if k > 0:
        cv.line((x + side * 14 * u * 0.7, y - 14 * u * 0.7), (x + (ex - x) * k, y + (ey - y) * k), 3, fade)
    # rule under the title, grows from the elbow
    k = ease(CALLOUT["rule"])
    if k > 0:                                                  # from the elbow to the far edge of the text
        far = tx + side * width
        cv.line((ex, ey), (ex + (far - ex) * k, ey), 3, fade)
    anchor = "ls" if side > 0 else "rs"
    shown, typing = _typed(title, lt, CALLOUT["title"])
    boxes = []
    if shown:
        boxes.append(cv.spans(tx, ey - 16 * u, [(shown, fnt_title, 64)], fade, anchor=anchor))
    shown_n, typing_n = _typed(note, lt, CALLOUT["note"])
    if shown_n:
        boxes.append(cv.spans(tx, ey + 46 * u, [(shown_n, fnt_note, 30)], fade, anchor=anchor))
    # blinking block cursor while typing
    cur = boxes[-1] if boxes else None
    if cur and (typing or typing_n) and int(lt * 6) % 2 == 0:
        cx = cur[2] + 6 * u if side > 0 else cur[2] + 6 * u
        cv.dt.rectangle((cx, cur[1] + 2 * u, cx + 12 * u, cur[3]), fill=(255, 255, 255, int(230 * fade)))
    if boxes:
        cv.plate((min(b[0] for b in boxes), boxes[0][1], max(b[2] for b in boxes), boxes[-1][3]), fade, pad=16)


def draw_frame(data: OverlayData, i: int, credits: bool = False) -> Image.Image:
    """credits: draw the attribution whatever the frame (a cover); otherwise it shows in the outro."""
    f = data.frames
    t = float(f["t"][i])
    cv = Canvas(data.size, data.style)
    W, H = data.size
    u = cv.u
    shots = data.shots
    kinds = [s["kind"] for s in shots]
    ride_t0 = next((s["t0"] for s in shots if s["kind"] in ("ride", "join", "highlight", "gap")), 0.0)
    ride_t1 = max((s["t1"] for s in shots if s["kind"] in ("ride", "join", "highlight", "gap")), default=0.0)
    intro = shots[0]
    outro = shots[-1] if kinds[-1] == "overview" else None

    # ---- intro zoom labels: strongest when the camera frames that level
    if intro["kind"] == "zoom" and data.zoom:
        tan_v = SENSOR_H_MM / 2 / data.lens_mm
        d = float(f["cam_dist"][i])
        for z in data.zoom:
            if not z.get("label"):
                continue
            level = z["extent_m"] / 2 / tan_v
            a = math.exp(-(math.log(d / level) / 0.45) ** 2) * window(t, intro["t0"], intro["t1"] + 0.3)
            box = cv.spans(W / 2, 0.30 * H, [(z["label"], data.style.number_font, 128)], a, anchor="ms")
            cv.plate(box, a)

    # ---- title: the last intro «level» after the zoom labels (e.g. Japan → Kyushu → by bike), intro only
    title_box = (0, 0, 0, 0)
    if data.title:
        a_in = window(t, intro["t1"] - 1.6, intro["t1"] - 0.05)
        if a_in > 0:
            cv.plate(cv.spans(W / 2, 0.30 * H, [(data.title, data.style.number_font, 110)], a_in, anchor="ms"), a_in)

    # ---- km counters while riding
    a_cnt = window(t, ride_t0, ride_t1 + 0.4, fade=0.5)
    if a_cnt > 0:
        day = int(f["day"][i])
        dmeta = next(d for d in data.days if d["day"] == day)
        km = float(f["km"][i])
        day_km = min(max(km - dmeta["km_start"], 0.0), dmeta["distance_km"])
        y = SAFE_BOTTOM / BASE_H * H - 40 * u
        left, right = 70 * u, W - 70 * u
        b1 = cv.tracked(left + 34 * u, y - 132 * u, f"{data.tr('day')} {day}", data.style.label_font, 34, a_cnt)
        cv.dot(left + 11 * u, y - 144 * u, 11, data.route_rgb.get(day, (255, 255, 255)), a_cnt)
        km_s = f" {data.tr('km')}"
        b2 = cv.spans(left, y, [(fmt_km(day_km), data.style.number_font, 112), (km_s, data.style.label_font, 40)], a_cnt)
        b3 = cv.tracked(right, y - 132 * u, data.tr("total"), data.style.label_font, 34, a_cnt, anchor="rs")
        b4 = cv.spans(right, y, [(fmt_km(km), data.style.number_font, 112), (km_s, data.style.label_font, 40)], a_cnt,
                      anchor="rs")
        cv.plate((left, min(b1[1], b2[1]), max(b1[2], b2[2]), b2[3]), a_cnt)
        cv.plate((min(b3[0], b4[0]), min(b3[1], b4[1]), right, b4[3]), a_cnt)
        if data.profile["km"].size > 1:
            _draw_profile(cv, data, km, a_cnt)
            _draw_climb(cv, data, km, a_cnt * (1.0 - day_card_alpha(shots, t)))   # the card owns that spot
    if "marker" in f and "marker_radius" in f:
        _draw_pulse(cv, data, i, window(t, ride_t0, ride_t1 + 0.4, fade=0.5))   # not over the intro labels

    # ---- day card at the start of every day
    for day, t0, t1 in day_card_windows(shots):
        a = window(t, t0, t1, fade=DAY_CARD_FADE)
        if a > 0:
            _draw_day_card(cv, data, day, a)

    # ---- outro totals (a day story also says where the day ended)
    story_day = shots[0].get("day")
    if outro and data.show_totals:
        # chained stories hand over a neutral frame: the totals leave before the trail greys out
        finale = bool(data.route_points)               # the last story: trip totals + the route, held to the end
        t_in = outro["t0"] + (FINALE_CARD_IN_S if finale else 0.4)           # finale: once the camera has backed off
        a = window(t, t_in, outro["t1"] - 0.7 if data.chain and not finale else outro["t1"] + 1)
        if a > 0 and finale:
            tt = data.totals
            _draw_finale(cv, data, tt["distance_km"], tt["days"], tt["ascent_m"], tt.get("moving_time_s"), a)
        elif a > 0:
            tt = data.totals
            last = story_day if story_day is not None else data.days[-1]["day"]   # Reels: where the trip is now
            first = story_day if story_day is not None else data.days[0]["day"]   # a story: its day; Reels: the trip
            finish = data.day_places.get(last, (None, None))[1]
            start = data.day_places.get(first, (None, None))[0]
            _draw_totals(cv, data, tt["distance_km"], tt["days"], tt["ascent_m"], finish, a, data.day_notes.get(last),
                         start)
    # ---- checkpoint labels, next to their pins, for a few seconds after the marker reaches them
    reached_at = {}
    for k, cp in enumerate(data.checkpoints):
        hit = np.flatnonzero(f["s"] >= cp["s"] - 1e-6)
        reached_at[k] = max(f["t"][hit[0]], ride_t0) if len(hit) else np.inf     # not during the intro
    passed = {k for k, cp in enumerate(data.checkpoints) if cp["s"] < f["s"][0] - 1e-6}   # day story: earlier days
    shown = [k for k in reached_at if (k not in passed and window(t, reached_at[k], reached_at[k] + 3.2) > 0)
             or (outro and t >= outro["t0"] and not data.chain)]           # chained: nothing to match across
    if shown:
        pts = np.array([data.checkpoints[k]["xyz"] for k in shown])
        xy, front = project(pts, f["cam"][i], f["target"][i], data.lens_mm, data.size)
        in_outro = outro is not None and t >= outro["t0"]
        top = SAFE_TOP / BASE_H * H if in_outro else LABELS_TOP / BASE_H * H    # outro: only the title box is off limits
        bottom = (SAFE_BOTTOM - 260) / BASE_H * H if in_outro else LABELS_BOTTOM / BASE_H * H   # totals live below
        for k, (x, y), ok in zip(shown, xy, front):
            if not ok or not (0 <= x <= W and top <= y <= bottom):
                continue
            cp = data.checkpoints[k]
            a = window(t, outro["t0"], outro["t1"] + 1) if in_outro else window(t, reached_at[k], reached_at[k] + 3.2)
            text = (cp["label"] if cp["label"] == data.tr("finish") else f"{data.tr('day')} {cp['day']}") \
                if in_outro else cp["label"]
            size = 32 if in_outro else 40
            side = -1 if x > 0.6 * W else 1                  # labels on the right half point left
            lx, ly = x + side * 26 * u, y - 46 * u
            anchor = "ls" if side > 0 else "rs"
            box = cv.spans(lx, ly, [(text, data.style.label_font, size)], 1.0, anchor=anchor, draw=False)
            if in_outro and (_overlaps(box, title_box, 10 * u) or box[1] < SAFE_TOP / BASE_H * H):
                ly = y + 60 * u                              # under the pin instead of over the title / top UI
                box = cv.spans(lx, ly, [(text, data.style.label_font, size)], 1.0, anchor=anchor, draw=False)
                if _overlaps(box, title_box, 10 * u):
                    continue
            if any(_overlaps(box, b, 30 * u) for b in cv.taken):    # totals / cards own their space (+ air)
                continue
            cv.line((x, y - 10 * u if ly < y else y + 10 * u), (lx - side * 6 * u, ly + 10 * u if ly < y else ly - 30 * u), 3, a)
            box = cv.spans(lx, ly, [(text, data.style.label_font, size)], a, anchor=anchor)
            cv.plate(box, a, pad=14)

    # ---- night between days: where the night was spent
    for s in shots:
        if s["kind"] == "night":
            a = window(t, s["t0"] - 0.1, s["t1"] + 0.3, fade=0.25)
            place = data.day_places.get(s["day"], (None, None))[1]
            if a > 0 and place:
                b1 = cv.tracked(W / 2, 0.42 * H, data.tr("night"), data.style.label_font, 34, a, anchor="ms")
                b2 = cv.spans(W / 2, 0.42 * H + 70 * u, [(place, data.style.label_font, 56)], a, anchor="ms")
                cv.plate((min(b1[0], b2[0]), b1[1], max(b1[2], b2[2]), b2[3]), a)

    # ---- opening (sunrise): «日出ずる国» / «Land of the Rising Sun» once the sun is up
    for s in shots:
        if s["kind"] == "opening" and s["day"] in data.openings:
            cap, note = data.openings[s["day"]]
            a = window(t, s["t0"] + 0.45 * (s["t1"] - s["t0"]), s["t1"] + 0.4, fade=0.5)
            if a > 0 and (cap or note):
                y = 0.27 * H
                parts = []
                if note:
                    parts.append(cv.spans(W / 2, y, [(note, data.style.number_font, 104)], a, anchor="ms"))
                    y += 70 * u
                if cap:
                    parts.append(cv.tracked(W / 2, y, cap, data.style.label_font, 34, a, anchor="ms"))
                cv.plate((min(b[0] for b in parts), parts[0][1], max(b[2] for b in parts), parts[-1][3]), a)

    # ---- highlight: an animated callout from the place (ring → leader → underline → typed text)
    for s in shots:
        if s["kind"] == "highlight" and s["t0"] - 0.1 <= t <= s["t1"] + 0.1:
            _draw_callout(cv, data, i, s, t)

    # ---- gap captions while crossing (optional: a light trail already says «transport»)
    for s in shots:
        if s["kind"] in ("join", "gap") and s["day"] in data.captions:
            a = window(t, s["t0"] - 0.3, s["t1"] + 0.2)
            if a > 0:
                box = cv.spans(W / 2, 0.62 * H, [(data.captions[s["day"]], data.style.label_font, 56)], a, anchor="ms")
                cv.plate(box, a)

    # chained day story: it opens on the neutral frame the previous one ended on (no text)

    # ---- places (peaks, towns): labelled whenever on screen, once the intro zoom and the title are done
    if data.places:
        a = _smooth((t - (intro["t1"] + 0.6)) / 0.8) if intro["kind"] == "zoom" or data.chain else 1.0
        if outro:                                   # the whole-route outro shows day pins only (no clutter)
            a *= 1.0 - _smooth((t - outro["t0"] + 0.2) / 0.5)
        if a > 0:
            _draw_places(cv, data, i, float(a))

    # ---- attribution: the outro (or the last seconds), gone before a chained story's neutral last frame
    if data.credits:
        end = shots[-1]["t1"]
        t0 = outro["t0"] + 0.4 if outro else end - 2.0
        a = 1.0 if credits else window(t, t0, end - 0.7 if data.chain else end + 1)
        if a > 0:
            _draw_credits(cv, data, a)

    return cv.compose()


EDGE_FADE = 0.06              # place labels fade out over this share of the frame near its edges
PLACE_MIN_S = 0.6             # label visibility runs / gaps shorter than this are smoothed away
PLACE_FADE_S = 0.25
PLACE_MAX_SPEED = 40.0        # design px per frame: a place sweeping across the screen faster stays unlabelled


def _overlaps(a, b, pad: float) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def _runs(v: np.ndarray) -> list[tuple[int, int, bool]]:
    edges = np.flatnonzero(np.diff(v.astype(int))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(v)]
    return [(int(a), int(b), bool(v[a])) for a, b in zip(starts, ends)]


def _debounce(v: np.ndarray, min_len: int) -> np.ndarray:
    """Fill short gaps between visible runs, then drop short visible runs."""
    v = v.copy()
    for a, b, on in _runs(v):
        if not on and 0 < a and b < len(v) and b - a < min_len:
            v[a:b] = True
    for a, b, on in _runs(v):
        if on and b - a < min_len:
            v[a:b] = False
    return v


def plan_places(data: OverlayData) -> None:
    """Lay out place labels for every frame in order (sequential: hysteresis + temporal smoothing),
    so the parallel per-frame drawing only paints. Sets data.place_layout / data.place_alpha."""
    f = data.frames
    W, H = data.size
    cv = Canvas(data.size, data.style)                 # used for text measurement only
    u = cv.u
    top, bottom = LABELS_TOP / BASE_H * H, LABELS_BOTTOM / BASE_H * H        # clear of profile and counters
    edge = EDGE_FADE * W
    pts = np.array([p["xyz"] for p in data.places])
    P, n = len(data.places), data.n
    side = [1] * P                                     # town labels: right of the dot (+1) or left (-1)
    down = [False] * P                                 # peak labels below the summit when there is no room
    layout: list[list[dict | None]] = [[None] * P for _ in range(n)]
    want = np.zeros((n, P), bool)
    edge_a = np.zeros((n, P))
    cards = day_card_windows(data.shots)
    card_box = day_card_box(data.size)
    for i in range(n):
        xy, front = project(pts, f["cam"][i], f["target"][i], data.lens_mm, data.size)
        t = float(f["t"][i])
        taken = [card_box] if any(t0 - PLACE_FADE_S <= t <= t1 + PLACE_FADE_S for _, t0, t1 in cards) else []
        for k, (p, (x, y), ok) in enumerate(zip(data.places, xy, front)):   # storyboard order = priority
            if not ok:
                continue
            a = float(_smooth(min(x, W - x) / edge) * _smooth(min(y - top, bottom - y) / edge))
            if a <= 0.01:
                continue
            if p["kind"] == "peak":
                parts = [(p["name"], data.style.label_font, 40)]
                if p["ele"]:
                    parts.append((f"  {fmt_int(p['ele'])} {data.tr('m')}", data.style.small_font, 28))
                room = y - top - 60 * u
                down[k] = room < (70 if down[k] else 40) * u
                stick = -90 * u if down[k] else float(np.clip(room, 50 * u, 120 * u))
                y_lab = y - stick + (40 * u if down[k] else 0)
                lx, anchor = x, "ms"
            else:
                parts = [(p["name"], data.style.label_font, 34)]
                if side[k] > 0 and x > 0.72 * W:
                    side[k] = -1
                elif side[k] < 0 and x < 0.52 * W:
                    side[k] = 1
                stick = 0.0
                y_lab = y - 18 * u if y - 60 * u > top else y + 50 * u
                lx, anchor = x + side[k] * (22 if p["kind"] == "town" else 30) * u, ("ls" if side[k] > 0 else "rs")
            box = cv.spans(lx, y_lab - 12 * u, parts, 1.0, anchor=anchor, draw=False)
            layout[i][k] = {"x": x, "y": y, "stick": stick, "lx": lx, "ly": y_lab - 12 * u, "anchor": anchor,
                            "parts": parts, "kind": p["kind"]}
            edge_a[i, k] = a
            if not any(_overlaps(box, b, 12 * u) for b in taken):
                taken.append(box)
                want[i, k] = True
    # screen speed of every place; a label needs its point to hold still-ish and to be clear of the edges
    pos = np.array([[(l["x"], l["y"]) if l else (np.nan, np.nan) for l in row] for row in layout])   # (n, P, 2)
    speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=2) / u
    steady = np.nan_to_num(speed, nan=np.inf) < PLACE_MAX_SPEED
    want &= steady & (edge_a > 0.3)
    min_len = max(1, int(PLACE_MIN_S * data.fps))
    fade = max(1, int(PLACE_FADE_S * data.fps))
    kernel = np.ones(fade) / fade
    alpha = np.zeros((n, P))
    for k in range(P):
        on = _debounce(want[:, k], min_len).astype(float)
        on = np.convolve(np.pad(on, (fade - 1, 0), mode="edge"), kernel, mode="valid")   # fade in / out
        alpha[:, k] = on * edge_a[:, k]
    data.place_layout, data.place_alpha = layout, alpha


def _draw_places(cv: Canvas, data: OverlayData, i: int, alpha: float) -> None:
    for k, lay in enumerate(data.place_layout[i]):
        a = alpha * float(data.place_alpha[i, k])
        if lay is None or a <= 0.01:
            continue
        x, y = lay["x"], lay["y"]
        if lay["kind"] == "peak":
            cv.line((x, y), (x, y - lay["stick"]), 3, a)
            cv.dot(x, y, 7, (255, 255, 255), a)
        else:
            cv.icon(lay["kind"], x, y, a)
        cv.plate(cv.spans(lay["lx"], lay["ly"], lay["parts"], a, anchor=lay["anchor"]), a, pad=12)


# --------------------------------------------------------------------------- batch

_DATA: OverlayData | None = None


def _init(trip: str, story: str | None):
    global _DATA
    _DATA = load_overlay_data(TripPaths(Path(trip), story))


def _job(args):
    i, out = args
    draw_frame(_DATA, i).save(out, compress_level=1)
    return i


def render_overlays(p: TripPaths, out_dir: Path | None = None, frames: list[int] | None = None,
                    workers: int | None = None, log=print) -> Path:
    """Write build/overlay/o_0001.png… for the given 0-based timeline frames (default: all), numbered in order."""
    import multiprocessing as mp

    out_dir = out_dir or p.build / "overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(np.load(p.build / "timeline.npz")["t"])
    idx = frames if frames is not None else list(range(n))
    jobs = [(i, str(out_dir / f"o_{k:04d}.png")) for k, i in enumerate(idx, start=1)]
    # macOS: fork after Blender rendered with Metal in this process deadlocks the children now and then
    method = "spawn" if sys.platform == "darwin" else "fork"
    with mp.get_context(method).Pool(workers or min(mp.cpu_count(), 48), initializer=_init, initargs=(str(p.trip), p.story)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=8), start=1):
            if k % 300 == 0 or k == len(jobs):
                log(f"  overlays: {k}/{len(jobs)}")
    return out_dir
