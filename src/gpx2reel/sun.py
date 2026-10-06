"""Solar position (NOAA approximation, ~0.1° for 1950–2050) and the sky / sun look for a given elevation."""
from __future__ import annotations

import numpy as np


def solar_position(lat: np.ndarray, lon: np.ndarray, epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth (degrees clockwise from north) and elevation (degrees) of the sun."""
    lat, lon, epoch = (np.asarray(v, float) for v in (lat, lon, epoch))
    jd = epoch / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = np.radians(m)
    c = (np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(2 * mr) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = np.radians(eps0 + 0.00256 * np.cos(np.radians(omega)))
    decl = np.arcsin(np.sin(eps) * np.sin(np.radians(app_long)))
    y = np.tan(eps / 2) ** 2
    l0r = np.radians(l0)
    eq_time = 4 * np.degrees(y * np.sin(2 * l0r) - 2 * e * np.sin(mr) + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
                             - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * mr))
    minutes_utc = np.mod(epoch, 86400.0) / 60.0
    true_solar = np.mod(minutes_utc + eq_time + 4 * lon, 1440)
    hour_angle = np.radians(true_solar / 4 - 180)
    latr = np.radians(lat)
    cos_zen = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(hour_angle)
    zen = np.arccos(np.clip(cos_zen, -1, 1))
    elev = 90 - np.degrees(zen)
    az = np.degrees(np.arctan2(np.sin(hour_angle), np.cos(hour_angle) * np.sin(latr) - np.tan(decl) * np.cos(latr))) + 180
    return np.mod(az, 360), elev


def sunrise_before(lat: float, lon: float, epoch: float, search_h: float = 12.0) -> float:
    """Last sunrise (elevation crossing 0 upwards) before `epoch`, to the minute."""
    t = np.arange(epoch - search_h * 3600, epoch + 60, 60.0)
    _, el = solar_position(np.full(len(t), lat), np.full(len(t), lon), t)
    up = np.flatnonzero((el[:-1] < 0) & (el[1:] >= 0))
    if not len(up):
        return float(epoch)
    i = up[-1]
    return float(t[i] + 60 * (-el[i]) / (el[i + 1] - el[i]))


def night_after(lat: float, lon: float, epoch: float, el_night: float = -9.0, search_h: float = 18.0) -> float:
    """First moment after `epoch` with the sun below el_night — sky_look no longer changes there, so two chained
    stories meeting at night look the same; the darkest moment when the sun never gets that low (white nights)."""
    t = np.arange(epoch, epoch + search_h * 3600, 60.0)
    _, el = solar_position(np.full(len(t), lat), np.full(len(t), lon), t)
    dark = np.flatnonzero(el <= el_night)
    return float(t[dark[0]] if len(dark) else t[int(np.argmin(el))])


def night_before(lat: float, lon: float, epoch: float, el_night: float = -9.0, search_h: float = 18.0) -> float:
    """Last moment before `epoch` with the sun below el_night (the darkest moment when it never gets that low)."""
    t = np.arange(epoch - search_h * 3600, epoch + 1, 60.0)
    _, el = solar_position(np.full(len(t), lat), np.full(len(t), lon), t)
    dark = np.flatnonzero(el <= el_night)
    return float(t[dark[-1]] if len(dark) else t[int(np.argmin(el))])


def sun_descends_to(lat: float, lon: float, epoch: float, el_deg: float, search_h: float = 14.0) -> float | None:
    """First moment after `epoch` when the sinking sun passes el_deg (None: it is already lower, or never)."""
    t = np.arange(epoch, epoch + search_h * 3600, 60.0)
    _, el = solar_position(np.full(len(t), lat), np.full(len(t), lon), t)
    hit = np.flatnonzero((el[:-1] >= el_deg) & (el[1:] < el_deg))
    if not len(hit):
        return None
    i = hit[0]
    return float(t[i] + 60 * (el[i] - el_deg) / (el[i] - el[i + 1]))


def _lin(hex_: str) -> np.ndarray:
    c = np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# sunset look, from a reference photo of the sun going down over the sea:
# the whole sky is orange (no blue), the disc is white-hot inside an orange halo, far ridges fade to red-brown
SUNSET_ZENITH = _lin("#a67040")
SUNSET_HORIZON = _lin("#f0621a")
SUNSET_LIGHT = np.array([1.0, 0.42, 0.14])
SEA_MIRROR = 0.35             # share of the sky water mirrors outside a sunset (a grazing view of a pale sky)
SUNSET_AMBIENT = 0.3         # share of the sky's light that reaches the scene: the shore is a silhouette
SUNSET_DISK = np.array([0.3, 0.22, 0.13, 1.0])       # × the disc's emission: the core clips to white, bloom stays orange
SUNSET_SKY_STRENGTH = 1.0
SUNRISE_DISKS = {                                    # style.sun_disc: the disc's colour in a sunrise opening
    "natural": np.array([*_lin("#FFB04A"), 1.0]),    # a low golden sun
    "hinomaru": np.array([*_lin("#BC002D"), 1.0]),   # the red disc of the Japanese flag
}


AFTERGLOW_ZENITH = _lin("#2b2338")    # once the sun is down: a dusky violet overhead…
AFTERGLOW_HORIZON = _lin("#a8381c")   # …and a band of red left along the horizon
AFTERGLOW_STRENGTH = 0.55
AFTERGLOW_EL = (-6.0, 0.0)            # the sunset palette fades into the afterglow over these sun elevations


def sunset_look(look: dict, w: np.ndarray, el_deg: np.ndarray | None = None) -> dict:
    """Blend a sky_look dict towards the sunset palette by weight w (per frame, 0..1). With el_deg the palette
    itself follows the sun below the horizon: orange while it is up, the afterglow by AFTERGLOW_EL[0]."""
    w = np.asarray(w, float)
    up = np.ones(len(w)) if el_deg is None else _smooth(el_deg, *AFTERGLOW_EL)
    out = dict(look)
    for key, day, dusk in (("sky_color", SUNSET_ZENITH, AFTERGLOW_ZENITH),
                           ("horizon_color", SUNSET_HORIZON, AFTERGLOW_HORIZON),
                           ("sun_color", SUNSET_LIGHT, SUNSET_LIGHT)):
        tgt = dusk[None] + (day - dusk)[None] * up[:, None]
        out[key] = look[key] + (tgt - look[key]) * w[:, None]
    strength = AFTERGLOW_STRENGTH + (SUNSET_SKY_STRENGTH - AFTERGLOW_STRENGTH) * up
    out["sky_strength"] = look["sky_strength"] + (strength - look["sky_strength"]) * w
    return out


def one_night(lat: float, lon: float, a: float, b: float, u: np.ndarray) -> np.ndarray:
    """Clock from `a` (evening) to `b` (morning) over u ∈ [0, 1] with a single night: rest days in between are
    skipped by jumping from the first night to the last one — all nights look the same (sky_look is constant
    below −8°), so the jump is invisible and a week off does not flicker day / night seven times."""
    u = np.asarray(u, float)
    if b - a < 12 * 3600:
        return a + (b - a) * u
    na, nb = night_after(lat, lon, a), night_before(lat, lon, b)
    if nb <= na:
        return a + (b - a) * u
    d1, d2 = na - a, b - nb
    k = 0.5                                           # half the way into the night (or holding it), half to dawn
    return np.where(u <= k, a + d1 * u / max(k, 1e-9), nb + d2 * (u - k) / max(1 - k, 1e-9))


def sun_vector(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Unit vector towards the sun in scene axes (x east, y north, z up)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], -1)


def _smooth(x, a, b):
    u = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


DAY_SKY = np.array([0.55, 0.68, 0.90])            # zenith
DAY_HORIZON = np.array([0.86, 0.90, 0.96])        # pale haze at the horizon
DUSK_SKY = np.array([0.95, 0.55, 0.35])
NIGHT_SKY = np.array([0.05, 0.07, 0.16])          # moonlit: the terrain stays readable, the trail glows
NIGHT_HORIZON = np.array([0.09, 0.11, 0.21])
SUN_DAY = np.array([1.0, 0.97, 0.92])
SUN_LOW = np.array([1.0, 0.55, 0.30])
MOON_COLOR = np.array([0.62, 0.72, 1.0])
MOON_ENERGY = 0.7                                  # a bright moon relative to the 3.5 sun: 1.2 s nights must read
MOON_AZ_EL = (130.0, 42.0)                         # fixed: the moon is a light, not a clock


def sky_look(el_deg: np.ndarray) -> dict:
    """Sun / moon energy and colour, sky zenith / horizon colours and strength for a sun elevation (per frame)."""
    el = np.asarray(el_deg, float)
    day = _smooth(el, -1.0, 12.0)[..., None]              # 0 = night … 1 = full day
    dusk = (_smooth(el, -8.0, 0.0) * (1 - _smooth(el, 4.0, 14.0)))[..., None]
    sky = NIGHT_SKY + (DAY_SKY - NIGHT_SKY) * day
    sky = sky + (DUSK_SKY - sky) * 0.6 * dusk
    horizon = NIGHT_HORIZON + (DAY_HORIZON - NIGHT_HORIZON) * day
    horizon = horizon + (DUSK_SKY - horizon) * 0.8 * dusk
    sun_col = SUN_LOW + (SUN_DAY - SUN_LOW) * _smooth(el, 2.0, 20.0)[..., None]
    return {
        "sun_energy": 3.5 * _smooth(el, -2.0, 8.0),
        "sun_color": sun_col,
        "sky_color": sky,
        "horizon_color": horizon,
        "sky_strength": 0.35 + 0.35 * day[..., 0] + 0.1 * dusk[..., 0],
        "moon_energy": MOON_ENERGY * (1 - _smooth(el, -6.0, 2.0)),
    }
