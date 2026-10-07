import numpy as np
import pytest

from gpx2reel import world as W
from gpx2reel.storyboard import Highlight, ZoomStage, skeleton
from gpx2reel.timeline import CameraRig, build_frames, day_clock_of, plan_shots
from tests.conftest import EXAG

FPS = 30


def make(setup, gap_mode: str):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.highlights.append(Highlight(poi="Вулкан", at_km=60, duration_s=3, camera="orbit"))
    sb.intro.duration_s = 3
    sb.intro.zoom = [ZoomStage(extent_km=2000)]
    for g in sb.gaps:
        g.mode = gap_mode
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    shots, _ = plan_shots(sb, route, path, arcs)
    cps = [{"s": float(path.s[0])}, {"s": float(path.s[len(path.s) // 2])}, {"s": float(path.s[-1])}]
    frames = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                          zoom=[{"extent_m": 2_000_000, "center_xy": [300_000.0, 500_000.0]}], checkpoints=cps,
                          day_rgba={d: (d / 3, 0, 0, 1) for d in (1, 2, 3)},
                          day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    return sb, shots, frames, grid, path


@pytest.fixture(scope="module")
def reel(setup):
    return make(setup, "straight")


@pytest.fixture(scope="module")
def joined(setup):
    return make(setup, "join")


def test_shots_fill_duration_in_order(reel):
    sb, shots, frames, _, _ = reel
    kinds = [s.kind for s in shots]
    assert kinds[:2] == ["zoom", "ride"] and kinds[-1] == "overview"
    assert "highlight" in kinds and "gap" in kinds
    assert all(abs(a.t1 - b.t0) < 1e-9 for a, b in zip(shots, shots[1:]))
    assert abs(shots[-1].t1 - sb.duration_s) < 0.01
    assert len(frames["t"]) == round(sb.duration_s * FPS)


def test_join_replaces_arc(joined):
    _, shots, f, _, path = joined
    kinds = [s.kind for s in shots]
    assert "join" in kinds and "gap" not in kinds
    assert np.all(np.diff(f["s"]) >= -1e-9)
    # the marker never jumps: per-frame steps stay small compared to the route
    step = np.linalg.norm(np.diff(f["marker"][:, :2], axis=0), axis=1)
    moving = np.isin(f["shot"][1:], [i for i, s in enumerate(shots) if s.kind in ("ride", "join")])
    assert step[moving].max() < 0.05 * np.ptp(path.xyz[:, :2], axis=0).max()


def test_progress_monotonic_and_km_follows(reel):
    _, _, f, _, _ = reel
    assert np.all(np.diff(f["s"]) >= -1e-9)
    assert np.all(np.diff(f["km"]) >= -1e-9)


def test_marker_on_ground_while_riding(reel):
    _, shots, f, grid, _ = reel
    ride = np.isin(f["shot"], [i for i, s in enumerate(shots) if s.kind == "ride"])
    ground = grid.height_at(f["marker"][ride, 0], f["marker"][ride, 1])
    assert np.allclose(f["marker"][ride, 2], ground, atol=0.5)


def test_highlight_stops_the_marker(reel):
    _, shots, f, _, _ = reel
    m = f["shot"] == next(i for i, s in enumerate(shots) if s.kind == "highlight")
    assert np.allclose(f["km"][m], 60, atol=0.05)
    assert np.ptp(f["marker"][m], axis=0).max() < 1e-6


def test_zoom_starts_wide_and_lands_on_the_route(reel):
    _, _, f, _, _ = reel
    z = np.flatnonzero(f["shot"] == 0)
    d = f["cam_dist"][z]
    assert d[0] > 1_000_000 and d[-1] < 0.2 * d[0]
    assert np.all(np.diff(d) <= 1e-3 * d[0])        # only zooms in


def test_camera_above_terrain_and_closer_at_highlight(reel):
    _, shots, f, grid, _ = reel
    assert np.all(f["cam"][:, 2] > grid.height_at(f["cam"][:, 0], f["cam"][:, 1]))
    h = np.flatnonzero(f["shot"] == next(i for i, s in enumerate(shots) if s.kind == "highlight"))
    r = np.flatnonzero(f["shot"] == next(i for i, s in enumerate(shots) if s.kind == "ride"))
    assert f["cam_dist"][h[len(h) // 2]] < f["cam_dist"][r[len(r) // 2]]


def test_sizes_follow_camera_distance(reel):
    _, _, f, _, _ = reel
    r = f["marker_radius"] / np.linalg.norm(f["cam"] - f["marker"], axis=1)
    assert np.allclose(r, r[0])


def test_checkpoints_pop_when_reached(reel):
    _, _, f, _, _ = reel
    cp = f["cp_scale"]
    assert cp[-1].min() > 0                        # all reached by the end
    first = np.argmax(cp[:, 1] > 0)
    assert first > 0 and cp[:first, 1].max() == 0  # hidden until the marker gets there
    assert np.all(f["day"][1:] >= f["day"][:-1])


def test_night_between_days_holds_camera_and_goes_dark(reel):
    _, shots, f, _, _ = reel
    nights = [i for i, s in enumerate(shots) if s.kind == "night"]
    assert len(nights) == 2                                  # 3 days
    m = np.flatnonzero(f["shot"] == nights[0])
    assert np.ptp(f["marker"][m], axis=0).max() < 1e-6
    assert np.ptp(f["cam_dist"][m]) < 0.35 * f["cam_dist"][m].mean()      # held, smoothing eases in/out
    assert f["sun_el"][m].min() < -5 and f["sky_strength"][m].min() < 0.45
    assert np.all(np.diff(f["epoch"][m]) > 0)


def test_daytime_sun_while_riding(reel):
    _, shots, f, _, _ = reel
    ride = np.isin(f["shot"], [i for i, s in enumerate(shots) if s.kind == "ride"])
    assert np.median(f["sun_el"][ride]) > 10                 # synthetic days start 09:00 local
    assert np.allclose(np.linalg.norm(f["sun_vec"], axis=1), 1)


def test_ride_camera_drifts_and_outro_orbits(reel):
    _, shots, f, _, _ = reel
    ride = np.flatnonzero(f["shot"] == next(i for i, s in enumerate(shots) if s.kind == "ride"))
    # the side angle swings: the camera offset direction relative to the heading is not constant
    off = f["cam"][ride, :2] - f["marker"][ride, :2]
    ang = np.unwrap(np.arctan2(off[:, 1], off[:, 0]))
    assert np.ptp(ang) > np.radians(5)
    out = np.flatnonzero(f["shot"] == len(shots) - 1)
    o = f["cam"][out, :2] - f["target"][out, :2]
    a = np.arctan2(o[:, 1], o[:, 0])
    assert 0.5 * np.radians(10) < abs(a[-1] - a[0]) < 1.5 * np.radians(10)   # CameraRig.outro_turn_deg


def test_sky_look_has_moon_and_horizon():
    from gpx2reel.sun import sky_look

    look = sky_look(np.array([-20.0, -3.0, 30.0]))
    assert look["moon_energy"][0] > 0.4 and look["moon_energy"][2] == 0
    assert look["horizon_color"].shape == (3, 3)
    assert np.all(look["horizon_color"][2] > look["sky_color"][2])        # day: horizon paler than the zenith
    assert look["sky_color"][0].mean() > 0.04                              # night is moonlit, not black


def test_sunrise_opening(setup):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.days[1].opening = "sunrise"
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    shots, _ = plan_shots(sb, route, path, arcs)
    f = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                     day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    i = next(i for i, s in enumerate(shots) if s.kind == "opening")
    assert shots[i + 1].kind == "ride" and shots[i + 1].day == 2 and shots[i - 1].kind in ("night", "gap", "join")
    m = np.flatnonzero(f["shot"] == i)
    el = f["sun_el"][m]
    assert el[0] < 0 < el[-1] and np.all(np.diff(f["epoch"][m]) > 0)          # the sun comes up during the shot
    night = np.flatnonzero(f["shot"] == i - 1) if shots[i - 1].kind == "night" else []
    if len(night):
        step = np.diff(f["epoch"][night]).max()                             # a night frame spans ~30 min
        assert abs(f["epoch"][m[0]] - f["epoch"][night[-1]]) < 2 * step         # night hands over at dawn
    ground = grid.height_at(f["cam"][m, 0], f["cam"][m, 1])
    mid = m[len(m) // 2]
    assert f["cam"][mid, 2] - ground[len(m) // 2] < 400                        # low camera
    look = (f["target"][mid] - f["cam"][mid])[:2]
    sun = f["sun_vec"][mid, :2]
    assert look @ sun / np.linalg.norm(look) / np.linalg.norm(sun) > 0.9       # looking at the sun


def test_sunrise_beach_opening(setup):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.days[1].opening = "sunrise_beach"
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    shots, _ = plan_shots(sb, route, path, arcs)
    f = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                     day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    i = next(i for i, s in enumerate(shots) if s.kind == "opening")
    assert shots[i].camera == "sunrise_beach"
    m = np.flatnonzero(f["shot"] == i)
    rig = CameraRig()
    early = m[int(rig.blend_s * FPS) + 2: len(m) // 2]          # after the fly-in, before the crane: low and level
    ground = grid.height_at(f["cam"][early, 0], f["cam"][early, 1])
    assert np.all(f["cam"][early, 2] - ground < rig.beach_height_m + 1)
    d = f["target"][early[-1]] - f["cam"][early[-1]]
    assert abs(d[2]) / np.linalg.norm(d) < 0.02
    d = f["target"][m[-1]] - f["cam"][m[-1]]                 # the crane ends looking down at the start
    assert d[2] / np.linalg.norm(d) < -0.4
    assert np.all(f["beach_show"][m[20:]] > 0.99) and f["beach_show"][: m[0]].max() == 0   # 0.5 s fade-in
    assert np.all(f["trail_width"][m[15:]] < 1e-6)             # the day before's line would cross the foreground
    ride = np.flatnonzero(f["shot"] == i + 1)
    assert f["trail_width"][ride[-1]] > 0


def test_sunset_closing(setup):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.days[0].closing = "sunset"
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    shots, _ = plan_shots(sb, route, path, arcs)
    f = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                     day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    i = next(i for i, s in enumerate(shots) if s.kind == "closing")
    assert shots[i - 1].kind == "ride" and shots[i - 1].day == 1 and shots[i].camera == "sunset"
    m = np.flatnonzero(f["shot"] == i)
    rig = CameraRig()
    assert np.all(np.diff(f["epoch"][m]) >= 0) and f["epoch"][m[-1]] > f["epoch"][m[0]]
    el = f["sun_el"][m]
    assert el[len(m) // 3] > 0 > el[-1] > -8                       # the sun goes down, the shot ends in the dusk
    assert f["sky_color"][m[-1]].mean() < f["sky_color"][m[len(m) // 3]].mean() * 0.5   # the sky has gone dark
    mid = m[int(rig.blend_s * FPS) + 2: len(m) - 2]
    look = (f["target"][mid] - f["cam"][mid])[:, :2]
    sun = f["sun_vec"][mid, :2]
    cos = np.sum(look * sun, axis=1) / np.linalg.norm(look, axis=1) / np.linalg.norm(sun, axis=1)
    assert np.all(cos > 0.93)                                      # looking at where the sun goes down
    ground = grid.height_at(f["cam"][mid, 0], f["cam"][mid, 1])
    assert np.all(f["cam"][mid, 2] - ground < rig.sunset_height_m + 1)
    assert np.all(f["sunset"][mid] > 0.99) and np.all(f["ambient"][mid] < 0.35) and f["sunset"][: m[0]].max() == 0
    assert np.all(f["sun_disk_rgba"][mid, 2] > f["sun_disk_rgba"][0, 2])   # white-hot, not the red rising sun
    up = mid[f["sun_el"][mid] > 0]
    assert np.all(f["sun_disk_k"][up] == 1) and f["sun_disk_k"][m[-1]] == 0  # full size until it has set
    nxt = np.flatnonzero(f["shot"] == i + 1)                       # the evening goes on from the sunset
    assert f["epoch"][nxt[0]] >= f["epoch"][m[-1]] - 1 and f["trail_width"][mid].min() > 0


def test_sunset_at_another_spot(setup):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.days[0].closing = "sunset"
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    end = path.xyz[path.day == 1][-1]
    spot = end + np.array([3000.0, 2000.0, 0.0])                # closing_at, a few km from the finish
    spot[2] = grid.height_at(spot[:1], spot[1:2])[0]
    shots, _ = plan_shots(sb, route, path, arcs, pois={"closing1": spot.tolist()})
    f = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                     day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    i = next(i for i, s in enumerate(shots) if s.kind == "closing")
    rig = CameraRig()
    m = np.flatnonzero(f["shot"] == i)
    mid = m[int(rig.blend_s * FPS) + 2: len(m) - 2]
    d_spot = np.linalg.norm(f["cam"][mid, :2] - spot[:2], axis=1)
    assert np.all(d_spot < rig.sunset_spot_back_m + 1)          # the camera flew to the spot…
    assert np.all(np.linalg.norm(f["marker"][mid, :2] - end[:2], axis=1) < 50)   # …the marker stays at the finish
    assert np.linalg.norm(f["cam"][m[-1] + int(2 * FPS)] - f["cam"][mid[-1]]) > 5000   # and back to the overview
    assert np.all(f["trail_width"][mid] < 1e-6)                 # the road along that shore would cross the frame


def test_day_story_rides_only_its_day(setup):
    route, _, grid, raw = setup
    sb = skeleton(route)
    sb.focus_day = 2
    sb.duration_s = 20
    modes = {g.after_day: g.mode for g in sb.gaps}
    path = W.join_gaps(route, raw, grid, EXAG, modes)
    arcs = {a["after_day"]: a["points"] for a in W.gap_arcs(route, path, modes)}
    shots, _ = plan_shots(sb, route, path, arcs)
    assert {s.day for s in shots if s.kind == "ride"} == {2}
    assert not any(s.kind == "night" for s in shots)
    assert abs(shots[-1].t1 - 20) < 0.01
    f = build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                     day_clock=day_clock_of(route), latlon=(53.1, 158.7))
    day2 = next(d for d in route["days"] if d["day"] == 2)
    assert abs(f["km"][0] - day2["km_start"]) < 0.1                 # starts with day 1 already drawn
    assert f["km"].max() <= day2["km_end"] + 0.1


def test_chained_day_stories_meet_on_the_same_frame(setup):
    import copy

    route, proj, grid, _ = setup
    sb = skeleton(route)
    for g in sb.gaps:
        g.mode = "join"

    def story(day):
        r = copy.deepcopy(route)
        r["days"] = [d for d in r["days"] if d["day"] <= day]
        r["gaps"] = [g for g in r["gaps"] if g["after_day"] < day]
        st = sb.model_copy(deep=True)
        st.focus_day, st.story_chain, st.duration_s = day, True, 15
        st.days = [p for p in st.days if p.day <= day]
        st.gaps = [g for g in st.gaps if g.after_day < day]
        modes = {g.after_day: g.mode for g in st.gaps}
        path = W.join_gaps(r, W.route_path(r, proj, grid, EXAG), grid, EXAG, modes)
        shots, _ = plan_shots(st, r, path, {})
        return build_frames(shots, path, grid, W.RIBBON_LIFT_M * EXAG, FPS, 1080 / 1920,
                            day_clock=day_clock_of(r), latlon=(53.1, 158.7), chain=True)

    a, b = story(1), story(2)
    assert np.allclose(a["cam"][-1], b["cam"][0], atol=5.0)          # metres, over a ~100 km view
    assert np.allclose(a["target"][-1], b["target"][0], atol=5.0)
    assert np.allclose(a["marker"][-1], b["marker"][0], atol=1.0)
    assert abs(a["s"][-1] - b["s"][0]) < 1e-6                        # same trail progress
    assert abs(a["sun_el"][-1] - b["sun_el"][0]) < 0.5                # the same night on both sides
    assert np.allclose(a["sky_color"][-1], b["sky_color"][0], atol=0.02)
    assert b["sun_el"][int(2.5 * FPS)] > b["sun_el"][0]              # and story 2 heads for the morning
    assert a["past_mix"][-1] > 0.99 and a["past_mix"][0] == 0          # story 1 greys its day before the cut


def test_rest_days_pass_as_one_night():
    from gpx2reel.sun import one_night, solar_position

    lat, lon = 32.0, 130.5
    a = 1790838000.0                                   # an afternoon; the next ride starts ~67 h later, at 08:40
    b = a + 67 * 3600
    u = np.linspace(0, 1, 400)
    t = one_night(lat, lon, a, b, u)
    _, el = solar_position(np.full(len(u), lat), np.full(len(u), lon), t)
    assert t[0] == a and abs(t[-1] - b) < 1 and np.all(np.diff(t) >= 0)
    rises = np.flatnonzero((el[:-1] < 0) & (el[1:] >= 0))
    sets = np.flatnonzero((el[:-1] >= 0) & (el[1:] < 0))
    assert len(rises) <= 1 and len(sets) <= 1          # one dusk, one dawn — not three days of flicker


def test_climbs_take_longer_than_flats():
    from gpx2reel.timeline import _effort

    s = np.linspace(0, 20, 1001)                                # km
    z = np.where(s < 10, 0.0, (s - 10) * 1000 * 0.06) * 1.6       # flat, then 6 % up (exaggerated 1.6)
    path = W.Path3D(np.column_stack([s * 1000, np.zeros_like(s), z + 3 * 1.6]), s, np.ones_like(s, int),
                    np.zeros_like(s, int), s)
    e = _effort(path, 3 * 1.6, 10.0)
    flat, climb = e[500] - e[0], e[1000] - e[500]
    assert 1.5 < climb / flat < 1.7                              # × (1 + 10 · 0.06)
    assert np.allclose(_effort(path, 3 * 1.6, 0.0), s)


def test_ferry_inside_a_day_is_drawn_as_transport():
    g = {"kind": "missing", "within_day": True, "after_day": 11}
    assert W.gap_mode(g, {"in11": "ferry"}) == "ferry"
    assert W.gap_mode(g, {11: "road"}) == "join"                  # the gap after the day is another one


def test_day_story_keeps_its_own_ferry(setup):
    from gpx2reel.story import story_storyboard
    from gpx2reel.storyboard import GapPlan

    route = setup[0]
    sb = skeleton(route)
    sb.gaps = [GapPlan(after_day=2, within_day=True, mode="ferry"), GapPlan(after_day=2, mode="road")]
    st, _ = story_storyboard(sb, route, 2, 20.0, seamless=True)
    assert st.gap_modes() == {"in2": "ferry"}                 # the gap after day 2 belongs to the next story


def test_finale_keeps_the_route_in_frame(setup):
    from gpx2reel.timeline import _keep_route_in_frame, zoom_poses

    _, _, grid, path = setup
    rig = CameraRig()
    c = path.xyz[:, :2].mean(axis=0)
    far = {"extent_m": 2_400_000.0, "center_xy": [c[0] + 900_000.0, c[1] + 600_000.0]}   # "Japan", centred far off
    ov = (np.array([c[0], c[1] - 50_000.0, 80_000.0]), np.array([c[0], c[1], 0.0]))
    poses = _keep_route_in_frame(zoom_poses([far], *ov, rig), path, 1080 / 1920, rig)
    cam, tgt = poses[0]
    half_w = np.linalg.norm(cam - tgt) * 12.0 / rig.lens_mm * 1080 / 1920
    assert abs(path.xyz[:, 0].mean() - tgt[0]) < half_w           # the route is inside the frame's width
    half_h = np.linalg.norm(cam - tgt) * 12.0 / rig.lens_mm
    assert path.xyz[:, 1].min() > tgt[1] - 0.16 * half_h          # and about above the middle (the card is below)
    assert np.allclose(poses[-1][1], ov[1])                        # the route overview itself is untouched


def test_finale_dots_shrink_away_as_the_camera_backs_out():
    from gpx2reel.timeline import _finale_dots

    k = _finale_dots(np.array([470e3, 500e3, 1000e3, 1500e3, 3000e3]), CameraRig())
    assert k[0] == k[1] == 1.0 and 0 < k[2] < 1 and k[3] == k[4] == 0.0
