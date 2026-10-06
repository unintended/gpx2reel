import numpy as np

from gpx2reel import overlay as O
from gpx2reel import poi as POI

SIZE = (540, 960)


def test_project_center_right_and_behind():
    cam, target = np.array([0.0, -1000, 500]), np.array([0.0, 0, 0])
    pts = np.array([[0, 0, 0], [200, 0, 0], [0, -3000, 0]], float)
    xy, front = O.project(pts, cam, target, 30.0, SIZE)
    assert np.allclose(xy[0], [SIZE[0] / 2, SIZE[1] / 2])
    assert xy[1, 0] > SIZE[0] / 2                      # +X is to the right when looking north
    assert front.tolist() == [True, True, False]


def test_window_fades():
    assert O.window(5, 2, 8) == 1.0
    assert O.window(2, 2, 8) == 0.0 and 0 < O.window(2.2, 2, 8) < 1


def _data(n=90, places=()):
    t = np.arange(n) / 30
    s = np.clip((t - 1) * 10, 0, 20)
    frames = {
        "t": t, "s": s, "km": s, "day": np.where(s < 10, 1, 2), "cam_dist": np.full(n, 5000.0),
        "cam": np.tile([0.0, -4000, 3000], (n, 1)), "target": np.zeros((n, 3)),
    }
    shots = [{"kind": "overview", "t0": 0, "t1": 1, "day": None, "label": "intro"},
             {"kind": "ride", "t0": 1, "t1": 2, "day": 1, "label": "day 1"},
             {"kind": "ride", "t0": 2, "t1": 3, "day": 2, "label": "day 2"}]
    days = [{"day": 1, "km_start": 0, "distance_km": 10, "date": "2026-09-21", "ascent_m": 300, "moving_time_s": 3600},
            {"day": 2, "km_start": 10, "distance_km": 10, "date": "2026-09-22", "ascent_m": 100, "moving_time_s": 2400}]
    km = np.linspace(0, 20, 50)
    profile = {"km": km, "ele": 100 + 50 * np.sin(km / 3), "day": np.where(km < 10, 1, 2)}
    return O.OverlayData(size=SIZE, fps=30, lens_mm=30, style=O.STYLES["bold_minimal"], title="Тест",
                         shots=shots, days=days, totals={"days": 2, "distance_km": 20, "ascent_m": 500},
                         checkpoints=[{"label": "Day 2 · 10 km", "day": 2, "s": 10, "xyz": [0, 0, 0]}],
                         zoom=[], captions={}, show_totals=True, route_rgb={1: (255, 0, 0), 2: (0, 0, 255)},
                         places=list(places), profile=profile, frames=frames)


def _alpha(img, box):
    return np.asarray(img.getchannel("A").crop(box)).max()


def test_counters_only_while_riding():
    d = _data()
    counters = (0, int(SIZE[1] * 0.6), SIZE[0], int(SIZE[1] * 0.8))
    assert _alpha(O.draw_frame(d, 5), (0, int(SIZE[1] * 0.6), SIZE[0], int(SIZE[1] * 0.72))) == 0   # intro: title band only
    assert _alpha(O.draw_frame(d, 45), counters) > 200          # riding


def test_checkpoint_label_appears_after_reaching():
    d = _data()
    # the checkpoint sits at the camera target → screen centre; reached at s = 10 (t = 2 s)
    centre = (SIZE[0] // 2, SIZE[1] // 2 - 60, SIZE[0], SIZE[1] // 2)
    assert _alpha(O.draw_frame(d, 45), centre) == 0
    assert _alpha(O.draw_frame(d, 75), centre) > 100


def test_peak_label_drawn_near_marker():
    d = _data(places=[{"name": "Асо", "kind": "peak", "ele": 1592, "xyz": np.array([300.0, 0, 0])}])
    O.plan_places(d)
    img = O.draw_frame(d, 36)                                         # before the day card (from 1.5 s) takes over
    xy, _ = O.project(np.array([[300.0, 0, 0]]), d.frames["cam"][36], d.frames["target"][36], 30, SIZE)
    x, y = xy[0].astype(int)
    assert _alpha(img, (x - 60, y - 90, x + 60, y - 10)) > 100          # label + stick above the summit


def test_poi_candidates_rank_and_roles(setup):
    route = setup[0]
    first = route["days"][0]["segments"][0]["points"][0]
    osm = {"elements": [
        {"lat": first[0] + 0.01, "lon": first[1], "tags": {"place": "town", "name": "Старт", "population": "5000"}},
        {"lat": 53.255, "lon": 158.83, "tags": {"natural": "volcano", "name": "Big", "ele": "2600"}},
        {"lat": 53.255, "lon": 158.84, "tags": {"natural": "peak", "name": "Small", "ele": "900"}},
        {"lat": 53.255, "lon": 158.85, "tags": {"natural": "peak", "name": "NoEle"}},
    ]}
    c = POI.candidates(route, osm)
    peaks = [x["name"] for x in c if x["kind"] == "peak"]
    assert peaks == ["Big", "Small"]
    assert "start" in next(x for x in c if x["kind"] == "town")["role"]


def test_pick_frames_keeps_duration():
    from gpx2reel.render import pick_frames

    sel, step = pick_frames(1800, 300)
    assert step == 6 and len(sel) == 300 and sel[0] == 1
    assert pick_frames(1800, None) == (list(range(1, 1801)), 1)


def test_debounce_smooths_flicker():
    v = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0], bool)
    out = O._debounce(v, 3)
    assert out[1:10].all()            # 1-frame gap filled
    assert not out[14]                # 1-frame blip dropped


def test_day_card_and_profile_while_riding():
    d = _data()
    top = (0, int(480 / O.BASE_H * SIZE[1]), SIZE[0], int(760 / O.BASE_H * SIZE[1]))
    band = (0, int(O.PROFILE_TOP / O.BASE_H * SIZE[1]), SIZE[0], int(O.PROFILE_BOTTOM / O.BASE_H * SIZE[1]))
    assert _alpha(O.draw_frame(d, 60), top) > 200          # day 1 card shortly after the ride starts
    assert _alpha(O.draw_frame(d, 85), band) > 100         # profile strip stays


def test_every_label_kind_has_an_icon():
    from gpx2reel.presets import LABEL_KINDS

    for kind in LABEL_KINDS:
        if kind == "peak":
            continue
        cv = O.Canvas(SIZE, O.STYLES["bold_minimal"])
        cv.icon(kind, 100, 100, 1.0)
        assert np.asarray(cv.text.getchannel("A")).max() == 255, kind


def test_day_card_windows_and_box():
    d = _data()
    assert O.day_card_windows(d.shots) == [(1, 1 + O.DAY_CARD_T[0], 2 + O.DAY_CARD_T[0]),          # till the next card
                                           (2, 2 + O.DAY_CARD_T[0], 2 + O.DAY_CARD_T[1])]
    x0, y0, x1, y1 = O.day_card_box(SIZE)
    assert 0 < x0 < SIZE[0] / 2 < x1 < SIZE[0] and 0 < y0 < y1 < SIZE[1]


def _shots(*spec):
    return [{"kind": k, "t0": t0, "t1": t1, "day": day, "label": k} for k, t0, t1, day in spec]


def test_day_card_leaves_when_a_highlight_starts():
    # day 11 of Kyushu: the highlight starts as the card would fade — it used to push the card into day 12's
    shots = _shots(("ride", 97.3, 101.5, 11), ("highlight", 101.5, 106.0, 11), ("ride", 106.0, 106.02, 11),
                   ("night", 106.02, 107.2, 11), ("ride", 107.2, 111.5, 12))
    (d11, a0, a1), (d12, b0, b1) = O.day_card_windows(shots)
    assert (d11, d12) == (11, 12) and a1 == 101.5 and a1 - a0 >= O.DAY_CARD_MIN_S and b0 > 107.2


def test_day_card_waits_for_an_early_highlight_or_is_dropped():
    late = _shots(("ride", 0.0, 0.2, 3), ("highlight", 0.2, 4.7, 3), ("ride", 4.7, 9.0, 3))
    assert O.day_card_windows(late) == [(3, 5.0, 9.0)]                 # after the highlight, the day still rides
    none = _shots(("ride", 0.0, 0.2, 3), ("highlight", 0.2, 4.7, 3), ("ride", 4.7, 5.5, 3), ("night", 5.5, 6.7, 3))
    assert O.day_card_windows(none) == []                              # no room left in the day


def test_no_pulse_over_the_intro():
    d = _data()
    d.frames["marker"] = np.zeros((d.n, 3))
    d.frames["marker_radius"] = np.full(d.n, 45.0)
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    assert _alpha(O.draw_frame(d, 10), (cx - 40, cy - 40, cx + 40, cy + 40)) == 0


def test_marker_pulse_rings_around_marker():
    d = _data()
    n = d.n
    d.frames["marker"] = np.zeros((n, 3))                    # at the camera target → screen centre
    d.frames["marker_radius"] = np.full(n, 45.0)
    img = O.draw_frame(d, 40)
    cx, cy = SIZE[0] // 2, SIZE[1] // 2
    ring = np.asarray(img.getchannel("A").crop((cx - 40, cy - 40, cx + 40, cy + 40)))
    assert ring.max() > 60


def test_intro_title_is_the_last_intro_level():
    d = _data()
    band = (0, int(SIZE[1] * 0.22), SIZE[0], int(SIZE[1] * 0.32))         # where the zoom labels / title sit
    intro_end = int(d.shots[0]["t1"] * d.fps)
    assert _alpha(O.draw_frame(d, intro_end - 15), band) > 100             # 0.5 s before the ride: title
    assert _alpha(O.draw_frame(d, intro_end + 9), band) < 40               # gone once the ride starts


def test_typewriter():
    assert O._typed("Mt Aso", 0.0, (1.0, 2.0)) == ("", False)
    assert O._typed("Mt Aso", 1.5, (1.0, 2.0)) == ("Mt ", True)
    assert O._typed("Mt Aso", 3.0, (1.0, 2.0)) == ("Mt Aso", False)


def test_callout_grows_from_the_highlight():
    d = _data()
    d.shots = [{"kind": "overview", "t0": 0, "t1": 0.5, "day": None, "label": "intro"},
               {"kind": "highlight", "t0": 0.5, "t1": 3, "day": 1, "label": "Mt Aso", "target": [0.0, 0.0, 0.0]}]
    d.highlight_notes = {"Mt Aso": "Active volcano"}
    O.plan_callouts(d)
    early, late = O.draw_frame(d, 16), O.draw_frame(d, 80)          # 0.03 s and 2.17 s into the highlight
    xy, _ = O.project(np.array([[0.0, 0, 0]]), d.frames["cam"][80], d.frames["target"][80], 30, SIZE)
    x, y = xy[0].astype(int)
    ring = (x - 20, y - 20, x + 20, y + 20)
    assert _alpha(late, ring) > 150                                   # ring at the point
    text = (0, 0, SIZE[0], y - 20)
    assert _alpha(late, text) > _alpha(early, text)                   # the label has appeared above it


def test_wetland_icon_is_not_a_lake():
    from gpx2reel.osm import _poi_kind
    from gpx2reel.overlay import Canvas, TextStyle

    assert _poi_kind({"natural": "wetland", "name": "Tadewara"}) == "wetland"
    imgs = {}
    for kind in ("lake", "wetland"):
        cv = Canvas((1080, 1920), TextStyle())
        cv.icon(kind, 540, 960, 1.0)
        imgs[kind] = np.asarray(cv.text)[900:1000, 480:600]
    def lit(a):                                                          # mean colour without the white outline
        px = a[a[..., 3] > 200][:, :3].astype(float)
        return px[px.min(axis=1) < 200].mean(axis=0)

    assert lit(imgs["lake"])[2] > lit(imgs["lake"])[1]                  # the lake is blue
    assert lit(imgs["wetland"])[1] > lit(imgs["wetland"])[2]            # the marsh is green, with reeds above
    above = slice(0, 48)                                                # rows above the lake's oval
    assert (imgs["wetland"][above, :, 3] > 0).any() and not (imgs["lake"][above, :, 3] > 0).any()


def test_climbs_are_found_in_the_profile():
    km = np.linspace(0, 48, 481)
    ele = np.where(km < 34, 10.0, 10 + (km - 34) / 14 * 680)       # day 11: flat, then 680 m up to Unzen
    ele[100:110] += 30                                               # a small bump on the flat
    found = O.climbs({"km": km, "ele": ele, "day": np.full(len(km), 11)})
    assert len(found) == 1
    k0, k1, e0, e1 = found[0]
    assert abs(k0 - 34) < 0.7 and abs(k1 - 48) < 0.2 and e1 - e0 > 650


def test_a_climb_starts_where_it_leaves_the_flat():
    km = np.linspace(0, 23, 231)
    ele = np.where(km < 9, 10 + 15 * np.sin(km * 2), 10 + (km - 9) / 14 * 680)   # rolling coast, then up
    (k0, k1, e0, e1), = O.climbs({"km": km, "ele": ele, "day": np.full(len(km), 11)})
    assert k0 > 8 and (e1 - e0) / ((k1 - k0) * 1000) > 0.045


def test_climb_counter_waits_for_the_day_card(monkeypatch):
    d = _data()
    seen = []
    monkeypatch.setattr(O, "_draw_climb", lambda cv, data, km, alpha: seen.append(alpha))
    O.draw_frame(d, 40)                                              # riding, the day 1 card (from 1.5 s) not up yet
    O.draw_frame(d, 60)                                              # the card is up
    assert seen[0] > 0 and seen[1] == 0


def test_finale_route_lines_three_places_each():
    pts = ["Beppu", "Aso", "Takachiho", "Miyazaki", "Kirishima", "Kagoshima", "Amakusa", "Unzen", "Nagasaki"]
    lines = O._route_lines(pts)
    assert lines == ["Beppu → Aso → Takachiho →", "Miyazaki → Kirishima → Kagoshima →", "Amakusa → Unzen → Nagasaki"]


def test_credits_in_the_last_seconds_and_on_covers():
    assert "ESA WorldCover" not in " ".join(O.credit_lines("satellite"))
    assert "ESA WorldCover" in " ".join(O.credit_lines("hybrid"))
    d = _data()
    d.credits = O.credit_lines("satellite")
    band = (0, int(O.SAFE_BOTTOM / O.BASE_H * SIZE[1]) + 20, SIZE[0], int(SIZE[1] * 0.9))
    assert _alpha(O.draw_frame(d, 15), band) == 0
    assert _alpha(O.draw_frame(d, 15, credits=True), band) > 100       # a cover
    assert _alpha(O.draw_frame(d, 85), band) > 100                    # the end of the video (no outro shot here)
    d.credits = []                                                     # style.credits: false
    assert _alpha(O.draw_frame(d, 85), band) == 0
