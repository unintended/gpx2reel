import pytest

from gpx2reel.ingest import ingest
from gpx2reel.storyboard import Highlight, Storyboard, plan_timing, skeleton, validate
from tests.fixtures.make_gpx import make_trip


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    folder = make_trip(tmp_path_factory.mktemp("trip") / "tracks")
    return ingest(sorted(folder.glob("*.gpx")))


def test_skeleton_is_valid(route):
    sb = skeleton(route, "Тест")
    errors, _ = validate(sb, route)
    assert errors == []
    assert len(sb.days) == 3 and len(sb.gaps) == 1


def test_timing_fills_duration(route):
    sb = skeleton(route)
    t = plan_timing(sb, route)
    total = t["fixed_s"] + sum(t["per_day_s"].values())
    assert abs(total - sb.duration_s) < 0.01
    # longest day gets the most time
    assert max(t["per_day_s"], key=t["per_day_s"].get) == 2


def test_unknown_preset_is_error(route):
    sb = skeleton(route)
    sb.highlights.append(Highlight(poi="Вулкан", at_km=50, duration_s=4, camera="spin360"))
    errors, _ = validate(sb, route)
    assert any("spin360" in e for e in errors)


def test_highlight_beyond_route(route):
    sb = skeleton(route)
    sb.highlights.append(Highlight(poi="Далеко", at_km=10_000, duration_s=4))
    errors, _ = validate(sb, route)
    assert any("beyond the route length" in e for e in errors)


def test_overbooked_duration(route):
    sb = skeleton(route)
    sb.duration_s = 10
    errors, _ = validate(sb, route)
    assert any("Not enough time" in e for e in errors)


def test_extra_keys_rejected():
    with pytest.raises(Exception):
        Storyboard.model_validate({"title": "x", "unknown_field": 1})


def test_day_story_storyboard(route):
    from gpx2reel.story import story_storyboard
    from gpx2reel.storyboard import ZoomStage

    sb = skeleton(route)
    sb.intro.zoom = [ZoomStage(extent_km=2000, label="Japan")]
    sb.highlights.append(Highlight(poi="Вулкан", at_km=60, duration_s=4))
    st, keep = story_storyboard(sb, route, 2, 20)
    assert st.focus_day == 2 and st.duration_s == 20 and st.platform == "instagram_story"
    assert [d.day for d in st.days] == [1, 2] and all(g.after_day < 2 for g in st.gaps)
    day2 = route["days"][1]
    assert keep == ([0] if day2["km_start"] <= 60 <= day2["km_end"] else [])
    assert not st.intro.play_zoom and st.intro.zoom             # the zoom opens only day 1's story (layers kept)
    errors, _ = validate(st, route)
    assert errors == []
    st1, _ = story_storyboard(sb, route, 1, 20)
    assert st1.intro.zoom and st1.intro.zoom[0].label == "Japan"


def test_long_place_names_warn(route):
    from gpx2reel.storyboard import MAX_PLACE_CHARS, skeleton, validate

    sb = skeleton(route)
    sb.days[0].finish = "Sun Royal Hotel, Kagoshima"
    sb.days[1].start = "Kagoshima"
    _, warns = validate(sb, route)
    assert len("Sun Royal Hotel, Kagoshima") > MAX_PLACE_CHARS
    assert any("Sun Royal Hotel" in w for w in warns) and not any("'Kagoshima'" in w for w in warns)


def test_sun_disc_choices_have_colours():
    from typing import get_args

    from gpx2reel.storyboard import Style
    from gpx2reel.sun import SUNRISE_DISKS

    assert Style().sun_disc == "natural" and Style().credits
    assert set(get_args(Style.model_fields["sun_disc"].annotation)) == set(SUNRISE_DISKS)
