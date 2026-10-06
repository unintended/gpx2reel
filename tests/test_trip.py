import pytest

from gpx2reel.trip import TripPaths, story_name


def test_main_reel_layout(tmp_path):
    p = TripPaths(tmp_path)
    assert p.config == tmp_path / "track.yaml"
    assert p.tracks == tmp_path / "tracks"
    assert p.build == tmp_path / ".work" / "track" / "build"
    assert p.route == p.build / "route.json"
    assert p.video() == tmp_path / "videos" / "track-reel.mp4"
    assert p.video(draft=True, tag="x") == tmp_path / "videos" / "drafts" / "track-reel-x.mp4"
    assert p.cover() == tmp_path / "videos" / "track-reel-cover.png"
    assert p.resolve("music/a.mp3") == tmp_path / "music" / "a.mp3"


def test_story_layout(tmp_path):
    p = TripPaths(tmp_path).for_story("day12")
    home = tmp_path / ".work" / "track" / "stories" / "day12"
    assert (p.home, p.config, p.tracks, p.build) == (home, home / "track.yaml", home / "tracks", home / "build")
    assert p.video() == tmp_path / "videos" / "track-day12.mp4"
    assert p.resolve("music/a.mp3") == tmp_path / "music" / "a.mp3"       # config paths: from the trip folder


def test_story_names():
    assert story_name(3) == "day03"
    assert story_name(12, chain=False) == "day12-standalone"
    assert story_name(9, suffix="hondo") == "day09-hondo"


def test_track_files(tmp_path):
    p = TripPaths(tmp_path)
    with pytest.raises(FileNotFoundError):
        p.track_files()
    p.tracks.mkdir()
    for name in ("b.FIT", "a.gpx", "c.zip", "notes.txt"):
        (p.tracks / name).write_text("")
    assert [f.name for f in p.track_files()] == ["a.gpx", "b.FIT", "c.zip"]


def test_day_story_folder(tmp_path):
    from gpx2reel.ingest import ingest, save_route
    from gpx2reel.storyboard import load
    from gpx2reel.story import make_day_story
    from tests.fixtures.make_gpx import make_trip

    p = TripPaths(tmp_path)
    make_trip(p.tracks)
    save_route(ingest(p.track_files()), p.route)
    (p.build / "terrain" / "ctx0").mkdir(parents=True)
    st = make_day_story(p, 2, build=False, log=lambda *_: None)
    assert st.name == "day02" and st.home == p.work / "stories" / "day02"
    assert [f.name for f in st.track_files()] == ["day1.gpx", "day2.gpx"]
    assert all(f.is_symlink() and f.resolve() == p.tracks / f.name for f in st.track_files())
    assert (st.build / "terrain" / "ctx0").resolve() == p.build / "terrain" / "ctx0"
    sb = load(st.config)
    assert sb.focus_day == 2 and sb.story_chain
    assert p.stories() == ["day02"]
    assert make_day_story(p, 2, build=False, chain=False, log=lambda *_: None).name == "day02-standalone"
