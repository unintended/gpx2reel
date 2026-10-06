"""Day story: a short Instagram story of one day of a longer trip, built in .work/track/stories/<name>/.

The day is ridden in full; earlier days are already drawn when it starts (storyboard.focus_day). The story
reuses the trip's config (style, labels, day places, that day's highlights / opening) and its downloaded
terrain (symlinks), so it costs only a scene + timeline + render. Its folder looks like a small trip:
track.yaml, tracks/ (symlinks to the trip's tracks of days 1..N) and build/.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .storyboard import Intro, Storyboard, dump
from .trip import TripPaths, story_name


def story_storyboard(sb: Storyboard, route: dict, day: int, duration_s: float,
                     seamless: bool = False) -> tuple[Storyboard, list[int]]:
    """Storyboard of the day story + indices (in sb.highlights) of the highlights it keeps."""
    d = next(x for x in route["days"] if x["day"] == day)
    st = sb.model_copy(deep=True)
    st.platform, st.duration_s, st.focus_day, st.story_chain = "instagram_story", duration_s, day, seamless
    st.days = [p for p in st.days if p.day <= day]
    st.gaps = [g for g in st.gaps if g.after_day < day or (g.within_day and g.after_day == day)]
    keep = [i for i, h in enumerate(sb.highlights) if d["km_start"] <= h.at_km <= d["km_end"]]
    st.highlights = [sb.highlights[i].model_copy(deep=True) for i in keep]
    if day != route["days"][0]["day"]:              # the zoom and the title open only the first story
        st.intro = Intro(duration_s=2.5 if seamless else 2.0, camera="overview", zoom=sb.intro.zoom,
                         play_zoom=False)                 # the zoom layers stay as far terrain
        st.title = ""
    if seamless:                                    # stories of equal length: clouds keep drifting across the cut
        st.clock_offset_s = duration_s * sum(1 for d in route["days"] if d["day"] < day)
    if day == route["days"][-1]["day"] and sb.outro.zoom_out:      # the last story ends the trip: the finale
        st.outro.duration_s = sb.outro.duration_s
    else:
        st.outro.duration_s = min(st.outro.duration_s, 3.5)
        st.outro.zoom_out, st.outro.route_points = False, []
    return st, keep


def _link(src: Path, dst: Path) -> None:
    """Relative symlink: the trip folder can be moved or copied to another machine as a whole."""
    import os

    if dst.is_symlink() or dst.exists():
        dst.unlink() if not dst.is_dir() or dst.is_symlink() else shutil.rmtree(dst)
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def make_day_story(p: TripPaths, day: int, duration_s: float = 20.0, build: bool = True, chain: bool = True,
                   log=print, suffix: str = "") -> TripPaths:
    """p: the trip (main reel paths). chain: a seamless story (the default) — see story_storyboard."""
    from .ingest import load_route

    route = load_route(p.route)
    sb = p.storyboard(route)
    if day not in {d["day"] for d in route["days"]}:
        raise ValueError(f"day {day} is not in the trip")
    st_p = p.for_story(story_name(day, chain, suffix))
    (st_p.build / "terrain").mkdir(parents=True, exist_ok=True)
    st_p.tracks.mkdir(exist_ok=True)

    # tracks of days 1..day (the files ingest kept: duplicates from a second device are already dropped)
    for f in st_p.tracks.iterdir():
        if f.is_symlink():
            f.unlink()
    for d in route["days"]:
        if d["day"] <= day:
            for name in d["files"]:
                _link(p.tracks / name, st_p.tracks / name)

    st, keep = story_storyboard(sb, route, day, duration_s, chain)
    dump(st, st_p.config)

    # terrain of the trip: files, intro-zoom layers, and the kept highlights' detail patches renumbered
    tsrc, tdst = p.build / "terrain", st_p.build / "terrain"
    for item in tsrc.iterdir():
        if not item.name.startswith("detail"):
            _link(item, tdst / item.name)
    for j, i in enumerate(keep):
        if (tsrc / f"detail{i}").exists():
            _link(tsrc / f"detail{i}", tdst / f"detail{j}")
    for name in ("roads.json", "poi.json"):
        if (p.build / name).exists():
            shutil.copy(p.build / name, st_p.build / name)
    for name in ("landcover", "osm"):
        if (p.build / name).exists():
            _link(p.build / name, st_p.build / name)
    log(f"Day {day} story: {st_p.home}")

    if build:
        from .ingest import IngestConfig, ingest, save_route

        save_route(ingest(st_p.track_files(), IngestConfig()), st_p.route)
        from .blender.scene import build_scene

        build_scene(st_p, log=log)
        from .blender.animate import build_reel

        build_reel(st_p, log=log)
    return st_p


def summary(p: TripPaths) -> str:
    tl = json.loads((p.build / "timeline.json").read_text(encoding="utf-8"))
    return f"{len(tl['shots'])} shots, {tl['shots'][-1]['t1']:.1f} s"
