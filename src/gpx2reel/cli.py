"""gpx2reel command line. Each command does one thing; every command takes the trip folder (docs/trip-layout.md),
and the per-film ones take `--story <name>` for a day story instead of the main reel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from .trip import TripPaths

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="GPX / FIT tracks → a 3D route video rendered in Blender.")

TRIP = typer.Argument(..., help="Trip folder (tracks in <trip>/tracks/)")
STORY = typer.Option(None, "--story", help="Day story name (e.g. day12, see `day-story`); default: the main reel")


def _paths(trip: Path, story: str | None = None) -> TripPaths:
    p = TripPaths(trip, story)
    if story and not p.home.is_dir():
        known = ", ".join(TripPaths(trip).stories()) or "none"
        raise typer.BadParameter(f"no story '{story}' in {p.trip} (stories: {known}); build it with `day-story`")
    return p


@app.command()
def ingest(
    trip: Path = TRIP,
    gap_threshold_m: float = typer.Option(500, help="Gap threshold, m"),
    day_mode: str = typer.Option("auto", help="auto | date | file"),
    preview: bool = typer.Option(True, help="Draw preview.png"),
):
    """Parse GPX / FIT / ZIP tracks: days, cleaning, gaps, stats → route.json."""
    from .ingest import IngestConfig, ingest as run, save_route, summary_text
    from .preview import render_preview

    p = _paths(trip)
    try:
        files = p.track_files()
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e))
    route = run(files, IngestConfig(gap_threshold_m=gap_threshold_m, day_mode=day_mode))
    save_route(route, p.route)
    typer.echo(summary_text(route))
    typer.echo(f"\n→ {p.route}")
    if preview:
        typer.echo(f"→ {render_preview(route, p.build / 'preview.png')}")


@app.command()
def summary(trip: Path = TRIP, story: str = STORY):
    """Show the summary of an ingested trip."""
    from .ingest import load_route, summary_text

    typer.echo(summary_text(load_route(_paths(trip, story).route)))


@app.command()
def sanitize(
    trip: Path = TRIP,
    out: Path = typer.Option(..., help="Output folder for the GPX files"),
    tolerance_m: float = typer.Option(5.0, help="Simplification tolerance (Ramer–Douglas–Peucker), m"),
    trim_m: float = typer.Option(0.0, help="Cut this many metres off the start and end of every day (privacy zone)"),
):
    """Shareable tracks: one plain GPX per ride day (lat / lon / ele / time only), simplified and trimmed."""
    from .ingest import IngestConfig, ingest as run
    from .sanitize import sanitize as run_sanitize

    p = _paths(trip)
    out = out.resolve()
    if out == p.trip or p.tracks.resolve() in (out, *out.parents):
        raise typer.BadParameter("--out must not be the trip or its tracks/ folder")
    written = run_sanitize(run(p.track_files(), IngestConfig()), out, tolerance_m, trim_m)
    for f in written:
        typer.echo(f"→ {f}  {f.stat().st_size / 1024:.0f} KB")


@app.command()
def skeleton(
    trip: Path = TRIP,
    title: str = typer.Option("My trip"),
    duration: float = typer.Option(45.0),
    force: bool = typer.Option(False, help="Overwrite an existing track.yaml"),
):
    """Write a minimal track.yaml for route.json (a starting point to edit)."""
    from .ingest import load_route
    from .storyboard import dump, skeleton as make

    p = _paths(trip)
    if p.config.exists() and not force:
        raise typer.BadParameter(f"{p.config} exists (use --force)")
    dump(make(load_route(p.route), title, duration), p.config)
    typer.echo(f"→ {p.config}")


@app.command()
def validate(trip: Path = TRIP, story: str = STORY,
             config: Path = typer.Option(None, help="Config to check instead of the film's track.yaml")):
    """Check track.yaml against the route. Exit code 1 on errors."""
    from .ingest import load_route
    from .storyboard import load, plan_timing, validate_file

    p = _paths(trip, story)
    sb_path = config or p.config
    route = load_route(p.route) if p.route.exists() else None
    errors, warns = validate_file(sb_path, route)
    for w in warns:
        typer.echo(f"⚠ {w}")
    for e in errors:
        typer.echo(f"✗ {e}")
    if errors:
        raise typer.Exit(1)
    if route:
        t = plan_timing(load(sb_path), route)
        per_day = ", ".join(f"d{k}: {v:.1f} s" for k, v in sorted(t["per_day_s"].items()))
        typer.echo(f"Timing: fixed shots {t['fixed_s']:.1f} s; riding per day — {per_day}")
    typer.echo(f"✓ {sb_path.name} is valid")


@app.command()
def terrain(
    trip: Path = TRIP,
    buffer_km: float = typer.Option(30.0, help="Margin around the route, km (the drone camera sees far)"),
    dem_zoom: int = typer.Option(12, help="12 ≈ 38 m/pixel"),
    ortho_zoom: int = typer.Option(13, help="13 ≈ 19 m/pixel, 14 ≈ 10 m/pixel"),
    imagery: bool = typer.Option(True, help="Download satellite imagery"),
    dry_run: bool = typer.Option(False, help="Only count the tiles"),
):
    """Download the elevation (DEM) and satellite imagery → terrain/. Needs the internet."""
    from .ingest import load_route
    from . import terrain as T

    p = _paths(trip)
    tdir = p.build / "terrain"
    route = load_route(p.route)
    if dry_run:
        b = T.expand_bbox(route["bbox"], buffer_km)
        typer.echo(T.estimate(b, dem_zoom, ortho_zoom))
        return
    area = dict(route)
    if p.config.exists():                             # sunset spots away from the route must lie on the terrain too:
        from .storyboard import load as _load         # beyond its edge the timeline would read the edge's height

        spots = [d.closing_at for d in _load(p.config).days if d.closing_at]
        if spots:
            b = route["bbox"]
            area["bbox"] = [min([b[0]] + [s[0] for s in spots]), min([b[1]] + [s[1] for s in spots]),
                            max([b[2]] + [s[0] for s in spots]), max([b[3]] + [s[1] for s in spots])]
    meta = T.download(area, tdir, buffer_km, dem_zoom, ortho_zoom, imagery, log=typer.echo)
    from . import coast as _coast

    _coast.download_coastline(meta["bbox"], tdir / "coastline.json", log=typer.echo)
    if p.config.exists():
        from .storyboard import load
        from .world import route_center

        sb = load(p.config)
        for i, h in enumerate(sb.highlights):
            if h.detail == "high_res_dem" and h.lat is not None and h.lon is not None:
                typer.echo(f"Detailed terrain for «{h.poi}»")
                T.download_detail((h.lat, h.lon), tdir / f"detail{i}", log=typer.echo)
        from . import coast

        full = sb.model_copy(update={"focus_day": None})
        for day, kind, sub, _, at in coast.coast_sets(full):      # beach sunrise / sunset: the shore in detail
            lat, lon = coast.set_anchor(route, day, kind, at)
            out = tdir / sub
            typer.echo(f"Shore for the {'sunrise' if kind == 'beach' else 'sunset'} of day {day}")
            T.download_detail((lat, lon), out, radius_km=coast.PATCH_RADIUS_KM, log=typer.echo)
            coast.download(lat, lon, out, log=typer.echo)
        for i, z in enumerate(sb.intro.zoom):
            typer.echo(f"Context for zoom {i}: {z.label or ''} {z.extent_km:.0f} km")
            T.download_context(z.center or route_center(route), z.extent_km, tdir / f"ctx{i}", log=typer.echo)


@app.command()
def scene(
    trip: Path = TRIP,
    story: str = STORY,
    step_m: float = typer.Option(100.0, help="Terrain grid step, m"),
    still: bool = typer.Option(False, help="Render check stills still_*.png"),
    samples: int = typer.Option(32),
):
    """Build the 3D scene: terrain, route, gaps, marker → scene.blend (needs bpy)."""
    from .blender.scene import build_scene

    build_scene(_paths(trip, story), step_m=step_m, stills=still, samples=samples, log=typer.echo)


@app.command()
def timeline(
    trip: Path = TRIP,
    story: str = STORY,
    stills: int = typer.Option(0, help="Render N evenly spaced frames into stills/"),
    samples: int = typer.Option(32),
):
    """Animate the scene from track.yaml: marker, camera, progress → reel.blend (needs bpy)."""
    from .blender.animate import build_reel

    build_reel(_paths(trip, story), stills=stills, samples=samples, log=typer.echo)


@app.command()
def overlays(trip: Path = TRIP, story: str = STORY):
    """Only the overlays (km, labels, title) → overlay/ — fast, no Blender."""
    from .overlay import render_overlays

    typer.echo(f"→ {render_overlays(_paths(trip, story), log=typer.echo)}")


@app.command()
def render(
    trip: Path = TRIP,
    story: str = STORY,
    draft: bool = typer.Option(False, help="25 % size, 8 samples → videos/drafts/"),
    frames: int = typer.Option(None, help="Render only ~N frames (every k-th): same length, choppy"),
    reuse_frames: bool = typer.Option(False, help="Don't re-render frames, only overlays and encoding"),
    samples: int = typer.Option(None, help="Samples (default: EEVEE 16, Cycles 32; 16 on a Mac)"),
    threads: int = typer.Option(0, help="CPU threads for Blender (0 = all / split between processes)"),
    jobs: int = typer.Option(None, help="Parallel Blender processes (default: 4 for a draft on ≥ 32 cores)"),
    engine: str = typer.Option("auto", help="auto (= cycles) | cycles | eevee"),
    tag: str = typer.Option("", help="Suffix for frames and video: frames-<tag>/, track-<name>-<tag>.mp4"),
    redo: str = typer.Option("", help="Re-render only the frames in these seconds: \"0-3,16-20\""),
):
    """Blender frames + overlays → videos/track-<name>.mp4 (a draft → videos/drafts/).
    Ctrl-C stops after the current frame, a second Ctrl-C at once."""
    from .render import pick_engine, render as run

    if samples is None:
        # EEVEE 16; Cycles 16 on the Mac (≈12 s/frame, 16 vs 32 is invisible: PSNR ≥ 40 dB), 32 elsewhere
        samples = 16 if pick_engine(engine) == "eevee" or sys.platform == "darwin" else 32
    run(_paths(trip, story), draft=draft, frames=frames, reuse_frames=reuse_frames, samples=samples, log=typer.echo,
        threads=threads, jobs=jobs, engine=engine, tag=tag, redo=redo)


@app.command()
def cover(
    trip: Path = TRIP,
    story: str = STORY,
    t: float = typer.Option(None, help="Second of the video; default: middle of the outro"),
    samples: int = typer.Option(64),
):
    """Cover: one full-size frame with its overlay → videos/track-<name>-cover.png."""
    from .render import cover as run

    run(_paths(trip, story), t=t, samples=samples, log=typer.echo)


@app.command("day-story")
def day_story(
    trip: Path = TRIP,
    day: int = typer.Option(..., help="Day number"),
    duration: float = typer.Option(20.0, help="Story length, s"),
    build: bool = typer.Option(True, help="Build the scene and the animation right away (needs bpy)"),
    standalone: bool = typer.Option(False, help="Not part of the seamless chain (day<NN>-standalone)"),
    suffix: str = typer.Option("", help="A variant of the same story to compare (day<NN>-<suffix>)"),
):
    """A story of one day: the route before it already drawn, the day ridden up close → story day<NN>.
    By default stories of one length chain into one film: each ends on the frame the next one starts with.
    Then: gpx2reel render <trip> --story day<NN> --draft."""
    from .story import make_day_story, summary

    st = make_day_story(_paths(trip), day, duration, build=build, chain=not standalone, suffix=suffix,
                        log=typer.echo)
    if build:
        typer.echo(summary(st))
    typer.echo(f"Story: {st.name}")


@app.command()
def stories(trip: Path = TRIP):
    """List the day stories built for this trip."""
    for name in _paths(trip).stories():
        typer.echo(name)


@app.command()
def compare(
    trip: Path = TRIP,
    story: str = STORY,
    frames: int = typer.Option(60, help="How many frames (evenly spread)"),
    samples: int = typer.Option(16),
):
    """The same frames in Cycles and EEVEE side by side with the time per frame → compare.mp4 (in the work dir)."""
    from .render import compare as run

    run(_paths(trip, story), frames=frames, samples=samples, log=typer.echo)


@app.command()
def profile(
    trip: Path = TRIP,
    story: str = STORY,
    engine: str = typer.Option("cycles", help="cycles | eevee"),
    samples: int = typer.Option(32),
    frames: int = typer.Option(5),
):
    """Measure how much of a frame goes into the scene update and how much into rendering."""
    from .render import profile as run

    run(_paths(trip, story), engine=engine, samples=samples, frames=frames, log=typer.echo)


@app.command()
def poi(
    trip: Path = TRIP,
    refresh: bool = typer.Option(False, help="Query OSM again (else the cached poi_raw.json)"),
    top: int = typer.Option(25, help="How many candidates of each kind to show"),
):
    """Label candidates along the route from OSM: peaks, volcanoes, towns → poi.json."""
    from .poi import run, table

    typer.echo(table(run(_paths(trip), refresh=refresh, log=typer.echo), top))


@app.command()
def landcover(trip: Path = TRIP, px_m: float = typer.Option(30.0, help="Pixel size for the main area, m")):
    """ESA WorldCover 10 m land cover → landcover/ (for style.terrain: stylized / hybrid)."""
    from .landcover import download

    download(_paths(trip), px_m=px_m, log=typer.echo)


@app.command()
def osm(trip: Path = TRIP):
    """Roads and rivers from OpenStreetMap → osm/vectors.json (for the stylized scene; needs pyosmium)."""
    from .osm import download

    typer.echo(f"→ {download(_paths(trip), log=typer.echo)}")


@app.command()
def presets(as_json: bool = typer.Option(False, "--json", help="Also the track.yaml schema with every field")):
    """List every preset track.yaml can use."""
    from .presets import ALL

    if as_json:
        from .storyboard import Storyboard

        typer.echo(json.dumps({**ALL, "track_yaml_schema": Storyboard.model_json_schema()}, ensure_ascii=False,
                              indent=2))
        return
    for group, items in ALL.items():
        typer.echo(f"\n{group}:")
        for k, v in items.items():
            typer.echo(f"  {k:<22} {v}")


@app.command()
def bike(
    out: Path = typer.Option(Path("avatar.glb"), help="Where to save the .glb"),
    color: str = typer.Option("periwinkle", help="Palette colour name or #RRGGBB"),
    size: str = typer.Option("M"),
    preview: Path = typer.Option(None, help="Render a PNG preview"),
    blend: Path = typer.Option(None, help="Save the .blend"),
    samples: int = typer.Option(48),
):
    """Model a Canyon Grizl procedurally (needs bpy)."""
    from .blender.bike import build_and_export

    build_and_export(out=out, color=color, size=size, preview=preview, blend=blend, samples=samples)
    typer.echo(f"→ {out}")
    if preview:
        typer.echo(f"→ {preview}")


if __name__ == "__main__":
    app()
