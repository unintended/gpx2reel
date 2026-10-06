"""Where things live in a trip folder — the one module that knows the layout (docs/trip-layout.md).

A film is the main reel or a day story. A story keeps its own config, tracks (symlinks) and build under
`.work/track/stories/<name>/`; the main reel's config and tracks are the trip's own. Videos of both go to
`videos/` named `track-<name>`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TOOL = "track"
CONFIG = f"{TOOL}.yaml"
TRACK_SUFFIXES = (".gpx", ".fit", ".zip")
MAIN = "reel"


def story_name(day: int, chain: bool = True, suffix: str = "") -> str:
    """day12 (a seamless chain story, the default), day12-standalone, + -<suffix> for a variant."""
    return f"day{day:02d}" + ("" if chain else "-standalone") + (f"-{suffix}" if suffix else "")


@dataclass(frozen=True)
class TripPaths:
    trip: Path
    story: str | None = None          # a day story's name; None = the main reel

    def __post_init__(self):
        object.__setattr__(self, "trip", Path(self.trip).resolve())

    @property
    def work(self) -> Path:
        """The tool's working state; everything under it can be deleted."""
        return self.trip / ".work" / TOOL

    @property
    def home(self) -> Path:
        """Folder with the film's config and tracks/: the trip itself or the story's folder."""
        return self.work / "stories" / self.story if self.story else self.trip

    @property
    def tracks(self) -> Path:
        return self.home / "tracks"

    @property
    def config(self) -> Path:
        return self.home / CONFIG

    @property
    def build(self) -> Path:
        """The film's intermediate files: route.json, terrain/, scene.blend, frames/, overlay/ …"""
        return self.home / "build" if self.story else self.work / "build"

    @property
    def route(self) -> Path:
        return self.build / "route.json"

    @property
    def name(self) -> str:
        return self.story or MAIN

    @property
    def videos(self) -> Path:
        return self.trip / "videos"

    def video(self, draft: bool = False, tag: str = "") -> Path:
        name = f"{TOOL}-{self.name}" + (f"-{tag}" if tag else "") + ".mp4"
        return self.videos / "drafts" / name if draft else self.videos / name

    def cover(self) -> Path:
        return self.videos / f"{TOOL}-{self.name}-cover.png"

    def resolve(self, path: str) -> Path:
        """A path written in a config: relative to the trip folder."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.trip / p

    def for_story(self, name: str) -> "TripPaths":
        return TripPaths(self.trip, name)

    def stories(self) -> list[str]:
        d = self.work / "stories"
        return sorted(x.name for x in d.iterdir() if (x / CONFIG).exists()) if d.is_dir() else []

    def track_files(self) -> list[Path]:
        files = sorted(f for f in self.tracks.iterdir() if f.suffix.lower() in TRACK_SUFFIXES) \
            if self.tracks.is_dir() else []
        if not files:
            raise FileNotFoundError(f"no GPX / FIT / ZIP tracks in {self.tracks}")
        return files

    def storyboard(self, route: dict | None = None):
        """The film's config, or a skeleton for the route when there is none yet."""
        from .storyboard import load, skeleton

        if self.config.exists():
            return load(self.config)
        if route is None:
            from .ingest import load_route

            route = load_route(self.route)
        return skeleton(route)
