# Kyushu by bike, September–October 2026

Twelve riding days around Kyushu, Japan: Beppu → Aso → Takachiho → the Miyazaki coast → Kirishima → Kagoshima →
Amakusa → Unzen → Nagasaki, 823 km and 8758 m of climbing (device totals), with a bus, a train and a ferry between.

`tracks/` holds one GPX per day made with `gpx2reel sanitize --trim-m 300` from the original Garmin, bike
computer and app recordings: position, elevation and time only, thinned to 5 m, 300 m cut off each day's ends.
Re-ingested they read 804 km (the thinning and trimming) and less climb (computed, not the devices' totals).

## What the config shows

[`track.yaml`](track.yaml) is a 100 s Reels cut of the whole trip
([concepts](../../docs/configuration.md)):

- **Intro zoom** Japan → Kyushu → the route, and the finale back out with the route line (`intro.zoom`,
  `outro.zoom_out`, `outro.route_points`).
- **Highlights** with orbits and high-resolution terrain: Aso and Sakurajima with a steam plume, the Amakusa
  bridges, and a tight steep orbit over the Unzen Jigoku fumarole field (`orbit_km`, `orbit_pitch_deg`).
- **Beach sunrise** opening day 6 at Aoshima, with the red sun disc (`opening: sunrise_beach`,
  `style.sun_disc: hinomaru`).
- **Sunset closing** of day 10, moved from the finish to a beach nearby (`closing: sunset`, `closing_at`).
- **Gaps:** a bus and a train drawn along roads (`mode: road`), a ferry inside day 11 (`within_day: true`).
- **Labels:** peaks with elevations, towns, a wetland, a viewpoint, sights; start / finish names on the day
  cards.

## Reproduce

From the repo root. The first run downloads about 150 MB of elevation and imagery tiles into
`~/.cache/gpx2reel/tiles`, plus OSM data for the coasts; all outputs go to `examples/kyushu-2026/.work/` and
`videos/` (both ignored by git).

```bash
.venv/bin/gpx2reel ingest examples/kyushu-2026
.venv/bin/gpx2reel validate examples/kyushu-2026
.venv/bin/gpx2reel terrain examples/kyushu-2026
.venv/bin/gpx2reel poi examples/kyushu-2026
.venv/bin/gpx2reel scene examples/kyushu-2026
.venv/bin/gpx2reel timeline examples/kyushu-2026
.venv/bin/gpx2reel render examples/kyushu-2026 --draft
```

The draft (25 % size, 8 samples) lands in `videos/drafts/track-reel.mp4`. A final render (`render` without
`--draft`) takes hours on a CPU.

A 20 s story of one day, with the days before it already drawn:

```bash
.venv/bin/gpx2reel day-story examples/kyushu-2026 --day 11
.venv/bin/gpx2reel render examples/kyushu-2026 --story day11 --draft
```

## Renders

The rendered videos are not in git; they are attached to the
[`example-kyushu-2026` release](https://github.com/unintended/gpx2reel/releases/tag/example-kyushu-2026).

| Video | Shows |
|---|---|
| [Day 10](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day10.mp4) ([cover](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day10-cover.png)) | sunset closing on a beach |
| [Day 11](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day11.mp4) ([cover](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day11-cover.png)) | ferry inside a day, Unzen fumaroles, climb counter |
| [Day 12](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day12.mp4) ([cover](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day12-cover.png)) | the last day and the finale zooming out to Japan |
| [Day 6, draft](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day06.mp4) | beach sunrise opening |
| [Day 7, draft](https://github.com/unintended/gpx2reel/releases/download/example-kyushu-2026/track-day07-style-band-puck.mp4) | orbit around Sakurajima with the steam plume |

Finals are 1080×1920; drafts are rendered at 25 %.
