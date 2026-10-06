---
name: gpx-reel
description: Turn a trip folder of GPX/FIT rides into a vertical 3D route video with gpx2reel — ingest, track.yaml, places, terrain, scene, timeline, draft render, day stories. Use when the user drops trip tracks and asks for a reel / story / video of a ride, or invokes /gpx-reel <trip folder>.
---

# gpx-reel

The conversation does the judgement (what the story is), the scripts do the mechanics. Never write one-off
code in the session: if something is missing, add it to gpx2reel. Architecture, knobs and pitfalls live in
`CLAUDE.md`; everything `track.yaml` may reference is listed by `gpx2reel presets`.

Run commands from the repo with `.venv/bin/gpx2reel`. `<trip>` is the trip folder (`docs/trip-layout.md`):
tracks in `<trip>/tracks/`, the config `<trip>/track.yaml`, videos land in `<trip>/videos/` (drafts in
`videos/drafts/`), everything intermediate in `<trip>/.work/track/`. Never write into `tracks/`, `music/`,
`refs/` or `footage/`.

## 1. Understand the trip

1. `gpx2reel ingest <trip>` — read the summary: days, km, climb, gaps, notes (dropped duplicate recordings,
   spikes). Look at `.work/track/build/preview.png`.
2. For every gap between days ask what happened only if the numbers don't say it: an overnight near the end
   point → `join` 0 s; bus / car → `road`, train / flight → `straight`, ferry → `ferry`, with a caption.
   A watch switched on late is fixed by adding the bike computer's FIT to `tracks/`, not by a gap mode.

## 2. track.yaml

1. `gpx2reel skeleton <trip> --title "…"` if there is no `track.yaml`.
2. Decide and write:
   - `platform` (Instagram story = 60 s), `duration_s`;
   - `intro.zoom`: country → region → route (`extent_km`, `center`, `label`), intro 5–6 s;
   - `style`: `color_mode` single / per_day, `day_night`, `haze`, `trail`, `ahead`, `language`, `sun_disc`,
     `credits`; look knobs `bloom`, `contrast`, `vignette`, `clouds` (defaults are the tuned look — change only
     on request);
   - `music`: a path relative to the trip folder (`music/<file>`);
   - gaps: modes, durations, captions;
   - highlights: 1–2 hero places at most, with `lat`/`lon`, `camera: orbit`, `detail: high_res_dem`,
     effects that are true to the place (a steaming crater gets `steam_plume`).
3. Day `start` / `finish` and labels: a town or a landmark in 1–2 words, never a hotel or a campsite — long names
   crowd the day card and the map (`validate` warns above 20 characters).
4. `gpx2reel poi <trip>` → pick 5–10 `labels` that tell the story: start / finish towns, the peaks the ride
   passes, famous sights. Names in `style.language`. Order = priority when labels collide.
5. `gpx2reel validate <trip>` must print ✓; read the timing line (every day should get ≥ 5 s).

## 3. Build and check

1. `gpx2reel terrain <trip>` (DEM + imagery, intro-zoom context, highlight detail patches, coast sets).
2. `gpx2reel scene <trip>` → `gpx2reel timeline <trip> --stills 8`. Look at `.work/track/build/stills/`: marker
   and trail readable, labels not colliding, night / dawn right, highlight framed. Fix via `track.yaml` or, if the
   tool is wrong, in the code (with a test).
3. `gpx2reel render <trip> --draft --frames 300` for a quick look, `--draft` for a smooth one →
   `videos/drafts/track-reel.mp4`.
4. `gpx2reel cover <trip>` → `videos/track-reel-cover.png`.

## Day stories (during a long trip)

1. `gpx2reel ingest <trip>` with the new day's tracks, then `gpx2reel day-story <trip> --day N` → story `dayNN`
   (seamless chain by default; `--standalone`, `--suffix x` for variants). `gpx2reel stories <trip>` lists them.
2. Every per-film command takes `--story dayNN`: `validate`, `timeline`, `render --draft`, `cover`, …
   The story's own config is `.work/track/stories/dayNN/track.yaml`; edit the trip's `track.yaml` and rebuild
   the story to keep them in step.
3. Final: `gpx2reel render <trip> --story dayNN` → `videos/track-dayNN.mp4`.

## 4. Hand over

Give the user the paths (draft, cover, stills) and what to decide next. Final renders are Cycles; on a Mac
(Metal) the first frame compiles kernels for ~1.5 min. To render on another machine copy the trip folder
without what render doesn't need: `scene.blend`, `*.blend1`, `*.npy`, and the `draft*/`, `frames*/`, `overlay/`,
`stills/` folders under `.work/track/`.
