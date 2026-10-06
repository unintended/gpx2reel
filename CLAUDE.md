# gpx2reel — developer guide

Turns multi-day GPX / FIT rides into a vertical 3D video (1080×1920): a glowing marker rides the route over real
terrain with satellite imagery, leaving a trail, with per-day and total km overlays. Rendered with Blender 5 (`bpy`).

User docs: `README.md`, `docs/configuration.md` (track.yaml by topic), `examples/kyushu-2026/`. Private, machine-
or trip-specific notes go in `CLAUDE.local.md` (gitignored).

## Approach

Not an end-user app. **Scripts + a skill + a Claude session**:
- scripts in this repo do deterministic mechanics (parse, download, build scene, render);
- the conversation does the judgement: analyse the trip, pick highlights, write `track.yaml`, take the user's
  edits, run `validate` (skill `gpx-reel` in `.claude/skills/`);
- `track.yaml` (pydantic `Storyboard`) is the only contract between the conversation and the render. Presets come
  only from `gpx2reel presets`. Never write one-off code in a session — add features to the scripts.

Development, downloads and drafts run fine on a many-core Linux box without a GPU; final renders are Cycles
(Metal on a Mac, CPU elsewhere).

## Trip folder

Canonical spec: `docs/trip-layout.md`. Only `trip.py` (`TripPaths`) knows it — never build paths by hand.

```
<trip>/tracks/*.gpx|*.fit|*.zip        inputs (read-only)      <trip>/music/  <trip>/refs/
<trip>/track.yaml                      the config; paths in it are relative to the trip folder
<trip>/videos/track-<name>.mp4         finished renders, covers track-<name>-cover.png
<trip>/videos/drafts/track-<name>.mp4  drafts
<trip>/.work/track/build/              the main reel's state: route.json, terrain/, scene.blend, frames/, overlay/ …
<trip>/.work/track/stories/<name>/     a day story: track.yaml, tracks/ (symlinks), build/
~/.cache/gpx2reel/                     tiles and OSM extracts shared across trips
```

A *film* is the main reel (`name = reel`) or a day story. Trip-level commands (`ingest`, `terrain`, `poi`,
`landcover`, `osm`, `sanitize`, `day-story`) take the trip; per-film commands (`validate`, `scene`, `timeline`,
`overlays`, `render`, `cover`, `compare`, `profile`, `summary`) take the trip plus `--story <name>`.
Dev-only outputs (`compare.mp4`, `profile.png`, stills, frames, overlays) stay in the film's build dir.

## Pipeline

`ingest → terrain → [landcover → osm] → poi → scene → timeline → render`

- `ingest` — GPX, FIT (`fitdecode`) and Garmin Connect «export original» ZIPs; the same ride recorded / exported
  several times (>50 % time overlap) is grouped: a Garmin recording is primary (its numbers are what Garmin
  Connect shows), then another device (FIT), then an app export (Komoot's GPX re-smooths altitude: one day read
  1558 m vs 1723 m); what the primary missed at the start / end (a watch switched on late) is stitched on from
  another recording, and its climb is added. A day's climb is the device's session total (`ascent_source: device`),
  else computed (1 m hysteresis for FIT, 3 m for GPX); days by local date (timezonefinder), jitter / spike
  cleaning, gap detection (overnight < 3 km / transfer / missing), km, ascent, moving time, `preview.png`.
  Within ~1 % of raw distance on real tracks.
- `sanitize` — `sanitize.py`: one plain GPX per ride day from the ingested route (lat / lon / ele / time only),
  thinned with RDP over (x, y, ele) (`geo.rdp_mask`, distances to the segment so an out-and-back's turn stays),
  optional privacy trim of each day's ends. Re-ingests as the same days. The default 5 m costs ~1.4 % of the
  distance on winding roads (chords; 10 m: −2.4 %) and a computed climb below the device's: ingest smooths over
  7 *points*.
- `storyboard.py` — the schema, `plan_timing` (seconds per day by `pacing`: ∝ km, √km or equal, min 3 s),
  `validate`.
- `presets.py` — registry of cameras, terrain styles, material presets, effects, text styles.
- `terrain` — Terrarium DEM (z12 ≈ 38 m/px) + Esri World Imagery (z13) tiles, cached in `~/.cache/gpx2reel/tiles`,
  `Dem.sample(lat, lon)`. Buffer 30 km around the route (`BUFFER_KM`: the drone camera sees far); `intro.zoom`
  adds coarse context layers `ctx<i>/`. Terrarium carries bathymetry; the scene clamps the sea to 0.
- `blender/bike.py` — procedural Canyon Grizl CF SL, size M geometry from canyon.com; palette colours by name
  (`periwinkle` = albedo #57608B, calibrated so a lit render reads #8992C7 on screen). Wheels / cranks are parented
  to empties `Grizl.WheelFront`, `Grizl.WheelRear`, `Grizl.Cranks`. Origin on the ground under the BB, +Y forward,
  Z up. As `avatar.model: procedural:grizl` it reads poorly from 20 km; the default marker is a `puck`.
- `scene` — `world.py` (numpy, tested): terrain grid in local UTM (step 100 m, UV per vertex into the ortho), route
  path resampled every 20 m on the grid surface; `join_gaps` bridges `join` gaps on the ground and adds the draw
  coordinate `s` (km + bridged gaps — animation runs on s, overlays show km); arcs for `straight` gaps →
  `world.npz`. `blender/scene.py`: terrain + ortho; intro-zoom context layers (each dropped below the finer one,
  finer layers feathered and faded in by camera distance); trail = geometry nodes over the path line (keep
  s ≤ Progress, material via Set Material — slot materials do not reach generated faces); marker (object colour
  drives emission); checkpoints at day starts + finish → `scene.blend`. `--still` renders the overview.
- `timeline` — `timeline.py` (numpy, tested): shots (zoom intro → days split by highlights → join / arc gaps →
  outro, seconds from `plan_timing`) → per-frame marker, camera, trail progress / width, checkpoint pop-ins, day
  colour. The camera distance follows the on-screen speed (~9 km of route per second → ≈20–25 km away, `d_min` at
  highlights); marker, trail and checkpoints keep their on-screen size, sized by their own distance to the camera
  (a low sunrise camera stands next to them). Wide ↔ ride transitions are flown (`blend_pose`). Knobs:
  `timeline.CameraRig`. `blender/animate.py` keyframes it → `reel.blend`; `--stills N`. Cameras `flyover` /
  `reveal` fall back to chase with a warning.
- `overlays` — `overlay.py` (Pillow, tested): per-frame transparent PNGs: day / total km counters, an elevation
  profile strip, day cards («Day 2 · 22 September», km, climb, moving time), checkpoint pins, zoom labels, title,
  gap captions, outro totals, place labels (`labels`: peaks get a stick, towns a dot). `plan_places` lays labels
  out in one sequential pass — priority = config order, hysteresis for side / stick direction, hidden while the
  point sweeps fast, runs and gaps < 0.6 s smoothed away — so they don't flicker. 3D → screen via
  `overlay.project` (matches Blender's TRACK_TO camera to the pixel). Instagram-story safe zones (`SAFE_TOP` /
  `SAFE_BOTTOM`). Text speaks `style.language` (en / ru). `style.credits` draws the data attribution (Esri,
  Mapzen / AWS, OpenStreetMap, + ESA WorldCover for stylized / hybrid) small under the outro and on covers.
  Fonts are vendored in `assets/fonts/` (Inter; a Noto Sans JP subset — kana + selected kanji, re-subset to add
  kanji) so every machine renders the same.
- Highlight callouts (`overlay._draw_callout`, timings in `CALLOUT`): a ring at the projected point, a leader with
  an elbow, a rule, then the name and `caption` type themselves in; the callout follows the point while the camera
  orbits, picks the side with room, and the day card waits until the highlight is over. On a big climb the overlay
  counts the metres gained (`overlay.climbs`, ≥ 300 m, under the profile).
- `poi` — `poi.py`: Overpass (overpass-api.de is often busy → falls back to a mirror) or the local OSM extract
  around the simplified route: peaks / volcanoes ≤ 10 km, towns ≤ 4 km, villages at start / finish → ranked
  `poi.json`. The session picks the labels.
- `render` — `render.py`: Blender frames (`--draft`: 25 %, 8 samples, JPEG) + overlays → ffmpeg →
  `videos/track-<name>[-<tag>].mp4` (drafts in `videos/drafts/`); `--frames N` renders every k-th frame and
  encodes at fps/k (same length, choppy, fast); `--reuse-frames` re-does only overlays + encode; `--redo
  120-180,900-960` re-renders only those seconds; `music` is mixed in with fades. Frames are rendered in our own
  loop: Cycles swallows KeyboardInterrupt, so Ctrl-C stops after the current frame (twice = now). `--jobs N`
  splits frames between N Blender processes (default 4 for drafts on ≥ 32 cores: one process keeps only ~25 of
  64 cores busy on a 270×480 frame; 1.45 → 0.8 s/frame; ~6 GB RAM each). Final frames render in one process.
- Speed — `scene.fast_cycles` (4 bounces, no caustics, adaptive threshold 0.05); final default 32 samples
  (1.55× faster than 64 at PSNR ≥ 44 dB; 16 samples 2.2× faster at ≥ 40 dB, still invisible), 16 on a Mac.
  `--engine eevee` (`scene.setup_eevee`) needs a GPU (hangs without one) and was ~6× faster on Apple Silicon, but
  Metal kept producing artefacts (cyan / magenta materials, streaks on the sea, flickering night light), so
  `--engine auto` = Cycles. `gpx2reel compare` renders the same frames in both engines side by side with s/frame;
  `gpx2reel profile` splits a frame into scene update vs render (the update is ~0).
- `cover` — one full-size frame + overlay → `videos/track-<name>-cover.png` (default: middle of the outro).

## Light, sky, sea

- `sun.py`: NOAA solar position from the real date / time / place (frame time from the day's track points;
  `timeline.frame_times`), sky and sun colour by elevation. `style.day_night` adds a `night` shot between days
  (`night_s`): dusk, a moonlit night with the glowing trail, dawn. Rest days between rides pass as one night
  (`sun.one_night`): the clock runs to the first night and jumps to the last. `style.haze`: terrain materials
  share the `Haze` group (distance fog keyed to the camera distance).
- Sunrise opening — `days[n].opening: sunrise` (+ `opening_s`, `opening_caption`, `opening_note`): the night
  before ends 20 min before the real sunrise (`sun.sunrise_before`); the `opening` shot runs the clock to the ride
  start (lingering at sunrise, u^3.2), camera 160 m up behind the start looking at the sun. The disc (`SunDisk`,
  keyed along the sun direction from the camera, bloom makes it glow) is `style.sun_disc`: `natural` (golden) or
  `hinomaru` (the red disc of the Japanese flag), colours in `sun.SUNRISE_DISKS`. It shrinks to nothing below the
  horizon (else it shows through the terrain's soft edges at night) and only exists in sunrise / sunset shots.
- Beach sunrise — `opening: sunrise_beach`: `terrain` fetches a 3 km high-res patch + OSM coastline / buildings +
  WorldCover into `terrain/opening<day>/` (`coast.py`); `scene` builds palms along the coast and a grove around
  the camera lane, houses (OSM footprints + WorldCover built-up cells, none in the lane), a water square with a
  shore fade and foam bands (`blender/coast.py`). Camera 14 m up, 110 m behind the start, level at the sun; the
  last 1.3 s crane up and tilt onto the start (exempt from camera smoothing, it would lift off before the tilt).
  Pitfalls: the DEM's shallows poke through the water (`flatten_sea`), a wide shore fade turns a 1 km cove into
  blotches (`SHORE_FADE_M`), rocks punch holes (`ISLET_M`), the detail patch sits 0.5 m up (`WATER_Z`).
- Sunset closing — `days[n].closing: sunset` (+ `closing_s`): the clock time-lapses to the evening; the shot starts
  `sunset_span_deg` above the skyline the camera sees (`timeline._sunset_span`) and runs to `sunset_end_deg`
  (−5°): the disc sets behind the sea or a ridge at full size, the palette fades into the afterglow
  (`sun.AFTERGLOW_*`), and the camera holds on the shore (the sun must set before the turn to the night
  overview). Camera 35 m up over the water 520 m east of the finish. The look (`sun.SUNSET_*`): the whole sky
  orange, a white-hot disc, world `Ambient` 0.3 — camera and mirror rays see the full sky, diffuse light a third,
  so the shore is a silhouette and the water still mirrors gold. `closing_at: [lat, lon]` moves the sunset to a
  better spot: the camera flies there, stands low behind palms, the sun touches the sea
  (`sunset_above_sea_deg`); the set is the full beach set in `terrain/closing<day>s/`. The spot must lie on the
  base terrain (`terrain` widens the area: beyond the grid the timeline reads the edge's height and the camera
  hangs in the air). From a hill a bay doesn't fit a portrait frame with the sun — keep the camera low.
- Open water — Esri stitches the sea from tiles of different dates (hard colour steps). `scene.open_water`: out on
  the water the imagery gives way to its own colour blurred over water only (`SEA_BLUR_M`); shallows within
  `OPEN_SEA_SHORE_M` keep it. Sea vs land comes from the OSM coastline of the whole area (`terrain/coastline.json`,
  `coast.sea_mask`) — the DEM's 0–2 m over the sea made seams and stepped coasts; without it, the DEM.
- Per-frame material values live on objects (`scene.object_clock` drives a custom property read by an OBJECT
  Attribute node), not on node sockets — EEVEE on Metal flashed the object colour otherwise; renders keep GPU
  textures (`texture_time_out` 0).

## Features by config key

- Gaps: `join` (on the ground), `straight` (arc), `road` (`routing.py`: OSRM driving route, cached in
  `roads.json`, trail in `TRANSPORT_RGBA`), `ferry`, `skip`; `within_day: true` for a gap inside a day
  (`Storyboard.gap_modes()` keys them "in<day>").
- Highlights — `detail: high_res_dem` + lat/lon: a 10×10 km patch (z14 DEM, z16 imagery ≈ 2 m/px,
  `terrain/detail<i>/`) built at 15 m, cut into the base terrain, the path follows the finest grid
  (`world.Surface`); `orbit` circles at `CameraRig.orbit_dist` (4 km) or `orbit_km` / `orbit_pitch_deg` (a
  1.2 km orbit at 28° hides behind the next hill). Effects: `steam_plume` (volumetric column, 4D noise),
  `fumaroles` (small wisps; density per wisp — a plume's 0.008 makes a 20 m wisp invisible).
- `style.climb_weight`: ride time by effort, × (1 + w·grade) (`timeline._effort`).
- `style.trail`: `band` (default; flat lozenge `BAND_FLAT` over a dark translucent casing, like a road line),
  `ribbon` (zero-thickness strip), `tube`; flat ones are `FLAT_WIDEN`× wider. With a Z-up curve normal the
  profile's Y is *up*, so flat profiles run along X (`tests/test_trail.py`). `avatar.model: puck` — a bevelled
  chip with a white ring.
- Look — `scene.add_grade`: Blender 5 compositor (`scene.compositing_node_group`): bloom from the **Emit pass**
  only (`style.bloom`; terrain never blooms), S-curve + saturation (`style.contrast`), vignette from a packed image
  scaled to the render size (`style.vignette`). View transform stays Standard. Sky = horizon → zenith gradient
  (`Horizon` / `Zenith`, keyed from `sun.sky_look`); a `Moon` sun lamp keeps nights readable. Trail attribute `age`
  → the last `HEAD_KM` glow hotter; the trail runs through `Haze` so far days don't float over hazed terrain.
  Camera drift (`CameraRig.drift_*`) gives parallax without cuts; `outro_turn_deg` turns the outro slowly.
  `style.clouds`: a cumulus grid at `CLOUD_BASE_M` with shadows, cleared around highlights, fading with distance.

## Day stories and formats

- The main reel (`platform: instagram_reels` ≤ 180 s, `pacing: mixed` = √km so long and short days stay
  comparable) and **day stories**: `gpx2reel day-story <trip> --day N` builds story `dayNN` — tracks of days 1..N
  as symlinks, a config derived with `focus_day: N` (earlier days pre-drawn, only day N ridden, no nights, the
  intro zoom only on day 1), terrain linked; then `gpx2reel render <trip> --story dayNN --draft`.
- Stories chain by default (`story_chain`): stories of one length flow into one film — each ends at night
  (`sun.night_after`) on the whole ridden path, the next starts on that exact frame. The joint is neutral (the
  first frame is the Instagram thumbnail): in the last ~1.6 s the stats and credits fade and the trail turns grey
  (`past_mix` → `PastMix` in the `Trail` material, `focus` point attribute = the story's day).
  `--standalone` builds `dayNN-standalone`; `--suffix x` a variant `dayNN-x` to compare.

## Stylized mode

`style.terrain: satellite` (default) | `stylized` (land-cover palette) | `hybrid` (satellite + the same 3D layers).
Everything else works in every mode: the style texture feeds `Haze`, the detail patch gets its own blurred style
texture, trees and ribbons sit on `world.Surface`. Needs the `stylized` extra (pyosmium).
- `landcover` — ESA WorldCover 10 m v200 (public COGs on S3, windows read over HTTP with rasterio, overviews for
  coarse layers) → `landcover/{main,ctx1}.npz` (30 m main area, 600 m for the regional zoom level; the widest
  zoom level stays satellite, softened).
- `osm` — Geofabrik extract (smallest region covering the area; an island polygon may miss the sea part and pull
  in the whole country — multi-GB, cached in `~/.cache/gpx2reel/osm`) read with pyosmium + a key filter →
  `osm/vectors.json` (motorway…secondary roads, rivers) and `poi_osm.json` (peaks, waterfalls, lakes, viewpoints,
  sights, towns); `poi` reads the extract when it exists.
- `stylize.py` bakes the palette (+ elevation tint, low-frequency noise, water mask in alpha) into a texture
  aligned with the terrain grid. `blender/stylized.py`: the tree layer is **static** (camera-driven density made
  trees flicker) — ~75 k trees chosen once with a seed (22 % of forest points, 190 m ±20 %, none within 700 m of
  the route, probability fading out 18→30 km from it); the only animated thing is one Scale input that shrinks
  the layer as the camera pulls back (45→110 km). Roads / rivers = draped ribbons, width ∝ camera distance,
  fading out beyond 90 km. Knobs: constants at the top of `blender/stylized.py`.
- Open ends: no lake polygons from OSM, no buildings at this scale, 30 m land cover reads as soft blotches.

## Checking a look

Stills at 50 % from `reel.blend` via a small bpy script (frame_set + render), compared as before / after sheets;
`render --draft --frames 300` for motion. Scene + timeline rebuild takes ~25 s (satellite).

Ideas: photo cards from EXIF GPS / time, cuts on music beats, 1:1 / 16:9 formats from the same scene (the
timeline takes the aspect; outputs would need their own build dirs).

## Conventions

- Python ≥ 3.11 without Blender; **3.13 with `bpy~=5.2`** (bpy 5.x wheels are cp313 only):
  `uv venv -p 3.13 .venv`, `uv pip install -e ".[blender,stylized,dev]"`; headless bpy on Ubuntu needs `libgl1`
  (+ libxi6, libxkbcommon0, libsm6 …). Tests: `.venv/bin/python -m pytest -q` (107 passing).
- Blender 5 API: geometry-nodes modifier inputs are `mod.properties.inputs.<Socket_N>.value`, not
  `mod["Socket_N"]`. `ShaderNodeMix` has several sockets named A/B — use `scene.mix_rgba` (index-based); a name
  lookup hits the float ones silently. `Material.use_nodes` is deprecated (warning only).
- User-facing text (CLI help, logs, validation) is English; overlays speak `style.language`.
- Keep API keys in `.env`. License: GPL-3.0-or-later.
