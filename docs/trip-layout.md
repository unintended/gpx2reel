# Trip folder layout

One folder holds one trip: the raw inputs from every device, each tool's config
and authored state, the finished videos, and each tool's working state. Tools that follow this
layout don't know about each other — they share only these conventions.

```
2026-09-kyushu/              any name; YYYY-MM-place sorts well
  tracks/                    *.fit *.gpx *.zip — as exported from the device or app
  footage/<device>/          insta360/, photos/, gopro/ … — may be symlinks to a card or disk
  music/                     audio tracks
  refs/                      reference images
  <tool>.yaml                each tool's own config (track.yaml, 360.yaml)
  <tool>/                    what a tool writes on the user's or agent's behalf
                             (notes, edits); never regenerated, never deleted
  videos/                    finished renders: <tool>-<name>.mp4 and stills
    drafts/                  drafts and previews, same naming
  .work/<tool>/              intermediate state; safe to delete, rebuilt on the next run
```

## Rules

- **Inputs are read-only.** A tool never writes into `tracks/`, `footage/`,
  `music/` or `refs/`.
- **A tool writes only** to `.work/<tool>/`, `<tool>/` and `videos/`.
  Everything under `.work/` can be deleted; deleting it costs only render
  time. `<tool>/` holds work that can't be rebuilt, so it is never deleted.
- **Outputs are prefixed with the tool's name** (`track-day03.mp4`,
  `360-descents.mp4`), so tools never collide and `ls videos/` groups them.
- **Paths inside configs are relative to the trip folder** (`music/tonna.mp3`),
  so the folder can be moved or copied as a whole.
- **Caches shared across trips** (map tiles, OSM extracts, models) live in the
  user cache dir, `~/.cache/<tool>/`, not in the trip.
- There is no shared manifest: title, language and the like stay in each
  tool's config.

## Tools

| Tool    | Config       | Authored | Reads                          | Work dir       |
|---------|--------------|----------|--------------------------------|----------------|
| `track` | `track.yaml` | —        | `tracks/`, `music/`, `refs/`   | `.work/track/` |
| `360`   | `360.yaml`   | `360/`   | `footage/insta360/`, `tracks/`, `music/` | `.work/360/` |
