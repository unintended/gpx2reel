"""Final assembly: Blender frames (reel.blend) + Pillow overlays → ffmpeg → videos/track-<name>.mp4
(drafts: videos/drafts/).

`frames=N` renders every k-th frame only (k ≈ total / N) and encodes at fps / k: same duration, choppy,
k times faster — for drafts. Frames and overlays are numbered 1..N in both cases.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import subprocess
from pathlib import Path

from .trip import TripPaths

DRAFT_PERCENT = 25
DRAFT_SAMPLES = 8


def frames_dir(p: TripPaths, draft: bool, tag: str = "") -> Path:
    return p.build / (("draft" if draft else "frames") + (f"-{tag}" if tag else ""))


def pick_frames(total: int, frames: int | None) -> tuple[list[int], int]:
    """Blender frame numbers (1-based) to render and the step between them."""
    step = max(1, round(total / frames)) if frames else 1
    return list(range(1, total + 1, step)), step


class _StopOnCtrlC:
    """Cycles swallows KeyboardInterrupt inside a frame; stop between frames instead (twice = now)."""

    def __init__(self, log):
        self.stop, self.log = False, log

    def __enter__(self):
        self.prev = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, *_):
        if self.stop:
            os._exit(130)
        self.stop = True
        self.log("\nStopping after the current frame (Ctrl-C again to stop now)")

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self.prev)


def _open_reel(p: TripPaths, draft: bool, samples: int, threads: int, engine: str = "cycles"):
    import bpy

    from .blender.scene import fast_cycles, setup_eevee, use_best_device

    bpy.ops.wm.open_mainfile(filepath=str(p.build / "reel.blend"))
    s = bpy.context.scene
    s.render.resolution_percentage = DRAFT_PERCENT if draft else 100
    s.render.use_persistent_data = True                  # terrain + textures are synced once
    if engine == "eevee":
        setup_eevee(s, DRAFT_SAMPLES if draft else samples)
        device = "EEVEE (GPU)"
        sysp = bpy.context.preferences.system         # never free GPU textures mid-render (Blender does it
        sysp.texture_time_out = 0                     # every minute by default; a frame after it could lose them)
    else:
        fast_cycles(s)
        s.cycles.samples = DRAFT_SAMPLES if draft else samples
        device = use_best_device(s)
    if threads:
        s.render.threads_mode, s.render.threads = "FIXED", threads
    s.render.image_settings.file_format = "JPEG" if draft else "PNG"
    if not draft:
        s.render.image_settings.compression = 0      # frames are temporary: saving fast beats small files
    if engine == "eevee" or sys.platform == "darwin":
        s.render.compositor_device = "GPU"            # bloom / grade on the GPU as well
    return s, device


def _render_items(s, out: Path, ext: str, items: list[tuple[int, int]], guard=None, log=None) -> None:
    """items: (output number, Blender frame)."""
    import time

    import bpy

    t0 = time.time()
    for n, (k, f) in enumerate(items, start=1):
        s.frame_set(f)
        s.render.filepath = str(out.resolve() / f"f_{k:04d}.{ext}")
        bpy.ops.render.render(write_still=True)
        if log and (n == 1 or n % 25 == 0 or n == len(items)):
            el = time.time() - t0
            log(f"  frame {n}/{len(items)}, {el / n:.1f} s/frame, ~{el / n * (len(items) - n) / 60:.0f} min left")
        if guard is not None and guard.stop:
            raise SystemExit(130)


def default_jobs(draft: bool) -> int:
    """Small draft frames leave most of a big CPU idle: split the frames between processes (not on the Mac GPU)."""
    import os
    import sys

    cores = os.cpu_count() or 1
    return 4 if draft and sys.platform != "darwin" and cores >= 32 else 1       # ~6 GB RAM per process


def render_frames(p: TripPaths, draft: bool, samples: int, sel: list[int], log=print, threads: int = 0,
                  jobs: int = 1, engine: str = "cycles", tag: str = "") -> Path:
    import time

    t_start = time.time()
    out = frames_dir(p, draft, tag)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ext = "jpg" if draft else "png"
    items = list(enumerate(sel, start=1))
    if jobs <= 1:
        s, device = _open_reel(p, draft, samples, threads, engine)
        log(f"Rendering {len(sel)} of {s.frame_end} frames, {s.render.resolution_x * s.render.resolution_percentage // 100}"
            f"×{s.render.resolution_y * s.render.resolution_percentage // 100}, {s.cycles.samples} samples, {device} → {out}")
        with _StopOnCtrlC(log) as guard:
            _render_items(s, out, ext, items, guard, log)
    else:
        _render_parallel(p, draft, samples, items, out, ext, threads, jobs, log, engine, tag)
    (out / "frames.json").write_text(json.dumps({"blender_frames": sel, "s_per_frame": (time.time() - t_start) / max(len(sel), 1)}))
    return out


def _render_parallel(p: TripPaths, draft: bool, samples: int, items: list, out: Path, ext: str, threads: int,
                     jobs: int, log, engine: str = "cycles", tag: str = "") -> None:
    """jobs Blender processes, each a contiguous share of the frames and of the cores."""
    import os
    import subprocess
    import sys
    import time

    threads = threads or max(1, (os.cpu_count() or jobs) // jobs)
    chunks = [items[i * len(items) // jobs:(i + 1) * len(items) // jobs] for i in range(jobs)]
    procs = []
    for j, chunk in enumerate(chunks):
        spec = out / f".items_{j}.json"
        spec.write_text(json.dumps(chunk))
        cmd = [sys.executable, "-m", "gpx2reel.render", "--worker", str(p.trip), str(spec), "--story", p.story or "",
               "--threads", str(threads), "--samples", str(samples), "--engine", engine, "--tag", tag] \
            + (["--draft"] if draft else [])
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out / f".worker_{j}.log", "w")))
    log(f"Rendering {len(items)} frames in {jobs} processes × {threads} threads → {out}")
    t0, last = time.time(), 0
    try:
        with _StopOnCtrlC(log) as guard:
            while any(p.poll() is None for p in procs):
                time.sleep(2)
                if guard.stop:
                    raise SystemExit(130)
                done = len(list(out.glob(f"f_*.{ext}")))
                if done and time.time() - last > 20:
                    last = time.time()
                    el = time.time() - t0
                    log(f"  frames {done}/{len(items)}, {el / done:.2f} s/frame, ~{el / done * (len(items) - done) / 60:.0f} min left")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    bad = [j for j, p in enumerate(procs) if p.returncode != 0]
    if bad:
        raise RuntimeError(f"render processes {bad} failed, see {out}/.worker_*.log")
    for f in out.glob(".items_*.json"):
        f.unlink()


def encode(p: TripPaths, draft: bool, fps: float, size: tuple[int, int], music: Path | None, log=print,
           tag: str = "", out: Path | None = None) -> Path:
    fdir = frames_dir(p, draft, tag)
    ext = "jpg" if draft else "png"
    w, h = (size[0] * DRAFT_PERCENT // 100, size[1] * DRAFT_PERCENT // 100) if draft else size
    out = out or p.video(draft, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-framerate", f"{fps:g}", "-i", str(fdir / f"f_%04d.{ext}"),
           "-framerate", f"{fps:g}", "-i", str(p.build / "overlay" / "o_%04d.png")]
    if music:
        cmd += ["-i", str(music)]
    cmd += ["-filter_complex", f"[1]scale={w}:{h}:flags=lanczos[o];[0][o]overlay=format=auto:shortest=1,"
                               f"format=yuv420p[v]", "-map", "[v]", "-r", "30"]
    if music:
        cmd += ["-map", "2:a", "-af", "afade=t=in:d=0.5,areverse,afade=t=in:d=2,areverse", "-shortest",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-crf", "23" if draft else "16", "-preset", "medium", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    log(f"→ {out}")
    return out


def pick_engine(engine: str) -> str:
    """auto: Cycles everywhere. EEVEE on the Mac was ~6× faster but kept failing in new ways from day 11 on
    (cyan / magenta materials, streaks on the sea, flickering light at night) — and Cycles is what the drafts
    and stills are checked in. EEVEE stays available with --engine eevee."""
    return "cycles" if engine == "auto" else engine


def render(p: TripPaths, draft: bool = False, frames: int | None = None, reuse_frames: bool = False,
           samples: int = 32, log=print, threads: int = 0, jobs: int | None = None, engine: str = "auto",
           tag: str = "", redo: str = "", out: Path | None = None) -> Path:
    """redo: "a-b,c-d" seconds — re-render only those frames into the existing frames, then overlays + encode.
    out: the video's path instead of videos/ (dev renders such as compare keep theirs in the build)."""
    from .overlay import render_overlays

    engine = pick_engine(engine)
    tl = json.loads((p.build / "timeline.json").read_text(encoding="utf-8"))
    total = int(round(tl["shots"][-1]["t1"] * tl["fps"]))
    sel, step = pick_frames(total, frames)
    m = p.storyboard().music
    music = p.resolve(m) if m else None
    fdir = frames_dir(p, draft, tag)
    done = fdir / "frames.json"
    if done.exists():
        have = json.loads(done.read_text())["blender_frames"]
    else:                                   # rendered before frames.json existed: trust a full set
        have = sel if fdir.exists() and len(list(fdir.glob("f_*"))) == len(sel) else None
    if redo and have == sel:
        spans = [tuple(float(x) for x in part.split("-")) for part in redo.split(",") if part]
        fps = tl["fps"]
        items = [(k, f) for k, f in enumerate(sel, start=1)
                 if any(a <= (f - 1) / fps <= b for a, b in spans)]
        log(f"Re-rendering {len(items)} frames in {redo} s")
        s_, device = _open_reel(p, draft, samples, threads, engine)
        with _StopOnCtrlC(log) as guard:
            _render_items(s_, fdir, "jpg" if draft else "png", items, guard, log)
    elif not reuse_frames or have != sel:
        if reuse_frames:
            log("Existing frames were rendered with a different --frames — rendering again")
        njobs = jobs if jobs is not None else (1 if engine == "eevee" else default_jobs(draft))
        render_frames(p, draft, samples, sel, log, threads, njobs, engine, tag)
    log("Overlays…")
    shutil.rmtree(p.build / "overlay", ignore_errors=True)
    render_overlays(p, frames=[f - 1 for f in sel], log=log)
    return encode(p, draft, tl["fps"] / step, tuple(tl["resolution"]), music, log, tag=tag, out=out)


def cover(p: TripPaths, t: float | None = None, samples: int = 32, log=print) -> Path:
    """One full-size frame with its overlay → videos/track-<name>-cover.png (default: middle of the outro — whole
    route + totals)."""
    import bpy
    from PIL import Image

    from .blender.scene import use_best_device
    from .overlay import draw_frame, load_overlay_data

    build = p.build
    tl = json.loads((build / "timeline.json").read_text(encoding="utf-8"))
    last = tl["shots"][-1]
    t = (last["t0"] + last["t1"]) / 2 if t is None else t
    frame = int(round(t * tl["fps"])) + 1
    bpy.ops.wm.open_mainfile(filepath=str(build / "reel.blend"))
    s = bpy.context.scene
    s.render.resolution_percentage = 100
    from .blender.scene import fast_cycles

    fast_cycles(s)
    s.cycles.samples = samples
    use_best_device(s)
    s.frame_set(frame)
    raw = build / "cover_raw.png"
    s.render.image_settings.file_format = "PNG"
    s.render.filepath = str(raw.resolve())
    bpy.ops.render.render(write_still=True)
    data = load_overlay_data(p)
    out = p.cover()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(Image.open(raw).convert("RGBA"), draw_frame(data, frame - 1, credits=True)).convert("RGB").save(out)
    log(f"→ {out}")
    return out


if __name__ == "__main__":                     # render worker: python -m gpx2reel.render --worker …
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("TRIP", "ITEMS"), required=True)
    ap.add_argument("--story", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--engine", default="cycles")
    ap.add_argument("--tag", default="")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    paths, spec = TripPaths(Path(a.worker[0]), a.story or None), Path(a.worker[1])
    scene, _ = _open_reel(paths, a.draft, a.samples, a.threads, a.engine)
    _render_items(scene, frames_dir(paths, a.draft, a.tag), "jpg" if a.draft else "png",
                  [tuple(x) for x in json.loads(spec.read_text())])


def profile(p: TripPaths, engine: str = "cycles", samples: int = 32, frames: int = 5, log=print) -> None:
    """Where a full-size frame's time goes: scene update (frame_set) vs render + save."""
    import time

    import bpy

    s, device = _open_reel(p, False, samples, 0, engine)
    s.render.filepath = str(p.build / "profile.png")
    step = max(1, s.frame_end // (frames + 1))
    for n, f in enumerate(range(step, s.frame_end, step)[:frames]):
        t0 = time.time()
        s.frame_set(f)
        bpy.context.evaluated_depsgraph_get().update()      # force the scene update the render would do
        t1 = time.time()
        bpy.ops.render.render(write_still=True)
        t2 = time.time()
        log(f"  frame {f}: prep {t1 - t0:.2f} s, render+write {t2 - t1:.2f} s" + ("  (first: compiling)" if n == 0 else ""))
    log(f"Device: {device}")


def _label_png(text: str, width: int, path: Path) -> Path:
    """Engine label as a transparent PNG (ffmpeg builds without freetype have no drawtext)."""
    from PIL import Image, ImageDraw, ImageFont

    fnt = ImageFont.truetype(str(Path(__file__).parent / "assets" / "fonts" / "Inter-SemiBold.otf"), 26)
    img = Image.new("RGBA", (width, 80), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=fnt)
    x0 = (width - w) / 2
    d.rounded_rectangle((x0 - 12, 18, x0 + w + 12, 62), radius=8, fill=(0, 0, 0, 140))
    d.text((x0, 50), text, font=fnt, fill=(255, 255, 255, 255), anchor="ls")
    img.save(path)
    return path


def compare(p: TripPaths, frames: int = 60, samples: int = 16, engines: tuple[str, ...] = ("cycles", "eevee"),
            reuse: bool = True, log=print) -> Path:
    """The same frames in each engine, then side by side with the engine and seconds per frame on top →
    build/compare.mp4 (the per-engine videos stay as build/compare-<engine>.mp4; reuse=True keeps existing ones)."""
    import time

    tl = json.loads((p.build / "timeline.json").read_text(encoding="utf-8"))
    w, h = tl["resolution"]
    outs, labels = [], []
    for e in engines:
        mp4, meta = p.build / f"compare-{e}.mp4", frames_dir(p, False, e) / "frames.json"
        if not (reuse and mp4.exists() and meta.exists()):
            render(p, frames=frames, samples=samples, engine=e, tag=e, log=log, jobs=1, out=mp4)
        spf = json.loads(meta.read_text()).get("s_per_frame")
        outs.append(mp4)
        labels.append(f"{e.upper()} · {samples} samples" + (f" · {spf:.1f} s/frame" if spf else ""))
    parts, cmd = [], ["ffmpeg", "-y", "-loglevel", "error"]
    for o in outs:
        cmd += ["-i", str(o)]
    for k, lab in enumerate(labels):
        cmd += ["-i", str(_label_png(lab, w // 2, p.build / f".label_{k}.png"))]
    n = len(outs)
    for k in range(n):
        parts.append(f"[{k}:v]scale={w // 2}:{h // 2}[s{k}];[s{k}][{n + k}:v]overlay=0:0[v{k}]")
    stack = "".join(f"[v{k}]" for k in range(n)) + f"hstack=inputs={n}[v]" if n > 1 else "[v0]null[v]"
    out = p.build / "compare.mp4"
    cmd += ["-filter_complex", ";".join(parts) + ";" + stack, "-map", "[v]", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    log(f"→ {out}")
    return out
