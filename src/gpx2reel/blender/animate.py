"""Keyframe timeline.py frames into build/scene.blend → build/reel.blend (+ key-frame stills)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..trip import TripPaths
from .scene import set_input


def _gauss_1d(v: np.ndarray, fps: int, sigma_s: float = 0.3) -> np.ndarray:
    from ..timeline import _gauss

    return _gauss(np.asarray(v, float), sigma_s * fps)


def _key_all(obj, data_path: str, values, setter) -> None:
    for f, v in enumerate(values, start=1):
        setter(v)
        obj.keyframe_insert(data_path, frame=f)


def _loc(o):
    return lambda v: setattr(o, "location", tuple(map(float, v)))


def _uniform_scale(o):
    return lambda v: setattr(o, "scale", (float(v),) * 3)


def apply_frames(frames: dict, fps: int) -> None:
    import bpy

    s = bpy.context.scene
    n = len(frames["t"])
    s.frame_start, s.frame_end = 1, n
    s.render.fps = fps
    O = bpy.data.objects

    cam = bpy.data.objects.new("Cam.Main", bpy.data.cameras.new("Cam.Main"))
    cam.data.lens = frames["lens_mm"]
    cam.data.sensor_fit = "VERTICAL"
    cam.data.clip_start = 1.0
    cam.data.clip_end = 1e7                           # intro zoom starts thousands of km away
    target = bpy.data.objects.new("Cam.Main.Target", None)
    for o in (cam, target):
        s.collection.objects.link(o)
    c = cam.constraints.new("TRACK_TO")
    c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    s.camera = cam
    _key_all(cam, "location", frames["cam"], _loc(cam))
    _key_all(target, "location", frames["target"], _loc(target))
    # clip_start follows the distance: keeps depth precision at 3000 km and at 1.5 km
    _key_all(cam.data, "clip_start", frames["cam_dist"] * 1e-3,
             lambda v: setattr(cam.data, "clip_start", max(1.0, float(v))))

    for name in ("Trail", "Trail.Ahead"):
        o = O.get(name)
        if o is None:
            continue
        p = set_input(o, "Progress", float(frames["s"][0]))
        _key_all(o, p, frames["s"], lambda v, o=o: set_input(o, "Progress", float(v)))
        p = set_input(o, "Width", float(frames["trail_width"][0]))
        _key_all(o, p, frames["trail_width"], lambda v, o=o: set_input(o, "Width", float(v)))

    marker = O.get("Marker")
    if marker is not None:
        lifted = frames["marker"] + np.column_stack([np.zeros((n, 2)), frames["marker_radius"]])
        _key_all(marker, "location", lifted, _loc(marker))
        _key_all(marker, "scale", frames["marker_radius"], _uniform_scale(marker))
        _key_all(marker, "color", frames["marker_rgba"], lambda v: setattr(marker, "color", tuple(map(float, v))))

    for k in range(frames["cp_scale"].shape[1]):
        o = O.get(f"Checkpoint.{k}")
        if o is not None:
            _key_all(o, "scale", frames["cp_scale"][:, k], _uniform_scale(o))

    bike = O.get("Grizl")
    if bike is not None:
        _key_all(bike, "location", frames["marker"], _loc(bike))
        _key_all(bike, "rotation_euler", frames["bike_rot_z"], lambda v: setattr(bike, "rotation_euler", (0, 0, float(v))))
        _key_all(bike, "scale", frames["bike_scale"], _uniform_scale(bike))
        for name, key in (("Grizl.WheelFront", "wheel"), ("Grizl.WheelRear", "wheel"), ("Grizl.Cranks", "crank")):
            o = O[name]
            _key_all(o, "rotation_euler", frames[key], lambda v, o=o: setattr(o, "rotation_euler", (float(v), 0, 0)))

    for o in O:                             # intro-zoom layers: fade in from 2× to 1× their framing distance
        if "fade_dist" in o:
            fd = float(o["fade_dist"])
            u = np.maximum.accumulate(np.clip((2 * fd - frames["cam_dist"]) / fd, 0, 1))   # once in, stays in
            a = u * u * (3 - 2 * u)
            if "solid_dist" in o:                    # hard edges near, feathered again on wide views
                sd = float(o["solid_dist"])
                k = np.clip((1.5 * sd - frames["cam_dist"]) / (0.5 * sd), 0, 1)
                solid = k * k * (3 - 2 * k)
            else:
                solid = (u >= 1).astype(float)                                            # edges solid after
            _key_all(o, "color", np.column_stack([solid, np.ones_like(a), np.ones_like(a), a]),
                     lambda v, o=o: setattr(o, "color", tuple(map(float, v))))

    if "sun_vec" in frames:
        _key_light(frames)

    m = bpy.data.materials.get("Trail")
    if m is not None and m.use_nodes and "PastMix" in m.node_tree.nodes and np.any(frames.get("past_mix", 0) > 0):
        sock = m.node_tree.nodes["PastMix"].outputs[0]
        for fr, v in enumerate(frames["past_mix"], start=1):
            sock.default_value = float(v)
            sock.keyframe_insert("default_value", frame=fr)

    if "fumarole_show" in frames:                         # fumaroles: only around their highlight
        for o in O:
            if o.name.startswith("Fumarole"):
                _key_all(o, "hide_render", frames["fumarole_show"] < 0.5,
                         lambda v, o=o: setattr(o, "hide_render", bool(v)))

    for m in bpy.data.materials:                        # coast sets: only around their opening / closing
        if m.name.startswith(("Coast.Water", "Coast.Surf")) and m.use_nodes and "Show" in m.node_tree.nodes:
            sock = m.node_tree.nodes["Show"].outputs[0]
            _key_all(sock, "default_value", frames["beach_show"], lambda v, s=sock: setattr(s, "default_value", float(v)))

    for o in O:                             # stylized roads / rivers: width follows the camera distance
        if "screen_width" in o and o.modifiers:
            sw = float(o["screen_width"])
            d = frames["cam_dist"]
            far = float(o.get("hide_beyond", 1e12))                # gone when the camera is far (intro zoom)
            k = np.clip((far - d) / (0.5 * far), 0, 1)
            w = d * sw * k * k * (3 - 2 * k)
            p = set_input(o, "Width", float(w[0]))
            _key_all(o, p, w, lambda v, o=o: set_input(o, "Width", float(v)))

    forest = O.get("Forest")                # static trees: only the whole layer scales away, smoothly
    if forest is not None and "layer_fade" in forest:
        lo, hi = forest["layer_fade"]
        k = np.clip((hi - frames["cam_dist"]) / (hi - lo), 0, 1)
        scale = _gauss_1d(k * k * (3 - 2 * k), fps)
        p = set_input(forest, "Scale", float(scale[0]))
        _key_all(forest, p, scale, lambda v: set_input(forest, "Scale", float(v)))

    gaps = O.get("Gaps")
    if gaps is not None:
        _key_all(gaps.data, "bevel_depth", frames["trail_width"] * 0.3,
                 lambda v: setattr(gaps.data, "bevel_depth", float(v)))


def _key_light(frames: dict) -> None:
    """Real sun, sky and haze per frame (see sun.py); haze distances follow the camera distance."""
    import bpy
    from mathutils import Vector

    disk = bpy.data.objects.get("SunDisk")
    if disk is not None:                       # far out along the sun direction, ~3° across (drama over realism)
        D = 150_000.0
        pos = frames["cam"] + frames["sun_vec"] * D
        _key_all(disk, "location", pos, _loc(disk))
        # only near / above the horizon: below it the disc drifted under the (finite, feathered) terrain and
        # showed through at night as a red ball crossing the frame
        k = np.clip((frames["sun_el"] + 1.5) / 2.0, 0, 1)
        k = frames["sun_disk_k"] if "sun_disk_k" in frames else k * k * (3 - 2 * k)
        _key_all(disk, "scale", D * math.tan(math.radians(1.5)) * k, _uniform_scale(disk))
        if "sun_disk_rgba" in frames:
            _key_all(disk, "color", frames["sun_disk_rgba"], lambda v: setattr(disk, "color", tuple(map(float, v))))
    sun = bpy.data.objects.get("Sun")
    if sun is not None:
        _key_all(sun, "rotation_euler", frames["sun_vec"],
                 lambda v: setattr(sun, "rotation_euler", Vector(tuple(map(float, v))).to_track_quat("Z", "Y").to_euler()))
        _key_all(sun.data, "energy", frames["sun_energy"], lambda v: setattr(sun.data, "energy", float(v)))
        _key_all(sun.data, "color", frames["sun_color"], lambda v: setattr(sun.data, "color", tuple(map(float, v))))
    moon = bpy.data.objects.get("Moon")
    if moon is not None and "moon_energy" in frames:
        _key_all(moon.data, "energy", frames["moon_energy"], lambda v: setattr(moon.data, "energy", float(v)))
    world = bpy.context.scene.world
    if world is not None and world.use_nodes:
        nodes = world.node_tree.nodes
        bg = nodes["Background"]
        st = bg.inputs["Strength"]
        zen = nodes["Zenith"].outputs[0] if "Zenith" in nodes else bg.inputs["Color"]
        hor = nodes["Horizon"].outputs[0] if "Horizon" in nodes else None
        if "Mirror" in nodes and "mirror" in frames:
            sock = nodes["Mirror"].outputs[0]
            _key_all(sock, "default_value", frames["mirror"], lambda v, s=sock: setattr(s, "default_value", float(v)))
        if "Ambient" in nodes and "ambient" in frames:
            sock = nodes["Ambient"].outputs[0]
            _key_all(sock, "default_value", frames["ambient"], lambda v, s=sock: setattr(s, "default_value", float(v)))
        horizon = frames.get("horizon_color", frames["sky_color"])
        for f, (rgb, hz, k) in enumerate(zip(frames["sky_color"], horizon, frames["sky_strength"]), start=1):
            zen.default_value = (*map(float, rgb), 1.0)
            zen.keyframe_insert("default_value", frame=f)
            if hor is not None:
                hor.default_value = (*map(float, hz), 1.0)
                hor.keyframe_insert("default_value", frame=f)
            st.default_value = float(k)
            st.keyframe_insert("default_value", frame=f)
    ng = bpy.data.node_groups.get("Haze")
    if ng is not None:
        d = frames["cam_dist"]
        if "horizon_color" in frames:           # aerial perspective fades towards the horizon colour
            haze = np.clip(frames["horizon_color"] * 0.97, 0, 1)
        else:
            haze = np.clip(frames["sky_color"] * 0.5 + 0.45 * np.minimum(frames["sky_strength"], 0.7)[:, None], 0, 1)
        for name, vals in (("Start", d * 0.8), ("Scale", d * 3.0), ("Amount", np.full(len(d), frames.get("haze", 0.6)))):
            sock = ng.nodes[name].outputs[0]
            for f, v in enumerate(vals, start=1):
                sock.default_value = float(v)
                sock.keyframe_insert("default_value", frame=f)
        sock = ng.nodes["HazeColor"].outputs[0]
        for f, rgb in enumerate(haze, start=1):
            sock.default_value = (*map(float, rgb), 1.0)
            sock.keyframe_insert("default_value", frame=f)


def build_reel(p: TripPaths, stills: int = 0, samples: int = 32, log=print) -> Path:
    """build/scene.blend + storyboard → build/reel.blend; stills = number of preview frames to render."""
    import bpy

    from ..ingest import load_route
    from ..timeline import build_frames, plan_shots, shot_table
    from ..world import load_world
    from .bike import hex_to_rgba

    build = p.build
    route = load_route(p.route)
    sb = p.storyboard(route)
    grid, path, arcs, meta = load_world(build / "world.npz")
    stale = []
    if abs(meta["exaggeration"] - sb.style.exaggeration) > 1e-9:
        stale.append("style.exaggeration")
    if meta.get("day_colors") != sb.style.colors_for(route["totals"]["days"]):
        stale.append("route colors")
    if [z["extent_m"] for z in meta.get("zoom", [])] != [z.extent_km * 1000 for z in sb.intro.zoom]:
        stale.append("intro.zoom")
    if stale:
        raise RuntimeError(f"{', '.join(stale)} changed since scene — rerun `gpx2reel scene`")

    shots, warns = plan_shots(sb, route, path, arcs, pois=meta.get("highlights"))
    for w in warns:
        log(f"⚠ {w}")
    log(shot_table(shots))
    res = tuple(sb.resolution)
    from ..timeline import day_clock_of
    from ..world import route_center

    frames = build_frames(shots, path, grid, meta["lift_m"], sb.fps, res[0] / res[1], zoom=meta.get("zoom"),
                          climb_weight=sb.style.climb_weight,
                          checkpoints=meta.get("checkpoints"),
                          day_rgba={d: hex_to_rgba(h) for d, h in enumerate(meta["day_colors"], start=1)},
                          day_clock=day_clock_of(route) if sb.style.day_night else None, latlon=route_center(route),
                          chain=sb.story_chain, sun_disc=sb.style.sun_disc)
    frames["haze"] = sb.style.haze
    np.savez_compressed(build / "timeline.npz", **{k: v for k, v in frames.items() if isinstance(v, np.ndarray)})
    (build / "timeline.json").write_text(json.dumps({
        "fps": sb.fps, "resolution": list(res), "lens_mm": frames["lens_mm"], "chain": sb.story_chain,
        "shots": [{"kind": s.kind, "t0": s.t0, "t1": s.t1, "day": s.day, "label": s.label,
                   **({"target": [float(v) for v in s.target]} if s.target is not None else {})} for s in shots],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    bpy.ops.wm.open_mainfile(filepath=str((build / "scene.blend").resolve()))
    apply_frames(frames, sb.fps)
    bpy.context.scene.cycles.samples = samples
    out = build / "reel.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()))
    log(f"{len(frames['t'])} frames → {out}")

    if stills:
        sdir = build / "stills"
        sdir.mkdir(exist_ok=True)
        for old in sdir.glob("frame_*.png"):
            old.unlink()
        s = bpy.context.scene
        for f in np.linspace(1, len(frames["t"]), stills + 2)[1:-1].round().astype(int):
            s.frame_set(int(f))
            s.render.filepath = str((sdir / f"frame_{f:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            log(f"→ {s.render.filepath}")
    return out
