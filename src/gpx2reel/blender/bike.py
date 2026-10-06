"""Procedural Canyon Grizl (gravel bike) built from real frame geometry.

Coordinate system: X = right, Y = forward, Z = up, metres. Origin on the ground,
under the bottom bracket. Wheels and cranks are parented to empties
(Grizl.WheelFront / Grizl.WheelRear / Grizl.Cranks) so the scene script can spin them.

Geometry: Canyon Grizl CF SL, size M, from canyon.com geometry sheet.
Colours are approximations of Canyon colourway names — pass #RRGGBB to override.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- data

GEOMETRY = {
    # mm / degrees — Grizl CF SL (canyon.com, frameset geometry sheet)
    "M": dict(stack=579, reach=402, hta=72.25, sta=73.5, st=522, ht=138, cs=435, wb=1037, bb_drop=75),
}

PALETTE = {
    # approximate RGB of Canyon colourways (by name) — tune from a photo if needed
    "periwinkle": "#57608B",        # matte periwinkle: albedo calibrated so a lit render reads #8992C7 on screen
    "floating_violet": "#9480CC",   # Grizl CF 6 / CF 7
    "lavender_gelato": "#C9B9E6",   # Grizl CF 8 Di2
    "purple_punch": "#7A3E9D",      # Grizl:ONfly
    "off_berry": "#5E2A4E",         # Grizl 5 / 6
    "stealth": "#1C1C1E",
}

RIM_R = 0.311          # ETRTO 622 bead seat radius
TIRE_W = 0.045         # 45 mm tyres
WHEEL_R = RIM_R + TIRE_W  # ~0.356 axle height


@dataclass
class Pts:
    bb: np.ndarray
    rear: np.ndarray
    front: np.ndarray
    ht_top: np.ndarray
    ht_bot: np.ndarray
    ht_dir: np.ndarray      # unit vector down the steering axis
    st_top: np.ndarray
    st_dir: np.ndarray      # unit vector up the seat tube


def frame_points(size: str = "M") -> Pts:
    g = GEOMETRY[size]
    mm = 0.001
    bb = np.array([0.0, 0.0, WHEEL_R - g["bb_drop"] * mm])
    cs_h = math.sqrt((g["cs"] * mm) ** 2 - (g["bb_drop"] * mm) ** 2)
    rear = np.array([0.0, -cs_h, WHEEL_R])
    front = np.array([0.0, rear[1] + g["wb"] * mm, WHEEL_R])
    ht_top = bb + np.array([0, g["reach"] * mm, g["stack"] * mm])
    a = math.radians(g["hta"])
    ht_dir = np.array([0, math.cos(a), -math.sin(a)])
    ht_bot = ht_top + ht_dir * g["ht"] * mm
    s = math.radians(g["sta"])
    st_dir = np.array([0, -math.cos(s), math.sin(s)])
    st_top = bb + st_dir * g["st"] * mm
    return Pts(bb, rear, front, ht_top, ht_bot, ht_dir, st_top, st_dir)


def hex_to_rgba(h: str) -> tuple[float, float, float, float]:
    h = PALETTE.get(h, h).lstrip("#")
    srgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, 1.0)


def catmull_rom(points: list, n: int = 8) -> list:
    """Smooth polyline through control points (for bars, fork blades)."""
    P = [np.asarray(p, float) for p in points]
    P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            out.append(
                0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
            )
    out.append(P[-2])
    return out


# --------------------------------------------------------------------------- blender helpers

class Builder:
    def __init__(self, color: str, matte: bool = True):
        import bpy

        self.bpy = bpy
        self.col = bpy.data.collections.new("Grizl")
        bpy.context.scene.collection.children.link(self.col)
        self.mats = {
            "paint": self._mat("Grizl.Paint", hex_to_rgba(color), rough=0.55 if matte else 0.32, coat=0.0 if matte else 0.6),
            "carbon": self._mat("Grizl.Black", hex_to_rgba("#141416"), rough=0.45),
            "rubber": self._mat("Grizl.Rubber", hex_to_rgba("#1b1a19"), rough=0.9),
            "metal": self._mat("Grizl.Metal", hex_to_rgba("#b8bcc2"), rough=0.28, metal=1.0),
            "dark_metal": self._mat("Grizl.DarkMetal", hex_to_rgba("#3a3c40"), rough=0.35, metal=1.0),
            "saddle": self._mat("Grizl.Saddle", hex_to_rgba("#101010"), rough=0.6),
        }
        self.root = self.empty("Grizl", (0, 0, 0))

    def _mat(self, name, rgba, rough=0.5, metal=0.0, coat=0.0):
        m = self.bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Base Color"].default_value = rgba
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
        if coat and "Coat Weight" in b.inputs:
            b.inputs["Coat Weight"].default_value = coat
            b.inputs["Coat Roughness"].default_value = 0.08
        m.diffuse_color = rgba
        return m

    def _link(self, obj, parent=None):
        self.col.objects.link(obj)
        obj.parent = parent or self.root
        return obj

    def empty(self, name, loc, parent=None):
        o = self.bpy.data.objects.new(name, None)
        o.empty_display_size = 0.05
        o.location = loc
        self.col.objects.link(o)
        if parent is not None or name != "Grizl":
            o.parent = parent if parent is not None else self.root
        return o

    def tube(self, name, pts, r, mat="paint", parent=None, r_end=None, resolution=12, flatten=1.0):
        """Tube along a polyline, radius r (optionally tapering to r_end)."""
        bpy = self.bpy
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = r
        cu.bevel_resolution = max(2, resolution // 4)
        cu.use_fill_caps = True
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        loc0 = np.zeros(3) if parent is None else np.array(parent.location)
        for i, p in enumerate(pts):
            q = np.asarray(p, float) - loc0
            sp.points[i].co = (*q, 1.0)
            if r_end is not None:
                sp.points[i].radius = 1 + (r_end / r - 1) * i / max(1, len(pts) - 1)
        obj = bpy.data.objects.new(name, cu)
        obj.data.materials.append(self.mats[mat])
        if flatten != 1.0:
            obj.scale = (flatten, 1, 1)
        return self._link(obj, parent)

    def mesh(self, name, verts, faces, mat, parent=None, smooth=True):
        me = self.bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        me.update()
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        me.materials.append(self.mats[mat])
        return self._link(self.bpy.data.objects.new(name, me), parent)

    def torus(self, name, R, r, mat, parent, axis_x=True, seg=64, ring=12):
        verts, faces = [], []
        for i in range(seg):
            u = 2 * math.pi * i / seg
            for j in range(ring):
                v = 2 * math.pi * j / ring
                rad = R + r * math.cos(v)
                # wheel plane = YZ, axle along X
                verts.append((r * math.sin(v), rad * math.cos(u), rad * math.sin(u)))
        for i in range(seg):
            for j in range(ring):
                a = i * ring + j
                b = ((i + 1) % seg) * ring + j
                c = ((i + 1) % seg) * ring + (j + 1) % ring
                d = i * ring + (j + 1) % ring
                faces.append((a, b, c, d))
        return self.mesh(name, verts, faces, mat, parent)

    def disc(self, name, R, thickness, mat, parent, x=0.0, seg=48, r_in=0.0):
        """Flat disc/ring in the YZ plane at lateral offset x."""
        verts, faces = [], []
        for side in (-1, 1):
            for i in range(seg):
                u = 2 * math.pi * i / seg
                verts.append((x + side * thickness / 2, R * math.cos(u), R * math.sin(u)))
                verts.append((x + side * thickness / 2, r_in * math.cos(u), r_in * math.sin(u)))
        n = seg * 2
        for i in range(seg):
            j = (i + 1) % seg
            o0, i0, o1, i1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
            faces.append((o0, o1, i1, i0))                       # left face
            faces.append((n + o0, n + i0, n + i1, n + o1))       # right face
            faces.append((o0, n + o0, n + o1, o1))               # outer rim
            if r_in > 0:
                faces.append((i0, i1, n + i1, n + i0))
        return self.mesh(name, verts, faces, mat, parent, smooth=False)

    def loft(self, name, sections, mat, parent=None, ring=16):
        """sections: list of (y, width, height, z) ellipses along Y."""
        verts, faces = [], []
        for y, w, h, z in sections:
            for j in range(ring):
                v = 2 * math.pi * j / ring
                verts.append((w / 2 * math.cos(v), y, z + h / 2 * math.sin(v)))
        for i in range(len(sections) - 1):
            for j in range(ring):
                a = i * ring + j
                b = (i + 1) * ring + j
                c = (i + 1) * ring + (j + 1) % ring
                d = i * ring + (j + 1) % ring
                faces.append((a, b, c, d))
        faces.append(tuple(range(ring))[::-1])
        faces.append(tuple(range((len(sections) - 1) * ring, len(sections) * ring)))
        return self.mesh(name, verts, faces, mat, parent)


# --------------------------------------------------------------------------- the bike

def build_grizl(color: str = "periwinkle", size: str = "M", matte: bool = True):
    """Build the bike into the current scene; returns the root empty."""
    P = frame_points(size)
    B = Builder(color, matte)
    X = np.array([1.0, 0, 0])

    def on_st(length):
        return P.bb + P.st_dir * length

    # ---- frame
    dt_top = P.ht_bot - P.ht_dir * 0.025
    B.tube("Frame.DownTube", [P.bb + [0, 0.01, 0.01], dt_top], 0.032, flatten=0.8)
    tt_front = P.ht_top + P.ht_dir * 0.022
    tt_rear = on_st(0.47)
    B.tube("Frame.TopTube", [tt_rear, tt_front], 0.02)
    B.tube("Frame.SeatTube", [P.bb, P.st_top], 0.019)
    B.tube("Frame.HeadTube", [P.ht_top - P.ht_dir * 0.012, P.ht_bot + P.ht_dir * 0.008], 0.027, r_end=0.031)
    B.tube("Frame.BB", [P.bb - X * 0.043, P.bb + X * 0.043], 0.024, mat="paint")
    ss_top = on_st(0.39)
    for side in (-1, 1):
        s = "R" if side > 0 else "L"
        axle = P.rear + X * side * 0.066
        B.tube(f"Frame.Chainstay.{s}", [axle, P.bb + X * side * 0.032 + [0, -0.03, 0.0]], 0.013, r_end=0.018)
        B.tube(f"Frame.Seatstay.{s}", [axle + [0, 0.005, 0.01], ss_top + X * side * 0.018], 0.0085, r_end=0.011)
        B.tube(f"Frame.Dropout.{s}", [axle - [0, 0.012, 0], axle + [0, 0.012, 0.012]], 0.009)

    # ---- fork (tapered, slightly raked blades)
    crown = P.ht_bot + P.ht_dir * 0.02
    B.tube("Fork.Crown", [crown - X * 0.06, crown + X * 0.06], 0.02, flatten=1.0)
    for side in (-1, 1):
        s = "R" if side > 0 else "L"
        axle = P.front + X * side * 0.058
        mid = crown + (axle - crown) * 0.55 + [0, 0.01, 0] + X * side * 0.055
        pts = catmull_rom([crown + X * side * 0.05, mid, axle], n=6)
        B.tube(f"Fork.Blade.{s}", pts, 0.016, r_end=0.009)

    # ---- cockpit
    steer_top = P.ht_top - P.ht_dir * 0.045
    B.tube("Cockpit.Spacers", [P.ht_top - P.ht_dir * 0.01, steer_top], 0.017, mat="carbon")
    stem_dir = np.array([0, math.cos(math.radians(8)), math.sin(math.radians(8))])
    clamp = steer_top + stem_dir * 0.09
    B.tube("Cockpit.Stem", [steer_top - stem_dir * 0.012, clamp], 0.016, mat="carbon")
    half = [
        (0.00, 0.000, 0.000), (0.10, 0.000, 0.000), (0.185, 0.010, 0.000), (0.205, 0.045, -0.006),
        (0.212, 0.075, -0.028), (0.220, 0.082, -0.070), (0.228, 0.058, -0.112), (0.234, 0.012, -0.128),
        (0.238, -0.045, -0.122),
    ]
    for side in (-1, 1):
        s = "R" if side > 0 else "L"
        pts = [clamp + np.array([side * x, y, z]) for x, y, z in catmull_rom(half, n=6)]
        B.tube(f"Cockpit.Bar.{s}", pts, 0.0125, mat="carbon")
        hood = clamp + np.array([side * 0.21, 0.07, -0.02])
        B.tube(f"Cockpit.Hood.{s}", [hood + [0, -0.015, 0.012], hood + [0, 0.03, 0.0], hood + [0, 0.035, -0.05]],
               0.013, mat="carbon", r_end=0.007)

    # ---- seatpost + saddle
    saddle_c = P.bb + P.st_dir * 0.735
    B.tube("Seatpost", [P.st_top - P.st_dir * 0.02, saddle_c - [0, 0, 0.035]], 0.0135, mat="carbon")
    sections = [
        (-0.125, 0.030, 0.020, 0.0), (-0.115, 0.125, 0.034, 0.004), (-0.08, 0.14, 0.036, 0.004),
        (-0.03, 0.12, 0.032, 0.0), (0.03, 0.06, 0.028, -0.004), (0.10, 0.038, 0.024, -0.004),
        (0.135, 0.030, 0.020, -0.006), (0.145, 0.012, 0.010, -0.006),
    ]
    saddle = B.loft("Saddle", sections, "saddle")
    saddle.location = saddle_c
    saddle.rotation_euler = (math.radians(-2), 0, 0)

    # ---- wheels (spinning groups)
    for name, axle, is_rear in (("Front", P.front, False), ("Rear", P.rear, True)):
        wheel = B.empty(f"Grizl.Wheel{name}", tuple(axle))
        B.torus(f"Wheel{name}.Tire", RIM_R + TIRE_W * 0.5, TIRE_W * 0.5, "rubber", wheel, seg=96, ring=14)
        B.torus(f"Wheel{name}.Rim", RIM_R - 0.012, 0.012, "carbon", wheel, seg=96, ring=10)
        B.tube(f"Wheel{name}.Hub", [(-0.05, 0, 0), (0.05, 0, 0)], 0.014, mat="dark_metal").parent = wheel
        B.disc(f"Wheel{name}.Rotor", 0.08, 0.002, "dark_metal", wheel, x=-0.045, r_in=0.055, seg=40)
        spokes = []
        for i in range(24):
            ang = 2 * math.pi * i / 24
            side = 1 if i % 2 else -1
            ang_hub = ang + side * 0.25
            hub_p = np.array([side * 0.03, 0.022 * math.cos(ang_hub), 0.022 * math.sin(ang_hub)])
            rim_p = np.array([0.0, (RIM_R - 0.022) * math.cos(ang), (RIM_R - 0.022) * math.sin(ang)])
            spokes.append((hub_p, rim_p))
        for i, (a, b) in enumerate(spokes):
            t = B.tube(f"Wheel{name}.Spoke{i:02d}", [a, b], 0.0012, mat="dark_metal", resolution=4)
            t.parent = wheel
        if is_rear:
            B.disc("WheelRear.Cassette", 0.052, 0.035, "dark_metal", wheel, x=0.035, r_in=0.014, seg=36)

    # ---- drivetrain (1x)
    cranks = B.empty("Grizl.Cranks", tuple(P.bb))
    B.disc("Cranks.Chainring", 0.083, 0.004, "dark_metal", cranks, x=0.048, r_in=0.03, seg=40)
    B.tube("Cranks.Spindle", [(-0.07, 0, 0), (0.07, 0, 0)], 0.012, mat="carbon").parent = cranks
    L = 0.1725
    for side, ang in ((1, math.radians(-35)), (-1, math.radians(145))):
        s = "R" if side > 0 else "L"
        end = np.array([side * 0.075, L * math.cos(ang), L * math.sin(ang)])
        B.tube(f"Cranks.Arm.{s}", [np.array([side * 0.07, 0, 0]), end], 0.011, mat="carbon", flatten=0.7).parent = cranks
        pedal = B.tube(f"Cranks.Pedal.{s}", [end, end + np.array([side * 0.09, 0, 0])], 0.012, mat="dark_metal")
        pedal.parent = cranks
    # chain: top and bottom runs between chainring and cassette
    ring_c, cas_c = P.bb + X * 0.048, P.rear + X * 0.035
    B.tube("Chain.Top", [ring_c + [0, 0, 0.083], cas_c + [0, 0, 0.045]], 0.0035, mat="dark_metal", resolution=4)
    B.tube("Chain.Bottom", [ring_c + [0, 0, -0.083], cas_c + [0, 0.02, -0.075]], 0.0035, mat="dark_metal", resolution=4)
    B.tube("Derailleur", [cas_c + [0, 0.0, -0.03], cas_c + [0, 0.02, -0.08]], 0.009, mat="dark_metal")

    # note: parts re-parented with `.parent = wheel/cranks` were built in that parent's local coords
    return B.root


# --------------------------------------------------------------------------- scene / export

def _curves_to_meshes(col):
    import bpy

    dg = bpy.context.evaluated_depsgraph_get()
    for obj in list(col.objects):
        if obj.type != "CURVE":
            continue
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
        new = bpy.data.objects.new(obj.name, me)
        new.matrix_world = obj.matrix_world.copy()
        col.objects.link(new)
        new.parent = obj.parent
        new.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
        new.location, new.rotation_euler, new.scale = obj.location, obj.rotation_euler, obj.scale
        for p in me.polygons:
            p.use_smooth = True
        bpy.data.objects.remove(obj)


def reset_scene():
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_preview(scene, samples: int = 48, res=(1400, 900)):
    import bpy

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    # world
    world = bpy.data.worlds.new("Studio")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.82, 0.84, 0.88, 1)
    bg.inputs["Strength"].default_value = 0.45
    scene.world = world
    # ground
    me = bpy.data.meshes.new("Ground")
    s = 20
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    g = bpy.data.objects.new("Ground", me)
    m = bpy.data.materials.new("GroundMat")
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.75, 0.76, 0.78, 1)
    m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
    me.materials.append(m)
    scene.collection.objects.link(g)
    # lights
    key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
    key.data.energy = 450
    key.data.size = 3
    key.location = (3.0, -1.0, 3.2)
    key.rotation_euler = (math.radians(50), 0, math.radians(70))
    scene.collection.objects.link(key)
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 1.8
    sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(-30))
    scene.collection.objects.link(sun)
    # camera: 3/4 drive-side view
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    cam.data.lens = 45
    cam.location = (2.9, 1.9, 1.35)
    scene.collection.objects.link(cam)
    target = bpy.data.objects.new("CamTarget", None)
    target.location = (0, 0.1, 0.55)
    scene.collection.objects.link(target)
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    scene.camera = cam


def build_and_export(out: Path | None, color: str = "periwinkle", size: str = "M",
                     preview: Path | None = None, blend: Path | None = None, samples: int = 48):
    import bpy

    reset_scene()
    root = build_grizl(color, size)
    col = bpy.data.collections["Grizl"]
    _curves_to_meshes(col)
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", use_active_collection=False)
    if preview:
        setup_preview(bpy.context.scene, samples=samples)
        bpy.context.scene.render.filepath = str(Path(preview).resolve())
        bpy.ops.render.render(write_still=True)
    if blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(blend).resolve()))
    return root
