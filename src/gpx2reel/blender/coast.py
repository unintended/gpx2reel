"""Beach set meshes for a coastal sunrise opening (placement: gpx2reel.coast): palms, houses, sea, surf.

Palms and houses are one merged mesh each (numpy geometry, ~250 palms / ~2000 houses stay cheap for EEVEE).
Seen against the sunrise they read as silhouettes, so shapes matter more than textures. The water disc and the
surf only live around the opening: animate keys their `Show` value (1 in the opening, fading into the ride).
"""
from __future__ import annotations

import math

import numpy as np

PALM_H = (10.0, 17.0)         # trunk height, m
FRONDS = 14
FROND_L = (4.5, 6.0)
WATER_HALF_M = 14000.0        # a square to past the horizon of a 14 m high camera (~13 km)
WATER_STEP_M = 50.0           # vertex spacing of the shore fade
WATER_Z = 1.2                 # over the detail patch's sea (+0.5, shallows a bit higher); the low shore it
                              # floods is where the shore fade keeps it transparent
SURF_OFFSETS_M = (6.0, 18.0, 34.0)   # foam bands seaward of the coastline
SURF_WIDTH_M = 7.0

WALLS = ["#b8ae9e", "#a39c90", "#9a9186", "#c2b6a2", "#8f8a82", "#b0a48e"]
ROOFS = ["#3a3f47", "#4a4f57", "#2f3a4a", "#5a4a42", "#6b6e72"]


def _lin(h: str):
    from .stylized import _lin as lin

    return lin(h)


def _build(name: str, verts: list, faces: list, mat_idx: list, colors: list | None = None):
    import bpy

    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, np.float32).tolist(), [], faces)
    me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    if colors is not None:
        a = me.color_attributes.new("tint", "FLOAT_COLOR", "FACE")
        a.data.foreach_set("color", np.asarray(colors, np.float32).ravel())
    me.update()
    return me


# --------------------------------------------------------------------------- palms

def palm_geometry(points: np.ndarray, lean_dirs: np.ndarray):
    """points (n, 4): x, y, z, rand; lean_dirs (n, 2): unit vectors palms lean towards (the sea).
    Returns verts, faces, material index per face (0 trunk, 1 fronds)."""
    V, F, M = [], [], []
    sides, rings = 6, 7
    for (x, y, z, r), ld in zip(points, lean_dirs):
        rng = np.random.default_rng(int(r * 1e6))
        H = PALM_H[0] + (PALM_H[1] - PALM_H[0]) * rng.random()
        lean = 0.12 + 0.22 * rng.random()
        side = np.array([-ld[1], ld[0]])
        base = len(V)
        top = None
        for k in range(rings):
            t = k / (rings - 1)
            c = np.array([x, y]) + ld * lean * H * t ** 1.7 + side * 0.4 * math.sin(math.pi * t) * (rng.random() - 0.5)
            rad = 0.28 - 0.12 * t
            for j in range(sides):
                a = 2 * math.pi * j / sides
                V.append([c[0] + rad * math.cos(a), c[1] + rad * math.sin(a), z - 0.5 + (H + 0.5) * t])
            top = np.array([c[0], c[1], z + H])
        for k in range(rings - 1):
            for j in range(sides):
                a, b = base + k * sides + j, base + k * sides + (j + 1) % sides
                F.append([a, b, b + sides, a + sides])
                M.append(0)
        # crown: fronds arch up and droop, a shallow V across
        for f in range(FRONDS):
            ang = 2 * math.pi * f / FRONDS + rng.normal(0, 0.18)
            L = FROND_L[0] + (FROND_L[1] - FROND_L[0]) * rng.random()
            droop = 0.8 + 0.7 * rng.random()
            d = np.array([math.cos(ang), math.sin(ang), 0.0])
            s = np.array([-d[1], d[0], 0.0])
            fb = len(V)
            n = 7
            for i in range(n):
                u = i / (n - 1)
                p = top + d * L * u + np.array([0, 0, L * (0.55 * u - droop * 0.9 * u * u)])
                w = 0.95 * math.sin(math.pi * min(u * 1.15, 1.0)) ** 0.6 + 0.04
                V.append(list(p - s * w + np.array([0, 0, -0.12 * w])))
                V.append(list(p + np.array([0, 0, 0.05])))
                V.append(list(p + s * w + np.array([0, 0, -0.12 * w])))
            for i in range(n - 1):
                a = fb + 3 * i
                F.append([a, a + 3, a + 4, a + 1]); M.append(1)
                F.append([a + 1, a + 4, a + 5, a + 2]); M.append(1)
    return V, F, M


def add_palms(points: np.ndarray, lean_dirs: np.ndarray):
    import bpy

    from .scene import _link

    if not len(points):
        return None
    V, F, M = palm_geometry(points, lean_dirs)
    me = _build("Coast.Palms", V, F, M)
    for name, hex_, rough in (("Palm.Trunk", "#5b4c3b", 0.9), ("Palm.Frond", "#2e4a22", 0.7)):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = _lin(hex_)
        b.inputs["Roughness"].default_value = rough
        if name == "Palm.Frond":                      # thin leaves glow a little when backlit
            b.inputs["Subsurface Weight"].default_value = 0.15
        me.materials.append(m)
    obj = _link(bpy.data.objects.new("Coast.Palms", me))
    obj.visible_shadow = True
    return obj


# --------------------------------------------------------------------------- houses

def house_geometry(boxes: np.ndarray, seed: int = 3):
    """boxes (n, 7): x, y, z, width, depth, height, rotation → verts, faces, material (0 walls, 1 roof), colours."""
    rng = np.random.default_rng(seed)
    V, F, M, C = [], [], [], []
    for x, y, z, w, d, h, rot in boxes:
        wall = _hex_rgb(WALLS[rng.integers(len(WALLS))])
        roof = _hex_rgb(ROOFS[rng.integers(len(ROOFS))])
        ca, sa = math.cos(rot), math.sin(rot)

        def P(u, v, zz):
            return [x + u * ca - v * sa, y + u * sa + v * ca, zz]

        hw, hd = w / 2, d / 2
        z0, z1 = z - 1.0, z + h
        rh = min(w, d) * 0.28
        o = 0.5
        b = len(V)
        for u, v in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
            V.append(P(u, v, z0))
        for u, v in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):
            V.append(P(u, v, z1))
        for j in range(4):
            F.append([b + j, b + (j + 1) % 4, b + 4 + (j + 1) % 4, b + 4 + j]); M.append(0); C.append(wall)
        e = len(V)                                     # eaves (overhang) and the ridge along the width
        for u, v in ((-hw - o, -hd - o), (hw + o, -hd - o), (hw + o, hd + o), (-hw - o, hd + o)):
            V.append(P(u, v, z1 - 0.2))
        V.append(P(-hw - o, 0, z1 + rh)); V.append(P(hw + o, 0, z1 + rh))
        r0, r1 = e + 4, e + 5
        F.append([e, e + 1, r1, r0]); M.append(1); C.append(roof)
        F.append([e + 2, e + 3, r0, r1]); M.append(1); C.append(roof)
        g = len(V)                                     # gables
        V.append(P(-hw, -hd, z1)); V.append(P(-hw, hd, z1)); V.append(P(-hw, 0, z1 + rh))
        V.append(P(hw, -hd, z1)); V.append(P(hw, hd, z1)); V.append(P(hw, 0, z1 + rh))
        F.append([g, g + 1, g + 2]); M.append(0); C.append(wall)
        F.append([g + 3, g + 5, g + 4]); M.append(0); C.append(wall)
    return V, F, M, C


def _hex_rgb(h: str):
    return list(_lin(h))


def add_houses(boxes: np.ndarray):
    import bpy

    from .scene import _link

    if not len(boxes):
        return None
    V, F, M, C = house_geometry(boxes)
    me = _build("Coast.Houses", V, F, M, C)
    for name, rough in (("House.Wall", 0.85), ("House.Roof", 0.55)):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = "tint"
        nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        me.materials.append(m)
    return _link(bpy.data.objects.new("Coast.Houses", me))


# --------------------------------------------------------------------------- sea and surf

def _show_value(nt, name: str = "Show"):
    v = nt.nodes.new("ShaderNodeValue")
    v.name = v.label = name
    v.outputs[0].default_value = 1.0
    return v


def _animated_noise(nt, scale: float, detail: float = 4.0):
    """4D noise in object space whose W drifts with the frame (the object's `clock`, see scene.object_clock)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = scale
    nz.inputs["Detail"].default_value = detail
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    from .scene import object_value

    nt.links.new(object_value(nt, "clock"), nz.inputs["W"])
    return nz


def add_water(center_xy, lines: list[np.ndarray], half: float = WATER_HALF_M, step: float = WATER_STEP_M):
    """A square of moving water over the satellite sea: ripples catch the low sun as a glitter path. Opacity per
    vertex (`shore`) fades out towards the coastline, so the satellite shallows and the surf show near the shore."""
    import bpy

    from ..coast import shore_alpha
    from .scene import _grid_quads, _link, _mesh

    k = int(2 * half / step) + 1
    g = np.linspace(-half, half, k)
    gx, gy = np.meshgrid(g, g)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    me = _mesh("Coast.Water", verts, faces=_grid_quads(k, k))
    a = me.attributes.new("shore", "FLOAT", "POINT")
    a.data.foreach_set("value", shore_alpha(lines, verts[:, :2] + np.asarray(center_xy)[None, :2]).astype(np.float32))
    obj = _link(bpy.data.objects.new("Coast.Water", me))
    from .scene import object_clock

    object_clock(obj, "clock", "frame / 90")
    obj.location = (float(center_xy[0]), float(center_xy[1]), WATER_Z)
    obj.visible_shadow = False

    m = bpy.data.materials.new("Coast.Water")
    m.use_nodes = True
    nt, L = m.node_tree, m.node_tree.links
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.005, 0.03, 0.045, 1)
    b.inputs["Roughness"].default_value = 0.14    # the sun path breaks into glitter
    b.inputs["IOR"].default_value = 1.33
    nz = _animated_noise(nt, scale=0.12)             # ~8 m ripples (object space, metres)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.3
    L.new(nz.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    # alpha: shore fade × rim fade × Show (keyed)
    shore = nt.nodes.new("ShaderNodeAttribute")
    shore.attribute_name = "shore"
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    ax = nt.nodes.new("ShaderNodeMath"); ax.operation = "ABSOLUTE"
    ay = nt.nodes.new("ShaderNodeMath"); ay.operation = "ABSOLUTE"
    edge = nt.nodes.new("ShaderNodeMath"); edge.operation = "MAXIMUM"
    L.new(sep.outputs["X"], ax.inputs[0]); L.new(sep.outputs["Y"], ay.inputs[0])
    L.new(ax.outputs[0], edge.inputs[0]); L.new(ay.outputs[0], edge.inputs[1])
    rim = nt.nodes.new("ShaderNodeMapRange")
    rim.inputs["From Min"].default_value, rim.inputs["From Max"].default_value = 0.7 * half, half
    rim.inputs["To Min"].default_value, rim.inputs["To Max"].default_value = 0.95, 0.0
    L.new(edge.outputs[0], rim.inputs["Value"])
    show = _show_value(nt)
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    L.new(rim.outputs["Result"], m1.inputs[0]); L.new(shore.outputs["Fac"], m1.inputs[1])
    L.new(m1.outputs[0], m2.inputs[0]); L.new(show.outputs[0], m2.inputs[1])
    L.new(m2.outputs[0], b.inputs["Alpha"])
    me.materials.append(m)
    return obj


def surf_geometry(lines: list[np.ndarray], center_xy, radius: float, step: float = 5.0):
    """Foam bands seaward of the coastlines (OSM: sea on the right) → verts, quad faces, `across` per vertex
    (0 at the band's edges, 1 in its middle: the foam thins out sideways instead of ending in a hard line)."""
    from ..coast import _resample

    V, F, A = [], [], []
    for line in lines:
        p, d = _resample(line, step)
        if len(p) < 2:
            continue
        near = np.linalg.norm(p - np.asarray(center_xy)[None, :2], axis=1) < radius
        right = np.column_stack([d[:, 1], -d[:, 0]])
        for off in SURF_OFFSETS_M:
            run = []
            for i in range(len(p)):
                if near[i]:
                    run.append(i)
                if (not near[i] or i == len(p) - 1) and len(run) > 1:
                    b = len(V)
                    for k in run:
                        for s, a in ((-0.5, 0.0), (0.0, 1.0), (0.5, 0.0)):
                            q = p[k] + right[k] * (off + s * SURF_WIDTH_M)
                            V.append([q[0], q[1], WATER_Z + 0.15])
                            A.append(a)
                    for j in range(len(run) - 1):
                        a0 = b + 3 * j
                        F.append([a0, a0 + 3, a0 + 4, a0 + 1])
                        F.append([a0 + 1, a0 + 4, a0 + 5, a0 + 2])
                if not near[i]:
                    run = []
    return V, F, A


def add_surf(lines: list[np.ndarray], center_xy, radius: float):
    import bpy

    from .scene import _link

    V, F, A = surf_geometry(lines, center_xy, radius)
    if not F:
        return None
    me = bpy.data.meshes.new("Coast.Surf")
    me.from_pydata(V, [], F)
    me.attributes.new("across", "FLOAT", "POINT").data.foreach_set("value", np.asarray(A, np.float32))
    obj = _link(bpy.data.objects.new("Coast.Surf", me))
    from .scene import object_clock

    object_clock(obj, "clock", "frame / 90")
    obj.visible_shadow = False
    m = bpy.data.materials.new("Coast.Surf")
    m.use_nodes = True
    nt, L = m.node_tree, m.node_tree.links
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.9, 0.93, 0.95, 1)
    b.inputs["Roughness"].default_value = 0.9
    nz = _animated_noise(nt, scale=0.08, detail=6.0)
    foam = nt.nodes.new("ShaderNodeMapRange")
    foam.inputs["From Min"].default_value, foam.inputs["From Max"].default_value = 0.46, 0.66
    L.new(nz.outputs["Fac"], foam.inputs["Value"])
    across = nt.nodes.new("ShaderNodeAttribute")
    across.attribute_name = "across"
    show = _show_value(nt)
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    L.new(foam.outputs["Result"], m1.inputs[0]); L.new(across.outputs["Fac"], m1.inputs[1])
    L.new(m1.outputs[0], mul.inputs[0]); L.new(show.outputs[0], mul.inputs[1])
    L.new(mul.outputs[0], b.inputs["Alpha"])
    me.materials.append(m)
    return obj
