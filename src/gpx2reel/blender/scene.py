"""Blender scene from world.py geometry: terrain (+ intro-zoom context layers), trail, marker, checkpoints.

Things the timeline animates (blender/animate.py):
  Trail / Trail.Ahead modifier inputs Progress (s, km) and Width (m) — see _trail_group;
  Marker location / scale / color (object colour drives its emission);
  Checkpoint.* scale (0 until the marker reaches them);
  Gaps curve bevel_depth — dash thickness;
  Grizl root empty + wheel / crank empties when avatar.model is the procedural bike.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from ..trip import TripPaths
from ..world import Path3D, TerrainGrid

AHEAD_ALPHA = 0.3
TRANSPORT_RGBA = (0.82, 0.85, 0.9, 1.0)
PAST_RGBA = (0.62, 0.64, 0.68, 1.0)      # the road already travelled (earlier days in a day story)
CONTEXT_DROP = 0.0007       # coarser layers sit this share of their extent below finer ones (coarse DEM
                            # is smoother: valleys come out higher and would poke through)
FEATHER = 0.08              # finer layers fade out over this share of their size (imagery zooms differ in tone)
DETAIL_STEP_M = 15.0        # grid of the high-res highlight patch (z14 DEM ≈ 8 m/px)
DETAIL_FEATHER = 0.12


def _mesh(name: str, verts: np.ndarray, faces: np.ndarray | None = None, edges: np.ndarray | None = None):
    import bpy

    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    if edges is not None and len(edges):
        me.edges.add(len(edges))
        me.edges.foreach_set("vertices", edges.astype(np.int32).ravel())
    if faces is not None and len(faces):
        me.loops.add(faces.size)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(len(faces))
        me.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    return me


def _grid_quads(ny: int, nx: int) -> np.ndarray:
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, 1:], idx[1:, :-1]
    return np.stack([a, b, c, d], -1).reshape(-1, 4)   # CCW seen from +Z


def _link(obj, col=None):
    import bpy

    (col or bpy.context.scene.collection).objects.link(obj)
    return obj


def _sphere(name: str, radius: float = 1.0, segments: int = 32, rings: int = 16):
    import bmesh
    import bpy

    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    return me


def mix_rgba(nt, blend_type: str = "MIX"):
    """ShaderNodeMix in colour mode → (node, factor, a, b, result). The node has several sockets called
    "A" / "B" (float, vector, colour…): name lookup returns the float one, so address them by index."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type, n.blend_type = "RGBA", blend_type
    return n, n.inputs[0], n.inputs[6], n.inputs[7], n.outputs[2]


def _emission_material(name: str, rgba=None, strength: float = 2.5, from_object: bool = False):
    """Principled with emission; colour fixed or taken from the object colour (animatable)."""
    import bpy

    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    if from_object:
        info = nt.nodes.new("ShaderNodeObjectInfo")
        nt.links.new(info.outputs["Color"], b.inputs["Base Color"])
        nt.links.new(info.outputs["Color"], b.inputs["Emission Color"])
    else:
        b.inputs["Base Color"].default_value = rgba
        b.inputs["Emission Color"].default_value = rgba
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = 0.4
    return m


# --------------------------------------------------------------------------- terrain

def object_clock(obj, prop: str, expression: str, variables=()) -> None:
    """A per-frame value for a material lives on the object (driven custom property), not on a node socket:
    EEVEE on Metal rebuilt the materials whose sockets changed every frame and now and then rendered a frame
    with a broken one (cyan flashes on day 11's steam and water). variables: (name, target, transform)."""
    obj[prop] = 0.0
    drv = obj.driver_add(f'["{prop}"]').driver
    for name, target, transform in variables:
        v = drv.variables.new()
        v.name, v.type = name, "TRANSFORMS"
        v.targets[0].id = target
        v.targets[0].transform_type = transform
        v.targets[0].transform_space = "WORLD_SPACE"
    drv.expression = expression


def object_value(nt, prop: str):
    """Shader output of an object's custom property (see object_clock)."""
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "OBJECT"
    a.attribute_name = prop
    return a.outputs["Fac"]


def add_terrain(grid: TerrainGrid, ortho: Path | None, name: str = "Terrain",
                drop_m: float = 0.0, feather: float = 0.0, holes: list[tuple] = (), sea_like=None, coast=None):
    """Terrain mesh; feather = edge band (share of the size) fading to transparent over the coarser layer below;
    holes = (x0, x1, y0, y1) rectangles left out (covered by an opaque finer patch)."""
    import bpy

    coast_dist = None
    if coast:                                            # the coastline decides sea or land, not the DEM's heights
        from ..coast import sea_mask

        grid.sea, coast_dist = sea_mask(coast, grid)
    gx, gy = np.meshgrid(grid.x, grid.y)
    verts = np.column_stack([gx.ravel(), gy.ravel(), grid.z.ravel() - drop_m])
    quads = _grid_quads(*grid.z.shape)
    for x0, x1, y0, y1 in holes:
        v = verts[quads]
        inside = ((v[..., 0] > x0) & (v[..., 0] < x1) & (v[..., 1] > y0) & (v[..., 1] < y1)).all(1)
        quads = quads[~inside]
    me = _mesh(name, verts, quads)
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", grid.uv.reshape(-1, 2)[quads.ravel()].astype(np.float32).ravel())
    me.shade_smooth()
    if grid.sea is not None and grid.sea.any():
        a = me.attributes.new("sea", "FLOAT", "POINT")
        a.data.foreach_set("value", grid.sea.astype(np.float32).ravel())
    deep = open_water(grid, ortho, coast_dist) if ortho is not None else None
    if deep is not None and sea_like is not None:        # a detail patch: the open sea in the base layer's tone
        deep = (deep[0], sea_tone_of(sea_like, grid))    # (its own imagery is of another date — a dark rectangle)
    if deep is not None and name == "Terrain":
        add_terrain.base_sea = (grid, deep[1])
    if deep is not None:
        me.attributes.new("deep", "FLOAT", "POINT").data.foreach_set("value", deep[0].ravel())
        c = me.color_attributes.new("sea_rgb", "FLOAT_COLOR", "POINT")
        c.data.foreach_set("color", deep[1].reshape(-1, 4).ravel())

    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    haze = nt.nodes.new("ShaderNodeGroup")
    haze.node_tree = haze_group()
    nt.links.new(haze.outputs["Color"], bsdf.inputs["Base Color"])
    if ortho is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(Path(ortho).resolve()))
        tex.extension = "EXTEND"
        if deep is not None:                 # open water: the imagery's own tone, blurred past its tile seams
            _open_sea(nt, tex, haze)
        else:
            nt.links.new(tex.outputs["Color"], haze.inputs["Color"])
    else:
        haze.inputs["Color"].default_value = (0.35, 0.4, 0.3, 1)
    if "sea" in me.attributes:
        _sea_gloss(nt, bsdf)
    if feather > 0:
        _feather_alpha(nt, bsdf, feather)
    me.materials.append(m)
    return _link(bpy.data.objects.new(name, me))


def _sea_gloss(nt, bsdf) -> None:
    """Sea (vertex attribute `sea`: the coastline, else the DEM) is glossy: sun glints on it, land stays matte."""
    sea = nt.nodes.new("ShaderNodeAttribute")
    sea.attribute_name = "sea"
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.95, SEA_ROUGHNESS
    nt.links.new(sea.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    # and a weaker mirror: looking towards a low sun, every bay turned milky white (day 12, Nagasaki)
    spec = nt.nodes.new("ShaderNodeMapRange")
    spec.inputs["To Min"].default_value, spec.inputs["To Max"].default_value = 0.5, SEA_SPECULAR
    nt.links.new(sea.outputs["Fac"], spec.inputs["Value"])
    nt.links.new(spec.outputs["Result"], bsdf.inputs["Specular IOR Level"])
    _, fac, a, b, tint = mix_rgba(nt)                  # at grazing angles any water mirrors almost all: tint it
    nt.links.new(sea.outputs["Fac"], fac)
    a.default_value = (1.0, 1.0, 1.0, 1.0)
    b.default_value = SEA_SPECULAR_TINT
    nt.links.new(tint, bsdf.inputs["Specular Tint"])


OPEN_SEA_M = (5.0, 25.0)     # depth over which the imagery gives way to the blurred sea (shallows keep their reefs)
OPEN_SEA_SHORE_M = (100.0, 500.0)    # the same by distance from land, where the DEM has no bathymetry (zeros)
SEA_BLUR_M = 2500.0          # Esri stitches the sea from tiles of different dates: blur the steps into a gradient


def open_water(grid: TerrainGrid, ortho: Path, coast_dist: np.ndarray | None = None):
    """(openness 0..1, linear RGBA per vertex) or None: how far a vertex is out on open water, and the imagery's
    colour there blurred over water only (normalised convolution: the land doesn't bleed into the sea)."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    ss = lambda v, lo, hi: (lambda u: u * u * (3 - 2 * u))(np.clip((v - lo) / (hi - lo), 0, 1))  # noqa: E731
    if coast_dist is not None:                         # water and distance from the OSM coastline (sea_mask)
        water = grid.sea
        if water.sum() < 100:
            return None
        openness = ss(coast_dist, *OPEN_SEA_SHORE_M) * water
    else:                                              # no coastline: guess the water from the DEM
        if grid.depth is None:
            return None
        water = (grid.depth > 0.5) | (grid.z <= 0)
        if water.sum() < 100:
            return None
        dist = distance_transform_edt(water) * float(grid.x[1] - grid.x[0])
        openness = np.maximum(ss(grid.depth, *OPEN_SEA_M), ss(dist, *OPEN_SEA_SHORE_M)) * water
    step = float(grid.x[1] - grid.x[0])
    if (openness > 0.5).sum() < 50:
        return None
    Image.MAX_IMAGE_PIXELS = None                 # the trip's imagery is ~190 Mpx; a 1/8 JPEG draft is plenty
    im = Image.open(ortho)
    im.draft("RGB", (im.size[0] // 8, im.size[1] // 8))
    px = np.asarray(im.convert("RGB"), np.float32) / 255
    h, w = px.shape[:2]
    cu = np.clip((grid.uv[..., 0] * w).astype(int), 0, w - 1)
    cv = np.clip(((1 - grid.uv[..., 1]) * h).astype(int), 0, h - 1)
    srgb = px[cv, cu]
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    sig = SEA_BLUR_M / step
    wt = gaussian_filter(water.astype(np.float32), sig)
    rgb = np.stack([gaussian_filter(lin[..., k] * water, sig) for k in range(3)], -1) / np.maximum(wt, 1e-4)[..., None]
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], -1).astype(np.float32)
    return openness.astype(np.float32), rgba


def sea_tone_of(base, grid: TerrainGrid) -> np.ndarray:
    """The base layer's blurred sea colour (base = (grid, rgba per vertex)) sampled at another grid's vertices."""
    from scipy.ndimage import map_coordinates

    bg, rgba = base
    gx, gy = np.meshgrid(grid.x, grid.y)
    fx = (gx - bg.x[0]) / (bg.x[1] - bg.x[0])
    fy = (gy - bg.y[0]) / (bg.y[1] - bg.y[0])
    out = np.stack([map_coordinates(rgba[..., k], [fy, fx], order=1, mode="nearest") for k in range(4)], -1)
    return out.astype(np.float32)


def _open_sea(nt, tex, haze) -> None:
    deep = nt.nodes.new("ShaderNodeAttribute")
    deep.attribute_name = "deep"
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "sea_rgb"
    _, fac, a, b, out = mix_rgba(nt)
    nt.links.new(deep.outputs["Fac"], fac)
    nt.links.new(tex.outputs["Color"], a)
    nt.links.new(col.outputs["Color"], b)
    nt.links.new(out, haze.inputs["Color"])


SEA_ROUGHNESS = 0.6          # a soft sheen: 0.32 blew a harbour seen towards the sun into a white sheet
SEA_SPECULAR = 0.2           # Principled specular level on the sea (0.5 is the default)
SEA_SPECULAR_TINT = (0.28, 0.36, 0.45, 1.0)   # the sky in the sea comes back a deep blue-grey, not milk-white


def haze_group():
    """Shared shader group: terrain colour → haze colour with distance from the camera (aerial perspective).
    The timeline keys its nodes: Start / Scale (m, follow the camera distance), Color, Amount."""
    import bpy

    ng = bpy.data.node_groups.get("Haze")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("Haze", "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Fog", in_out="OUTPUT", socket_type="NodeSocketFloat")     # haze share 0..1
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    cam = N.new("ShaderNodeCameraData")
    vals = {}
    for name, v in (("Start", 20_000.0), ("Scale", 60_000.0), ("Amount", 0.6)):
        n = N.new("ShaderNodeValue")
        n.name = n.label = name
        n.outputs[0].default_value = v
        vals[name] = n
    col = N.new("ShaderNodeRGB")
    col.name = col.label = "HazeColor"
    col.outputs[0].default_value = (0.75, 0.82, 0.92, 1)
    sub = N.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; sub.use_clamp = False
    L.new(cam.outputs["View Distance"], sub.inputs[0]); L.new(vals["Start"].outputs[0], sub.inputs[1])
    mx = N.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; mx.inputs[1].default_value = 0.0
    L.new(sub.outputs[0], mx.inputs[0])
    div = N.new("ShaderNodeMath"); div.operation = "DIVIDE"
    L.new(mx.outputs[0], div.inputs[0]); L.new(vals["Scale"].outputs[0], div.inputs[1])
    neg = N.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1.0
    L.new(div.outputs[0], neg.inputs[0])
    ex = N.new("ShaderNodeMath"); ex.operation = "EXPONENT"
    L.new(neg.outputs[0], ex.inputs[0])
    one = N.new("ShaderNodeMath"); one.operation = "SUBTRACT"; one.inputs[0].default_value = 1.0
    L.new(ex.outputs[0], one.inputs[1])
    amt = N.new("ShaderNodeMath"); amt.operation = "MULTIPLY"
    L.new(one.outputs[0], amt.inputs[0]); L.new(vals["Amount"].outputs[0], amt.inputs[1])
    _, fac, a, b, res = mix_rgba(ng)
    L.new(amt.outputs[0], fac)
    L.new(gin.outputs["Color"], a)
    L.new(col.outputs["Color"], b)
    L.new(res, gout.inputs["Color"])
    L.new(amt.outputs[0], gout.inputs["Fog"])
    return ng


def _feather_alpha(nt, bsdf, feather: float) -> None:
    """Alpha = smoothstep of the distance to the mesh border (generated coords are 0..1 over the bbox)."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    edge = []
    for axis in ("X", "Y"):
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(sep.outputs[axis], inv.inputs[1])
        mn = nt.nodes.new("ShaderNodeMath")
        mn.operation = "MINIMUM"
        nt.links.new(sep.outputs[axis], mn.inputs[0])
        nt.links.new(inv.outputs["Value"], mn.inputs[1])
        edge.append(mn)
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MINIMUM"
    nt.links.new(edge[0].outputs["Value"], both.inputs[0])
    nt.links.new(edge[1].outputs["Value"], both.inputs[1])
    fade = nt.nodes.new("ShaderNodeMapRange")
    fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Max"].default_value = feather
    nt.links.new(both.outputs["Value"], fade.inputs["Value"])
    # object colour (animated by the timeline): R = 1 makes the edges solid once the zoom is over,
    # alpha fades the whole layer in while the intro zooms towards it
    info = nt.nodes.new("ShaderNodeObjectInfo")
    rgb = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(info.outputs["Color"], rgb.inputs["Color"])
    solid = nt.nodes.new("ShaderNodeMath")
    solid.operation = "MAXIMUM"
    nt.links.new(fade.outputs["Result"], solid.inputs[0])
    nt.links.new(rgb.outputs["Red"], solid.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(solid.outputs["Value"], mul.inputs[0])
    nt.links.new(info.outputs["Alpha"], mul.inputs[1])
    # shadow rays pass the soft edge: half-transparent, it shaded the layer below as a dark frame around this one
    # (day 11, frame 57); where the layer is solid it still shades its own valleys
    lp = nt.nodes.new("ShaderNodeLightPath")
    band = nt.nodes.new("ShaderNodeMath"); band.operation = "LESS_THAN"; band.inputs[1].default_value = 0.98
    nt.links.new(mul.outputs["Value"], band.inputs[0])
    kill = nt.nodes.new("ShaderNodeMath"); kill.operation = "MULTIPLY"
    nt.links.new(lp.outputs["Is Shadow Ray"], kill.inputs[0]); nt.links.new(band.outputs["Value"], kill.inputs[1])
    keep = nt.nodes.new("ShaderNodeMath"); keep.operation = "SUBTRACT"; keep.inputs[0].default_value = 1.0
    nt.links.new(kill.outputs["Value"], keep.inputs[1])
    out = nt.nodes.new("ShaderNodeMath"); out.operation = "MULTIPLY"
    nt.links.new(mul.outputs["Value"], out.inputs[0]); nt.links.new(keep.outputs["Value"], out.inputs[1])
    nt.links.new(out.outputs["Value"], bsdf.inputs["Alpha"])


# --------------------------------------------------------------------------- trail

def _trail_group(style: str):
    """Geometry nodes: keep points with s <= Progress (or > with Invert), turn the line into a tube/ribbon."""
    import bpy

    ng = bpy.data.node_groups.new(f"Trail.{style}", "GeometryNodeTree")
    I = ng.interface
    I.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for nm, default in (("Progress", 0.0), ("Width", 30.0)):
        sk = I.new_socket(nm, in_out="INPUT", socket_type="NodeSocketFloat")
        sk.default_value = default
    I.new_socket("Invert", in_out="INPUT", socket_type="NodeSocketBool")
    I.new_socket("Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    I.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    s_attr = N.new("GeometryNodeInputNamedAttribute")
    s_attr.data_type = "FLOAT"
    s_attr.inputs["Name"].default_value = "s"
    cmp = N.new("FunctionNodeCompare")
    cmp.data_type, cmp.operation = "FLOAT", "GREATER_THAN"
    L.new(s_attr.outputs["Attribute"], cmp.inputs["A"])
    L.new(gin.outputs["Progress"], cmp.inputs["B"])
    xor = N.new("FunctionNodeBooleanMath")
    xor.operation = "XOR"
    L.new(cmp.outputs["Result"], xor.inputs[0])
    L.new(gin.outputs["Invert"], xor.inputs[1])
    delete = N.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"
    L.new(gin.outputs["Geometry"], delete.inputs["Geometry"])
    L.new(xor.outputs["Boolean"], delete.inputs["Selection"])
    # age = km behind the marker (≥ 0): the material makes the head of the trail glow hotter
    age = N.new("ShaderNodeMath")
    age.operation = "SUBTRACT"
    L.new(gin.outputs["Progress"], age.inputs[0])
    L.new(s_attr.outputs["Attribute"], age.inputs[1])
    age_pos = N.new("ShaderNodeMath")
    age_pos.operation = "MAXIMUM"
    age_pos.inputs[1].default_value = 0.0
    L.new(age.outputs["Value"], age_pos.inputs[0])
    store = N.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT", "POINT"
    store.inputs["Name"].default_value = "age"
    L.new(delete.outputs["Geometry"], store.inputs["Geometry"])
    L.new(age_pos.outputs["Value"], store.inputs["Value"])
    to_curve = N.new("GeometryNodeMeshToCurve")
    L.new(store.outputs["Geometry"], to_curve.inputs["Mesh"])

    half = N.new("ShaderNodeMath")
    half.operation = "MULTIPLY"
    half.inputs[1].default_value = 0.5 if style == "tube" else 0.5 * FLAT_WIDEN
    L.new(gin.outputs["Width"], half.inputs[0])
    if style == "tube":
        prof = N.new("GeometryNodeCurvePrimitiveCircle")
        prof.inputs["Resolution"].default_value = 10
        L.new(half.outputs["Value"], prof.inputs["Radius"])
    elif style == "band":                                     # flattened ellipse: with a Z-up normal profile Y is up
        circ = N.new("GeometryNodeCurvePrimitiveCircle")
        circ.inputs["Resolution"].default_value = 16
        L.new(half.outputs["Value"], circ.inputs["Radius"])
        prof = N.new("GeometryNodeTransform")
        prof.inputs["Scale"].default_value = (1.0, BAND_FLAT, 1.0)
        L.new(circ.outputs["Curve"], prof.inputs["Geometry"])
        normal = N.new("GeometryNodeSetCurveNormal")
        set_normal_z_up(normal)
        L.new(to_curve.outputs["Curve"], normal.inputs["Curve"])
        to_curve = normal
    else:
        neg = N.new("ShaderNodeMath")
        neg.operation = "MULTIPLY"
        neg.inputs[1].default_value = -1.0
        L.new(half.outputs["Value"], neg.inputs[0])
        a, b = N.new("ShaderNodeCombineXYZ"), N.new("ShaderNodeCombineXYZ")
        L.new(neg.outputs["Value"], a.inputs["X"])
        L.new(half.outputs["Value"], b.inputs["X"])
        prof = N.new("GeometryNodeCurvePrimitiveLine")
        L.new(a.outputs["Vector"], prof.inputs["Start"])
        L.new(b.outputs["Vector"], prof.inputs["End"])
        normal = N.new("GeometryNodeSetCurveNormal")
        set_normal_z_up(normal)
        L.new(to_curve.outputs["Curve"], normal.inputs["Curve"])
        to_curve = normal
    sweep = N.new("GeometryNodeCurveToMesh")
    L.new(to_curve.outputs[0], sweep.inputs["Curve"])
    L.new(prof.outputs[0], sweep.inputs["Profile Curve"])
    if "Fill Caps" in sweep.inputs:
        sweep.inputs["Fill Caps"].default_value = style != "ribbon"
    lift = N.new("ShaderNodeCombineXYZ")                      # tube / band rest on the ground
    if style != "ribbon":
        rest = N.new("ShaderNodeMath")
        rest.operation = "MULTIPLY"
        rest.inputs[1].default_value = 1.0 if style == "tube" else BAND_FLAT * 1.15
        L.new(half.outputs["Value"], rest.inputs[0])
        L.new(rest.outputs["Value"], lift.inputs["Z"])
    setp = N.new("GeometryNodeSetPosition")
    L.new(sweep.outputs["Mesh"], setp.inputs["Geometry"])
    L.new(lift.outputs["Vector"], setp.inputs["Offset"])
    smooth = N.new("GeometryNodeSetShadeSmooth")
    L.new(setp.outputs["Geometry"], smooth.inputs[0])
    setm = N.new("GeometryNodeSetMaterial")                   # slot materials do not reach generated faces
    L.new(smooth.outputs[0], setm.inputs["Geometry"])
    L.new(gin.outputs["Material"], setm.inputs["Material"])
    if style != "band":
        L.new(setm.outputs["Geometry"], gout.inputs["Geometry"])
        return ng
    # band: a dark translucent casing on the ground under it, like a road line on a map — reads on any imagery
    cw = N.new("ShaderNodeMath")
    cw.operation = "MULTIPLY"
    cw.inputs[1].default_value = BAND_CASING * FLAT_WIDEN / 2
    L.new(gin.outputs["Width"], cw.inputs[0])
    cneg = N.new("ShaderNodeMath")
    cneg.operation = "MULTIPLY"
    cneg.inputs[1].default_value = -1.0
    L.new(cw.outputs["Value"], cneg.inputs[0])
    ca, cb = N.new("ShaderNodeCombineXYZ"), N.new("ShaderNodeCombineXYZ")
    L.new(cneg.outputs["Value"], ca.inputs["X"])
    L.new(cw.outputs["Value"], cb.inputs["X"])
    cline = N.new("GeometryNodeCurvePrimitiveLine")
    L.new(ca.outputs["Vector"], cline.inputs["Start"])
    L.new(cb.outputs["Vector"], cline.inputs["End"])
    csweep = N.new("GeometryNodeCurveToMesh")
    L.new(to_curve.outputs[0], csweep.inputs["Curve"])
    L.new(cline.outputs["Curve"], csweep.inputs["Profile Curve"])
    csetm = N.new("GeometryNodeSetMaterial")
    csetm.inputs["Material"].default_value = _casing_material()
    L.new(csweep.outputs["Mesh"], csetm.inputs["Geometry"])
    join = N.new("GeometryNodeJoinGeometry")
    L.new(csetm.outputs["Geometry"], join.inputs[0])
    L.new(setm.outputs["Geometry"], join.inputs[0])
    L.new(join.outputs[0], gout.inputs["Geometry"])
    return ng


FLAT_WIDEN = 1.2            # a flat trail seen at an angle looks thinner than a tube of the same width
BAND_FLAT = 0.18             # band thickness / width
BAND_CASING = 1.8            # casing width / band width
CASING_ALPHA = 0.45


def _casing_material():
    import bpy

    m = bpy.data.materials.get("Trail.Casing") or bpy.data.materials.new("Trail.Casing")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
    b.inputs["Roughness"].default_value = 1.0
    b.inputs["Alpha"].default_value = CASING_ALPHA
    return m


def set_normal_z_up(node) -> None:
    """Set Curve Normal → Z Up (Blender 5: a menu input socket; older: the `mode` property)."""
    if "Mode" in node.inputs:
        node.inputs["Mode"].default_value = "Z Up"
    else:
        node.mode = "Z_UP"


TRAIL_EMISSION = 0.9
HEAD_EMISSION = 2.0          # extra emission right behind the marker, fading over HEAD_KM
HEAD_KM = 1.5
HEAD_WHITE = 0.22            # how much the head pales towards white


def _trail_material(name: str, alpha: float, head: bool = True):
    """Day colour with emission; head=True: the last HEAD_KM glow hotter and whiter (attribute `age`)."""
    import bpy

    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    L = nt.links
    b = nt.nodes["Principled BSDF"]
    col = nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name = "day_color"
    # day stories: earlier days are the neutral «road so far»; PastMix (keyed) greys today's trail too at the end
    foc = nt.nodes.new("ShaderNodeAttribute")
    foc.attribute_name = "focus"
    pm = nt.nodes.new("ShaderNodeValue")
    pm.name = pm.label = "PastMix"
    pm.outputs[0].default_value = 0.0
    keep = nt.nodes.new("ShaderNodeMath"); keep.operation = "SUBTRACT"; keep.inputs[0].default_value = 1.0
    L.new(pm.outputs[0], keep.inputs[1])
    today = nt.nodes.new("ShaderNodeMath"); today.operation = "MULTIPLY"
    L.new(foc.outputs["Fac"], today.inputs[0]); L.new(keep.outputs[0], today.inputs[1])
    _, pfac, pa, pb, day_col = mix_rgba(nt)
    L.new(today.outputs[0], pfac)
    pa.default_value = PAST_RGBA
    L.new(col.outputs["Color"], pb)
    color = day_col
    if head:                                                               # the head pales a little
        age = nt.nodes.new("ShaderNodeAttribute")
        age.attribute_name = "age"
        div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = -HEAD_KM
        L.new(age.outputs["Fac"], div.inputs[0])
        ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"      # 1 at the marker → 0 behind
        L.new(div.outputs[0], ex.inputs[0])
        _, fac, a, bb, color = mix_rgba(nt)
        bb.default_value = (1, 1, 1, 1)
        pale = nt.nodes.new("ShaderNodeMath"); pale.operation = "MULTIPLY"; pale.inputs[1].default_value = HEAD_WHITE
        L.new(ex.outputs[0], pale.inputs[0])
        L.new(pale.outputs[0], fac)
        L.new(day_col, a)
    # aerial haze: far parts of the trail fade into the horizon like the terrain (colour and emission)
    haze = nt.nodes.new("ShaderNodeGroup")
    haze.node_tree = haze_group()
    L.new(color, haze.inputs["Color"])
    L.new(haze.outputs["Color"], b.inputs["Base Color"])
    L.new(haze.outputs["Color"], b.inputs["Emission Color"])
    clear = nt.nodes.new("ShaderNodeMath"); clear.operation = "SUBTRACT"; clear.inputs[0].default_value = 1.0
    L.new(haze.outputs["Fog"], clear.inputs[1])
    strength = nt.nodes.new("ShaderNodeMath"); strength.operation = "MULTIPLY"
    if head:
        hot = nt.nodes.new("ShaderNodeMath"); hot.operation = "MULTIPLY_ADD"
        hot.inputs[1].default_value, hot.inputs[2].default_value = HEAD_EMISSION, TRAIL_EMISSION
        L.new(ex.outputs[0], hot.inputs[0])
        L.new(hot.outputs[0], strength.inputs[0])
    else:
        strength.inputs[0].default_value = TRAIL_EMISSION
    L.new(clear.outputs[0], strength.inputs[1])
    L.new(strength.outputs[0], b.inputs["Emission Strength"])
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Alpha"].default_value = alpha
    return m


def set_input(obj, socket: str, value) -> str:
    """Set a geometry-nodes modifier input (Blender 5 API); returns the data path to keyframe it."""
    mod = obj.modifiers[0]
    key = mod.node_group.interface.items_tree[socket].identifier
    getattr(mod.properties.inputs, key).value = value
    return f'modifiers["{mod.name}"].properties.inputs.{key}.value'


def add_trail(path: Path3D, day_rgba: dict[int, tuple], style: str, ahead: str, width_m: float,
              focus_day: int | None = None):
    """Trail behind the marker (+ 'Trail.Ahead' when the route ahead is shown faint). focus_day (a day story):
    earlier days are drawn in PAST_RGBA."""
    import bpy

    i = np.flatnonzero(path.piece[1:] == path.piece[:-1])
    me = _mesh("Trail", path.xyz, edges=np.column_stack([i, i + 1]))
    a = me.attributes.new("s", "FLOAT", "POINT")
    a.data.foreach_set("value", path.s.astype(np.float32))
    rgba = np.array([day_rgba[int(d)] for d in path.day], np.float32)
    rgba[path.transport] = TRANSPORT_RGBA                    # bus / car legs read as "not ridden"
    a = me.color_attributes.new("day_color", "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", rgba.ravel())
    focus = np.ones(len(path.day), np.float32) if focus_day is None else (path.day == focus_day).astype(np.float32)
    a = me.attributes.new("focus", "FLOAT", "POINT")
    a.data.foreach_set("value", focus)
    ng = _trail_group(style)
    objs = []
    for name, invert, alpha in (("Trail", False, 1.0), ("Trail.Ahead", True, AHEAD_ALPHA)):
        if invert and ahead != "faint":
            continue
        obj = _link(bpy.data.objects.new(name, me))
        obj.modifiers.new("Trail", "NODES").node_group = ng
        set_input(obj, "Material", _trail_material(name, alpha, head=not invert))
        set_input(obj, "Invert", invert)
        set_input(obj, "Progress", float(path.s[-1]))
        set_input(obj, "Width", float(width_m))
        objs.append(obj)
    return objs


# --------------------------------------------------------------------------- effects

def add_steam_plume(ground_xyz, height_m: float = 1800.0, radius_m: float = 450.0, fps: int = 30,
                    frame_offset: float = 0.0, density: float = 0.008, name: str = "SteamPlume"):
    """Volumetric steam column over a crater: 4D noise drifting with time, fading to the top and the sides."""
    import bpy

    obj = _link(bpy.data.objects.new(name, _sphere(name, segments=24, rings=12)))
    obj.location = (ground_xyz[0], ground_xyz[1], ground_xyz[2] + height_m / 2)
    obj.scale = (radius_m, radius_m, height_m / 2)
    obj.rotation_euler = (0.18, 0.1, 0.0)                     # leaning with the wind
    m = bpy.data.materials.new("SteamPlume")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    out = nt.nodes["Material Output"]
    vol = nt.nodes.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    vol.inputs["Anisotropy"].default_value = 0.35
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 2.2
    noise.inputs["Detail"].default_value = 6.0
    nt.links.new(object_value(nt, "clock"), noise.inputs["W"])
    object_clock(obj, "clock", f"(frame + {frame_offset:.1f}) / {fps * 3:.1f}")   # slow billowing
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.42, 0.68
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    rise = nt.nodes.new("ShaderNodeMapRange")                     # denser at the vent, thin at the top
    rise.inputs["From Min"].default_value, rise.inputs["From Max"].default_value = -1.0, 1.0
    rise.inputs["To Min"].default_value, rise.inputs["To Max"].default_value = 1.0, 0.0
    sph = nt.nodes.new("ShaderNodeTexGradient")
    sph.gradient_type = "SPHERICAL"
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    m3 = nt.nodes.new("ShaderNodeMath"); m3.operation = "MULTIPLY"; m3.inputs[1].default_value = density
    L = nt.links
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    L.new(tc.outputs["Object"], sep.inputs["Vector"])
    L.new(tc.outputs["Object"], sph.inputs["Vector"])
    L.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    L.new(sep.outputs["Z"], rise.inputs["Value"])
    L.new(ramp.outputs["Color"], m1.inputs[0])
    L.new(rise.outputs["Result"], m1.inputs[1])
    L.new(m1.outputs["Value"], m2.inputs[0])
    L.new(sph.outputs["Fac"], m2.inputs[1])
    L.new(m2.outputs["Value"], m3.inputs[0])
    L.new(m3.outputs["Value"], vol.inputs["Density"])
    L.new(vol.outputs["Volume"], out.inputs["Volume"])
    obj.data.materials.append(m)
    obj.visible_shadow = False
    return obj


FUMAROLES = 11                # vents scattered around a highlight point (Unzen Jigoku: a valley of them)
FUMAROLE_SPREAD_M = 170.0
FUMAROLE_H_M = (80.0, 200.0)
FUMAROLE_R_M = (18.0, 36.0)
FUMAROLE_DEPTH = 8.0          # nominal optical depth across a wisp (the noise and the falloff thin it ~5×)


def add_fumaroles(center_xyz, surface, fps: int = 30, frame_offset: float = 0.0, seed: int = 4):
    """A field of small steam vents: the volcano's column scaled down to ~10–20 m wisps, each billowing on
    its own clock; the density is set per wisp so each one is equally see-through whatever its size."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(FUMAROLES):
        r, a = FUMAROLE_SPREAD_M * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        x, y = center_xyz[0] + r * math.cos(a), center_xyz[1] + r * math.sin(a)
        z = float(surface.height_at(np.array([x]), np.array([y]))[0])
        rad = rng.uniform(*FUMAROLE_R_M)
        out.append(add_steam_plume((x, y, z), name="Fumarole", height_m=rng.uniform(*FUMAROLE_H_M), radius_m=rad, fps=fps,
                                   frame_offset=frame_offset + rng.uniform(0, 300), density=FUMAROLE_DEPTH / (2 * rad)))
    return out


CLOUD_BASE_M = 3200.0        # cloud layer height before exaggeration (the 20 km ride camera looks down on it)
CLOUD_SCALE_M = 22_000.0     # size of the noise cells: a few cumulus fields per frame at 20 km
CLOUD_CLEAR_M = (12_000.0, 28_000.0)   # no clouds this close to a highlight point (the orbit camera flies low)
CLOUD_MARKER_M = (5_000.0, 12_000.0)   # …nor over the marker (a cloud between the camera and the dot hides the story)
CLOUD_DRIFT_M_S = 9.0
CLOUD_FADE_M = (70_000.0, 150_000.0)   # the layer fades out as the camera gets this far (specks on the wide views)
CLOUD_NEAR_M = (2_500.0, 7_000.0)      # …and within this distance of the camera (it dips through the layer at d_min)


def add_clouds(grid: TerrainGrid, exaggeration: float, coverage: float, fps: int, clear_xy=(), cell_m: float = 2000.0,
               origin_xy=(0.0, 0.0), frame_offset: float = 0.0):
    """Flat cumulus layer: a coarse grid at CLOUD_BASE_M with a noise-driven alpha (Transparent ↔ white
    Diffuse), casting soft shadows on the terrain. Vertex attribute `clear` fades it out near highlights."""
    import bpy

    z = CLOUD_BASE_M * exaggeration
    xs = np.arange(grid.x[0], grid.x[-1] + cell_m, cell_m)
    ys = np.arange(grid.y[0], grid.y[-1] + cell_m, cell_m)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    me = _mesh("Clouds", verts, _grid_quads(len(ys), len(xs)))
    clear = np.ones(len(verts), np.float32)
    for cx, cy in clear_xy:
        d = np.hypot(verts[:, 0] - cx, verts[:, 1] - cy)
        u = np.clip((d - CLOUD_CLEAR_M[0]) / (CLOUD_CLEAR_M[1] - CLOUD_CLEAR_M[0]), 0, 1)
        clear = np.minimum(clear, (u * u * (3 - 2 * u)).astype(np.float32))
    a = me.attributes.new("clear", "FLOAT", "POINT")
    a.data.foreach_set("value", clear)
    obj = _link(bpy.data.objects.new("Clouds", me))
    obj.visible_diffuse = obj.visible_glossy = False           # no bounce light from the layer
    m = bpy.data.materials.new("Clouds")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    N.remove(N["Principled BSDF"])
    out = N["Material Output"]
    tc = N.new("ShaderNodeTexCoord")
    mapping = N.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / CLOUD_SCALE_M,) * 3
    # the pattern is anchored to UTM (origin_xy = the scene origin there): every story of a trip sees the same
    # clouds over the same places; frame_offset continues the drift across chained stories
    ox, oy = origin_xy[0] / CLOUD_SCALE_M, origin_xy[1] / CLOUD_SCALE_M
    loc = N.new("ShaderNodeCombineXYZ")
    loc.inputs["Y"].default_value = oy
    L.new(object_value(nt, "drift"), loc.inputs["X"])
    L.new(loc.outputs["Vector"], mapping.inputs["Location"])
    # Location is applied after Scale (texture units), so the speed is in units of CLOUD_SCALE_M
    object_clock(obj, "drift", f"{ox!r} + (frame + {frame_offset!r}) * {CLOUD_DRIFT_M_S / fps / CLOUD_SCALE_M!r}")
    L.new(tc.outputs["Object"], mapping.inputs["Vector"])
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.55
    L.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")                          # coverage moves the threshold of the noise
    lo = 0.74 - 0.22 * coverage
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = lo, min(lo + 0.13, 1.0)
    L.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    clr = N.new("ShaderNodeAttribute")
    clr.attribute_name = "clear"
    fac = N.new("ShaderNodeMath"); fac.operation = "MULTIPLY"
    L.new(ramp.outputs["Color"], fac.inputs[0]); L.new(clr.outputs["Fac"], fac.inputs[1])
    # keep the sky clear over the marker: its xy comes in as the layer's custom properties (follow_marker)
    marker = N.new("ShaderNodeCombineXYZ")
    L.new(object_value(nt, "marker_x"), marker.inputs["X"])
    L.new(object_value(nt, "marker_y"), marker.inputs["Y"])
    pos = N.new("ShaderNodeCombineXYZ")
    sx = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sx.inputs["Vector"])
    L.new(sx.outputs["X"], pos.inputs["X"]); L.new(sx.outputs["Y"], pos.inputs["Y"])
    dist = N.new("ShaderNodeVectorMath"); dist.operation = "DISTANCE"
    L.new(pos.outputs["Vector"], dist.inputs[0]); L.new(marker.outputs["Vector"], dist.inputs[1])
    away = N.new("ShaderNodeMapRange")
    away.interpolation_type = "SMOOTHSTEP"
    away.inputs["From Min"].default_value, away.inputs["From Max"].default_value = CLOUD_MARKER_M
    L.new(dist.outputs["Value"], away.inputs["Value"])
    fac2 = N.new("ShaderNodeMath"); fac2.operation = "MULTIPLY"
    L.new(fac.outputs[0], fac2.inputs[0]); L.new(away.outputs["Result"], fac2.inputs[1])
    fac = fac2
    cam = N.new("ShaderNodeCameraData")
    far = N.new("ShaderNodeMapRange")
    far.interpolation_type = "SMOOTHSTEP"
    far.inputs["From Min"].default_value, far.inputs["From Max"].default_value = CLOUD_FADE_M
    far.inputs["To Min"].default_value, far.inputs["To Max"].default_value = 0.82, 0.0
    L.new(cam.outputs["View Distance"], far.inputs["Value"])
    near = N.new("ShaderNodeMapRange")
    near.interpolation_type = "SMOOTHSTEP"
    near.inputs["From Min"].default_value, near.inputs["From Max"].default_value = CLOUD_NEAR_M
    L.new(cam.outputs["View Distance"], near.inputs["Value"])
    vis = N.new("ShaderNodeMath"); vis.operation = "MULTIPLY"
    L.new(far.outputs["Result"], vis.inputs[0]); L.new(near.outputs["Result"], vis.inputs[1])
    dens = N.new("ShaderNodeMath"); dens.operation = "MULTIPLY"
    L.new(fac.outputs[0], dens.inputs[0]); L.new(vis.outputs[0], dens.inputs[1])
    diff = N.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (1, 1, 1, 1)
    trans = N.new("ShaderNodeBsdfTransparent")
    mix = N.new("ShaderNodeMixShader")
    L.new(dens.outputs[0], mix.inputs["Fac"]); L.new(trans.outputs[0], mix.inputs[1]); L.new(diff.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    return obj


def add_sun_disk():
    """Emissive disc the timeline places far away in the sun's direction from the camera (seen in sunrise shots)."""
    import bpy

    obj = _link(bpy.data.objects.new("SunDisk", _sphere("SunDisk", segments=48, rings=24)))
    obj.data.materials.append(_emission_material("SunDisk", strength=14.0, from_object=True))
    obj.color = (1.0, 1.0, 1.0, 1.0)           # keyed by the timeline: sun.SUNRISE_DISKS, white-hot at a sunset
    obj.visible_shadow = False
    obj.visible_diffuse = False                # it must not light the terrain: the Sun lamp does that
    obj.visible_glossy = False                 # water mirrors it as a hard column; the lamp's own glint is the path
    for attr in ("hide_probe_volume", "hide_probe_sphere", "hide_probe_plane"):   # EEVEE: nor through its probes
        if hasattr(obj, attr):
            setattr(obj, attr, True)
    obj.scale = (0.0, 0.0, 0.0)
    return obj


# --------------------------------------------------------------------------- coast sets (sunrise_beach, sunset)

def add_sunset_set(d: Path, route: dict, day: int, proj, surface, exag: float, log=print) -> None:
    """Around the day's finish, for the sunset closing: the town as silhouettes and moving water for the sun path.
    No palms or surf — the camera looks across a harbour, not along a beach."""
    import json

    from .. import coast as CO
    from .. import landcover as LC
    from . import coast as BC

    cj = json.loads((d / "coast.json").read_text())
    lat, lon = CO.day_end(route, day)
    cx, cy = proj.to_xy(np.array([lat]), np.array([lon]))
    c = np.array([cx[0], cy[0], 0.0])
    lc = LC.LandCover.load(d / "landcover.npz") if (d / "landcover.npz").exists() else None
    houses = CO.house_boxes(cj, lc, proj, surface, c, exag) if lc is not None else np.zeros((0, 7))
    BC.add_houses(houses)
    BC.add_water(c[:2], CO.coast_lines_xy(cj, proj))
    log(f"Sunset of day {day}: {len(houses)} houses")



def add_beach_set(d: Path, route: dict, day: int, proj, surface, exag: float, log=print, at=None) -> None:
    """Palms, houses, water and surf: around the day's start for a beach sunrise, or around `at` (lat, lon) for a
    sunset spot away from the finish — then the camera lane runs towards the setting sun."""
    import json

    from .. import coast as CO
    from .. import landcover as LC
    from . import coast as BC

    cj = json.loads((d / "coast.json").read_text())
    lat, lon = CO.set_anchor(route, day, "beach", at)
    cx, cy = proj.to_xy(np.array([lat]), np.array([lon]))
    c = np.array([cx[0], cy[0], 0.0])
    lines = CO.coast_lines_xy(cj, proj)
    palms = CO.palm_points(lines, surface, c, exag)
    # palms lean towards the sea: away from the land side of the nearest coastline point
    if len(palms):
        pts = np.concatenate([CO._resample(l, 5.0)[0] for l in lines])
        dirs = np.concatenate([CO._resample(l, 5.0)[1] for l in lines])
        from scipy.spatial import cKDTree

        k = cKDTree(pts).query(palms[:, :2])[1]
        lean = np.column_stack([dirs[k, 1], -dirs[k, 0]])
    else:
        lean = np.zeros((0, 2))
    lc = LC.LandCover.load(d / "landcover.npz") if (d / "landcover.npz").exists() else None
    houses = CO.house_boxes(cj, lc, proj, surface, c, exag) if lc is not None else np.zeros((0, 7))
    from ..timeline import CameraRig

    if at is None:
        sun_d, back = CO.sunrise_dir(route, day), CameraRig().beach_back_m
    else:
        sun_d, back = CO.sunset_dir(route, day), CameraRig().sunset_spot_back_m
    houses = CO.clear_lane(houses, c, sun_d, back)
    grove = CO.grove_points(c, sun_d, back, surface, exag)
    g_lean = np.tile(sun_d, (len(grove), 1))              # the grove leans towards the sea too
    palms, lean = np.concatenate([palms.reshape(-1, 4), grove]), np.concatenate([lean.reshape(-1, 2), g_lean])
    BC.add_palms(palms, lean)
    BC.add_houses(houses)
    BC.add_water(c[:2], lines)
    BC.add_surf(lines, c[:2], CO.SET_RADIUS_M * 2)
    log(f"Beach of day {day}: {len(palms)} palms, {len(houses)} houses")


# --------------------------------------------------------------------------- marker, checkpoints

def add_marker(path: Path3D, rgba, radius_m: float):
    import bpy

    obj = _link(bpy.data.objects.new("Marker", _sphere("Marker")))
    obj.data.materials.append(_emission_material("Marker", strength=4.0, from_object=True))
    obj.color = rgba
    obj.visible_shadow = False            # a 20 km marker would shade half of Kyushu in the intro
    obj.location = tuple(path.xyz[0])
    obj.scale = (radius_m,) * 3
    return obj


PUCK_H = 0.42               # game-chip marker: height / radius
PUCK_RING = (0.58, 0.76)     # white ring on the top face, radii / radius


def add_puck(path: Path3D, rgba, radius_m: float):
    """Flat game-chip marker: a bevelled disc in the object colour with a white ring on top. Same object name and
    animation as the sphere (animate lifts it by the radius, so the mesh stands on z = -1)."""
    import bmesh
    import bpy

    me = bpy.data.meshes.new("Marker")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=1.0, radius2=1.0, depth=PUCK_H)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, -1 + PUCK_H / 2))
    rims = [e for e in bm.edges if all(abs(v.co.xy.length - 1) < 1e-4 for v in e.verts)]
    bmesh.ops.bevel(bm, geom=rims, offset=0.12, segments=3, affect="EDGES")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    m = _emission_material("Marker", strength=4.0, from_object=True)
    nt, L = m.node_tree, m.node_tree.links
    b = nt.nodes["Principled BSDF"]
    info = next(n for n in nt.nodes if n.type == "OBJECT_INFO")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    r = nt.nodes.new("ShaderNodeVectorMath"); r.operation = "LENGTH"
    xy = nt.nodes.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["X"], xy.inputs["X"]); L.new(sep.outputs["Y"], xy.inputs["Y"])
    L.new(xy.outputs["Vector"], r.inputs[0])
    ring = nt.nodes.new("ShaderNodeMapRange"); ring.interpolation_type = "SMOOTHSTEP"
    # 1 inside PUCK_RING: distance from the ring's middle against its half width
    mid = nt.nodes.new("ShaderNodeMath"); mid.operation = "ABSOLUTE"
    off = nt.nodes.new("ShaderNodeMath"); off.operation = "SUBTRACT"
    off.inputs[1].default_value = sum(PUCK_RING) / 2
    L.new(r.outputs["Value"], off.inputs[0]); L.new(off.outputs[0], mid.inputs[0])
    hw = (PUCK_RING[1] - PUCK_RING[0]) / 2
    ring.inputs["From Min"].default_value, ring.inputs["From Max"].default_value = hw, hw * 0.8
    L.new(mid.outputs[0], ring.inputs["Value"])
    _, fac, a, w, col = mix_rgba(nt)
    w.default_value = (1, 1, 1, 1)
    L.new(info.outputs["Color"], a)
    L.new(ring.outputs["Result"], fac)
    L.new(col, b.inputs["Base Color"])
    L.new(col, b.inputs["Emission Color"])
    obj = _link(bpy.data.objects.new("Marker", me))
    obj.data.materials.append(m)
    obj.color = rgba
    obj.visible_shadow = False
    obj.location = tuple(path.xyz[0])
    obj.scale = (radius_m,) * 3
    return obj


def add_checkpoints(points: list[dict], radius_m: float):
    """points: {'label', 's', 'xyz'}; hidden (scale 0) until the timeline reaches them."""
    import bpy

    me = _sphere("Checkpoint", segments=24, rings=12)
    me.materials.append(_emission_material("Checkpoint", (1, 1, 1, 1), strength=3.0))
    out = []
    for k, p in enumerate(points):
        o = _link(bpy.data.objects.new(f"Checkpoint.{k}", me))
        o.location = tuple(p["xyz"])
        o.scale = (radius_m,) * 3
        o["s"], o["label"] = float(p["s"]), p["label"]
        out.append(o)
    return out


# --------------------------------------------------------------------------- gaps

def add_gap_arcs(arcs: list[np.ndarray], radius_m: float, dash_m: float):
    import bpy

    if not arcs:
        return None
    cu = bpy.data.curves.new("Gaps", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius_m
    cu.bevel_resolution = 2
    for pts in arcs:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        k = 0.0
        while k < s[-1]:
            t = np.linspace(k, min(k + dash_m, s[-1]), 6)
            dash = np.column_stack([np.interp(t, s, pts[:, j]) for j in range(3)])
            sp = cu.splines.new("POLY")
            sp.points.add(len(dash) - 1)
            sp.points.foreach_set("co", np.column_stack([dash, np.ones(len(dash))]).astype(np.float32).ravel())
            k += 2 * dash_m
    m = bpy.data.materials.new("Gaps")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1)
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    b.inputs["Emission Strength"].default_value = 0.8
    cu.materials.append(m)
    return _link(bpy.data.objects.new("Gaps", cu))


# --------------------------------------------------------------------------- bike, light, camera

def heading_rotation(direction_xy) -> float:
    """Z rotation that turns the bike's +Y (forward) to direction_xy."""
    return math.atan2(-float(direction_xy[0]), float(direction_xy[1]))


def add_bike(path: Path3D, color: str, scale: float, lift_m: float):
    from .bike import build_grizl

    root = build_grizl(color)
    p = path.xyz[0].copy()
    p[2] -= lift_m                    # path is lifted for the ribbon; the bike stands on the ground
    root.location = tuple(p)
    root.rotation_euler = (0, 0, heading_rotation(path.heading()[0]))
    root.scale = (scale,) * 3
    return root


def _sun_lamp(name: str, elevation_deg: float, azimuth_deg: float, energy: float, angle_deg: float = 1.0):
    import bpy

    lamp = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    lamp.data.energy = energy
    lamp.data.angle = math.radians(angle_deg)
    # light travels along the lamp's local -Z; azimuth is where the sun is, clockwise from north
    lamp.rotation_euler = (math.radians(90 - elevation_deg), 0, math.radians(180 - azimuth_deg))
    return _link(lamp)


def add_light(sun_elevation_deg: float = 38, sun_azimuth_deg: float = 225):
    """Sun + moon lamps and a gradient sky (nodes Zenith / Horizon, keyed by the timeline with sun.sky_look)."""
    import bpy

    from ..sun import DAY_HORIZON, DAY_SKY, MOON_AZ_EL, MOON_COLOR

    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.7
    zen, hor = nt.nodes.new("ShaderNodeRGB"), nt.nodes.new("ShaderNodeRGB")
    zen.name = zen.label = "Zenith"
    hor.name = hor.label = "Horizon"
    zen.outputs[0].default_value = (*DAY_SKY, 1)
    hor.outputs[0].default_value = (*DAY_HORIZON, 1)
    tc = nt.nodes.new("ShaderNodeTexCoord")               # Generated = view direction for a world shader
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
    up = nt.nodes.new("ShaderNodeMapRange")                # horizon band: 0 at -3° … 1 at +25° above it
    up.interpolation_type = "SMOOTHSTEP"
    up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = -0.05, 0.42
    nt.links.new(sep.outputs["Z"], up.inputs["Value"])
    _, fac, a, b, res = mix_rgba(nt)
    nt.links.new(up.outputs["Result"], fac)
    nt.links.new(hor.outputs[0], a)
    nt.links.new(zen.outputs[0], b)
    # Ambient (keyed): how much of the sky lights the scene. The camera always sees the sky itself in full —
    # at a sunset the sky is bright orange while the shore in front of it stays a dark silhouette
    amb = nt.nodes.new("ShaderNodeValue")
    amb.name = amb.label = "Ambient"
    amb.outputs[0].default_value = 1.0
    lp = nt.nodes.new("ShaderNodeLightPath")
    # mirror rays see the sky at `Mirror` (keyed): a pale horizon mirrored at grazing angles turned far bays
    # milk-white (day 12, Nagasaki); at a sunset 1 — the sea's gold is the sky's
    mir = nt.nodes.new("ShaderNodeValue")
    mir.name = mir.label = "Mirror"
    mir.outputs[0].default_value = 1.0
    gl = nt.nodes.new("ShaderNodeMix")                   # glossy ? Mirror : Ambient
    gl.data_type = "FLOAT"
    nt.links.new(lp.outputs["Is Glossy Ray"], gl.inputs[0])
    nt.links.new(amb.outputs[0], gl.inputs[2])
    nt.links.new(mir.outputs[0], gl.inputs[3])
    k = nt.nodes.new("ShaderNodeMath")                   # the camera always sees the full sky
    k.operation = "MAXIMUM"
    nt.links.new(lp.outputs["Is Camera Ray"], k.inputs[0])
    nt.links.new(gl.outputs[0], k.inputs[1])
    sc = nt.nodes.new("ShaderNodeVectorMath")
    sc.operation = "SCALE"
    nt.links.new(res, sc.inputs[0])
    nt.links.new(k.outputs[0], sc.inputs["Scale"])
    nt.links.new(sc.outputs["Vector"], bg.inputs["Color"])
    bpy.context.scene.world = world
    moon = _sun_lamp("Moon", MOON_AZ_EL[1], MOON_AZ_EL[0], 0.0, angle_deg=3.0)
    moon.data.color = tuple(MOON_COLOR)
    return _sun_lamp("Sun", sun_elevation_deg, sun_azimuth_deg, 3.5)


def add_overview_camera(path: Path3D, grid: TerrainGrid, res: tuple[int, int], pitch_deg: float = 60, lens: float = 40):
    """Camera from the south, tilted, framing the whole route."""
    import bpy

    cam = bpy.data.objects.new("Cam.Overview", bpy.data.cameras.new("Cam.Overview"))
    cam.data.lens = lens
    cam.data.sensor_fit = "VERTICAL"
    lo, hi = path.xyz[:, :2].min(0), path.xyz[:, :2].max(0)
    c = (lo + hi) / 2
    ex, ey = hi - lo
    pitch = math.radians(pitch_deg)
    tan_v = cam.data.sensor_height / 2 / lens
    tan_h = tan_v * res[0] / res[1]
    d = 1.12 * max(ey * math.sin(pitch) / 2 / tan_v, ex / 2 / tan_h)
    z0 = float(np.median(path.xyz[:, 2]))
    cam.location = (c[0], c[1] - d * math.cos(pitch), z0 + d * math.sin(pitch))
    cam.rotation_euler = (math.pi / 2 - pitch, 0, 0)
    cam.data.clip_start = 10
    cam.data.clip_end = 4 * d + 2 * float(grid.z.max())
    return _link(cam)


# --------------------------------------------------------------------------- render settings

def setup_render(res: tuple[int, int], fps: int, samples: int = 32):
    """Cycles; Metal GPU on macOS, CPU elsewhere."""
    import bpy

    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.render.fps = fps
    s.view_settings.view_transform = "Standard"      # keep satellite colours as they are
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    fast_cycles(s)
    use_best_device(s)


def add_grade(bloom: float, contrast: float, vignette: float) -> None:
    """Compositor look: bloom from the emission pass only (trail, marker, checkpoints — never the terrain),
    a mild S-curve with a little saturation, and a vignette from a packed image scaled to the render size
    (so drafts at 25 % and the final look the same). Zero switches a stage off."""
    import bpy

    s = bpy.context.scene
    s.render.compositor_device = "CPU"                # the GPU compositor needs a display; CPU is fast enough
    s.render.use_compositing = True
    s.view_layers[0].use_pass_emit = bloom > 0
    nt = bpy.data.node_groups.new("Grade", "CompositorNodeTree")
    s.compositing_node_group = nt
    nt.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = nt.nodes, nt.links
    rl = N.new("CompositorNodeRLayers")
    out = N.new("NodeGroupOutput")
    img = rl.outputs["Image"]
    if bloom > 0:
        glare = N.new("CompositorNodeGlare")
        glare.inputs["Type"].default_value = "Bloom"
        glare.inputs["Threshold"].default_value = 0.0
        glare.inputs["Strength"].default_value = bloom
        glare.inputs["Size"].default_value = 0.7
        L.new(rl.outputs["Emission"], glare.inputs["Image"])
        add = N.new("ShaderNodeMix")
        add.data_type, add.blend_type = "RGBA", "ADD"
        add.inputs["Factor"].default_value = 1.0
        L.new(img, add.inputs[6])
        L.new(glare.outputs["Glare"], add.inputs[7])
        img = add.outputs[2]
    if contrast > 0:
        curve = N.new("CompositorNodeCurveRGB")
        c = curve.mapping.curves[3]
        c.points.new(0.25, 0.25 - 0.07 * contrast)
        c.points.new(0.75, 0.75 + 0.07 * contrast)
        curve.mapping.update()
        L.new(img, curve.inputs["Image"])
        hs = N.new("CompositorNodeHueSat")
        hs.inputs["Saturation"].default_value = 1.0 + 0.16 * contrast
        L.new(curve.outputs["Image"], hs.inputs["Image"])
        img = hs.outputs["Image"]
    if vignette > 0:
        w, h = 270, 480
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / 1.42
        v = 1 - vignette * np.clip((r - 0.45) / 0.6, 0, 1) ** 2
        tex = bpy.data.images.new("Vignette", w, h, alpha=False, float_buffer=True)
        tex.pixels.foreach_set(np.dstack([v, v, v, np.ones_like(v)]).astype(np.float32)[::-1].ravel())
        tex.pack()
        im = N.new("CompositorNodeImage")
        im.image = tex
        sc = N.new("CompositorNodeScale")
        sc.inputs["Type"].default_value = "Render Size"
        L.new(im.outputs["Image"], sc.inputs["Image"])
        mul = N.new("ShaderNodeMix")
        mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        L.new(img, mul.inputs[6])
        L.new(sc.outputs["Image"], mul.inputs[7])
        img = mul.outputs[2]
    L.new(img, out.inputs[0])


def fast_cycles(scene) -> None:
    """Cycles light paths trimmed for this kind of scene (measured on Kyushu frames: 64 → 32 samples with a
    0.05 adaptive threshold renders 1.55× faster at PSNR ≥ 44 dB vs the defaults; fewer bounces cost nothing)."""
    c = scene.cycles
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces, c.transmission_bounces = 4, 2, 2, 2
    c.caustics_reflective = c.caustics_refractive = False
    c.volume_step_rate = 2.0
    c.use_adaptive_sampling, c.adaptive_threshold = True, 0.05


def setup_eevee(scene, samples: int = 32) -> None:
    """EEVEE (rasterised, GPU): seconds → fractions of a second per frame on the Mac. Untested on the devbox (no
    GPU): volumes need a range reaching the steam over Aso from the drone camera."""
    scene.render.engine = "BLENDER_EEVEE"
    e = scene.eevee
    e.taa_render_samples = samples
    e.use_shadows = True
    # no screen-space tracing: on the Mac it streaked the glossy sea with light stripes and the fast GI spread
    # the marker's glow over the terrain as a red halo, both changing every frame (day 11's night overview).
    # Reflections come from the world probe — stable; the sunset's glitter path is the sun lamp's own specular
    e.use_raytracing = False
    e.use_fast_gi = False
    e.use_volume_custom_range = True
    e.volumetric_start, e.volumetric_end = 10.0, 200_000.0
    e.volumetric_sample_distribution = 0.9
    e.volumetric_tile_size = "8"          # 4 cost too much on the Mac (3.5 s per frame); the steam is soft anyway
    e.volumetric_samples = 32
    e.use_volumetric_shadows = False
    e.shadow_resolution_scale = 0.5
    e.shadow_pool_size = "2048"           # 512 MB overflowed on a 200 km terrain: single frames came out darker


def use_best_device(scene) -> str:
    """Metal GPU on macOS, CPU elsewhere. Device preferences are not saved in the .blend: call before rendering."""
    import bpy

    scene.cycles.device = "CPU"
    scene.render.compositor_device = "CPU"
    if sys.platform == "darwin":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = True
        scene.cycles.device = "GPU"
    return scene.cycles.device


def render_still(camera, out: Path):
    import bpy

    s = bpy.context.scene
    s.camera = camera
    s.render.filepath = str(Path(out).resolve())
    bpy.ops.render.render(write_still=True)
    return out




# --------------------------------------------------------------------------- entry

def checkpoints_of(path: Path3D, lift_m: float) -> list[dict]:
    """Start of every day and the finish, on the ground."""
    out = []
    for d in np.unique(path.day):
        i = int(np.flatnonzero(path.day == d)[0])
        out.append({"label": f"Day {int(d)}", "day": int(d), "s": float(path.s[i]), "xyz": path.xyz[i] - [0, 0, lift_m]})
    out.append({"label": "Finish", "day": int(path.day[-1]), "s": float(path.s[-1]), "xyz": path.xyz[-1] - [0, 0, lift_m]})
    return out


def build_scene(p: TripPaths, step_m: float = 100.0, stills: bool = False, samples: int = 32, log=print) -> Path:
    """route.json + track.yaml + terrain/ → build/world.npz, build/scene.blend (+ overview still)."""
    import json

    import bpy

    from .. import world as W
    from ..ingest import load_route
    from ..terrain import Dem
    from .bike import hex_to_rgba, reset_scene

    build = p.build
    route = load_route(p.route)
    sb = p.storyboard(route)
    tdir = build / "terrain"
    exag = sb.style.exaggeration
    lift = W.RIBBON_LIFT_M * exag

    proj = W.projection_of(route)
    grid = W.terrain_grid(proj, Dem.load(tdir), json.loads((tdir / "terrain.json").read_text()), exag, step_m)
    modes = sb.gap_modes()
    from ..routing import road_gaps
    roads = {}
    for d, poly in road_gaps(route, modes, build / "roads.json", log).items():
        rx, ry = proj.to_xy(np.array([p[0] for p in poly]), np.array([p[1] for p in poly]))
        roads[d] = np.column_stack([rx, ry])
    details = []                                         # high-res patches around highlights (detail: high_res_dem)
    for i, h in enumerate(sb.highlights):
        d = tdir / f"detail{i}"
        if h.detail == "high_res_dem" and (d / "terrain.json").exists():
            details.append((i, d, W.terrain_grid(proj, Dem.load(d), json.loads((d / "terrain.json").read_text()),
                                                 exag, DETAIL_STEP_M)))
    from .. import coast as CO

    for day, _kind, sub, tag, _at in CO.coast_sets(sb):  # beach sunrise / sunset: high-res patch around the shore
        d = tdir / sub
        if (d / "terrain.json").exists():
            g = W.terrain_grid(proj, Dem.load(d), json.loads((d / "terrain.json").read_text()), exag, DETAIL_STEP_M)
            if (d / "coast.json").exists():
                CO.flatten_sea(g, CO.coast_lines_xy(json.loads((d / "coast.json").read_text()), proj), exag)
            details.append((tag, d, g))
        else:
            log(f"⚠ no {d} — run `gpx2reel terrain` (day {day} will have no close-up shore)")
    surface = W.Surface(grid, [g for _, _, g in details])
    path = W.join_gaps(route, W.route_path(route, proj, surface, exag), surface, exag, modes, roads=roads)
    poi_meta = {}
    for h in sb.highlights:
        if h.lat is not None and h.lon is not None:
            hx, hy = proj.to_xy(np.array([h.lat]), np.array([h.lon]))
            poi_meta[h.poi] = [float(hx[0]), float(hy[0]), float(surface.height_at(hx, hy)[0])]
    for dp in sb.days:                                  # a sunset spot away from the finish: where the closing looks
        if dp.closing and dp.closing_at:
            hx, hy = proj.to_xy(np.array([dp.closing_at[0]]), np.array([dp.closing_at[1]]))
            poi_meta[f"closing{dp.day}"] = [float(hx[0]), float(hy[0]), float(surface.height_at(hx, hy)[0])]
    arcs = W.gap_arcs(route, path, modes)
    layers = []                                          # intro-zoom context, widest first
    zoom_meta = []
    for i, z in enumerate(sb.intro.zoom):
        c = z.center or W.route_center(route)
        cx, cy = proj.to_xy(np.array([c[0]]), np.array([c[1]]))
        zoom_meta.append({"extent_m": z.extent_km * 1000, "center_xy": [float(cx[0]), float(cy[0])], "label": z.label})
        d = tdir / f"ctx{i}"
        if (d / "terrain.json").exists():
            layers.append((d, W.terrain_grid(proj, Dem.load(d), json.loads((d / "terrain.json").read_text()),
                                             exag, z.extent_km * 1000 / 450)))
        else:
            log(f"⚠ no {d} — run `gpx2reel terrain` (zoom {z.label or i} will have no backdrop)")
    checkpoints = checkpoints_of(path, lift)
    n_days = route["totals"]["days"]
    day_hex = sb.style.colors_for(n_days)
    W.save_world(build / "world.npz", grid, path, arcs, {
        "exaggeration": exag, "step_m": step_m, "projection": proj.to_dict(), "lift_m": lift,
        "zoom": zoom_meta, "day_colors": day_hex,
        "checkpoints": [{**c, "xyz": [float(v) for v in c["xyz"]]} for c in checkpoints],
        "highlights": poi_meta,
    })
    log(f"Terrain {grid.z.shape[1]}×{grid.z.shape[0]} (step {step_m:.0f} m), path {len(path.s)} points, "
        f"arc gaps: {len(arcs)}, zoom layers: {len(layers)}")

    reset_scene()
    ortho = tdir / "ortho.jpg"
    # a finer layer fades in when the camera gets within 2× the distance its zoom level is framed from
    from ..timeline import CameraRig, overview_pose
    rig = CameraRig()
    level_dist = [z["extent_m"] / 2 / (12.0 / rig.lens_mm) for z in zoom_meta]
    ov_cam, ov_tgt = overview_pose(path, sb.resolution[0] / sb.resolution[1], rig)
    holes = [W.Surface.rect(g, DETAIL_FEATHER * len(g.x) + 2) for _, _, g in details]
    add_terrain.base_sea = None
    coast = None                                         # the trip's coastline: sea / land for every layer near it
    if (tdir / "coastline.json").exists():
        from .. import coast as CO

        coast = CO.coast_lines_xy(json.loads((tdir / "coastline.json").read_text()), proj)
    main = add_terrain(grid, ortho if ortho.exists() else None, feather=FEATHER if layers else 0.0, holes=holes,
                       coast=coast)
    for i, d, g in details:
        o = d / "ortho.jpg"
        det = add_terrain(g, o if o.exists() else None, name=f"Detail.{i}", feather=DETAIL_FEATHER,
                          sea_like=getattr(add_terrain, "base_sea", None), coast=coast)
        det.color = (0.0, 1.0, 1.0, 1.0)             # R = 0: keep the feathered edge (see _feather_alpha)
        det.location.z = 0.5                         # above the base grid in the feather band
    if layers:
        main["fade_dist"] = float(np.linalg.norm(ov_cam - ov_tgt))
        # hard edges only while the camera is close; wide views feather them again. From the terrain size, not
        # the route, so chained day stories agree at the cut
        main["solid_dist"] = 0.5 * float(max(np.ptp(grid.x), np.ptp(grid.y)))
    for k, (d, g) in enumerate(layers):
        o = d / "ortho.jpg"
        ctx = add_terrain(g, o if o.exists() else None, name=f"Context.{k}",
                          drop_m=CONTEXT_DROP * zoom_meta[k]["extent_m"], feather=FEATHER if k > 0 else 0.0)
        if k > 0:
            ctx["fade_dist"] = level_dist[k]
            ctx["solid_dist"] = 0.5 * float(max(np.ptp(g.x), np.ptp(g.y)))
    if sb.style.terrain in ("stylized", "hybrid"):
        from .stylized import apply as stylize

        stylize(sb.style.terrain, build, proj, grid, layers, exag, path.xyz[:, :2], log,
                details=details, surface=surface)
    extent = float(np.ptp(path.xyz[:, :2], axis=0).max())
    day_rgba = {d: hex_to_rgba(h) for d, h in enumerate(day_hex, start=1)}
    trail = add_trail(path, day_rgba, sb.style.trail, sb.style.ahead, width_m=extent / 250, focus_day=sb.focus_day)
    add_checkpoints(checkpoints, radius_m=extent / 300)
    gaps = add_gap_arcs([a["points"] for a in arcs], radius_m=extent / 1000, dash_m=max(150.0, extent / 400))
    if sb.avatar.model.startswith("procedural:grizl"):
        add_bike(path, sb.avatar.color, sb.avatar.scale, lift)
    elif sb.avatar.model == "puck":
        add_puck(path, day_rgba[1], radius_m=extent / 150)
    else:
        add_marker(path, day_rgba[1], radius_m=extent / 150)
    for day, kind, sub, _tag, at in CO.coast_sets(sb):
        d = tdir / sub
        if not (d / "coast.json").exists():
            continue
        if kind == "sunset":
            add_sunset_set(d, route, day, proj, surface, exag, log)
        else:
            add_beach_set(d, route, day, proj, surface, exag, log, at=at)
    for h in sb.highlights:
        if "fumaroles" in h.effects and h.poi in poi_meta:
            add_fumaroles(poi_meta[h.poi], surface, fps=sb.fps, frame_offset=sb.clock_offset_s * sb.fps)
        if "steam_plume" in h.effects and h.poi in poi_meta:
            add_steam_plume(poi_meta[h.poi], height_m=1800.0, radius_m=450.0, fps=sb.fps,
                            frame_offset=sb.clock_offset_s * sb.fps)
    if sb.style.clouds > 0:
        clouds = add_clouds(grid, exag, sb.style.clouds, sb.fps, clear_xy=[p[:2] for p in poi_meta.values()],
                            origin_xy=(proj.x0, proj.y0), frame_offset=sb.clock_offset_s * sb.fps)
        follow = bpy.data.objects.get("Marker") or bpy.data.objects.get("Grizl")
        for axis in ("X", "Y"):                            # the clear hole over the marker follows it
            object_clock(clouds, f"marker_{axis.lower()}", "m", [("m", follow, f"LOC_{axis}")])
    add_light()
    add_sun_disk()
    res = tuple(sb.resolution)
    overview = add_overview_camera(path, grid, res)
    overview.data.clip_end = 1e7
    bpy.context.scene.camera = overview
    setup_render(res, sb.fps, samples)
    add_grade(sb.style.bloom, sb.style.contrast, sb.style.vignette)
    blend = build / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend.resolve()))
    log(f"→ {blend}")
    if stills:
        log(f"→ {render_still(overview, build / 'still_overview.png')}")
    return blend
