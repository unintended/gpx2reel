"""Stylized / hybrid ground (style.terrain): land-cover palette texture, 3D low-poly trees, roads, rivers.

Trees: forest points are scattered once (numpy, from WorldCover); geometry nodes keep a share of them that
falls off as 1/d² from the active camera and scale each tree with d, so the forest keeps the same on-screen
density and tree size whether the camera is 2 km or 25 km away — without millions of instances.
Roads / rivers: draped polylines swept into flat ribbons whose width follows the camera distance
(object property "screen_width" → animate.py keys the modifier's Width input).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .. import landcover as LC
from ..world import TerrainGrid
from .scene import _link, _mesh, set_input, set_normal_z_up

TREE_SPACING_M = 150.0         # scatter grid for forest points
# The tree layer is static: which trees exist, where and how big is decided once (seeded). Tuned for the
# 15–25 km drone altitude; the whole layer shrinks away uniformly when the camera pulls back (intro/outro).
TREE_KEEP = 0.22               # share of forest points that get a tree (≈ one per 320 m)
TREE_HEIGHT_M = 190.0          # ±20 %
ROUTE_CLEAR_M = 700.0          # no trees this close to the route (they would hide the trail)
CORRIDOR_KM = (18.0, 30.0)     # tree probability fades from 1 to 0 between these distances from the route
LAYER_FADE_M = (45_000.0, 110_000.0)   # layer scale 1 → 0 as the camera distance grows (animate.py keys it)
ROAD_SCREEN = 0.0016           # ribbon width / camera distance (motorway ×1.7)
RIVER_SCREEN = 0.0013
LIFT_M = 2.0
RIBBON_HIDE_BEYOND = 90_000.0  # roads / rivers fade out as the camera goes further (a 3000 km view is a smudge)

FOLIAGE = ["#4E8552", "#5F9659", "#72A463", "#7FAB5E"]
TRUNK = "#6B5443"
ROAD = "#F3EEE4"
RIVER = "#5E9DC4"


def _lin(h: str, a: float = 1.0):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return (*[v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c], a)


# --------------------------------------------------------------------------- ground material

def style_material(obj, texture_png: Path, grid: TerrainGrid, name: str):
    """Replace the terrain colour with the baked style texture (StyleUV = grid rectangle)."""
    import bpy

    me = obj.data
    # StyleUV from vertex positions: u = x across the grid, v = y up
    co = np.zeros(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    loops = np.zeros(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loops)
    u = (co[loops, 0] - grid.x[0]) / (grid.x[-1] - grid.x[0])
    v = (co[loops, 1] - grid.y[0]) / (grid.y[-1] - grid.y[0])
    uv = me.uv_layers.new(name="StyleUV")
    uv.data.foreach_set("uv", np.column_stack([u, v]).astype(np.float32).ravel())

    m = obj.material_slots[0].material
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    haze = next((n for n in nt.nodes if n.type == "GROUP" and n.node_tree and n.node_tree.name == "Haze"), None)
    target = haze.inputs["Color"] if haze is not None else bsdf.inputs["Base Color"]   # keep the aerial haze
    for link in list(target.links):
        nt.links.remove(link)
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "StyleUV"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(texture_png).resolve()))
    tex.extension = "EXTEND"
    tex.interpolation = "Cubic"
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], target)
    # water (texture alpha) is glossy
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.92
    rough.inputs["To Max"].default_value = 0.34          # glossy but not a blown-out sun glint
    nt.links.new(tex.outputs["Alpha"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    tex.image.alpha_mode = "CHANNEL_PACKED"
    return m


def tint_satellite(obj, saturation: float = 0.8, value: float = 1.08):
    """Soften a satellite layer so it sits next to the stylized palette (the widest intro level)."""
    m = obj.material_slots[0].material
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    links = list(bsdf.inputs["Base Color"].links)
    if not links:
        return
    src = links[0].from_socket
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = saturation
    hsv.inputs["Value"].default_value = value
    nt.links.new(src, hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])


# --------------------------------------------------------------------------- trees

def forest_points(proj, grid: TerrainGrid, lc: LC.LandCover, route_xy: np.ndarray | None = None,
                  seed: int = 7, height=None) -> np.ndarray:
    """(n, 5): x, y, z, rand, distance to the route — jittered grid points on tree cover inside the grid."""
    rng = np.random.default_rng(seed)
    xs = np.arange(grid.x[0], grid.x[-1], TREE_SPACING_M)
    ys = np.arange(grid.y[0], grid.y[-1], TREE_SPACING_M)
    gx, gy = np.meshgrid(xs, ys)
    x = gx.ravel() + rng.uniform(-0.45, 0.45, gx.size) * TREE_SPACING_M
    y = gy.ravel() + rng.uniform(-0.45, 0.45, gy.size) * TREE_SPACING_M
    lat, lon = proj.to_latlon(x, y)
    cls = lc.sample(np.asarray(lat), np.asarray(lon))
    keep = (cls == LC.TREE) | (cls == LC.MANGROVE)
    x, y = x[keep], y[keep]
    z = (height or grid).height_at(x, y)
    keep = z > 1.0
    x, y, z = x[keep], y[keep], z[keep]
    if route_xy is not None and len(route_xy):
        from scipy.spatial import cKDTree

        rd = cKDTree(route_xy).query(np.column_stack([x, y]))[0]
    else:
        rd = np.full(len(x), 1e9)
    return np.column_stack([x, y, z, rng.random(len(x)), rd])


def _tree_meshes():
    """Two low-poly tree shapes 1 m tall, origin at the base: a conifer and a round broadleaf."""
    import bmesh
    import bpy

    out = []
    for kind in ("conifer", "broadleaf"):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=0.05, radius2=0.04, depth=0.25,
                              matrix=_translate(0, 0, 0.125))
        trunk_faces = list(bm.faces)
        if kind == "conifer":
            bmesh.ops.create_cone(bm, cap_ends=True, segments=7, radius1=0.3, radius2=0.0, depth=0.85,
                                  matrix=_translate(0, 0, 0.15 + 0.425))
        else:
            bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.33, matrix=_translate(0, 0, 0.62))
        me = bpy.data.meshes.new(f"Tree.{kind}")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.material_index = 1 if p.index < len(trunk_faces) else 0
        out.append(me)
    return out


def _translate(x, y, z):
    from mathutils import Matrix

    return Matrix.Translation((x, y, z))


def _foliage_material():
    import bpy

    m = bpy.data.materials.new("Tree.Foliage")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.8
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "INSTANCER"
    attr.attribute_name = "tint"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].color = _lin(FOLIAGE[0])
    els[1].color = _lin(FOLIAGE[2])
    e = els.new(0.4)
    e.color = _lin(FOLIAGE[1])
    e = els.new(0.75)
    e.color = _lin(FOLIAGE[3])
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def _plain_material(name, hex_):
    import bpy

    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = _lin(hex_)
    b.inputs["Roughness"].default_value = 0.8
    return m


def select_trees(points: np.ndarray, seed: int = 11) -> np.ndarray:
    """Static subset of forest points: seeded share, clear of the route, soft corridor edge."""
    rng = np.random.default_rng(seed)
    rd = points[:, 4]
    lo, hi = CORRIDOR_KM[0] * 1000, CORRIDOR_KM[1] * 1000
    p = TREE_KEEP * np.clip((hi - rd) / (hi - lo), 0, 1)
    keep = (rng.random(len(points)) < p) & (rd > ROUTE_CLEAR_M)
    return points[keep]


def add_trees(points: np.ndarray):
    """Instances on fixed points; per-tree shape / size / rotation / tint from the point's seeded rand.
    One animated modifier input (Scale) shrinks the whole layer uniformly — nothing depends on the camera."""
    import bpy

    me = _mesh("Forest.Points", points[:, :3])
    a = me.attributes.new("rand", "FLOAT", "POINT")
    a.data.foreach_set("value", points[:, 3].astype(np.float32))
    obj = _link(bpy.data.objects.new("Forest", me))

    trees = _tree_meshes()
    fol, trunk = _foliage_material(), _plain_material("Tree.Trunk", TRUNK)
    coll = bpy.data.collections.new("TreeShapes")
    bpy.context.scene.collection.children.link(coll)
    for tm in trees:
        tm.materials.append(fol)
        tm.materials.append(trunk)
        coll.objects.link(bpy.data.objects.new(tm.name, tm))
    coll.hide_render = True
    coll.hide_viewport = True

    ng = bpy.data.node_groups.new("Forest", "GeometryNodeTree")
    I = ng.interface
    I.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    sc = I.new_socket("Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    sc.default_value = 1.0
    I.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    def math(op, a, b=None, value=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        L.new(a, n.inputs[0])
        if b is not None:
            L.new(b, n.inputs[1])
        elif value is not None:
            n.inputs[1].default_value = value
        return n.outputs[0]

    rnd = N.new("GeometryNodeInputNamedAttribute")
    rnd.data_type = "FLOAT"
    rnd.inputs["Name"].default_value = "rand"
    r = rnd.outputs["Attribute"]
    shapes = N.new("GeometryNodeCollectionInfo")
    shapes.inputs["Collection"].default_value = coll
    shapes.inputs["Separate Children"].default_value = True
    shapes.inputs["Reset Children"].default_value = True
    inst = N.new("GeometryNodeInstanceOnPoints")
    L.new(gin.outputs["Geometry"], inst.inputs["Points"])
    L.new(shapes.outputs[0], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True
    L.new(math("GREATER_THAN", math("FRACT", math("MULTIPLY", r, value=37.0)), value=0.6), inst.inputs["Instance Index"])
    var = math("ADD", math("MULTIPLY", math("FRACT", math("MULTIPLY", r, value=91.0)), value=0.4), value=0.8)
    size = math("MULTIPLY", math("MULTIPLY", var, value=TREE_HEIGHT_M), gin.outputs["Scale"])
    L.new(size, inst.inputs["Scale"])
    rot = N.new("ShaderNodeCombineXYZ")
    L.new(math("MULTIPLY", r, value=628.3), rot.inputs["Z"])
    L.new(rot.outputs[0], inst.inputs["Rotation"])
    tint = N.new("GeometryNodeStoreNamedAttribute")
    tint.data_type = "FLOAT"
    tint.domain = "INSTANCE"
    tint.inputs["Name"].default_value = "tint"
    L.new(inst.outputs["Instances"], tint.inputs["Geometry"])
    L.new(math("FRACT", math("MULTIPLY", r, value=13.0)), tint.inputs["Value"])
    L.new(tint.outputs["Geometry"], gout.inputs[0])
    obj.modifiers.new("Forest", "NODES").node_group = ng
    obj["layer_fade"] = list(LAYER_FADE_M)
    return obj


# --------------------------------------------------------------------------- roads, rivers

def _ribbon_group(name: str):
    import bpy

    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    I = ng.interface
    I.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    w = I.new_socket("Width", in_out="INPUT", socket_type="NodeSocketFloat")
    w.default_value = 20.0
    I.new_socket("Material", in_out="INPUT", socket_type="NodeSocketMaterial")
    I.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    to_curve = N.new("GeometryNodeMeshToCurve")
    L.new(gin.outputs["Geometry"], to_curve.inputs["Mesh"])
    normal = N.new("GeometryNodeSetCurveNormal")
    set_normal_z_up(normal)
    L.new(to_curve.outputs["Curve"], normal.inputs["Curve"])
    rad = N.new("GeometryNodeSetCurveRadius")
    wattr = N.new("GeometryNodeInputNamedAttribute")
    wattr.data_type = "FLOAT"
    wattr.inputs["Name"].default_value = "w"
    L.new(normal.outputs["Curve"], rad.inputs["Curve"])
    L.new(wattr.outputs["Attribute"], rad.inputs["Radius"])
    half = N.new("ShaderNodeMath")
    half.operation = "MULTIPLY"
    half.inputs[1].default_value = 0.5
    L.new(gin.outputs["Width"], half.inputs[0])
    neg = N.new("ShaderNodeMath")
    neg.operation = "MULTIPLY"
    neg.inputs[1].default_value = -1.0
    L.new(half.outputs[0], neg.inputs[0])
    a, b = N.new("ShaderNodeCombineXYZ"), N.new("ShaderNodeCombineXYZ")
    L.new(neg.outputs[0], a.inputs["Y"])
    L.new(half.outputs[0], b.inputs["Y"])
    prof = N.new("GeometryNodeCurvePrimitiveLine")
    L.new(a.outputs[0], prof.inputs["Start"])
    L.new(b.outputs[0], prof.inputs["End"])
    sweep = N.new("GeometryNodeCurveToMesh")
    L.new(rad.outputs["Curve"], sweep.inputs["Curve"])
    L.new(prof.outputs["Curve"], sweep.inputs["Profile Curve"])
    setm = N.new("GeometryNodeSetMaterial")
    L.new(sweep.outputs["Mesh"], setm.inputs["Geometry"])
    L.new(gin.outputs["Material"], setm.inputs["Material"])
    L.new(setm.outputs["Geometry"], gout.inputs["Geometry"])
    return ng


def add_ribbons(name: str, lines: list[tuple[np.ndarray, float]], grid: TerrainGrid, exaggeration: float,
                color: str, screen_width: float, rough: float = 0.7, height=None):
    """lines: (xy (n, 2), width multiplier). Draped on the grid, lifted a little."""
    import bpy

    if not lines:
        return None
    verts, edges, w = [], [], []
    off = 0
    for xy, mult in lines:
        z = (height or grid).height_at(xy[:, 0], xy[:, 1]) + LIFT_M * exaggeration
        verts.append(np.column_stack([xy, z]))
        k = np.arange(len(xy) - 1) + off
        edges.append(np.column_stack([k, k + 1]))
        w.append(np.full(len(xy), mult))
        off += len(xy)
    me = _mesh(name, np.concatenate(verts), edges=np.concatenate(edges))
    a = me.attributes.new("w", "FLOAT", "POINT")
    a.data.foreach_set("value", np.concatenate(w).astype(np.float32))
    obj = _link(bpy.data.objects.new(name, me))
    obj.modifiers.new(name, "NODES").node_group = _ribbon_group(name)
    mat = _plain_material(name, color)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = rough
    set_input(obj, "Material", mat)
    obj["screen_width"] = screen_width
    obj["hide_beyond"] = RIBBON_HIDE_BEYOND
    set_input(obj, "Width", 20.0)
    return obj


def add_osm(vectors: dict, proj, grid: TerrainGrid, exaggeration: float, log=print, height=None):
    from ..osm import ROAD_RANK, polylines_xy

    inside = lambda xy: (grid.x[0] < xy[:, 0].mean() < grid.x[-1]) and (grid.y[0] < xy[:, 1].mean() < grid.y[-1])  # noqa: E731
    roads = [(xy, 1.0 + 0.35 * (ROAD_RANK.get(t.get("highway"), 1) - 1))
             for xy, t in polylines_xy(vectors.get("roads", []), proj) if inside(xy)]
    rivers = [(xy, 1.0) for xy, t in polylines_xy(vectors.get("rivers", []), proj) if inside(xy)]
    log(f"OSM: {len(roads)} roads, {len(rivers)} rivers")
    add_ribbons("Roads", roads, grid, exaggeration, ROAD, ROAD_SCREEN, rough=0.6, height=height)
    add_ribbons("Rivers", rivers, grid, exaggeration * 0.7, RIVER, RIVER_SCREEN, rough=0.2, height=height)


# --------------------------------------------------------------------------- entry

def apply(style: str, build: Path, proj, grid: TerrainGrid, layers: list, exaggeration: float,
          route_xy: np.ndarray | None = None, log=print, details: list = (), surface=None) -> None:
    """Called by scene.build_scene after the terrain objects exist. layers: [(ctx dir, grid)] widest first;
    details: [(index, dir, grid)] high-res highlight patches; surface: heights incl. those patches."""
    height = surface or grid
    import json

    import bpy
    from PIL import Image, ImageFilter

    from ..stylize import style_texture

    lc_dir = build / "landcover"
    main_lc = lc_dir / "main.npz"
    if not main_lc.exists():
        raise RuntimeError("no build/landcover — run `gpx2reel landcover`")
    lc = LC.LandCover.load(main_lc)
    if style == "stylized":
        tex = style_texture(proj, grid, lc, 4096, exaggeration)
        png = lc_dir / "style_main.png"
        Image.fromarray(tex, "RGBA").save(png)
        style_material(bpy.data.objects["Terrain"], png, grid, "Terrain")
        for i, _d, g in details:
            obj = bpy.data.objects.get(f"Detail.{i}")
            if obj is not None:
                p = lc_dir / f"style_detail{i}.png"
                # the highlight camera is ~4 km away: soften land-cover pixel edges (10–30 m blocks)
                img = Image.fromarray(style_texture(proj, g, lc, 2048, exaggeration), "RGBA")
                img.filter(ImageFilter.GaussianBlur(6)).save(p)
                style_material(obj, p, g, obj.name)
        for k, (d, g) in enumerate(layers):
            obj = bpy.data.objects.get(f"Context.{k}")
            ctx_lc = lc_dir / f"{d.name}.npz"
            if obj is None:
                continue
            if ctx_lc.exists():
                t = style_texture(proj, g, LC.LandCover.load(ctx_lc), 2048, exaggeration)
                p = lc_dir / f"style_{d.name}.png"
                Image.fromarray(t, "RGBA").save(p)
                style_material(obj, p, g, obj.name)
            else:
                tint_satellite(obj)
    pts = select_trees(forest_points(proj, grid, lc, route_xy, height=height))
    log(f"Forest: {len(pts)} trees (static)")
    add_trees(pts)
    vec = build / "osm" / "vectors.json"
    if vec.exists():
        add_osm(json.loads(vec.read_text(encoding="utf-8")), proj, grid, exaggeration, log, height=height)
    else:
        log("⚠ no build/osm/vectors.json — no roads or rivers (`gpx2reel osm`)")
