"""Trail geometry in Blender: flat styles lie in the ground plane, the tube stands on it."""
import numpy as np
import pytest

bpy = pytest.importorskip("bpy")


@pytest.mark.parametrize("style, max_height", [("ribbon", 0.0), ("band", 0.25), ("tube", 1.0)])
def test_trail_profile_orientation(style, max_height):
    from gpx2reel.blender import scene as S

    bpy.ops.wm.read_factory_settings(use_empty=True)
    xyz = np.column_stack([np.linspace(0, 1000, 20), np.zeros(20), np.zeros(20)])
    me = S._mesh("T", xyz, edges=np.column_stack([np.arange(19), np.arange(1, 20)]))
    me.attributes.new("s", "FLOAT", "POINT").data.foreach_set("value", xyz[:, 0].astype(np.float32))
    o = S._link(bpy.data.objects.new("T", me))
    o.modifiers.new("Trail", "NODES").node_group = S._trail_group(style)
    S.set_input(o, "Progress", 2000.0)
    S.set_input(o, "Width", 30.0)
    ev = o.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    v = np.array([p.co[:] for p in ev.vertices])
    width, height = np.ptp(v[:, 1]), np.ptp(v[:, 2])
    assert width >= 30.0 - 1e-3                       # across the route, in the ground plane
    assert height <= max_height * width + 1e-3
    assert v[:, 2].min() >= -1e-3                     # nothing sinks below the path
