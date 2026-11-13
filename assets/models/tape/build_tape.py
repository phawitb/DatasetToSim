"""Build the clear packing-tape roll in Blender and export MuJoCo-ready OBJ parts.

Run: blender --background --python assets/models/tape/build_tape.py

Dimensions come from tape.json, the user's measurements of the real roll: 53 mm outer
diameter, 24 mm wide, 25 mm core hole, 30 mm core outer diameter (2.5 mm wall).
"""
import json
from pathlib import Path
import bpy

HERE = Path(__file__).resolve().parent
DIMENSIONS = json.loads((HERE / 'tape.json').read_text())
OUTER_DIAMETER = DIMENSIONS['outer_diameter_m']
CORE_OUTER_DIAMETER = DIMENSIONS['core_outer_diameter_m']
CORE_INNER_DIAMETER = DIMENSIONS['core_inner_diameter_m']
LINER = 0.0004           # printed paper liner on the inside of the core
WIDTH = DIMENSIONS['width_m']
SEGMENTS = 96

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0


def material(name, rgba, transmission=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Transmission Weight'].default_value = transmission
    bsdf.inputs['Alpha'].default_value = rgba[3]
    mat.diffuse_color = rgba
    return mat


def tube(name, outer_d, inner_d, mat, bevel=0.0):
    """Closed, watertight annular cylinder centred on the origin, axis +Z."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=SEGMENTS, radius=outer_d / 2, depth=WIDTH)
    outer = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(vertices=SEGMENTS, radius=inner_d / 2, depth=WIDTH * 1.2)
    cutter = bpy.context.active_object
    boolean = outer.modifiers.new('hole', 'BOOLEAN')
    boolean.operation = 'DIFFERENCE'
    boolean.object = cutter
    boolean.solver = 'EXACT'
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier='hole')
    bpy.data.objects.remove(cutter)
    if bevel:
        mod = outer.modifiers.new('edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = 'ANGLE'
        bpy.ops.object.modifier_apply(modifier='edges')
    outer.name = outer.data.name = name
    outer.data.materials.append(mat)
    bpy.ops.object.shade_auto_smooth()
    return outer


clear = material('clear_tape', (0.93, 0.91, 0.84, 0.55), transmission=0.9, roughness=0.15)
cardboard = material('cardboard_core', (0.55, 0.40, 0.26, 1.0), roughness=0.8)
liner = material('printed_liner', (0.88, 0.90, 0.95, 1.0), roughness=0.7)

parts = [
    tube('tape_roll', OUTER_DIAMETER, CORE_OUTER_DIAMETER, clear, bevel=0.0006),
    tube('tape_core', CORE_OUTER_DIAMETER, CORE_INNER_DIAMETER + 2 * LINER, cardboard),
    tube('tape_liner', CORE_INNER_DIAMETER + 2 * LINER, CORE_INNER_DIAMETER, liner),
]

for obj in parts:
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(filepath=str(HERE / f'{obj.name}.obj'), export_selected_objects=True,
                          forward_axis='Y', up_axis='Z', export_triangulated_mesh=True,
                          export_materials=False, export_uv=False, export_normals=True)

bpy.ops.wm.save_as_mainfile(filepath=str(HERE / 'tape.blend'))
print('TAPE_BUILD_OK', [(o.name, len(o.data.polygons)) for o in parts])
