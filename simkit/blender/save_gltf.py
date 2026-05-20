import bpy
import os

# Ensure you're in object mode
if bpy.ops.object.mode_set.poll():
    bpy.ops.object.mode_set(mode='OBJECT')

# Get the only Armature in the scene
armatures = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']
if len(armatures) != 1:
    raise ValueError("Expected exactly one armature in the scene.")
armature = armatures[0]

# Find skinned meshes that are affected by the armature
skinned_meshes = []
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        has_armature_mod = any(mod.type == 'ARMATURE' and mod.object == armature for mod in obj.modifiers)
        if has_armature_mod:
            skinned_meshes.append(obj)

if not skinned_meshes:
    raise ValueError("No mesh objects found with an Armature modifier linked to the armature.")

# Deselect everything
bpy.ops.object.select_all(action='DESELECT')

# Select the armature and its skinned meshes
armature.select_set(True)
for mesh in skinned_meshes:
    mesh.select_set(True)

# Set the armature as the active object
bpy.context.view_layer.objects.active = armature

# Export path
export_path = os.path.join(bpy.path.abspath("//"), "armature_export.glb")

# Export selected objects to GLB
bpy.ops.export_scene.gltf(
    filepath=export_path,
    export_format='GLB',
    use_selection=True,
    export_apply=True,
)

print(f"Exported to {export_path}")
