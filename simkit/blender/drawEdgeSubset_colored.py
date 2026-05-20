

import bpy
import math
import numpy as np
from PIL import Image
import mathutils
def drawEdgeSubset_colored(mesh, E, r, scalars, colormap_path, vminmax=None):
       # Load colormap PNG as numpy array
    colormap = Image.open(colormap_path).convert('RGBA')
    cmap_np = np.array(colormap) / 255.0  # normalize to [0, 1]
    width = cmap_np.shape[1]

    # Normalize scalars
    if vminmax is None:
        scalar_min, scalar_max = np.min(scalars), np.max(scalars)
    else:
        scalar_min, scalar_max = vminmax[0], vminmax[1]
    norm_scalars = (scalars - scalar_min) / (scalar_max - scalar_min + 1e-8)

    # Create edge cylinder template
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=r, depth=1.0, location=(1e10,1e10,1e10))
    template = bpy.context.object
    template.name = "edge_template"
    edge_objects = []
    for i, (v1, v2) in enumerate(E):
        # Positions
        p1 = mesh.matrix_world @ mesh.data.vertices[v1].co
        p2 = mesh.matrix_world @ mesh.data.vertices[v2].co
        direction = p2 - p1
        length = direction.length
        center = (p1 + p2) / 2

        # Duplicate cylinder
        bpy.ops.object.select_all(action='DESELECT')
        template.select_set(True)
        bpy.context.view_layer.objects.active = template
        bpy.ops.object.duplicate()
        obj = bpy.context.object
        obj.location = center
        obj.scale = (1, 1, length)

        # Orient to match edge
        direction.normalize()
        up = mathutils.Vector((0, 0, 1))
        quat = up.rotation_difference(direction)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = quat

        # Sample color from colormap
        s = norm_scalars[i]
        u = int(s * (width - 1))
        color = cmap_np[0, u]  # assuming 1-pixel high horizontal colormap
        r_col, g_col, b_col, a_col = color

        # Create material with flat color
        mat = bpy.data.materials.new(name=f"edge_color_{i}")
        mat.use_nodes = False
        mat.diffuse_color = (r_col, g_col, b_col, 1.0)

        # Assign material
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        edge_objects.append(obj)

    # save blend file
    # import os
    # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test_colormap.blend')
 
    # Remove template
    bpy.data.objects.remove(template)
    return edge_objects