import bpy
import numpy as np

def readNumpyPoints(P, location, rotation_euler, scale, point_colors=None, radius=0.01):
    """
    Create a point cloud as a collection of small spheres from numpy array

    Parameters:
    - P: (N, 3) numpy array of points
    - location: tuple (3,) - world translation
    - rotation_euler: tuple (3,) - world rotation in degrees
    - scale: tuple (3,) - world scale
    - point_colors: optional (N, 3) RGB values in [0, 1]
    - radius: radius of each sphere
    """
    # Convert rotation to radians
    x = np.radians(rotation_euler[0])
    y = np.radians(rotation_euler[1])
    z = np.radians(rotation_euler[2])
    angle = (x, y, z)

    # Create an empty parent object
    parent = bpy.data.objects.new("point_cloud_parent", None)
    parent.location = location
    parent.rotation_euler = angle
    parent.scale = scale
    bpy.data.collections[0].objects.link(parent)
    created_objs = []
    for i, point in enumerate(P):
        # Create a UV sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
        sphere = bpy.context.object
        sphere.name = f"point_{i}"
        sphere.location = point
        sphere.parent = parent

        # Optional color
        if point_colors is not None:
            color = point_colors[i]
            mat = bpy.data.materials.new(name=f"point_color_{i}")
            mat.diffuse_color = (color[0], color[1], color[2], 1.0)
            mat.use_nodes = False
            sphere.data.materials.append(mat)
        created_objs.append(sphere)

    return parent,created_objs 