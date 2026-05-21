import numpy as np
from .edge_lengths import edge_lengths
from .triangle_areas import triangle_areas
from .tetrahedron_volumes import tetrahedron_volumes

def volume(V, F):
    """
    Compute the volume of a simplex defined with nodes V and faces F.

    Parameters
    ----------
    V : (n, 3) array
        Nodes of the mesh
    F : (m, 2|3|4) array
        Simpleces of the mesh (either edges, triangles or tets)
    """
    dim = V.shape[1]

    t = F.shape[1]

    if t == 2:
        vol = edge_lengths(V, F).reshape(-1, 1)
    if t == 3:
        vol = triangle_areas(V, F).reshape(-1, 1)
    elif t == 4:
        vol = tetrahedron_volumes(V, F).reshape(-1, 1)
    else:
        ValueError("Only F.shape[1] == 2, 3 or 4 are supported")
    return vol


