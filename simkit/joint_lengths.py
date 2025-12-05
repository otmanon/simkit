import igl
import numpy as np

from .edge_lengths import edge_lengths
from .volume import volume

def joint_lengths(V, E):
    """
    Compute the volume of a simplex defined with nodes V and faces F.

    Parameters
    ----------
    V : (n, 3) array
        Nodes of the mesh
    E : (m, 3) array
        Joints of the mesh indexing V, organized as (v_left, v_center, v_right)
    """
    dim = V.shape[1]

    E1 = np.concatenate([E[:, [0]], E[:, [1]]], axis=1)
    E2 = np.concatenate([E[:, [1]], E[:, [2]]], axis=1)
    
    vol0 = volume(V, E1)
    vol1 = volume(V, E2)
    vol = (vol0 + vol1) / 2
    return vol.flatten()