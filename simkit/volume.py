"""Per-simplex size measure for 1D segments, triangles, or tetrahedra.

Dispatches to edge length, triangle area, or tet volume depending on the
number of vertices per simplex in ``F``.
"""

import numpy as np

from .edge_lengths import edge_lengths
from .triangle_areas import triangle_areas
from .tetrahedron_volumes import tetrahedron_volumes


def volume(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-simplex length, area, or volume from mesh nodes and connectivity.

    Parameters
    ----------
    V : np.ndarray (n, 3)
        Mesh vertex positions.
    F : np.ndarray (m, 2), (m, 3), or (m, 4)
        Simplex indices: segments (2), triangles (3), or tetrahedra (4).

    Returns
    -------
    vol : np.ndarray (m, 1)
        Per-simplex measure (edge length, triangle area, or tet volume).
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
