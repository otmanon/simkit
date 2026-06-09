"""Average simplex volume along the two edges of each joint."""

import numpy as np

from .edge_lengths import edge_lengths
from .volume import volume


def joint_lengths(V: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Mean volume of the two edge simplices at each joint.

    Each joint ``(v_left, v_center, v_right)`` is split into edges
    ``(v_left, v_center)`` and ``(v_center, v_right)``; their volumes are
    averaged.

    Parameters
    ----------
    V : np.ndarray (n, dim)
        Mesh vertex positions.
    E : np.ndarray (m, 3)
        Joint indices into ``V`` as ``(v_left, v_center, v_right)``.

    Returns
    -------
    vol : np.ndarray (m,)
        Per-joint mean edge-simplex volume (flattened 1D).
    """
    dim = V.shape[1]

    E1 = np.concatenate([E[:, [0]], E[:, [1]]], axis=1)
    E2 = np.concatenate([E[:, [1]], E[:, [2]]], axis=1)

    vol0 = volume(V, E1)
    vol1 = volume(V, E2)
    vol = (vol0 + vol1) / 2
    return vol.flatten()
