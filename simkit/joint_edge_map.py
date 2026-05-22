"""Map each joint (vertex triple) to its two incident mesh edges."""

import numpy as np


def joint_edge_map(edges: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Edge indices for the two segments of each joint.

    Parameters
    ----------
    edges : np.ndarray (M, 2)
        Mesh edges as vertex index pairs ``(v1, v2)``.
    joints : np.ndarray (N, 3)
        Joints as ``(v_left, v_center, v_right)``.

    Returns
    -------
    joint_edges : np.ndarray (N, 2)
        For each joint, indices into ``edges`` for ``(v_left, v_center)`` and
        ``(v_center, v_right)``.
    """
    edges = np.asarray(edges, dtype=int)
    joints = np.asarray(joints, dtype=int)

    # Build a dictionary from sorted vertex pairs -> edge index
    edge_map = {tuple(sorted(e)): idx for idx, e in enumerate(edges)}

    joint_edges = np.zeros((len(joints), 2), dtype=int)

    for j_idx, (i, j, k) in enumerate(joints):
        e1 = edge_map[tuple(sorted((i, j)))]
        e2 = edge_map[tuple(sorted((j, k)))]
        joint_edges[j_idx, 0] = e1
        joint_edges[j_idx, 1] = e2

    return joint_edges
