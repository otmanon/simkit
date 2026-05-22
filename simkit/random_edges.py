"""Random sampling of undirected edges and length-2 joint paths."""

from typing import Optional

import numpy as np


def random_undirected_edges(
    num_vertices: int,
    num_edges: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Sample unique undirected edges ``(i, j)`` with ``i < j``.

    Parameters
    ----------
    num_vertices : int
        Number of vertices.
    num_edges : int
        Number of edges to sample (clamped to the number of available pairs).
    rng : np.random.RandomState, optional
        Random state for reproducibility. If ``None``, uses ``np.random``.

    Returns
    -------
    edges : np.ndarray (num_edges, 2)
        Vertex index pairs with ``i < j`` and no self-loops.
    """

    np.random.seed(0)
    iu, ju = np.triu_indices(num_vertices, k=1)
    all_pairs = np.stack([iu, ju], axis=1)
    max_edges = all_pairs.shape[0]
    if num_edges > max_edges:
        num_edges = max_edges
    chosen = np.random.choice(max_edges, size=num_edges, replace=False)
    return all_pairs[chosen]


def random_undirected_joints(
    num_vertices: int,
    num_edges: int,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Sample unique undirected joints ``[neighbor, i, other_neighbor]``.

    A joint is a length-2 path centered at vertex ``i`` with two distinct
    neighbors. Neighbor order is canonicalized by combinatorial pairs, so
    ``(a, i, b)`` and ``(b, i, a)`` are the same joint.

    Parameters
    ----------
    num_vertices : int
        Number of vertices.
    num_edges : int
        Number of joints to sample (clamped to the number of available joints).
    rng : np.random.RandomState, optional
        Random state for reproducibility. If ``None``, uses ``np.random``.

    Returns
    -------
    joints : np.ndarray (num_edges, 3)
        Rows ``[neighbor, i, other_neighbor]``. Empty if ``num_vertices < 3``.
    """

    np.random.seed(0)
    if num_vertices < 3:
        return np.zeros((0, 3), dtype=int)

    all_joints = []
    vertices = np.arange(num_vertices)
    for i in range(num_vertices):
        neighbors = vertices[vertices != i]
        iu, ju = np.triu_indices(neighbors.shape[0], k=1)
        neighbor_pairs = np.stack([neighbors[iu], neighbors[ju]], axis=1)
        centers = np.full((neighbor_pairs.shape[0], 1), i, dtype=int)
        joints_i = np.concatenate([neighbor_pairs[:, [0]], centers, neighbor_pairs[:, [1]]], axis=1)
        all_joints.append(joints_i)

    all_joints = np.concatenate(all_joints, axis=0)
    max_joints = all_joints.shape[0]
    if num_edges > max_joints:
        num_edges = max_joints
    chosen = np.random.choice(max_joints, size=num_edges, replace=False)
    return all_joints[chosen]
