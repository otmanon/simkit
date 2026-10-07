"""Sparse meshless models of heterogeneous deformable solids (Faure et al. 2011).

Implementation of the SIGGRAPH 2011 paper *"Sparse Meshless Models of Complex
Deformable Solids"* (Faure, Gilles, Bousquet, Pai), adapted from the paper's
voxel-grid setting to a **triangle mesh** carrying a per-element Young's
modulus.

The paper's key idea: instead of geometry-only kernels (RBFs, barycentric
coordinates), build *material-aware* shape functions so that a **sparse** set
of control frames can resolve a heterogeneous stiffness map. Two points
connected by **stiff** material move together (rigidly); points connected by
**compliant** material can move very differently. This is captured by a
**compliance distance** -- a shortest-path metric where each step costs
``compliance x length`` with ``compliance = 1 / E``.

Everything downstream flows from that one metric:

* :func:`simkit.compliance_graph`          -- the weighted mesh-edge graph.
* :func:`simkit.compliance_distances`      -- Dijkstra shortest paths on it.
* :func:`simkit.compliance_node_sampling`  -- farthest-point sampling + Lloyd
  relaxation in the compliance metric (Sec. 4.4); seeds more nodes in soft
  regions and fewer in stiff ones.
* :func:`simkit.voronoi_labels`            -- the Voronoi partition.
* :func:`simkit.voronoi_shape_functions`   -- the material-aware, partition-of-
  unity, interpolating, compact-support skinning weights ``W`` (Sec. 4.3).
* :func:`simkit.lbs_jacobian` (``sparse=True``) -- assembles ``W`` into a
  sparse linear-blend-skinning subspace ``B`` (affine frames, Sec. 3); point
  frames are the simpler ``kron(W, I)``.
* :func:`simkit.lbs_affine_coordinates`    -- reduced coordinates of a global
  affine map, for the patch test.

This module provides the one-call entry point that chains them.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy as sp
import scipy.sparse

from .compliance_distances import compliance_distances
from .compliance_graph import compliance_graph
from .compliance_node_sampling import compliance_node_sampling
from .lbs_jacobian import lbs_jacobian
from .voronoi_labels import voronoi_labels
from .voronoi_shape_functions import voronoi_shape_functions


def sparse_meshless_methods_basis(
    X: np.ndarray,
    T: np.ndarray,
    ym: np.ndarray,
    *,
    n_nodes: int = 16,
    frame_order: int = 1,
    support_scale: float = 1.25,
    lloyd_iters: int = 8,
    seed_index: Optional[int] = None,
    seed_nodes: Optional[Sequence[int]] = None,
    edge_reduce: str = "mean",
    return_distances: bool = False,
):
    """Build a sparse meshless model (Faure et al. 2011) on a heterogeneous mesh.

    Given a mesh ``(X, T)`` and a per-element Young's modulus ``ym``, this
    places ``n_nodes`` control frames using the compliance-distance metric,
    computes the material-aware Voronoi skinning weights, and assembles the
    sparse linear-blend skinning subspace ``B``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, 3)
        Triangle connectivity.
    ym : np.ndarray (t,) or (t, 1)
        Per-element Young's modulus (heterogeneous material map).
    n_nodes : int, optional
        Number of control nodes / frames (the model's sparsity). Soft regions
        automatically receive proportionally more of them.
    frame_order : {0, 1}, optional
        ``1`` = affine frames (default, the paper's recommended compromise):
        ``dim * (dim + 1)`` DOFs per node, assembled with
        :func:`simkit.lbs_jacobian` (``sparse=True``). ``0`` = point /
        translation frames: ``dim`` DOFs per node, ``B = kron(W, I)``.
    support_scale : float, optional
        Widens or tightens the shape-function supports (see
        :func:`simkit.voronoi_shape_functions`).
    lloyd_iters : int, optional
        Lloyd relaxation passes during node distribution.
    seed_index : int, optional
        Deterministic first farthest-point seed.
    seed_nodes : sequence of int, optional
        Vertex indices to pre-place and hold fixed (e.g. one per rigid bone).
    edge_reduce : {"mean", "min", "max"}, optional
        Edge compliance aggregation (see :func:`simkit.compliance_graph`).
    return_distances : bool, optional
        If ``True``, also return the ``(k, n)`` node-to-vertex compliance
        distance field as a fifth output (handy for debugging / visualisation).

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Material-aware skinning weights (partition of unity, compact support).
    B : scipy.sparse.csr_matrix (n*dim, r)
        Sparse linear-blend-skinning subspace, ``x = B z`` with
        ``r = k * dim * (dim + 1)`` (affine frames) or ``u = B z`` with
        ``r = k * dim`` (point frames).
    labels : np.ndarray (t,)
        Voronoi region (control-frame index) each triangle cell belongs to.
    nodes : np.ndarray (k,)
        Control-frame vertex indices.
    D_nodes : np.ndarray (k, n)
        Compliance distance from each control frame to every vertex. Returned
        **only** when ``return_distances=True``.

    Raises
    ------
    ValueError
        If ``frame_order`` is not 0 or 1.

    Examples
    --------
    >>> import numpy as np, simkit
    >>> X, T = ...                       # triangle mesh
    >>> ym = np.where(stiff_elements, 1e4, 1.0)
    >>> W, B, labels, nodes = simkit.sparse_meshless_methods_basis(X, T, ym, n_nodes=12)
    >>> z = simkit.lbs_affine_coordinates(len(nodes), np.eye(2), np.zeros(2))
    >>> np.allclose((B @ z).reshape(-1, 2), X)   # rest state is reproduced
    True
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)

    G = compliance_graph(X, T, ym, reduce=edge_reduce)
    nodes = compliance_node_sampling(
        X, G, n_nodes, seed_index=seed_index, seed_nodes=seed_nodes, lloyd_iters=lloyd_iters
    )
    D_nodes = compliance_distances(G, nodes)  # (k, n)

    _, labels = voronoi_labels(D_nodes, T)
    W = voronoi_shape_functions(D_nodes, nodes, support_scale=support_scale)
    if frame_order == 1:
        B = lbs_jacobian(X, W, sparse=True)
    elif frame_order == 0:
        B = sp.sparse.kron(W, sp.sparse.identity(X.shape[1]), format="csr")
    else:
        raise ValueError("frame_order must be 0 (point frames) or 1 (affine frames)")

    if return_distances:
        return W, B, labels, nodes, D_nodes
    return W, B, labels, nodes
