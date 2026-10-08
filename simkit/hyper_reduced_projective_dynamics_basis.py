"""Sparse skinning subspace of Hyper-Reduced Projective Dynamics (Brandt et al. 2018).

*Hyper-Reduced Projective Dynamics* (Brandt, Eisemann, Hildebrandt, SIGGRAPH
2018) builds its reduced spaces without training data or modal analysis: a
set of sample vertices is chosen by farthest-point sampling in the mesh
(geodesic) distance, every sample gets a compactly supported radial basis
function of that distance as a skinning weight, the weights are normalised to
a partition of unity, and each sample carries an affine frame, giving a
**sparse** linear-blend-skinning basis. The same construction, applied to the
constraint centres, yields the paper's subspace for the constraint
projections. This module builds the position-space basis; it is the
geometry-only counterpart of :func:`simkit.sparse_meshless_methods_basis`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy as sp
import scipy.sparse

from .compliance_distances import compliance_distances
from .compliance_graph import compliance_graph
from .compliance_node_sampling import compliance_node_sampling
from .lbs_jacobian import lbs_jacobian


def _wendland_c2(x: np.ndarray) -> np.ndarray:
    """Compactly supported Wendland C2 kernel ``(1 - x)^4 (4 x + 1)`` on ``[0, 1]``."""
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - x) ** 4 * (4.0 * x + 1.0)


def hyper_reduced_projective_dynamics_basis(
    X: np.ndarray,
    T: np.ndarray,
    n_samples: int,
    support_scale: float = 2.0,
    seed_index: Optional[int] = None,
    return_distances: bool = False,
):
    """Sparse LBS subspace of Hyper-Reduced Projective Dynamics (Brandt et al. 2018).

    1. ``n_samples`` sample vertices are picked by farthest-point sampling in
       the geodesic distance along mesh edges (Dijkstra).
    2. Every sample ``a`` gets the truncated radial basis function
       ``w_a(v) = phi(d_a(v) / r)`` of its geodesic distance ``d_a``, with the
       Wendland C2 kernel ``phi(x) = (1 - x)^4 (4 x + 1)`` for ``x < 1`` and
       ``0`` beyond, so the weight is smooth and compactly supported.
    3. The support radius is ``r = support_scale * d_cover`` where ``d_cover``
       is the farthest any vertex lies from its nearest sample, so every vertex
       is covered and neighbouring supports overlap.
    4. The weights are normalised to a partition of unity and assembled into
       the sparse affine linear-blend-skinning basis
       :func:`simkit.lbs_jacobian` ``(sparse=True)``.

    The construction uses geometry only. Compare
    :func:`simkit.sparse_meshless_methods_basis`, which replaces the geodesic
    distance by the compliance distance so the weights respect material
    interfaces, and :func:`simkit.skinning_eigenmodes` with a per-element
    ``mu``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, 3)
        Triangle connectivity.
    n_samples : int
        Number of sample vertices (frames). The subspace has
        ``n_samples * dim * (dim + 1)`` DOFs.
    support_scale : float, optional
        Support radius as a multiple of the sampling's covering distance.
        ``2.0`` (default) gives overlapping, smooth weights; ``1.0`` is the
        tightest support that still covers every vertex.
    seed_index : int, optional
        First farthest-point seed. Defaults to the vertex with the smallest
        first coordinate (deterministic).
    return_distances : bool, optional
        If ``True`` also return the ``(k, n)`` sample-to-vertex geodesic
        distances as a fourth output.

    Returns
    -------
    W : scipy.sparse.csr_matrix (n, k)
        Non-negative, compactly supported skinning weights whose rows sum to 1.
    B : scipy.sparse.csr_matrix (n*dim, k*dim*(dim+1))
        Sparse linear-blend-skinning subspace, ``x = B z``.
    samples : np.ndarray (k,)
        Sample vertex indices.
    D : np.ndarray (k, n)
        Geodesic distance from every sample to every vertex. Returned
        **only** when ``return_distances=True``.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T)
    n = X.shape[0]

    G = compliance_graph(X, T, np.ones(T.shape[0]))  # unit material: plain edge lengths
    samples = compliance_node_sampling(X, G, n_samples, seed_index=seed_index, lloyd_iters=0)
    D = compliance_distances(G, samples)  # (k, n) geodesic distances
    k = len(samples)

    finite = np.where(np.isfinite(D), D, np.inf)
    d_cover = finite.min(axis=0)
    d_cover = d_cover[np.isfinite(d_cover)].max() if np.any(np.isfinite(d_cover)) else 1.0
    r = support_scale * max(d_cover, np.finfo(float).tiny)

    raw = _wendland_c2(finite / r)
    raw[finite >= r] = 0.0
    raw[~np.isfinite(D)] = 0.0
    W = raw.T  # (n, k)

    row_sum = W.sum(axis=1)
    empty = row_sum <= 1e-12  # only possible on a disconnected mesh
    if np.any(empty):
        W[empty, :] = 1.0 / k
        row_sum = W.sum(axis=1)
    W = W / row_sum[:, None]
    W[W < 1e-9] = 0.0
    W = sp.sparse.csr_matrix(W)

    B = lbs_jacobian(X, W, sparse=True)
    if return_distances:
        return W, B, samples, D
    return W, B, samples
