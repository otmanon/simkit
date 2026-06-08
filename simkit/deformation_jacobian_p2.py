"""Deformation-Jacobian operator for quadratic (P2) simplicial elements."""

import numpy as np
import scipy as sp

from .p2_shape_functions import p2_num_nodes, _SIMPLEX_EDGE_LOCAL


def _p2_shape_gradients_batched(L: np.ndarray, s: int) -> np.ndarray:
    """Barycentric gradients ``dN/dL`` of the P2 shape functions, batched.

    Vectorized counterpart of :func:`p2_shape_functions` (gradients only) that
    evaluates many points at once. The loops here are over the *constant* number
    of simplex corners and edges, not over elements.

    Parameters
    ----------
    L : np.ndarray (m, s)
        Barycentric coordinates of ``m`` evaluation points.
    s : int
        Number of simplex corners (3 triangle, 4 tet).

    Returns
    -------
    dNdL : np.ndarray (m, n_nodes, s)
    """
    L = np.asarray(L, dtype=float)
    m = L.shape[0]
    edges_local = _SIMPLEX_EDGE_LOCAL[s]
    n_nodes = s + len(edges_local)

    dNdL = np.zeros((m, n_nodes, s))

    # Corner nodes: dN_i/dL_j = (4 L_i - 1) delta_ij.
    for i in range(s):
        dNdL[:, i, i] = 4.0 * L[:, i] - 1.0

    # Edge-midpoint nodes: N = 4 L_a L_b, dN/dL_a = 4 L_b, dN/dL_b = 4 L_a.
    for e, (a, b) in enumerate(edges_local):
        node = s + e
        dNdL[:, node, a] = 4.0 * L[:, b]
        dNdL[:, node, b] = 4.0 * L[:, a]

    return dNdL


# Reference-element P1 shape-function gradients dL/dxi (rows = corner, cols =
# reference axis). Identical to the ``H`` used in ``deformation_jacobian`` /
# ``deformation_gradient``; here it converts barycentric gradients to
# reference-axis gradients via the chain rule.
_H_P1 = {
    2: np.array([[-1.0, -1.0],
                 [1.0, 0.0],
                 [0.0, 1.0]]),
    3: np.array([[-1.0, -1.0, -1.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]),
}


def deformation_jacobian_p2(
    V2: np.ndarray,
    T2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray = None,
) -> "sp.sparse.csc_matrix":
    """Sparse operator ``J`` mapping P2 nodal positions to per-cubature ``F``.

    Builds the constant linear map such that ``F = (J @ x).reshape(-1, dim, dim)``
    gives the deformation gradient at every cubature point of every element,
    where ``x`` is ``V2`` flattened in C order.

    Is this matrix constant?
    ------------------------
    **Yes.** Like the P1 operator it is a fixed sparse matrix, built once from
    the rest geometry and the (fixed) reference-space quadrature points, and is
    independent of the deformed state. The key difference from P1: P2
    shape-function gradients are *linear* in the reference coordinates, so ``F``
    is **no longer constant within an element** -- it varies from one cubature
    point to the next. That is why ``J`` maps to a *stack* of per-cubature-point
    ``F`` blocks (``t * n_quad`` of them) rather than one ``F`` per element.
    Because the P2 midpoint nodes sit exactly at edge midpoints, the geometric
    rest map is still affine, so the reference->rest Jacobian is constant per
    element and is taken from the corner nodes alone.

    How P2 changes the integration of an energy
    -------------------------------------------
    With P1, each element has a single constant ``F``, so the total elastic
    energy is a one-point quadrature::

        E = Σ_t vol_t · ψ(F_t)

    With P2, ``F`` varies inside the element, so the integral becomes a sum over
    the ``n_quad`` cubature points::

        E = Σ_t Σ_q w_{t,q} · ψ(F_{t,q})

    No energy code changes are needed -- treat each cubature point as a
    pseudo-element. Using this operator together with the ``weights`` from
    :func:`gauss_legendre_quadrature`::

        x = V2.reshape(-1, 1)
        F = (J @ x).reshape(-1, dim, dim)      # (t*n_quad, dim, dim)
        psi = some_energy_element_F(F, mu)     # unchanged P1 energy code
        E = float((weights.reshape(-1, 1) * psi).sum())

    and likewise forces are ``J.T @ (weights ⊙ P)`` and the Hessian is
    ``J.T @ blockdiag(weights ⊙ d2psi) @ J``. The quadrature ``order`` used to
    build ``bary``/``weights`` must be high enough for the chosen energy's
    integrand (e.g. degree 2 for linear elasticity, whose density is quadratic
    in ``F``).

    Parameters
    ----------
    V2 : np.ndarray (n2, dim)
        Quadratic-mesh vertex positions (``dim`` is 2 or 3).
    T2 : np.ndarray (t, n_nodes)
        Quadratic connectivity (6 nodes for triangles, 10 for tets), as produced
        by :func:`linear_to_quadratic_elements`.
    bary : np.ndarray (t, n_quad, dim+1)
        Barycentric coordinates of the cubature points (from
        :func:`gauss_legendre_quadrature`).
    weights : np.ndarray (t, n_quad), optional
        Cubature weights. Accepted for interface symmetry but **not** used to
        build the operator: the operator is independent of the weights, which
        enter later as the energy's per-cubature ``vol``.

    Returns
    -------
    J : scipy.sparse.csc_matrix (t*n_quad*dim*dim, n2*dim)
        Constant deformation-Jacobian operator.
    """
    V2 = np.asarray(V2, dtype=float)
    T2 = np.asarray(T2)
    bary = np.asarray(bary, dtype=float)

    dim = V2.shape[1]
    s = dim + 1                      # number of simplex corners
    n2 = V2.shape[0]
    t = T2.shape[0]
    n_nodes = T2.shape[1]
    n_quad = bary.shape[1]

    if n_nodes != p2_num_nodes(s):
        raise ValueError(
            f"T2 has {n_nodes} columns; expected {p2_num_nodes(s)} for dim {dim}"
        )

    H_p1 = _H_P1[dim]

    n_rows = t * n_quad * dim * dim
    n_cols = n2 * dim

    # Affine reference->rest Jacobian from the corner nodes only, for every
    # element at once. XH = dX/dxi, XHi = dxi/dX; both constant over an element.
    Xc = V2[T2[:, :s]]                              # (t, s, dim)
    XH = np.einsum("esd,sk->edk", Xc, H_p1)        # (t, dim, dim)
    XHi = np.linalg.inv(XH)                         # (t, dim, dim)

    # Spatial shape-function gradients at every cubature point of every element.
    # bary is (t, n_quad, s); flatten the (element, quad) axes to one batch.
    m = t * n_quad
    dNdL = _p2_shape_gradients_batched(
        bary.reshape(m, s), s
    )                                              # (m, n_nodes, s)

    # Chain rule: dN/dX = (dN/dL)(dL/dxi)(dxi/dX) = dNdL @ H_p1 @ XHi.
    dNdxi = np.einsum("mns,sk->mnk", dNdL, H_p1)   # (m, n_nodes, dim)
    dNdxi = dNdxi.reshape(t, n_quad, n_nodes, dim)
    G = np.einsum("eqnk,ekj->eqnj", dNdxi, XHi)    # (t, n_quad, n_nodes, dim)
    G = G.reshape(m, n_nodes, dim)                 # row-block order is e*n_quad+q

    # Vectorized triplet assembly. For each (block m, node, i, j):
    #   row = m*(dim*dim) + i*dim + j
    #   col = global_node*dim + i
    #   val = G[m, node, j]            (F[i,j] = Σ_node u_node[i] * G[node, j])
    gnodes = np.repeat(T2, n_quad, axis=0)         # (m, n_nodes) global node ids

    m_idx = np.arange(m)[:, None, None, None]
    i_idx = np.arange(dim)[None, None, :, None]
    j_idx = np.arange(dim)[None, None, None, :]

    rows = m_idx * (dim * dim) + i_idx * dim + j_idx
    cols = gnodes[:, :, None, None] * dim + i_idx
    vals = np.broadcast_to(G[:, :, None, :], (m, n_nodes, dim, dim))

    J = sp.sparse.csc_matrix(
        (vals.ravel(),
         (np.broadcast_to(rows, vals.shape).ravel(),
          np.broadcast_to(cols, vals.shape).ravel())),
        shape=(n_rows, n_cols),
    )
    return J
