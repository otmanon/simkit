"""Deformation-Jacobian operator for quadratic (P2) simplicial elements."""

import numpy as np
import scipy as sp

from .p2_shape_functions import p2_shape_functions, p2_num_nodes


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

    # Triplet lists for sparse assembly (explicit loops, no vectorization).
    rows = []
    cols = []
    vals = []

    for e in range(t):
        # Affine reference->rest Jacobian from the corner nodes only. XH = dX/dxi
        # and XHi = dxi/dX are constant over the element.
        corners = T2[e, :s]
        Xc = V2[corners]                 # (s, dim)
        XH = Xc.T @ H_p1                  # (dim, dim)
        XHi = np.linalg.inv(XH)          # (dim, dim)

        for q in range(n_quad):
            L = bary[e, q]               # (s,)
            _, dNdL = p2_shape_functions(L, s)   # (n_nodes, s)

            # Chain rule: dN/dX = (dN/dL)(dL/dxi)(dxi/dX) = dNdL @ H_p1 @ XHi.
            dNdxi = dNdL @ H_p1          # (n_nodes, dim)
            G = dNdxi @ XHi              # (n_nodes, dim) spatial shape gradients

            # Row offset for this element's q-th cubature point.
            block = (e * n_quad + q) * (dim * dim)

            # F[i, j] = Σ_node u_node[i] * G[node, j]; place the coefficients.
            for node in range(n_nodes):
                global_node = int(T2[e, node])
                for i in range(dim):
                    col = global_node * dim + i
                    for j in range(dim):
                        row = block + i * dim + j
                        rows.append(row)
                        cols.append(col)
                        vals.append(G[node, j])

    J = sp.sparse.csc_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_rows, n_cols),
    )
    return J
