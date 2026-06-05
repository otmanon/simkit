"""Consistent (quadrature-assembled) mass matrix for quadratic (P2) elements.

The lumped :func:`simkit.massmatrix` dispatches on the number of vertices per
simplex (2/3/4) and cannot consume a P2 element (6/10 nodes). This builds the
consistent P2 mass matrix directly from the P2 shape functions and a quadrature
rule::

    M_ij = Σ_q w_q ρ N_i(x_q) N_j(x_q)

assembled element by element and scattered into the global node ordering.
"""

import numpy as np
import scipy as sp

from .p2_shape_functions import p2_shape_functions, p2_num_nodes


def p2_massmatrix(
    V2: np.ndarray,
    T2: np.ndarray,
    bary: np.ndarray,
    weights: np.ndarray,
    rho=1.0,
) -> "sp.sparse.csc_matrix":
    """Consistent P2 mass matrix over the quadratic node set.

    Written with explicit loops (no vectorization) for clarity.

    Parameters
    ----------
    V2 : np.ndarray (n2, dim)
        Quadratic-mesh vertex positions.
    T2 : np.ndarray (t, n_nodes)
        Quadratic connectivity (6 nodes for triangles, 10 for tets).
    bary : np.ndarray (t, n_quad, dim+1)
        Barycentric cubature points (from :func:`gauss_legendre_quadrature`).
    weights : np.ndarray (t, n_quad)
        Physical cubature weights (from :func:`gauss_legendre_quadrature`); these
        already include the element measure.
    rho : float or np.ndarray (t, 1), optional
        Mass density, scalar or per-element. Default 1.

    Returns
    -------
    M : scipy.sparse.csc_matrix (n2, n2)
        Consistent scalar-node mass matrix. As with :func:`simkit.massmatrix`,
        callers acting on flattened ``(n2, dim)`` positions should ``kron`` it
        with ``identity(dim)``.

    Notes
    -----
    The total mass ``M.sum()`` equals ``Σ_e ρ_e · vol_e`` because the P2 shape
    functions form a partition of unity (``Σ_i N_i ≡ 1``).
    """
    V2 = np.asarray(V2, dtype=float)
    T2 = np.asarray(T2)
    bary = np.asarray(bary, dtype=float)
    weights = np.asarray(weights, dtype=float)

    dim = V2.shape[1]
    s = dim + 1
    n2 = V2.shape[0]
    t = T2.shape[0]
    n_nodes = T2.shape[1]
    n_quad = bary.shape[1]

    if n_nodes != p2_num_nodes(s):
        raise ValueError(
            f"T2 has {n_nodes} columns; expected {p2_num_nodes(s)} for dim {dim}"
        )

    # Normalize rho to a per-element array for uniform indexing.
    rho_arr = np.asarray(rho, dtype=float).reshape(-1)
    if rho_arr.size == 1:
        rho_arr = np.full(t, float(rho_arr[0]))

    rows = []
    cols = []
    vals = []

    for e in range(t):
        for q in range(n_quad):
            L = bary[e, q]
            N, _ = p2_shape_functions(L, s)      # (n_nodes,)
            w = weights[e, q] * rho_arr[e]

            # Outer product N_i N_j weighted by w, scattered to global indices.
            for a in range(n_nodes):
                ga = int(T2[e, a])
                for b in range(n_nodes):
                    gb = int(T2[e, b])
                    rows.append(ga)
                    cols.append(gb)
                    vals.append(w * N[a] * N[b])

    M = sp.sparse.csc_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n2, n2),
    )
    return M
