"""Per-cubature-point deformation gradient for quadratic (P2) elements.

The P2 analogue of :func:`simkit.deformation_gradient`. Where the P1 version
returns one constant ``F`` per element, the P2 version returns one ``F`` per
*cubature point*, because P2 shape-function gradients vary within an element.
"""

import numpy as np

from .p2_shape_functions import p2_shape_functions, p2_num_nodes


# Reference-element P1 shape-function gradients dL/dxi (see deformation_gradient
# and deformation_jacobian_p2 for the matching construction).
_H_P1 = {
    2: np.array([[-1.0, -1.0],
                 [1.0, 0.0],
                 [0.0, 1.0]]),
    3: np.array([[-1.0, -1.0, -1.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]),
}


def deformation_gradient_p2(
    V2: np.ndarray,
    T2: np.ndarray,
    bary: np.ndarray,
    U2: np.ndarray,
) -> np.ndarray:
    """Per-cubature-point deformation gradient ``F = dx/dX`` for P2 elements.

    Mirrors :func:`simkit.deformation_gradient`: builds, per element, the P2
    shape-function gradients from the rest geometry and applies them to the
    deformed positions. Unlike the P1 version, ``F`` is evaluated at each
    cubature point because it is not constant over a quadratic element.

    Written with explicit loops (no vectorization) for clarity.

    Parameters
    ----------
    V2 : np.ndarray (n2, dim)
        Rest positions of the quadratic mesh (``dim`` is 2 or 3).
    T2 : np.ndarray (t, n_nodes)
        Quadratic connectivity (6 nodes for triangles, 10 for tets).
    bary : np.ndarray (t, n_quad, dim+1)
        Barycentric cubature points (from :func:`gauss_legendre_quadrature`).
    U2 : np.ndarray (n2, dim)
        Deformed positions of the quadratic mesh.

    Returns
    -------
    F : np.ndarray (t, n_quad, dim, dim)
        Deformation gradient at every cubature point of every element. Flatten
        the first two axes (``F.reshape(-1, dim, dim)``) to feed the existing
        element-tier energy functions, exactly matching
        ``(deformation_jacobian_p2(V2, T2, bary) @ U2.reshape(-1, 1))``.

    Example
    -------
    ```python
    F = deformation_gradient_p2(V2, T2, bary, U2)   # (t, n_quad, dim, dim)
    psi = arap_energy_element_F(F.reshape(-1, dim, dim), mu)
    E = float((weights.reshape(-1, 1) * psi).sum())
    ```
    """
    V2 = np.asarray(V2, dtype=float)
    U2 = np.asarray(U2, dtype=float)
    T2 = np.asarray(T2)
    bary = np.asarray(bary, dtype=float)

    dim = V2.shape[1]
    s = dim + 1
    t = T2.shape[0]
    n_nodes = T2.shape[1]
    n_quad = bary.shape[1]

    if n_nodes != p2_num_nodes(s):
        raise ValueError(
            f"T2 has {n_nodes} columns; expected {p2_num_nodes(s)} for dim {dim}"
        )

    H_p1 = _H_P1[dim]
    F = np.zeros((t, n_quad, dim, dim))

    for e in range(t):
        nodes = T2[e]
        Xc = V2[nodes[:s]]               # corner rest positions (s, dim)
        Ue = U2[nodes]                   # deformed positions of all P2 nodes

        # Affine reference->rest Jacobian inverse, constant over the element.
        XH = Xc.T @ H_p1                 # (dim, dim)
        XHi = np.linalg.inv(XH)          # (dim, dim)

        for q in range(n_quad):
            _, dNdL = p2_shape_functions(bary[e, q], s)  # (n_nodes, s)
            G = (dNdL @ H_p1) @ XHi                        # dN/dX (n_nodes, dim)
            # F[i, j] = Σ_node U_node[i] * G[node, j].
            F[e, q] = Ue.T @ G

    return F
