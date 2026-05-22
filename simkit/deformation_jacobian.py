"""Sparse linear operators mapping flattened positions to deformation gradients."""

import numpy as np
import scipy as sp

from .simplex_vertex_map import simplex_vertex_map


def deformation_jacobian(X: np.ndarray, T: np.ndarray) -> "sp.sparse.csc_matrix":
    """Sparse operator J mapping flattened positions to stacked gradients.

    Builds the constant linear map such that ``J @ x`` gives the stacked
    per-element deformation gradients, where ``x`` is the vertex positions
    flattened with the default C order. Useful because ``J`` is built once from
    the rest geometry and reused every simulation step.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Simplex indices.

    Returns
    -------
    J : scipy.sparse.csc_matrix (dim*dim*t, dim*n)
        Deformation Jacobian operator.

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 3)   # row-major Fij blocks per element
    ```
    """
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]
    dim = X.shape[1]

    # Reference-element shape-function gradient, set by simplex dimension.
    if dim == 1:
        H = np.array([[-1],
                      [1]])
    if dim == 2:
        H = np.array([[-1, -1],
                      [1, 0],
                      [0, 1]])
    if dim == 3:
        H = np.array([[-1, -1, -1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    # Per-element gradient operator D from the rest geometry (see
    # deformation_gradient for the derivation).
    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    # Replicate D across the dim spatial components: De[:, dim*i:dim*(i+1), :]
    # places D in the rows for component i and the matching strided columns.
    De = np.zeros((nt, dim * dim, dt * dim))
    for i in range(dim):
        Di = np.zeros((nt, dim, dt * dim))
        Ii = np.arange(dt) * dim + i
        Di[:, :, Ii] = D
        De[:, dim * i:dim * (i + 1), :] = Di

    # Scatter the dense per-element blocks De into a global sparse matrix.
    Ii = np.arange(dim * dim * nt).reshape(nt, dim * dim, 1)
    Ii = np.repeat(Ii, dim * dt, axis=2)
    Ji = np.arange(dim * dt * nt).reshape(nt, 1, dim * dt)
    Ji = np.repeat(Ji, dim * dim, axis=1)
    H_sparse = sp.sparse.csc_matrix(
        (De.flatten(), (Ii.flatten(), Ji.flatten())),
        (nt * dim * dim, nt * dim * dt),
    )

    # Compose with the simplex->vertex gather (lifted to dim components) so the
    # operator acts on global flattened positions rather than per-element ones.
    S = simplex_vertex_map(T)
    Se = sp.sparse.kron(S, sp.sparse.identity(dim))
    J = H_sparse @ Se
    return J


def membrane_deformation_jacobian(X: np.ndarray, T: np.ndarray) -> "sp.sparse.csc_matrix":
    """Deformation Jacobian for membrane (triangle-in-3D) elements.

    Same construction as :func:`deformation_jacobian`, intended for surface
    elements where the gradient maps into the in-plane reference directions.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, 3)
        Triangle indices.

    Returns
    -------
    J : scipy.sparse.csc_matrix
        Deformation Jacobian operator.

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 2)
    ```

    Notes
    -----
    The body below is identical to :func:`deformation_jacobian`; the membrane
    specialization (an ``H`` that drops to ``dim-1`` reference axes) is not yet
    implemented here. Flagged rather than changed to preserve current numerics.
    """
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]
    dim = X.shape[1]

    if dim == 1:
        H = np.array([[-1],
                      [1]])
    if dim == 2:
        H = np.array([[-1, -1],
                      [1, 0],
                      [0, 1]])
    if dim == 3:
        H = np.array([[-1, -1, -1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    De = np.zeros((nt, dim * dim, dt * dim))
    for i in range(dim):
        Di = np.zeros((nt, dim, dt * dim))
        Ii = np.arange(dt) * dim + i
        Di[:, :, Ii] = D
        De[:, dim * i:dim * (i + 1), :] = Di

    Ii = np.arange(dim * dim * nt).reshape(nt, dim * dim, 1)
    Ii = np.repeat(Ii, dim * dt, axis=2)
    Ji = np.arange(dim * dt * nt).reshape(nt, 1, dim * dt)
    Ji = np.repeat(Ji, dim * dim, axis=1)
    H_sparse = sp.sparse.csc_matrix(
        (De.flatten(), (Ii.flatten(), Ji.flatten())),
        (nt * dim * dim, nt * dim * dt),
    )

    S = simplex_vertex_map(T)
    Se = sp.sparse.kron(S, sp.sparse.identity(dim))
    J = H_sparse @ Se
    return J