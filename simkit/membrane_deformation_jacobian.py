"""Membrane (triangle-in-3D) deformation gradient and its sparse Jacobian.

Builds per-triangle in-plane deformation gradients from a local tangent basis
on the rest geometry, and assembles the constant sparse operator that maps
flattened vertex positions to stacked gradient blocks.
"""

import numpy as np
import scipy as sp

import simkit as sk

from .interweaving_matrix import interweaving_matrix


def interweaving_matrix(t: int, d: int) -> sp.sparse.csc_matrix:
    """Permutation matrix mapping C-order vectorization to Fortran order.

    Parameters
    ----------
    t : int
        Number of blocks (e.g. simplex vertices).
    d : int
        Dimension of each block (e.g. spatial or in-plane axes).

    Returns
    -------
    M : scipy.sparse.csc_matrix (t*d, t*d)
        Sparse interweaving / reordering matrix.
    """
    ii = np.arange(t * d)

    i = ii.reshape(t, d)
    j = ii.reshape(t, d, order='F')
    v = np.ones(ii.shape)
    M = sp.sparse.csc_matrix((v, (i.flatten(), j.flatten())), shape=(t * d, t * d))
    return M


def membrane_deformation_gradient(
    Y: np.ndarray, X: np.ndarray, T: np.ndarray
) -> np.ndarray:
    """Per-triangle membrane deformation gradient in the local tangent basis.

    Constructs an orthonormal in-plane frame on the rest triangle, inverts the
    rest edge matrix in that frame, and applies it to deformed edge vectors.

    Parameters
    ----------
    Y : np.ndarray (n, 3)
        Deformed vertex positions.
    X : np.ndarray (n, 3)
        Rest vertex positions.
    T : np.ndarray (t, 3)
        Triangle indices.

    Returns
    -------
    F : np.ndarray (t, 3, 2)
        Per-triangle deformation gradient (3 spatial × 2 in-plane components).
    """
    X0 = X[T[:, 0]]
    X1 = X[T[:, 1]]
    X2 = X[T[:, 2]]

    V1 = X1 - X0
    V2 = X2 - X0

    dim = 2
    dt = 3
    U1 = V1 / np.linalg.norm(V1, axis=1)[:, None]
    N = np.cross(V1, V2)
    N = N / np.linalg.norm(N, axis=1)[:, None]
    U2 = np.cross(N, U1)

    # edge vectors in the local basis
    E2D = np.array([[(U1 * V1).sum(axis=1), (U1 * V2).sum(axis=1)],
                    [(U2 * V1).sum(axis=1), (U2 * V2).sum(axis=1)]]).transpose(2, 0, 1)

    E2Di = np.linalg.inv(E2D)

    v1 = (Y[T[:, 1]] - Y[T[:, 0]])[:, :, None]
    v2 = (Y[T[:, 2]] - Y[T[:, 0]])[:, :, None]
    G = np.concatenate([v1, v2], axis=2)
    F = G @ E2Di
    return F


def membrane_deformation_jacobian(
    X: np.ndarray, T: np.ndarray
) -> sp.sparse.csc_matrix:
    """Sparse operator mapping flattened positions to membrane gradients.

    Builds the constant linear map such that ``J @ x`` gives stacked
    per-triangle membrane deformation gradients from rest geometry ``X``.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Rest vertex positions (must have three columns).
    T : np.ndarray (t, 3)
        Triangle indices (must have three columns).

    Returns
    -------
    J : scipy.sparse.csc_matrix
        Membrane deformation Jacobian operator.
    """
    assert X.shape[1] == 3
    assert T.shape[1] == 3
    X0 = X[T[:, 0]]
    X1 = X[T[:, 1]]
    X2 = X[T[:, 2]]

    V1 = X1 - X0
    V2 = X2 - X0

    dim = 2
    dt = 3
    U1 = V1 / np.linalg.norm(V1, axis=1)[:, None]
    N = np.cross(V1, V2)
    N = N / np.linalg.norm(N, axis=1)[:, None]
    U2 = np.cross(N, U1)

    # edge vectors in the local basis
    E2D = np.array([[(U1 * V1).sum(axis=1), (U1 * V2).sum(axis=1)],
                    [(U2 * V1).sum(axis=1), (U2 * V2).sum(axis=1)]]).transpose(2, 0, 1)

    E2Di = np.linalg.inv(E2D)

    H = np.array([[-1, -1],
                  [1, 0],
                  [0, 1]])
    D = (H[None, :, :] @ E2Di).transpose(0, 2, 1)
    # D = D.reshape(-1, dim*(dim+1))
    # D = D.reshape(-1, dim+1, dim).transpose(0, 2, 1)
    G = np.kron(D, np.eye(dim + 1))
    M = interweaving_matrix(3, 2).toarray()
    G = M[None, :, :] @ G
    Q = sp.sparse.block_diag(G)
    S = sk.simplex_vertex_map(T)

    # nt = T.shape[0]
    # De = np.zeros((nt, (dim+1)*dim,  dt*dim))

    # for i in range(dim):
    #     Di = np.zeros((nt, dim, dt*dim))
    #     Ii = np.arange(dt)*dim + i
    #     Di[:, :, Ii] = D

    #     De[:, dim*i:dim*(i + 1), :] = Di

    Se = sp.sparse.kron(S, sp.sparse.identity(dim + 1))
    J = Q @ Se
    return J
