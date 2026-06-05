"""Per-element gradient operator from mesh geometry and nodal values."""

from typing import Tuple

import numpy as np
import scipy as sp


def grad(
    X: np.ndarray, F: np.ndarray, U: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-element gradients of nodal field ``U`` on simplices ``F``.

    Constructs the constant gradient operator from rest positions ``X`` and
    applies it to the nodal values ``U`` on triangles (``t=3``) or tets
    (``t=4``).

    Parameters
    ----------
    X : np.ndarray (n, d)
        Rest (or reference) vertex positions.
    F : np.ndarray (t, s)
        Simplex indices; ``s=3`` for triangles, ``s=4`` for tetrahedra.
    U : np.ndarray (n, ...)
        Nodal values to differentiate (typically deformed positions).

    Returns
    -------
    grad : np.ndarray
        Per-element gradient of ``U`` (squeezed batch dimensions).
    HXHi : np.ndarray (t, s, d)
        Reference-to-rest map ``H @ pinv(X @ H)`` used in the operator.
    """
    t = F.shape[0]  # simplex size
    n = X.shape[0]
    dt = F.shape[1]

    TU = (U[F]).transpose([0, 2, 1])
    if F.shape[1] == 3:
        # triangle mesh!
        H = np.array([[-1.0, -1],
                      [1., 0],
                      [0, 1.0]], dtype=U.dtype)

    # for each triangle, get the three vertex positions dealing with it
    elif F.shape[1] == 4:
        # tet mesh!
        H = np.array([[-1, -1, -1],
                      [1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]], dtype=U.dtype)

    Tx = X[F].transpose([0, 2, 1])
    XH = Tx @ H
    XHi = np.linalg.pinv(XH)
    HXHi = H @ XHi

    tu = TU.reshape(-1, 3)

    grad = TU @ HXHi

    d = X.shape[1]

    # J = np.repeat(np.F[0, :], d)

    Fe = np.repeat(F[:, None, :], 2, axis=1)
    Fe = Fe * 2
    Fe[:, 1, :] += 1
    J = Fe.reshape(-1)
    I = np.arange(J.shape[0])
    vals = np.ones(I.shape)
    Pr = sp.sparse.csc_matrix((vals, (I, J)), shape=(d * dt * t, d * n))

    u = HXHi[:, :, 0]
    # add 0 between every column of u
    u = np.repeat(u, 2, axis=1)
    # only keep

    v = HXHi[:, :, 1]

    # grad = torch.squeeze(grad, dim=1)

    # squeze first two dims
    grad = np.squeeze(grad)
    return grad, HXHi


