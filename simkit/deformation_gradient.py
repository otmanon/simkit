"""Per-element deformation gradient F from rest and deformed positions."""

from typing import Optional

import numpy as np


def deformation_gradient(
    X: np.ndarray, T: np.ndarray, U: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-element deformation gradient mapping rest to deformed geometry.

    Builds, per simplex, the constant gradient operator from the rest shape and
    applies it to the deformed positions to give ``F = dx/dX``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Simplex indices.
    U : np.ndarray (n, dim), optional
        Deformed vertex positions. NOTE: there is no default; ``U`` must be
        supplied (passing ``None`` raises ``TypeError`` when indexed).

    Returns
    -------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradient.

    Example
    -------
    ```python
    F = deformation_gradient(X, T, U)   # F[e] is the dim x dim gradient
    ```
    """
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    dim = X.shape[1]

    # H is the reference-element shape-function gradient (rows = local vertex,
    # cols = reference axis), set by the simplex dimension.
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

    # Per-element gradient operator D from the rest geometry:
    #   XH maps reference axes to rest edges; its inverse pulls back, and
    #   D = (H XH^{-1})^T turns nodal positions into a spatial gradient.
    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    # Apply the rest-built operator to the deformed positions.
    UT = U[T]
    F = (D @ UT).transpose(0, 2, 1)
    return F