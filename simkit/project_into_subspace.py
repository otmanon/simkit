"""Mass-weighted least-squares projection onto a subspace basis."""

from typing import Optional, Union

import numpy as np
import scipy as sp


def project_into_subspace(
    y: np.ndarray,
    B: Union[np.ndarray, sp.sparse.spmatrix],
    M: Optional[sp.sparse.spmatrix] = None,
    BMB: Optional[Union[np.ndarray, sp.sparse.spmatrix]] = None,
    BMy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Minimize ``0.5 || B z - y ||^2_M`` over subspace coordinates ``z``.

    Solves the normal equations ``(B^T M B) z = B^T M y``. Optional
    precomputed ``BMB`` and ``BMy`` avoid repeated products when projecting
    many vectors with the same basis.

    Parameters
    ----------
    y : np.ndarray (n, 1) or (n,)
        Target vector in the full space.
    B : np.ndarray or scipy.sparse matrix (n, r)
        Subspace basis columns.
    M : scipy.sparse matrix (n, n), optional
        Mass matrix defining the weighted norm. Defaults to the identity.
    BMB : np.ndarray or scipy.sparse matrix (r, r), optional
        Precomputed ``B^T M B``.
    BMy : np.ndarray (r, 1) or (r,), optional
        Precomputed ``B^T M y``.

    Returns
    -------
    z : np.ndarray (r, 1)
        Subspace coordinates minimizing the weighted least-squares error.
    """
    if M is None:
        M = sp.sparse.identity(y.shape[0])

    if BMy is None:
        BMy = B.T @ M @ y

    if BMB is None:
        BMB = B.T @ M @ B

    if sp.sparse.issparse(BMB):
        z = sp.sparse.linalg.spsolve(BMB, BMy).reshape(-1, 1)
    else:
        z = np.linalg.solve(BMB, BMy).reshape(-1, 1)

    return z
