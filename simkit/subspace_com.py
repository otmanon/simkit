"""Center of mass in a reduced displacement subspace."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp

from .massmatrix import massmatrix


def subspace_com(
    z: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    return_SB: bool = False,
    SB: Optional[sp.sparse.spmatrix] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, sp.sparse.spmatrix]]:
    """Compute center of mass for a configuration ``x = B z``.

    Mass is lumped from the mesh mass matrix; ``SB`` maps reduced coordinates
    to the mass-weighted average position.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    B : np.ndarray or scipy.sparse matrix (n * dim, r)
        Reduced displacement basis.
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity.
    return_SB : bool, optional
        If True, also return the mass-weighted subspace map ``SB``. Default
        False.
    SB : scipy.sparse matrix, optional
        Precomputed ``SB``. Built from ``B`` and the mass matrix if None.

    Returns
    -------
    com : np.ndarray (1, dim)
        Center of mass.
    SB : scipy.sparse matrix, optional
        Mass-weighted subspace map. Returned only if ``return_SB`` is True.
    """
    dim = X.shape[1]

    if SB is None:
        M = massmatrix(X, T)
        m = M.diagonal()
        total_mass = m.sum()
        rho = m / total_mass
        S = sp.sparse.kron(rho, np.identity(dim))
        SB = S @ B

    com = (SB @ z).reshape(-1, dim)
    if return_SB:
        return com, SB
    else:
        return com
