"""Generalized eigenmodes of the ARAP Hessian under the mass matrix.

Solves the sparse generalized eigenproblem for the lowest ``k`` modes, with
optional elimination of constrained (boundary) DOFs.
"""

from typing import Optional, Tuple

import numpy as np
import scipy as sp

from .energies.linear_elasticity import linear_elasticity_hessian

from .eigs import eigs
from .massmatrix import massmatrix


def linear_modal_analysis(
    X: np.ndarray,
    T: np.ndarray,
    k: int,
    bI: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` linear vibration modes from ARAP stiffness and mass.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, dim+1)
        Simplex indices.
    k : int
        Number of modes to compute.
    bI : np.ndarray (n_b,), optional
        Indices of constrained vertices. Their DOFs are eliminated before the
        eigen solve and re-inserted as zeros in the full basis.

    Returns
    -------
    E : np.ndarray (k,)
        Generalized eigenvalues (squared frequencies), real part only.
    B : np.ndarray (n*dim, k)
        Eigenvectors (modal displacements), real part only.
    """
    n = X.shape[0]
    dim = X.shape[1]
    mu = np.ones((T.shape[0], 1))
    H = linear_elasticity_hessian(X=X, T=T, mu=mu, lam=mu)
    M = massmatrix(X, T)
    M = sp.sparse.kron(M, sp.sparse.identity(dim))

    if bI is not None:
        bI = np.array(bI)
        assert bI.ndim == 1
        bIr = np.repeat(bI[:, None], X.shape[1], axis=1)

        bIe = bIr * X.shape[1] + np.arange(dim)

        Ii = np.setdiff1d(np.arange(n * dim), bIe)
        H = H[Ii, :][:, Ii]
        M = sp.sparse.diags(M.diagonal()[Ii,], 0)

        [Ei, Bi] = eigs(H, k=k, M=M)
        B = np.zeros((n * dim, k))
        B[Ii, :] = Bi
        E = Ei
    else:
        [E, B] = eigs(H, k=k, M=M)

    E = E.real
    B = B.real

    return E, B
