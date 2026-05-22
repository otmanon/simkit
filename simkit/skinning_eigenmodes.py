"""Laplacian eigenmodes for linear blend skinning and the LBS Jacobian.

Solves a generalized eigenproblem on the Dirichlet Laplacian with optional
boundary reduction or equality constraints, then forms the skinning Jacobian
from the mode weights.
"""

from typing import Optional, Tuple

import numpy as np
import scipy as sp

from .lbs_jacobian import lbs_jacobian
from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix
from .eigs import eigs


def skinning_eigenmodes(
    X: np.ndarray,
    T: np.ndarray,
    k: int,
    mu: float = 1,
    bI: Optional[np.ndarray] = None,
    Aeq: Optional[sp.sparse.spmatrix] = None,
) -> Tuple[np.ndarray, np.ndarray, sp.sparse.spmatrix]:
    """Compute ``k`` skinning eigenmodes and the LBS Jacobian.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (nt, simplex_size)
        Mesh simplices.
    k : int
        Number of modes to compute.
    mu : float, optional
        Stiffness parameter passed to the Dirichlet Laplacian.
    bI : np.ndarray, optional
        Pinned vertex indices; eigenproblem is solved on the free subset and
        embedded back into full vertex space.
    Aeq : scipy.sparse matrix, optional
        Equality constraint matrix for a saddle-point eigenproblem.

    Returns
    -------
    W : np.ndarray (n, k)
        Eigenmode weights per vertex (skinning weights).
    E : np.ndarray (k,) or (k, k)
        Eigenvalues (real part when complex).
    B : scipy.sparse matrix
        Linear blend skinning Jacobian from ``(X, W)``.
    """
    M = massmatrix(X, T)
    L = dirichlet_laplacian(X, T, mu=mu)
    if bI is not None:
        assert(isinstance(bI, np.ndarray))
        Ii = np.setdiff1d(np.arange(X.shape[0]), bI)
        L = L[Ii, :][:, Ii]
        M = sp.sparse.diags(M.diagonal()[Ii,], 0)

        [E, Wi] = eigs(L, k, M=M)

        Wi = Wi.real
        E = E.real
        W = np.zeros((X.shape[0], k))
        W[Ii, :] = Wi
    elif Aeq is not None:
        Z = sp.sparse.csc_matrix((Aeq.shape[0], Aeq.shape[0]))
        H = sp.sparse.bmat([[L, Aeq.T], [Aeq, Z]]).tocsc()
        K = sp.sparse.block_diag([M, Z]).tocsc()

        [E, W] = sp.sparse.linalg.eigs(H, k=k, M=K, which='LM', sigma=0)
        E = E.real
        W = W.real
        W = W[:X.shape[0], :]
    else:
        [E, W] = eigs(L, k, M=M)

        E = E.real
        W = W.real
    B = lbs_jacobian(X, W)

    return W, E, B
