"""Compressed vibration modes via L1-regularized generalized eigenproblems.

Iteratively solves sparse vibration modes following "Compressed Vibration Modes
of Elastic Bodies" (https://www.sciencedirect.com/science/article/abs/pii/S0167839617300377)
by minimizing ``0.5 x^T H x + l ||x||_1`` subject to mass normalization and
orthogonality to previously found modes.
"""

from typing import Any, Optional, Union

import cvxopt
import numpy as np
import scipy as sp


def sp2cvxopt(A: Union[np.ndarray, sp.sparse.spmatrix]) -> Any:
    """Convert a SciPy or NumPy matrix to a CVXOPT matrix.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Dense or sparse matrix to convert.

    Returns
    -------
    A_cvxopt : cvxopt.matrix or cvxopt.spmatrix
        CVXOPT representation of ``A``.
    """
    if sp.sparse.issparse(A):
        i, j, v = sp.sparse.find(A)
        A_cvxopt = cvxopt.spmatrix(v, i, j)
    else:
        A_cvxopt = cvxopt.matrix(A)
    return A_cvxopt


def cvxopt2np(A: Any) -> np.ndarray:
    """Convert a CVXOPT matrix to a NumPy array.

    Parameters
    ----------
    A : cvxopt.matrix or cvxopt.spmatrix
        CVXOPT matrix to convert.

    Returns
    -------
    A_np : np.ndarray
        Dense NumPy array copy of ``A``.
    """
    if isinstance(A, cvxopt.spmatrix):
        return np.array(A)
    else:
        return np.array(A)


def eigs_iccm(
    H: sp.sparse.spmatrix,
    l: float,
    k: int,
    M: Optional[sp.sparse.spmatrix] = None,
    tolerance: float = 1e-6,
    max_iters: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """Solve for ``k`` compressed (sparse) vibration modes of stiffness ``H``.

    Each mode minimizes ``0.5 x^T H x + l ||x||_1`` subject to ``x^T M x = 1``
    and orthogonality to previously computed modes.

    Parameters
    ----------
    H : scipy.sparse matrix (n, n)
        Stiffness matrix.
    l : float
        L1 regularization parameter.
    k : int
        Number of modes to compute.
    M : scipy.sparse matrix (n, n), optional
        Mass matrix.
    tolerance : float, optional
        Convergence tolerance on the energy decrement. Default ``1e-6``.
    max_iters : int, optional
        Maximum iterations per mode. Default 100.
    verbose : bool, optional
        If ``True``, print per-iteration energy decrement. Default ``False``.

    Returns
    -------
    U : np.ndarray (n, k)
        Stacked eigenvectors (one column per mode).
    """
    dim_x = H.shape[0]
    U = np.zeros((dim_x, 0))

    cvxopt.solvers.options['show_progress'] = False

    # energy function to measure convergence
    def energy_func(x: np.ndarray) -> np.ndarray:
        e = 0.5 * x.T @ H @ x + l * np.abs(x).sum()
        return e.flatten()

    for mode in range(k):
        ck = np.random.randn(dim_x, 1)
        ck = ck / np.sqrt(ck.T @ M @ ck)

        for i in range(max_iters):
            e_prev = energy_func(ck)
            H_exp = sp.sparse.bmat([[H, -H], [-H, H]]).tocsc()

            rhs_exp = l * np.concatenate((M.sum(axis=1), M.sum(axis=1)), axis=0)

            un_eq_row = np.concatenate((M @ ck, -M @ ck), axis=0)
            un_eq_rhs = np.ones((1, 1))

            oc_eq_row = np.concatenate((M @ U, -M @ U), axis=0)
            oc_eq_rhs = np.zeros((U.shape[1], 1))

            A = np.concatenate((un_eq_row, oc_eq_row), axis=1).T
            b = np.concatenate((un_eq_rhs, oc_eq_rhs), axis=0)

            in_eq_mat = sp.sparse.identity(dim_x * 2).tocsc()
            in_eq_rhs = np.zeros((dim_x * 2, 1))

            # confert from scipy csc matrix to cvxopt sparse matrix
            P = sp2cvxopt(H_exp)
            q = sp2cvxopt(rhs_exp)
            A = sp2cvxopt(A)
            b = sp2cvxopt(b)
            G = sp2cvxopt(-in_eq_mat)
            h = sp2cvxopt(in_eq_rhs)
            res = cvxopt.solvers.qp(P, q, G, h, A, b)

            u = res['x'][:dim_x] - res['x'][dim_x:]
            u = cvxopt2np(u)

            ck = u / np.sqrt(u.T @ M @ u)

            e_curr = energy_func(ck)
            decrement = np.abs(e_curr - e_prev)

            if verbose:
                print(f"mode : {mode} , iter : {i} , decrement : {decrement}")
            if np.linalg.norm(decrement) < tolerance:
                break

        U = np.concatenate((U, ck.reshape(-1, 1)), axis=1)
    return U
