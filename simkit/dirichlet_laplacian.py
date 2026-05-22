"""Dirichlet (deformation-gradient) Laplacian on a simplicial mesh.

Builds ``J^T diag(vol * mu) J`` from the deformation Jacobian, optionally
averaging the per-coordinate blocks into a scalar vertex Laplacian.
"""

from __future__ import annotations

import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .volume import volume


def dirichlet_laplacian(
    X: np.ndarray,
    T: np.ndarray,
    mu: float | np.ndarray | None = 1,
    vector: bool = False,
) -> sp.sparse.csc_matrix:
    """Dirichlet energy Hessian ``J^T diag(vol * mu) J`` (or averaged vertex form).

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (t, dim+1)
        Simplex indices.
    mu : float or np.ndarray (t, 1), optional
        Per-element stiffness weight; scalar broadcasts to all elements.
    vector : bool, optional
        If ``False`` (default), average the ``dim`` per-coordinate Laplacian
        blocks into an ``(n, n)`` vertex operator. If ``True``, return the full
        ``(n*dim, n*dim)`` block-diagonal form.

    Returns
    -------
    L : scipy.sparse.csc_matrix
        Laplacian matrix, either ``(n, n)`` or ``(n*dim, n*dim)`` depending on
        ``vector``.

    Raises
    ------
    AssertionError
        If ``mu`` is an array whose length does not match the number of
        elements.
    """
    if mu is not None:
        if isinstance(mu, int) or isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu
        else:
            mu = mu.reshape(-1, 1)

        assert mu.shape[0] == T.shape[0]

    vol = volume(X, T)

    a = vol * mu

    dim = X.shape[1]
    n = X.shape[0]
    ae = np.repeat(a, dim * dim)
    A = sp.sparse.diags(ae)
    J = deformation_jacobian(X, T)

    H = J.T @ A @ J
    L = H

    if not vector:
        L = sp.sparse.csc_matrix((n, n))
        for i in range(dim):
            Ii = np.arange(n) * dim + i
            L = L + H[Ii, :][:, Ii]
        L = L / dim
    return L
