"""Rotation–strain (RS) coordinates via Jacobian fitting and polar decomposition.

Extracts a per-element rotation from a displacement field, forms the
rotation-only target ``Y = R - I``, and fits vertex displacements that
reproduce ``Y`` in Jacobian space using a precomputed factorization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .dirichlet_penalty import dirichlet_penalty
from .volume import volume
from .membrane_deformation_jacobian import membrane_deformation_jacobian


class RSPrecompute:
    """Precompute for fitting displacements to rotation-only Jacobian targets.

    Builds the deformation Jacobian ``J``, volume-weighted normal equations,
    and a sparse factorization of ``J^T Vol J + H_pin``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (nt, simplex_size)
        Mesh simplices.
    pinned : np.ndarray, optional
        Pinned vertex indices. If ``None``, pins vertices near the mesh mean.

    Attributes
    ----------
    X : np.ndarray (n, dim)
        Rest positions.
    T : np.ndarray (nt, simplex_size)
        Mesh simplices.
    J : scipy.sparse matrix
        Deformation Jacobian (membrane or solid).
    K : scipy.sparse matrix
        Volume-weighted transpose ``J^T Vol``.
    factorization : callable
        Sparse solve handle for the penalized normal system.
    """

    def __init__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        pinned: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.T = T
        dim = X.shape[1]

        if X.shape[1] == 3 and T.shape[1] == 3:
            self.J = membrane_deformation_jacobian(X, T)
            dd = 6
        else:
            self.J = deformation_jacobian(X, T)
            dd = dim * dim
        if pinned is None:
            mean_X = X.mean(axis=0).reshape(-1, dim)
            pinned = np.where(np.linalg.norm(X - mean_X, axis=1) < 0.01)[0]

        H_pin, _b = dirichlet_penalty(pinned, X[pinned], X.shape[0], 1e8)

        J = self.J
        vol = volume(X, T)
        Vol = sp.sparse.diags(vol.flatten())
        Vol = sp.sparse.kron(Vol, sp.sparse.identity(dd))
        L = J.T @ Vol @ J
        self.K = J.T @ Vol
        A = L + H_pin
        self.factorization = sp.sparse.linalg.factorized(A.tocsc())

    def fit_displacements_to_jacobian(self, Y: np.ndarray) -> np.ndarray:
        """Solve for vertex displacements whose Jacobian matches ``Y``.

        Parameters
        ----------
        Y : np.ndarray (*, dim, dim) or (*, 3, 2)
            Per-element rotation targets (flattened internally).

        Returns
        -------
        u : np.ndarray (n*dim, 1)
            Fitted displacement vector.
        """
        return self.factorization(self.K @ Y.reshape(-1, 1))



def rotation_strain_coordinates(
    X: np.ndarray,
    T: np.ndarray,
    u: np.ndarray,
    pinned: Optional[np.ndarray] = None,
    pre: Optional[RSPrecompute] = None,
    return_pre: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, RSPrecompute]]:
    """Map a displacement field to rotation–strain coordinates.

    Decomposes ``F = I + grad u`` (or membrane ``F``) into rotation ``R``,
    sets ``Y = R - I``, and fits ``u_rs`` so ``J u_rs ≈ Y``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (nt, simplex_size)
        Mesh simplices. Must be triangles for 2D, or tetrahedra for 3D.
    u : np.ndarray (n, dim) or (n*dim,)
        Input displacement field.
    pinned : np.ndarray, optional
        Pinned vertex indices passed to :class:`RSPrecompute` if ``pre`` is
        ``None``.
    pre : RSPrecompute, optional
        Reusable precompute. Built from ``(X, T, pinned)`` when ``None``.
    return_pre : bool, optional
        If ``True``, return ``(u_rs, pre)``; otherwise only ``u_rs``.

    Returns
    -------
    u_rs : np.ndarray (n, dim)
        Rotation–strain displacement coordinates.
    pre : RSPrecompute, optional
        Precompute object (only if ``return_pre`` is ``True``).
    """

    dim = X.shape[1]
    u = u.reshape(-1, 1)

    if pre is None:
        pre = RSPrecompute(X, T, pinned)

    grad_u = (pre.J @ u).reshape(-1, dim, dim)
    I = np.identity(dim)[None, ...]

    F = grad_u + I

    symmetric = (F + F.transpose(0, 2, 1)) / 2.0
    antisymmetric = (F - F.transpose(0, 2, 1)) / 2.0

    if dim == 2:
        w = -antisymmetric[:, 0, 1]
        R = np.array([[np.cos(w), -np.sin(w)],
                        [np.sin(w),  np.cos(w)]]
                    ).transpose(2, 0, 1)
    else:
        # safer: use polar instead of axis-angle approx
        U, _, Vt = np.linalg.svd(F)
        R = U @ Vt

    Y = R - I

    # -------------------------------------------------
    # Fit back to displacement coordinates
    # -------------------------------------------------
    u_rs = pre.fit_displacements_to_jacobian(Y).reshape(-1, dim)

    if return_pre:
        return u_rs, pre
    else:
        return u_rs
