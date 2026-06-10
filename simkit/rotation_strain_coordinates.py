"""Rotation–strain (RS) coordinates via Jacobian fitting.

Splits a per-element deformation gradient ``F = I + grad u`` into a rotation
``R`` and a symmetric stretch ``S``. The rotation is taken from the *linear*
field's axis-angle (the axial vector of the antisymmetric part of ``grad u``)
and turned into a finite rotation by the matrix exponential (Rodrigues). The
rotation–strain target ``Y = R @ S - I`` keeps **both** the rotation and the
strain, and vertex displacements reproducing ``Y`` in Jacobian space are fit
using a precomputed factorization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .dirichlet_penalty import dirichlet_penalty
from .volume import volume


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

    Decomposes ``F = I + grad u`` into a rotation ``R`` and a symmetric stretch
    ``S = I + sym(grad u)``. ``R`` is the matrix exponential of the linear
    field's axis-angle (the axial vector of the antisymmetric part of
    ``grad u``). Sets the rotation–strain target ``Y = R @ S - I`` and fits
    ``u_rs`` so ``J u_rs ≈ Y``.

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

    symmetric = (F + F.transpose(0, 2, 1)) / 2.0      # S = I + sym(grad u)  (strain)
    antisymmetric = (F - F.transpose(0, 2, 1)) / 2.0  # skew(grad u) = rotation generator

    # Rotation from the linear field's axis-angle, exponentiated to a finite
    # rotation R = exp(antisymmetric).
    if dim == 2:
        # The 2D skew is [[0, -w], [w, 0]]; its exponential is a planar rotation.
        w = -antisymmetric[:, 0, 1]
        R = np.array([[np.cos(w), -np.sin(w)],
                        [np.sin(w),  np.cos(w)]]
                    ).transpose(2, 0, 1)
    else:
        # Rodrigues: A is already the skew matrix [w]x, so A @ A is the K^2 term.
        A = antisymmetric
        w = np.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], axis=1)  # axial vector
        theta = np.linalg.norm(w, axis=1)
        small = theta < 1e-8
        safe_theta = np.where(small, 1.0, theta)
        # sinθ/θ → 1 and (1 - cosθ)/θ² → 1/2 as θ → 0 (exact, numerically safe limits).
        c1 = np.where(small, 1.0, np.sin(theta) / safe_theta)
        c2 = np.where(small, 0.5, (1.0 - np.cos(theta)) / safe_theta ** 2)
        R = I + c1[:, None, None] * A + c2[:, None, None] * (A @ A) # Rodrigues formula!

    # Keep both rotation and strain: target is R @ S - I.
    Y = R @ symmetric - I

    # -------------------------------------------------
    # Fit back to displacement coordinates
    # -------------------------------------------------
    u_rs = pre.fit_displacements_to_jacobian(Y).reshape(-1, dim)

    if return_pre:
        return u_rs, pre
    else:
        return u_rs
