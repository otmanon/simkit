r"""Force-dual modes: a subspace fitted to the displacements random forces produce.

*Force-Dual Modes: Subspace Design from Stochastic Forces* (Benchekroun,
Grinspun, Chiaramonte, Etter, SIGGRAPH Asia 2025) builds a reduced space from a
force distribution instead of from the stiffness alone. Random forces
``f ~ N(0, Sigma_F)`` pushed through the solid linearised at rest, ``H u = f``,
give displacements ``u ~ N(0, Sigma_U)`` with
``Sigma_U = H^{-1} Sigma_F H^{-1}``; the subspace is the best rank-``k`` fit to
that distribution, i.e. its top principal directions, in the mass norm.

This module covers a **diagonal** force covariance, given as a variance per
DOF. White noise weighted by mass everywhere (``variance = diag(M)``) recovers
:func:`simkit.linear_modal_analysis`; variance on a few DOFs gives their
Green's functions; variance confined to a region (a limb, a contact patch)
gives modes that answer forces there.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg


def force_dual_modes(
    H: sp.sparse.spmatrix,
    variance: np.ndarray,
    k: int,
    M: Optional[sp.sparse.spmatrix] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Top ``k`` force-dual modes for the diagonal force covariance ``diag(variance)``.

    Only the DOFs where ``variance > 0`` are touched: with
    ``Sigma_F = L L^T``, ``L`` the supported columns of
    ``diag(sqrt(variance))``, the displacement covariance is ``G G^T`` with
    ``G = H^{-1} L``, and with ``M = N N^T``, ``N = diag(sqrt(diag(M)))``, the
    thin SVD ``N G = U S V^T`` gives ``B = N^{-1} U[:, :k]`` and
    ``D = S[:k]^2`` (the paper's Eqs. 35-36). The cost is one sparse LU of
    ``H`` and one solve per supported DOF, so this is the route for a variance
    on a region; it is a dense inverse for a variance on every DOF.

    Parameters
    ----------
    H : scipy.sparse matrix (d, d)
        Symmetric positive-definite stiffness, typically the elastic Hessian
        at rest restricted to the free DOFs (pinning is what makes it
        definite; a free body's rigid modes make the solve singular).
    variance : np.ndarray (d,)
        Diagonal of the force covariance, per DOF (interleaved, like ``H``).
        Zero where no force acts.
    k : int
        Number of modes to return. At most the number of supported DOFs, the
        rank of the displacement distribution.
    M : scipy.sparse matrix (d, d), optional
        Lumped (diagonal) mass matrix: the norm the modes are fitted and
        orthonormalised in. Defaults to the identity.

    Returns
    -------
    D : np.ndarray (k,)
        Displacement variance along each mode, descending: the top
        eigenvalues of ``Sigma_U M``.
    B : np.ndarray (d, k)
        The modes, ``B.T @ M @ B = I``, each signed so its largest-magnitude
        entry is positive (an SVD's signs are otherwise arbitrary).

    Raises
    ------
    ValueError
        If ``H``, ``variance`` and ``M`` disagree on ``d``; if ``variance`` is
        negative anywhere (it is a covariance diagonal); if ``M`` is not
        diagonal with a positive diagonal, since ``N^{-1}`` is only cheap and
        exact for a lumped mass matrix; or if ``k`` exceeds the number of
        supported DOFs, beyond which every mode would be an arbitrary null
        direction.

    Notes
    -----
    With ``variance = diag(M)`` on every DOF, ``Sigma_U M = (H^{-1} M)^2``,
    whose eigenvectors solve ``H \phi = \gamma M \phi``: these are exactly the
    linear modal analysis modes, with ``D = 1 / \gamma^2``. With variance on
    every DOF the paper's other route, the sparse generalized eigenproblem
    ``H diag(variance)^{-1} H B = M B D^{-1}``, is cheaper; it is not
    implemented here.

    Example
    -------
    ```python
    # 16 modes answering forces on the vertices in `region` only.
    variance = np.zeros(2 * n)
    variance[2 * region] = mass[region]
    variance[2 * region + 1] = mass[region]
    D, B = force_dual_modes(H_free, variance[free], 16, M=M_free)
    ```
    """
    variance = np.asarray(variance, dtype=float).ravel()
    d = variance.shape[0]
    if H.shape != (d, d):
        raise ValueError(f"H is {H.shape} but variance has {d} entries; they must agree")
    if np.any(variance < 0.0):
        raise ValueError("variance must be non-negative: it is the diagonal of a covariance")
    if M is None:
        root_mass = np.ones(d)
    else:
        M = sp.sparse.csr_matrix(M)
        if M.shape != (d, d):
            raise ValueError(f"M is {M.shape} but variance has {d} entries; they must agree")
        mass = M.diagonal()
        if (M - sp.sparse.diags(mass)).count_nonzero() != 0 or np.any(mass <= 0.0):
            raise ValueError("M must be a lumped (diagonal, positive) mass matrix")
        root_mass = np.sqrt(mass)
    support = np.flatnonzero(variance > 0.0)
    if not 0 < k <= support.size:
        raise ValueError(
            f"k = {k}, but the variance supports {support.size} DOFs, which is the rank of the "
            "displacement distribution; ask for at most that many modes"
        )

    # Sigma_F = L L^T; G = H^{-1} L is the displacement covariance's square root.
    L = np.zeros((d, support.size))
    L[support, np.arange(support.size)] = np.sqrt(variance[support])
    G = sp.sparse.linalg.splu(sp.sparse.csc_matrix(H)).solve(L)

    # Top left singular vectors of N G, mapped back by N^{-1}, are M-orthonormal.
    U, S, _ = np.linalg.svd(root_mass[:, None] * G, full_matrices=False)
    B = U[:, :k] / root_mass[:, None]
    D = S[:k] ** 2

    largest = np.abs(B).argmax(axis=0)
    B = B * np.sign(B[largest, np.arange(k)])[None, :]
    return D, B
