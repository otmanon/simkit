"""Per-mode scaling limits for actuation from Dirichlet energy density.

Given deformation modes ``D``, caps each mode's contribution so the induced
Dirichlet energy density does not exceed ``max_s``.
"""

import numpy as np

from .deformation_jacobian import deformation_jacobian


def limit_actuation_dirichlet_energy(
    X: np.ndarray, T: np.ndarray, D: np.ndarray, max_s: float
) -> np.ndarray:
    """Per-mode scale factors ``a`` so ``||J @ (a_k D_k)||`` is bounded.

    For each column of ``D``, computes the Frobenius norm of the per-element
    deformation-gradient increment ``J @ D_k`` and sets
    ``a_k = max_s / max_e density``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Simplex indices.
    D : np.ndarray (n*dim, num_modes)
        Actuation modes in stacked coordinate form.
    max_s : float
        Maximum allowed Dirichlet energy density per mode.

    Returns
    -------
    a : np.ndarray (num_modes,)
        Per-mode scaling factors (at most 1 when the mode already fits).
    """
    J = deformation_jacobian(X, T)
    dim = X.shape[1]
    JD = (J @ D).reshape(T.shape[0], dim, dim, D.shape[1])
    dirichlet_energy_density = np.sqrt(np.sum(JD**2, axis=(1, 2)))
    a = max_s / np.max(dirichlet_energy_density, axis=0)
    a = a.reshape(-1,)
    return a
