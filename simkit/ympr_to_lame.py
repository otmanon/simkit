"""Convert Young's modulus and Poisson's ratio to Lamé parameters.

Isotropic linear elasticity uses shear modulus ``mu`` and first Lamé parameter
``lam``; these follow from ``(ym, pr)`` via the standard textbook relations.
"""

from __future__ import annotations

import numpy as np


def ympr_to_lame(
    ym: float | np.ndarray, pr: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Lamé parameters ``(mu, lam)`` from Young's modulus and Poisson's ratio.

    Parameters
    ----------
    ym : float or np.ndarray
        Young's modulus.
    pr : float or np.ndarray
        Poisson's ratio.

    Returns
    -------
    mu : float or np.ndarray
        Shear modulus ``ym / (2 * (1 + pr))``.
    lam : float or np.ndarray
        First Lamé parameter
        ``ym * pr / ((1 + pr) * (1 - 2 * pr))``.
    """
    mu = ym / (2 * (1 + pr))
    lam = ym * pr / ((1 + pr) * (1 - 2 * pr))
    return mu, lam
