"""Macklin-Mueller stable Neo-Hookean elastic energy.

Implements the stable Neo-Hookean formulation with a quadratic
volume-preserving term from Macklin and Mueller, "A Constraint-based
Formulation of Stable Neo-Hookean Materials" (MIG 2021):
https://dl.acm.org/doi/10.1145/3487983.3488289

Also equivalent to the one used in the Dynamic Deformables course notes (Equation 6.9)
https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf 


Per-element energy density (3D form; the 2D form drops the third determinant
factor):

    psi(F) = mu * (1 - det(F))
           + (lam / 2) * (1 - det(F))**2
           + (mu / 2) * (||F||_F**2 - dim)

The ``(1 - det(F))`` term is the volume-preserving constraint and the
quadratic ``(1 - det(F))**2`` term is its penalty; together they replace
the ``log(det(F))`` of the classical Neo-Hookean form, yielding an energy
that is finite and well-behaved at inverted configurations.

Follows the standardized three-tier layout (see :mod:`simkit.energies.arap`
for the reference). This energy has only the deformation gradient (``F``)
representation, so there is no ``_S`` tier:

Element tier (``*_element_F``)
    Per-element density and derivative blocks. Material parameters ``mu`` and
    ``lam`` only: no quadrature weight ``vol``, no summation, no operator.

Global explicit tier (``*_x``)
    Takes a prebuilt deformation Jacobian ``J`` and weights ``vol``, calls the
    element tier, weights, and assembles.

Self-contained tier (no suffix)
    Builds ``J`` and ``vol`` from rest geometry ``(X, T)``.

Notes
-----
Closed-form gradient and Hessian transcribed from generated C++. The
``mat2py`` ordering helpers reindex from the column-major (MATLAB/C++) flat
layout to the row-major ``F`` layout used throughout the library.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..mat2py import (
    _4vector_2D_ordering_,
    _9vector_3D_ordering_,
    _4x4matrix_2D_ordering_,
    _9x9matrix_3D_ordering_,
)
from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..psd_project import psd_project
from ..symmetric_stretch_map import symmetric_stretch_map


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def macklin_mueller_neo_hookean_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Macklin-Mueller Neo-Hookean energy density.

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        psi = (
            mu * (-F_00 * F_11 + F_01 * F_10 + 1.0)
            + (lam * (-(F_00 * F_11) + F_01 * F_10 + 1.0) ** 2) / 2.0
            + (mu * (F_00 ** 2 + F_01 ** 2 + F_10 ** 2 + F_11 ** 2 - 2.0)) / 2.0
        )
    elif dim == 3:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_02 = F[:, 0, 2].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        F_12 = F[:, 1, 2].reshape(-1, 1)
        F_20 = F[:, 2, 0].reshape(-1, 1)
        F_21 = F[:, 2, 1].reshape(-1, 1)
        F_22 = F[:, 2, 2].reshape(-1, 1)
        det = (
            -F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22
            - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0
        )
        psi = (
            mu * det
            + (lam * det ** 2) / 2.0
            + (mu * (
                F_00 ** 2 + F_01 ** 2 + F_02 ** 2
                + F_10 ** 2 + F_11 ** 2 + F_12 ** 2
                + F_20 ** 2 + F_21 ** 2 + F_22 ** 2 - 3.0)) / 2.0
        )
    else:
        raise ValueError("Macklin-Mueller Neo-Hookean supports dim=2 or dim=3")
    return psi


def macklin_mueller_neo_hookean_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element first Piola-Kirchhoff stress (gradient w.r.t. ``F``).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    P : np.ndarray (t, dim, dim)
        Per-element PK1 stress blocks. No quadrature weighting applied.
    """
    dim = F.shape[-1]
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        det = -F_00 * F_11 + F_01 * F_10 + 1.0
        P = np.hstack([
            F_00 * mu - F_11 * mu - F_11 * lam * det,
            F_01 * mu + F_10 * mu + F_01 * lam * det,
            F_01 * mu + F_10 * mu + F_10 * lam * det,
            -F_00 * mu + F_11 * mu - F_00 * lam * det,
        ])[:, _4vector_2D_ordering_].reshape(-1, 2, 2)
    elif dim == 3:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_02 = F[:, 0, 2].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        F_12 = F[:, 1, 2].reshape(-1, 1)
        F_20 = F[:, 2, 0].reshape(-1, 1)
        F_21 = F[:, 2, 1].reshape(-1, 1)
        F_22 = F[:, 2, 2].reshape(-1, 1)
        det = (
            -F_00 * F_11 * F_22 + F_00 * F_12 * F_21 + F_01 * F_10 * F_22
            - F_01 * F_12 * F_20 - F_02 * F_10 * F_21 + F_02 * F_11 * F_20 + 1.0
        )
        P = np.hstack([
            -mu * (F_11 * F_22 - F_12 * F_21) + F_00 * mu - lam * (F_11 * F_22 - F_12 * F_21) * det,
            mu * (F_01 * F_22 - F_02 * F_21) + F_10 * mu + lam * (F_01 * F_22 - F_02 * F_21) * det,
            -mu * (F_01 * F_12 - F_02 * F_11) + F_20 * mu - lam * (F_01 * F_12 - F_02 * F_11) * det,
            mu * (F_10 * F_22 - F_12 * F_20) + F_01 * mu + lam * (F_10 * F_22 - F_12 * F_20) * det,
            -mu * (F_00 * F_22 - F_02 * F_20) + F_11 * mu - lam * (F_00 * F_22 - F_02 * F_20) * det,
            mu * (F_00 * F_12 - F_02 * F_10) + F_21 * mu + lam * (F_00 * F_12 - F_02 * F_10) * det,
            -mu * (F_10 * F_21 - F_11 * F_20) + F_02 * mu - lam * (F_10 * F_21 - F_11 * F_20) * det,
            mu * (F_00 * F_21 - F_01 * F_20) + F_12 * mu + lam * (F_00 * F_21 - F_01 * F_20) * det,
            -mu * (F_00 * F_11 - F_01 * F_10) + F_22 * mu - lam * (F_00 * F_11 - F_01 * F_10) * det,
        ])[:, _9vector_3D_ordering_].reshape(-1, 3, 3)
    else:
        raise ValueError("Macklin-Mueller Neo-Hookean supports dim=2 or dim=3")
    return P


def macklin_mueller_neo_hookean_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. ``F`` (vectorized blocks).

    Parameters
    ----------
    F : np.ndarray (t, dim, dim)
        Per-element deformation gradients.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.

    Returns
    -------
    H : np.ndarray (t, dim*dim, dim*dim)
        Per-element Hessian blocks in vectorized ``F`` layout. No quadrature
        weighting applied. Not PSD-projected; projection happens in the global
        tier.
    """
    dim = F.shape[-1]
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)

        t0 = np.zeros((F.shape[0], 4))
        t0[:, [0]] = mu + (F_11 ** 2) * lam
        t0[:, [1]] = -F_01 * F_11 * lam
        t0[:, [2]] = -F_10 * F_11 * lam
        t0[:, [3]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam

        t1 = np.zeros((F.shape[0], 4))
        t1[:, [0]] = -F_01 * F_11 * lam
        t1[:, [1]] = mu + (F_01 ** 2) * lam
        t1[:, [2]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
        t1[:, [3]] = -F_00 * F_01 * lam

        t2 = np.zeros((F.shape[0], 4))
        t2[:, [0]] = -F_10 * F_11 * lam
        t2[:, [1]] = lam + mu - F_00 * F_11 * lam + F_01 * F_10 * lam * 2.0
        t2[:, [2]] = mu + (F_10 ** 2) * lam
        t2[:, [3]] = -F_00 * F_10 * lam

        t3 = np.zeros((F.shape[0], 4))
        t3[:, [0]] = -lam - mu + F_00 * F_11 * lam * 2.0 - F_01 * F_10 * lam
        t3[:, [1]] = -F_00 * F_01 * lam
        t3[:, [2]] = -F_00 * F_10 * lam
        t3[:, [3]] = mu + (F_00 ** 2) * lam

        H = np.hstack([t0, t1, t2, t3])[:, _4x4matrix_2D_ordering_].reshape(-1, 4, 4)
    elif dim == 3:
        la = lam
        F1_1 = F[:, 0, 0].reshape(-1, 1)
        F1_2 = F[:, 0, 1].reshape(-1, 1)
        F1_3 = F[:, 0, 2].reshape(-1, 1)
        F2_1 = F[:, 1, 0].reshape(-1, 1)
        F2_2 = F[:, 1, 1].reshape(-1, 1)
        F2_3 = F[:, 1, 2].reshape(-1, 1)
        F3_1 = F[:, 2, 0].reshape(-1, 1)
        F3_2 = F[:, 2, 1].reshape(-1, 1)
        F3_3 = F[:, 2, 2].reshape(-1, 1)

        det = (
            -F1_1 * F2_2 * F3_3 + F1_1 * F2_3 * F3_2 + F1_2 * F2_1 * F3_3
            - F1_2 * F2_3 * F3_1 - F1_3 * F2_1 * F3_2 + F1_3 * F2_2 * F3_1 + 1.0
        )

        t1 = np.zeros((F.shape[0], 9))
        t1[:, [0]] = mu + la * (F2_2 * F3_3 - F2_3 * F3_2) ** 2.0
        t1[:, [1]] = -la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [2]] = la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [3]] = -la * (F2_1 * F3_3 - F2_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [4]] = -F3_3 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F3_3 * la * det
        t1[:, [5]] = F2_3 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F2_3 * la * det
        t1[:, [6]] = la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t1[:, [7]] = F3_2 * mu - la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F3_2 * la * det
        t1[:, [8]] = -F2_2 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F2_2 * la * det

        t2 = np.zeros((F.shape[0], 9))
        t2[:, [0]] = -la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t2[:, [1]] = mu + la * (F1_2 * F3_3 - F1_3 * F3_2) ** 2.0
        t2[:, [2]] = -la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [3]] = F3_3 * mu + la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_3 - F2_3 * F3_1) + F3_3 * la * det
        t2[:, [4]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [5]] = -F1_3 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) - F1_3 * la * det
        t2[:, [6]] = -F3_2 * mu - la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_2 - F2_2 * F3_1) - F3_2 * la * det
        t2[:, [7]] = la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t2[:, [8]] = F1_2 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) + F1_2 * la * det

        t3 = np.zeros((F.shape[0], 9))
        t3[:, [0]] = la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_2 * F3_3 - F2_3 * F3_2)
        t3[:, [1]] = -la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_2 * F3_3 - F1_3 * F3_2)
        t3[:, [2]] = mu + la * (F1_2 * F2_3 - F1_3 * F2_2) ** 2.0
        t3[:, [3]] = -F2_3 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_3 - F2_3 * F3_1) - F2_3 * la * det
        t3[:, [4]] = F1_3 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_3 - F1_3 * F3_1) + F1_3 * la * det
        t3[:, [5]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t3[:, [6]] = F2_2 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_2 - F2_2 * F3_1) + F2_2 * la * det
        t3[:, [7]] = -F1_2 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_2 - F1_2 * F3_1) - F1_2 * la * det
        t3[:, [8]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)

        t4 = np.zeros((F.shape[0], 9))
        t4[:, [0]] = -la * (F2_1 * F3_3 - F2_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t4[:, [1]] = F3_3 * mu + la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_3 - F2_3 * F3_1) + F3_3 * la * det
        t4[:, [2]] = -F2_3 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_3 - F2_3 * F3_1) - F2_3 * la * det
        t4[:, [3]] = mu + la * (F2_1 * F3_3 - F2_3 * F3_1) ** 2.0
        t4[:, [4]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [5]] = la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [6]] = -la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t4[:, [7]] = -F3_1 * mu + la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1) - F3_1 * la * det
        t4[:, [8]] = F2_1 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1) + F2_1 * la * det

        t5 = np.zeros((F.shape[0], 9))
        t5[:, [0]] = -F3_3 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F3_3 * la * det
        t5[:, [1]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t5[:, [2]] = F1_3 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_3 - F1_3 * F3_1) + F1_3 * la * det
        t5[:, [3]] = -la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t5[:, [4]] = mu + la * (F1_1 * F3_3 - F1_3 * F3_1) ** 2.0
        t5[:, [5]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t5[:, [6]] = F3_1 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1) + F3_1 * la * det
        t5[:, [7]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t5[:, [8]] = -F1_1 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1) - F1_1 * la * det

        t6 = np.zeros((F.shape[0], 9))
        t6[:, [0]] = F2_3 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F2_3 * la * det
        t6[:, [1]] = -F1_3 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) - F1_3 * la * det
        t6[:, [2]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t6[:, [3]] = la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t6[:, [4]] = -la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t6[:, [5]] = mu + la * (F1_1 * F2_3 - F1_3 * F2_1) ** 2.0
        t6[:, [6]] = -F2_1 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1) - F2_1 * la * det
        t6[:, [7]] = F1_1 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1) + F1_1 * la * det
        t6[:, [8]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F2_3 - F1_3 * F2_1)

        t7 = np.zeros((F.shape[0], 9))
        t7[:, [0]] = la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2)
        t7[:, [1]] = -F3_2 * mu - la * (F1_2 * F3_3 - F1_3 * F3_2) * (F2_1 * F3_2 - F2_2 * F3_1) - F3_2 * la * det
        t7[:, [2]] = F2_2 * mu + la * (F1_2 * F2_3 - F1_3 * F2_2) * (F2_1 * F3_2 - F2_2 * F3_1) + F2_2 * la * det
        t7[:, [3]] = -la * (F2_1 * F3_2 - F2_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1)
        t7[:, [4]] = F3_1 * mu + la * (F1_1 * F3_3 - F1_3 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1) + F3_1 * la * det
        t7[:, [5]] = -F2_1 * mu - la * (F1_1 * F2_3 - F1_3 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1) - F2_1 * la * det
        t7[:, [6]] = mu + la * (F2_1 * F3_2 - F2_2 * F3_1) ** 2.0
        t7[:, [7]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t7[:, [8]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1)

        t8 = np.zeros((F.shape[0], 9))
        t8[:, [0]] = F3_2 * mu - la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_2 * F3_3 - F2_3 * F3_2) + F3_2 * la * det
        t8[:, [1]] = la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_2 * F3_3 - F1_3 * F3_2)
        t8[:, [2]] = -F1_2 * mu - la * (F1_2 * F2_3 - F1_3 * F2_2) * (F1_1 * F3_2 - F1_2 * F3_1) - F1_2 * la * det
        t8[:, [3]] = -F3_1 * mu + la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_3 - F2_3 * F3_1) - F3_1 * la * det
        t8[:, [4]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F1_1 * F3_3 - F1_3 * F3_1)
        t8[:, [5]] = F1_1 * mu + la * (F1_1 * F2_3 - F1_3 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1) + F1_1 * la * det
        t8[:, [6]] = -la * (F1_1 * F3_2 - F1_2 * F3_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t8[:, [7]] = mu + la * (F1_1 * F3_2 - F1_2 * F3_1) ** 2.0
        t8[:, [8]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1)

        t9 = np.zeros((F.shape[0], 9))
        t9[:, [0]] = -F2_2 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_2 * F3_3 - F2_3 * F3_2) - F2_2 * la * det
        t9[:, [1]] = F1_2 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F3_3 - F1_3 * F3_2) + F1_2 * la * det
        t9[:, [2]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_2 * F2_3 - F1_3 * F2_2)
        t9[:, [3]] = F2_1 * mu - la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_3 - F2_3 * F3_1) + F2_1 * la * det
        t9[:, [4]] = -F1_1 * mu + la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_3 - F1_3 * F3_1) - F1_1 * la * det
        t9[:, [5]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F2_3 - F1_3 * F2_1)
        t9[:, [6]] = la * (F1_1 * F2_2 - F1_2 * F2_1) * (F2_1 * F3_2 - F2_2 * F3_1)
        t9[:, [7]] = -la * (F1_1 * F2_2 - F1_2 * F2_1) * (F1_1 * F3_2 - F1_2 * F3_1)
        t9[:, [8]] = mu + la * (F1_1 * F2_2 - F1_2 * F2_1) ** 2.0

        H = np.hstack([t1, t2, t3, t4, t5, t6, t7, t8, t9])[:, _9x9matrix_3D_ordering_].reshape(-1, 9, 9)
    else:
        raise ValueError("Macklin-Mueller Neo-Hookean supports dim=2 or dim=3")
    return H


# --------------------------------------------------------------------------- #
# Element tier: symmetric stretch (S) representation                          #
# --------------------------------------------------------------------------- #
# The Macklin-Mueller density psi(F) = mu*(1 - det F) + (lam/2)*(1 - det F)**2
# + (mu/2)*(||F||_F**2 - dim) is isotropic, so for a symmetric stretch S the
# energy equals psi(F) evaluated at F = S (the rotation factors out). The
# mixed/MFEM solver works in the d(d+1)/2 independent components ``a`` of S; the
# embedding ``S_flat = C0 @ a`` (duplicating the off-diagonals) lets us reuse
# the trusted F-tier derivatives by change of variables:
#     g_a = C0.T @ g_F ,      H_a = C0.T @ H_F @ C0
# which automatically applies the correct off-diagonal doubling. ``C0`` is the
# per-element symmetric-stretch embedding, taken from ``symmetric_stretch_map``
# so the component ordering matches the constraint used by the MFEM solver.
def _stretch_embedding_map(dim: int) -> np.ndarray:
    """Dense ``(dim*dim, dim*(dim+1)//2)`` compact-to-full stretch embedding."""
    Se, _ = symmetric_stretch_map(1, dim)
    return np.asarray(Se.todense())


def _stretch_compact_to_full(S: np.ndarray) -> np.ndarray:
    """Reshape compact ``(t, k)`` or full ``(t, dim, dim)`` stretch to full matrices."""
    if S.ndim == 3:
        return S
    t, k = S.shape
    dim = 2 if k == 3 else 3 if k == 6 else None
    if dim is None:
        raise ValueError("Compact stretch must have 3 (2D) or 6 (3D) components, got " + str(k))
    C0 = _stretch_embedding_map(dim)
    return (S @ C0.T).reshape(t, dim, dim)


def macklin_mueller_neo_hookean_energy_element_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Macklin-Mueller Neo-Hookean energy density in the stretch ``S``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch, full matrices or compact components
        (``k = 3`` in 2D, ``6`` in 3D), with the ordering of
        :func:`simkit.symmetric_stretch_map`.
    mu, lam : np.ndarray (t, 1)
        Per-element shear modulus and first Lame parameter.

    Returns
    -------
    psi : np.ndarray (t, 1)
        Per-element energy densities. No quadrature weighting applied.
    """
    Sf = _stretch_compact_to_full(np.asarray(S))
    return macklin_mueller_neo_hookean_energy_element_F(Sf, mu, lam)


def macklin_mueller_neo_hookean_gradient_element_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element gradient of the density w.r.t. the stretch ``S``.

    For full-matrix input the gradient matches the ``F`` tier exactly; for
    compact input it is mapped to the independent components via ``C0.T``.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch (full or compact form).
    mu, lam : np.ndarray (t, 1)
        Per-element shear modulus and first Lame parameter.

    Returns
    -------
    g : np.ndarray (t, dim, dim) or (t, k)
        Per-element gradient, matching the input representation. No quadrature
        weighting applied.
    """
    S = np.asarray(S)
    if S.ndim == 3:
        return macklin_mueller_neo_hookean_gradient_element_F(S, mu, lam)
    t, k = S.shape
    dim = 2 if k == 3 else 3
    Sf = _stretch_compact_to_full(S)
    Pf = macklin_mueller_neo_hookean_gradient_element_F(Sf, mu, lam).reshape(t, dim * dim)
    C0 = _stretch_embedding_map(dim)
    return Pf @ C0


def macklin_mueller_neo_hookean_hessian_element_S(S: np.ndarray, mu: np.ndarray, lam: np.ndarray, psd: bool = True) -> np.ndarray:
    """Per-element Hessian of the density w.r.t. the stretch ``S``.

    Unlike the ``F`` tier (projected later in the global assembly), the ``S``
    Hessian is consumed directly by the mixed solver, so it is PSD-projected
    here by default.

    Parameters
    ----------
    S : np.ndarray (t, dim, dim) or (t, k)
        Per-element symmetric stretch (full or compact form).
    mu, lam : np.ndarray (t, 1)
        Per-element shear modulus and first Lame parameter.
    psd : bool, optional
        If ``True`` (default), project each per-element block to PSD.

    Returns
    -------
    H : np.ndarray (t, b, b)
        Per-element Hessian blocks, where ``b = dim*dim`` for full input and
        ``b = k`` for compact input. No quadrature weighting applied.
    """
    S = np.asarray(S)
    if S.ndim == 3:
        Hf = macklin_mueller_neo_hookean_hessian_element_F(S, mu, lam)
        return psd_project(Hf) if psd else Hf
    t, k = S.shape
    dim = 2 if k == 3 else 3
    Sf = _stretch_compact_to_full(S)
    Hf = macklin_mueller_neo_hookean_hessian_element_F(Sf, mu, lam)
    C0 = _stretch_embedding_map(dim)
    H = np.einsum('ji,tjk,kl->til', C0, Hf, C0)
    return psd_project(H) if psd else H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def macklin_mueller_neo_hookean_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled Macklin-Mueller Neo-Hookean energy at positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total Macklin-Mueller Neo-Hookean energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = macklin_mueller_neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def macklin_mueller_neo_hookean_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled Macklin-Mueller Neo-Hookean gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    P = macklin_mueller_neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def macklin_mueller_neo_hookean_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled Macklin-Mueller Neo-Hookean Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Current vertex positions.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        If ``True`` (default), project each per-element block to the nearest
        positive semi-definite matrix before assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    He = macklin_mueller_neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def macklin_mueller_neo_hookean_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled Macklin-Mueller Neo-Hookean energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`macklin_mueller_neo_hookean_energy_x` evaluated at
    ``x_bar + u`` but avoids recomputing ``J @ x_bar`` on every call. The
    reference ``x_bar`` is arbitrary (not required to be the rest pose).

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)`` — the flattened deformation
        gradient at the reference configuration.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    E : float
        Total Macklin-Mueller Neo-Hookean energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = macklin_mueller_neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def macklin_mueller_neo_hookean_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled Macklin-Mueller Neo-Hookean gradient w.r.t. displacement ``u``.

    Because ``dF/du = J`` (the same as ``dF/dX``), this returns the same
    assembled vector as :func:`macklin_mueller_neo_hookean_gradient_x` at
    ``x_bar + u``.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    P = macklin_mueller_neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def macklin_mueller_neo_hookean_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled Macklin-Mueller Neo-Hookean Hessian w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)``.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    vol : np.ndarray (t, 1)
        Per-element quadrature weights.
    psd : bool, optional
        If ``True`` (default), project each per-element block to the nearest
        positive semi-definite matrix before assembly.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    He = macklin_mueller_neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def macklin_mueller_neo_hookean_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """Macklin-Mueller Neo-Hookean energy, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions. Used to build ``J`` and ``vol``.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    E : float
        Total Macklin-Mueller Neo-Hookean energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return macklin_mueller_neo_hookean_energy_x(U, J, mu, lam, vol)


def macklin_mueller_neo_hookean_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Macklin-Mueller Neo-Hookean gradient, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.

    Returns
    -------
    g : np.ndarray (n*dim, 1)
        Assembled energy gradient.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return macklin_mueller_neo_hookean_gradient_x(U, J, mu, lam, vol)


def macklin_mueller_neo_hookean_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Macklin-Mueller Neo-Hookean Hessian, building the operator and weights from rest geometry.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, dim+1)
        Element connectivity.
    mu : np.ndarray (t, 1)
        Per-element shear modulus.
    lam : np.ndarray (t, 1)
        Per-element first Lame parameter.
    U : np.ndarray (n, dim), optional
        Current vertex positions. Defaults to ``X``.
    psd : bool, optional
        Project per-element blocks to PSD before assembly. Default ``True``.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (n*dim, n*dim)
        Assembled energy Hessian.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return macklin_mueller_neo_hookean_hessian_x(U, J, mu, lam, vol, psd=psd)
