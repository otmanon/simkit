"""Smith-Goes-Kim stable Neo-Hookean elastic energy.

Implements the final stable Neo-Hookean formulation from Smith, de Goes,
and Kim, "Stable Neo-Hookean Flesh Simulation" (TOG / SIGGRAPH 2018):
https://dl.acm.org/doi/10.1145/3180491

Per-element energy density:

    psi(F) = (mu / 2) * (I_C - dim)
           - (mu / 2) * log(I_C + 1)
           + (lam / 2) * (det(F) - alpha)**2

    alpha  = 1 + dim * mu / ((dim + 1) * lam)        (dim-dependent shift)
    I_C    = ||F||_F**2 = sum_{i,j} F_{ij}**2

The ``alpha`` shift makes the rest configuration (``F = I``) a critical
point of the energy: ``dpsi/dF |_{F=I} = 0``. Unlike Macklin-Mueller, the
rest energy ``psi(F=I)`` is a non-zero constant (this is a feature of the
formulation, not a bug). The log term penalises shrink toward ``F = 0``
without diverging there, so the energy remains finite even at collapsed
configurations -- the "stable" property of the paper.

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
Closed-form gradient and Hessian derived symbolically by
``scripts/derive_stable_neo_hookean.py``. The ``mat2py`` ordering helpers
reindex from the column-major (MATLAB/C++) flat layout to the row-major
``F`` layout used throughout the library.
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


# --------------------------------------------------------------------------- #
# Element tier: deformation gradient (F) representation                       #
# --------------------------------------------------------------------------- #
def stable_neo_hookean_energy_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Per-element Smith-Goes-Kim stable Neo-Hookean energy density.

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
        Note: ``psi(F=I)`` is a non-zero constant; the rest state is a
        stationary point (gradient zero) but not the zero of the energy.
    """
    dim = F.shape[-1]
    F = F.reshape(-1, dim, dim)
    mu = np.asarray(mu).reshape(-1, 1)
    lam = np.asarray(lam).reshape(-1, 1)
    alpha = 1.0 + dim * mu / ((dim + 1) * lam)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        I_C = F_00 ** 2 + F_01 ** 2 + F_10 ** 2 + F_11 ** 2
        J = F_00 * F_11 - F_01 * F_10
        psi = (
            (mu / 2.0) * (I_C - 2.0)
            - (mu / 2.0) * np.log(I_C + 1.0)
            + (lam / 2.0) * (J - alpha) ** 2
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
        I_C = (
            F_00 ** 2 + F_01 ** 2 + F_02 ** 2
            + F_10 ** 2 + F_11 ** 2 + F_12 ** 2
            + F_20 ** 2 + F_21 ** 2 + F_22 ** 2
        )
        J = (
            F_00 * F_11 * F_22 - F_00 * F_12 * F_21
            - F_01 * F_10 * F_22 + F_01 * F_12 * F_20
            + F_02 * F_10 * F_21 - F_02 * F_11 * F_20
        )
        psi = (
            (mu / 2.0) * (I_C - 3.0)
            - (mu / 2.0) * np.log(I_C + 1.0)
            + (lam / 2.0) * (J - alpha) ** 2
        )
    else:
        raise ValueError("Stable Neo-Hookean supports dim=2 or dim=3")
    return psi


def stable_neo_hookean_gradient_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
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
    alpha = 1.0 + dim * mu / ((dim + 1) * lam)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)
        I_C = F_00 ** 2 + F_01 ** 2 + F_10 ** 2 + F_11 ** 2
        J = F_00 * F_11 - F_01 * F_10
        # A = mu * I_C / (I_C + 1)     (per-element scalar)
        # D = lam * (J - alpha)        (per-element scalar)
        A = mu * I_C / (I_C + 1.0)
        D = lam * (J - alpha)
        # column-major flat: (00, 10, 01, 11);  cof_ij = dJ/dF_ij
        P = np.hstack([
            A * F_00 + D * F_11,    # d psi / d F_00,  cof_00 = F_11
            A * F_10 + D * (-F_01), # d psi / d F_10,  cof_10 = -F_01
            A * F_01 + D * (-F_10), # d psi / d F_01,  cof_01 = -F_10
            A * F_11 + D * F_00,    # d psi / d F_11,  cof_11 = F_00
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
        I_C = (
            F_00 ** 2 + F_01 ** 2 + F_02 ** 2
            + F_10 ** 2 + F_11 ** 2 + F_12 ** 2
            + F_20 ** 2 + F_21 ** 2 + F_22 ** 2
        )
        J = (
            F_00 * F_11 * F_22 - F_00 * F_12 * F_21
            - F_01 * F_10 * F_22 + F_01 * F_12 * F_20
            + F_02 * F_10 * F_21 - F_02 * F_11 * F_20
        )
        A = mu * I_C / (I_C + 1.0)
        D = lam * (J - alpha)
        # cofactors (row-major naming):  cof_ij = dJ/dF_ij
        c00 = F_11 * F_22 - F_12 * F_21
        c01 = F_12 * F_20 - F_10 * F_22
        c02 = F_10 * F_21 - F_11 * F_20
        c10 = F_02 * F_21 - F_01 * F_22
        c11 = F_00 * F_22 - F_02 * F_20
        c12 = F_01 * F_20 - F_00 * F_21
        c20 = F_01 * F_12 - F_02 * F_11
        c21 = F_02 * F_10 - F_00 * F_12
        c22 = F_00 * F_11 - F_01 * F_10
        # column-major flat: (00, 10, 20, 01, 11, 21, 02, 12, 22)
        P = np.hstack([
            A * F_00 + D * c00,
            A * F_10 + D * c10,
            A * F_20 + D * c20,
            A * F_01 + D * c01,
            A * F_11 + D * c11,
            A * F_21 + D * c21,
            A * F_02 + D * c02,
            A * F_12 + D * c12,
            A * F_22 + D * c22,
        ])[:, _9vector_3D_ordering_].reshape(-1, 3, 3)
    else:
        raise ValueError("Stable Neo-Hookean supports dim=2 or dim=3")
    return P


def stable_neo_hookean_hessian_element_F(F: np.ndarray, mu: np.ndarray, lam: np.ndarray) -> np.ndarray:
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
    alpha = 1.0 + dim * mu / ((dim + 1) * lam)

    if dim == 2:
        F_00 = F[:, 0, 0].reshape(-1, 1)
        F_01 = F[:, 0, 1].reshape(-1, 1)
        F_10 = F[:, 1, 0].reshape(-1, 1)
        F_11 = F[:, 1, 1].reshape(-1, 1)

        I_C = F_00 ** 2 + F_01 ** 2 + F_10 ** 2 + F_11 ** 2
        den = I_C + 1.0
        J = F_00 * F_11 - F_01 * F_10
        # per-element scalars
        A = mu * I_C / den              # coefficient of the identity in H
        B = 2.0 * mu / (den ** 2)       # coefficient of F (x) F
        D = lam * (J - alpha)           # coefficient of d cof / dF
        E = lam                         # coefficient of cof (x) cof
        # cofactors (row-major naming)
        c00 = F_11
        c01 = -F_10
        c10 = -F_01
        c11 = F_00

        # H[a,b] = A * delta(a,b) + B * F[a] * F[b] + E * cof[a] * cof[b]
        #       + D * dCof[a] / dF[b]
        # in column-major flatten order (00, 10, 01, 11).
        t1 = np.zeros((F.shape[0], 4))
        t1[:, [0]] = A + B * F_00 * F_00 + E * c00 * c00
        t1[:, [1]] = B * F_00 * F_10 + E * c00 * c10
        t1[:, [2]] = B * F_00 * F_01 + E * c00 * c01
        t1[:, [3]] = B * F_00 * F_11 + E * c00 * c11 + D    # dC00/dF11 = +1

        t2 = np.zeros((F.shape[0], 4))
        t2[:, [0]] = B * F_10 * F_00 + E * c10 * c00
        t2[:, [1]] = A + B * F_10 * F_10 + E * c10 * c10
        t2[:, [2]] = B * F_10 * F_01 + E * c10 * c01 - D    # dC10/dF01 = -1
        t2[:, [3]] = B * F_10 * F_11 + E * c10 * c11

        t3 = np.zeros((F.shape[0], 4))
        t3[:, [0]] = B * F_01 * F_00 + E * c01 * c00
        t3[:, [1]] = B * F_01 * F_10 + E * c01 * c10 - D    # dC01/dF10 = -1
        t3[:, [2]] = A + B * F_01 * F_01 + E * c01 * c01
        t3[:, [3]] = B * F_01 * F_11 + E * c01 * c11

        t4 = np.zeros((F.shape[0], 4))
        t4[:, [0]] = B * F_11 * F_00 + E * c11 * c00 + D    # dC11/dF00 = +1
        t4[:, [1]] = B * F_11 * F_10 + E * c11 * c10
        t4[:, [2]] = B * F_11 * F_01 + E * c11 * c01
        t4[:, [3]] = A + B * F_11 * F_11 + E * c11 * c11

        H = np.hstack([t1, t2, t3, t4])[:, _4x4matrix_2D_ordering_].reshape(-1, 4, 4)
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

        I_C = (
            F_00 ** 2 + F_01 ** 2 + F_02 ** 2
            + F_10 ** 2 + F_11 ** 2 + F_12 ** 2
            + F_20 ** 2 + F_21 ** 2 + F_22 ** 2
        )
        den = I_C + 1.0
        J = (
            F_00 * F_11 * F_22 - F_00 * F_12 * F_21
            - F_01 * F_10 * F_22 + F_01 * F_12 * F_20
            + F_02 * F_10 * F_21 - F_02 * F_11 * F_20
        )
        A = mu * I_C / den
        B = 2.0 * mu / (den ** 2)
        D = lam * (J - alpha)
        E = lam
        # cofactors (row-major naming)
        c00 = F_11 * F_22 - F_12 * F_21
        c01 = F_12 * F_20 - F_10 * F_22
        c02 = F_10 * F_21 - F_11 * F_20
        c10 = F_02 * F_21 - F_01 * F_22
        c11 = F_00 * F_22 - F_02 * F_20
        c12 = F_01 * F_20 - F_00 * F_21
        c20 = F_01 * F_12 - F_02 * F_11
        c21 = F_02 * F_10 - F_00 * F_12
        c22 = F_00 * F_11 - F_01 * F_10

        # column-major flat: (00, 10, 20, 01, 11, 21, 02, 12, 22)
        # H[a,b] = A*delta(a,b) + B*F[a]*F[b] + E*cof[a]*cof[b] + D*dCof[a,b]
        # row r=0: d/dF_00,  dCof = (0, 0, 0, 0, F_22, -F_12, 0, -F_21, F_11)
        t1 = np.zeros((F.shape[0], 9))
        t1[:, [0]] = A + B * F_00 * F_00 + E * c00 * c00
        t1[:, [1]] = B * F_00 * F_10 + E * c00 * c10
        t1[:, [2]] = B * F_00 * F_20 + E * c00 * c20
        t1[:, [3]] = B * F_00 * F_01 + E * c00 * c01
        t1[:, [4]] = B * F_00 * F_11 + E * c00 * c11 + D * F_22
        t1[:, [5]] = B * F_00 * F_21 + E * c00 * c21 - D * F_12
        t1[:, [6]] = B * F_00 * F_02 + E * c00 * c02
        t1[:, [7]] = B * F_00 * F_12 + E * c00 * c12 - D * F_21
        t1[:, [8]] = B * F_00 * F_22 + E * c00 * c22 + D * F_11

        # row r=1: d/dF_10,  dCof = (0, 0, 0, -F_22, 0, F_02, F_21, 0, -F_01)
        t2 = np.zeros((F.shape[0], 9))
        t2[:, [0]] = B * F_10 * F_00 + E * c10 * c00
        t2[:, [1]] = A + B * F_10 * F_10 + E * c10 * c10
        t2[:, [2]] = B * F_10 * F_20 + E * c10 * c20
        t2[:, [3]] = B * F_10 * F_01 + E * c10 * c01 - D * F_22
        t2[:, [4]] = B * F_10 * F_11 + E * c10 * c11
        t2[:, [5]] = B * F_10 * F_21 + E * c10 * c21 + D * F_02
        t2[:, [6]] = B * F_10 * F_02 + E * c10 * c02 + D * F_21
        t2[:, [7]] = B * F_10 * F_12 + E * c10 * c12
        t2[:, [8]] = B * F_10 * F_22 + E * c10 * c22 - D * F_01

        # row r=2: d/dF_20,  dCof = (0, 0, 0, F_12, -F_02, 0, -F_11, F_01, 0)
        t3 = np.zeros((F.shape[0], 9))
        t3[:, [0]] = B * F_20 * F_00 + E * c20 * c00
        t3[:, [1]] = B * F_20 * F_10 + E * c20 * c10
        t3[:, [2]] = A + B * F_20 * F_20 + E * c20 * c20
        t3[:, [3]] = B * F_20 * F_01 + E * c20 * c01 + D * F_12
        t3[:, [4]] = B * F_20 * F_11 + E * c20 * c11 - D * F_02
        t3[:, [5]] = B * F_20 * F_21 + E * c20 * c21
        t3[:, [6]] = B * F_20 * F_02 + E * c20 * c02 - D * F_11
        t3[:, [7]] = B * F_20 * F_12 + E * c20 * c12 + D * F_01
        t3[:, [8]] = B * F_20 * F_22 + E * c20 * c22

        # row r=3: d/dF_01,  dCof = (0, -F_22, F_12, 0, 0, 0, 0, F_20, -F_10)
        t4 = np.zeros((F.shape[0], 9))
        t4[:, [0]] = B * F_01 * F_00 + E * c01 * c00
        t4[:, [1]] = B * F_01 * F_10 + E * c01 * c10 - D * F_22
        t4[:, [2]] = B * F_01 * F_20 + E * c01 * c20 + D * F_12
        t4[:, [3]] = A + B * F_01 * F_01 + E * c01 * c01
        t4[:, [4]] = B * F_01 * F_11 + E * c01 * c11
        t4[:, [5]] = B * F_01 * F_21 + E * c01 * c21
        t4[:, [6]] = B * F_01 * F_02 + E * c01 * c02
        t4[:, [7]] = B * F_01 * F_12 + E * c01 * c12 + D * F_20
        t4[:, [8]] = B * F_01 * F_22 + E * c01 * c22 - D * F_10

        # row r=4: d/dF_11,  dCof = (F_22, 0, -F_02, 0, 0, 0, -F_20, 0, F_00)
        t5 = np.zeros((F.shape[0], 9))
        t5[:, [0]] = B * F_11 * F_00 + E * c11 * c00 + D * F_22
        t5[:, [1]] = B * F_11 * F_10 + E * c11 * c10
        t5[:, [2]] = B * F_11 * F_20 + E * c11 * c20 - D * F_02
        t5[:, [3]] = B * F_11 * F_01 + E * c11 * c01
        t5[:, [4]] = A + B * F_11 * F_11 + E * c11 * c11
        t5[:, [5]] = B * F_11 * F_21 + E * c11 * c21
        t5[:, [6]] = B * F_11 * F_02 + E * c11 * c02 - D * F_20
        t5[:, [7]] = B * F_11 * F_12 + E * c11 * c12
        t5[:, [8]] = B * F_11 * F_22 + E * c11 * c22 + D * F_00

        # row r=5: d/dF_21,  dCof = (-F_12, F_02, 0, 0, 0, 0, F_10, -F_00, 0)
        t6 = np.zeros((F.shape[0], 9))
        t6[:, [0]] = B * F_21 * F_00 + E * c21 * c00 - D * F_12
        t6[:, [1]] = B * F_21 * F_10 + E * c21 * c10 + D * F_02
        t6[:, [2]] = B * F_21 * F_20 + E * c21 * c20
        t6[:, [3]] = B * F_21 * F_01 + E * c21 * c01
        t6[:, [4]] = B * F_21 * F_11 + E * c21 * c11
        t6[:, [5]] = A + B * F_21 * F_21 + E * c21 * c21
        t6[:, [6]] = B * F_21 * F_02 + E * c21 * c02 + D * F_10
        t6[:, [7]] = B * F_21 * F_12 + E * c21 * c12 - D * F_00
        t6[:, [8]] = B * F_21 * F_22 + E * c21 * c22

        # row r=6: d/dF_02,  dCof = (0, F_21, -F_11, 0, -F_20, F_10, 0, 0, 0)
        t7 = np.zeros((F.shape[0], 9))
        t7[:, [0]] = B * F_02 * F_00 + E * c02 * c00
        t7[:, [1]] = B * F_02 * F_10 + E * c02 * c10 + D * F_21
        t7[:, [2]] = B * F_02 * F_20 + E * c02 * c20 - D * F_11
        t7[:, [3]] = B * F_02 * F_01 + E * c02 * c01
        t7[:, [4]] = B * F_02 * F_11 + E * c02 * c11 - D * F_20
        t7[:, [5]] = B * F_02 * F_21 + E * c02 * c21 + D * F_10
        t7[:, [6]] = A + B * F_02 * F_02 + E * c02 * c02
        t7[:, [7]] = B * F_02 * F_12 + E * c02 * c12
        t7[:, [8]] = B * F_02 * F_22 + E * c02 * c22

        # row r=7: d/dF_12,  dCof = (-F_21, 0, F_01, F_20, 0, -F_00, 0, 0, 0)
        t8 = np.zeros((F.shape[0], 9))
        t8[:, [0]] = B * F_12 * F_00 + E * c12 * c00 - D * F_21
        t8[:, [1]] = B * F_12 * F_10 + E * c12 * c10
        t8[:, [2]] = B * F_12 * F_20 + E * c12 * c20 + D * F_01
        t8[:, [3]] = B * F_12 * F_01 + E * c12 * c01 + D * F_20
        t8[:, [4]] = B * F_12 * F_11 + E * c12 * c11
        t8[:, [5]] = B * F_12 * F_21 + E * c12 * c21 - D * F_00
        t8[:, [6]] = B * F_12 * F_02 + E * c12 * c02
        t8[:, [7]] = A + B * F_12 * F_12 + E * c12 * c12
        t8[:, [8]] = B * F_12 * F_22 + E * c12 * c22

        # row r=8: d/dF_22,  dCof = (F_11, -F_01, 0, -F_10, F_00, 0, 0, 0, 0)
        t9 = np.zeros((F.shape[0], 9))
        t9[:, [0]] = B * F_22 * F_00 + E * c22 * c00 + D * F_11
        t9[:, [1]] = B * F_22 * F_10 + E * c22 * c10 - D * F_01
        t9[:, [2]] = B * F_22 * F_20 + E * c22 * c20
        t9[:, [3]] = B * F_22 * F_01 + E * c22 * c01 - D * F_10
        t9[:, [4]] = B * F_22 * F_11 + E * c22 * c11 + D * F_00
        t9[:, [5]] = B * F_22 * F_21 + E * c22 * c21
        t9[:, [6]] = B * F_22 * F_02 + E * c22 * c02
        t9[:, [7]] = B * F_22 * F_12 + E * c22 * c12
        t9[:, [8]] = A + B * F_22 * F_22 + E * c22 * c22

        H = np.hstack([t1, t2, t3, t4, t5, t6, t7, t8, t9])[:, _9x9matrix_3D_ordering_].reshape(-1, 9, 9)
    else:
        raise ValueError("Stable Neo-Hookean supports dim=2 or dim=3")
    return H


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def stable_neo_hookean_energy_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled stable Neo-Hookean energy at positions ``X``.

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
        Total stable Neo-Hookean energy.
    """
    dim = X.shape[1]
    F = (J @ X.reshape(-1, 1)).reshape(-1, dim, dim)
    psi = stable_neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def stable_neo_hookean_gradient_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled stable Neo-Hookean gradient w.r.t. positions ``X``.

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
    P = stable_neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def stable_neo_hookean_hessian_x(X: np.ndarray, J: sp.sparse.spmatrix, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled stable Neo-Hookean Hessian w.r.t. positions ``X``.

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
    He = stable_neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def stable_neo_hookean_energy_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> float:
    """Assembled stable Neo-Hookean energy at displacement ``u`` from a reference ``x_bar``.

    Equivalent to :func:`stable_neo_hookean_energy_x` evaluated at
    ``x_bar + u`` but avoids recomputing ``J @ x_bar`` on every call. The
    reference ``x_bar`` is arbitrary (not required to be the rest pose).

    Parameters
    ----------
    u : np.ndarray (n, dim)
        Displacement from the reference configuration.
    J : scipy.sparse matrix (t*dim*dim, n*dim)
        Deformation Jacobian.
    Jx_bar : np.ndarray (t*dim*dim, 1)
        Precomputed ``J @ x_bar.reshape(-1, 1)`` -- the flattened deformation
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
        Total stable Neo-Hookean energy.
    """
    dim = u.shape[1]
    F = (J @ u.reshape(-1, 1) + Jx_bar).reshape(-1, dim, dim)
    psi = stable_neo_hookean_energy_element_F(F, mu, lam)
    E = float((np.asarray(vol).reshape(-1, 1) * psi).sum())
    return E


def stable_neo_hookean_gradient_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Assembled stable Neo-Hookean gradient w.r.t. displacement ``u``.

    Because ``dF/du = J`` (the same as ``dF/dX``), this returns the same
    assembled vector as :func:`stable_neo_hookean_gradient_x` at
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
    P = stable_neo_hookean_gradient_element_F(F, mu, lam)
    P = P * np.asarray(vol).reshape(-1, 1, 1)
    g = J.transpose() @ P.reshape(-1, 1)
    return g


def stable_neo_hookean_hessian_u(u: np.ndarray, J: sp.sparse.spmatrix, Jx_bar: np.ndarray, mu: np.ndarray, lam: np.ndarray, vol: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled stable Neo-Hookean Hessian w.r.t. displacement ``u``.

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
    He = stable_neo_hookean_hessian_element_F(F, mu, lam)
    He = He * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    H = sp.sparse.block_diag(He)
    Q = J.transpose() @ H @ J
    return Q


# --------------------------------------------------------------------------- #
# Self-contained tier: builds J and vol from rest geometry                    #
# --------------------------------------------------------------------------- #
def stable_neo_hookean_energy(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> float:
    """Stable Neo-Hookean energy, building the operator and weights from rest geometry.

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
        Total stable Neo-Hookean energy.
    """
    if U is None:
        U = X
    J = deformation_jacobian(X, T)
    vol = volume(X, T)
    return stable_neo_hookean_energy_x(U, J, mu, lam, vol)


def stable_neo_hookean_gradient(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Stable Neo-Hookean gradient, building the operator and weights from rest geometry.

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
    return stable_neo_hookean_gradient_x(U, J, mu, lam, vol)


def stable_neo_hookean_hessian(X: np.ndarray, T: np.ndarray, mu: np.ndarray, lam: np.ndarray, U: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Stable Neo-Hookean Hessian, building the operator and weights from rest geometry.

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
    return stable_neo_hookean_hessian_x(U, J, mu, lam, vol, psd=psd)
