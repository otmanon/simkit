"""Polar stretch (symmetric factor) from deformation gradients."""

import numpy as np
import scipy as sp

from .deformation_jacobian import deformation_jacobian
from .polar_svd import polar_svd
from .rotation_gradient import rotation_gradient_F


def stretch(F: np.ndarray) -> np.ndarray:
    """Symmetric stretch factor from the polar decomposition ``F = R S``.

    Parameters
    ----------
    F : np.ndarray (t, d, d)
        Batch of deformation gradients.

    Returns
    -------
    s : np.ndarray (t * d * d, 1)
        Stacked symmetric stretch matrices ``S`` flattened column-wise.
    """
    [R, S] = polar_svd(F)
    s = S.reshape(-1, 1)
    return s


