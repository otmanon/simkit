"""All-pairs Euclidean distances between two point sets."""

from typing import Tuple, Union

import numpy as np

from .pairwise_displacement import pairwise_displacement


def pairwise_distance(
    X: np.ndarray,
    Y: np.ndarray,
    return_displacement: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Pairwise Euclidean distance ``R[i, j] = ||X[i, :] - Y[j, :]||``.

    Parameters
    ----------
    X : np.ndarray (n, d)
        First point set.
    Y : np.ndarray (m, d)
        Second point set.
    return_displacement : bool, optional
        If True, also return the displacement tensor from
        :func:`pairwise_displacement`. Default False.

    Returns
    -------
    R : np.ndarray (n, m)
        Distance from each row of ``X`` to each row of ``Y``.
    D : np.ndarray (n, m, d), optional
        Displacement vectors (only if ``return_displacement`` is True).
    """
    D = pairwise_displacement(X, Y)

    R = np.linalg.norm(D, axis=2)
    if return_displacement:
        return R, D
    else:
        return R
