"""Translate and scale vertex positions into a centered unit box."""

from typing import Tuple, Union

import numpy as np


def normalize_and_center(
    X: np.ndarray, return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float]]:
    """Center ``X`` at the origin and scale it to fit in ``[-1, 1]^dim``.

    Modifies ``X`` in place: subtracts the mean, then scales so the largest
    axis extent maps to 2 (half-width 1).

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions; updated in place.
    return_params : bool, optional
        If True, also return the translation and scale applied. Default False.

    Returns
    -------
    X : np.ndarray (n, dim)
        Normalized positions (same array as the input).
    t : np.ndarray (dim,), optional
        Translation subtracted from ``X`` (only if ``return_params`` is True).
    scale : float, optional
        Multiplicative scale applied after centering (only if
        ``return_params`` is True).
    """

    t = -X.mean(axis=0)
    X += t
    scale = 2 / max(X.max(axis=0) - X.min(axis=0))
    X *= scale

    if return_params:
        return X, t, scale
    else:
        return X
