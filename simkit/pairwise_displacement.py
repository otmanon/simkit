"""All-pairs displacement vectors between two point sets."""

import numpy as np


def pairwise_displacement(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise displacement ``D[i, j, :] = X[i, :] - Y[j, :]``.

    Parameters
    ----------
    X : np.ndarray (n, d)
        First point set.
    Y : np.ndarray (m, d)
        Second point set.

    Returns
    -------
    D : np.ndarray (n, m, d)
        Displacement from each row of ``Y`` to each row of ``X``.
    """
    assert X.ndim == 2
    assert Y.ndim == 2
    D = X[:, None, :]
    D = np.repeat(D, Y.shape[0], axis=1)
    D = D - Y[None, :, :]

    return D
