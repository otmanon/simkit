"""Unit face normals and their derivatives from triangle corner positions."""

import numpy as np

from .area_normals import area_normal_element, area_normal_gradient_element


def normals(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Unit face normal for each triangle in ``F``.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangle indices.

    Returns
    -------
    n : np.ndarray (m, 3)
        Unit normal per triangle.
    """
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return normal_element(x0, x1, x2)


def normals_gradient(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Gradient of unit face normals w.r.t. triangle corner coordinates.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangle indices.

    Returns
    -------
    dn_dx : np.ndarray (m, 3, 9)
        Derivative of each unit normal w.r.t. the stacked corners
        ``[x0, x1, x2]``.
    """
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return normal_gradient_element(x0, x1, x2)


def normal_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Unit normal ``n = a / |a|`` from area normal ``a``.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each triangle.

    Returns
    -------
    n : np.ndarray (m, 3)
        Unit normal per triangle.
    """
    area_normal = area_normal_element(x0, x1, x2)
    double_area = np.linalg.norm(area_normal, axis=1).reshape(-1, 1)
    normal = area_normal / double_area
    return normal


def normal_gradient_element(
    x0: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """Gradient of the unit normal w.r.t. the nine corner coordinates.

    Applies the chain rule through ``n = a / |a|`` using the area-normal
    gradient from :func:`area_normal_gradient_element`.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each triangle.

    Returns
    -------
    dn_dx : np.ndarray (m, 3, 9)
        Derivative of the unit normal w.r.t. the stacked corners
        ``[x0, x1, x2]``.
    """
    area_normal = area_normal_element(x0, x1, x2)
    double_area = np.linalg.norm(area_normal, axis=1).reshape(-1, 1)

    dan_dx = area_normal_gradient_element(x0, x1, x2)

    I = np.identity(3)[None, :, :]
    term_1 = I / double_area[:, :, None]
    term_2 = area_normal[:, :, None] @ area_normal[:, None, :] / double_area[:, :, None] ** 3
    dn_dan = term_1 - term_2

    dn_dx = dn_dan @ dan_dx

    return dn_dx
