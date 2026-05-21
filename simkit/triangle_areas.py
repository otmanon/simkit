"""Triangle area, gradient, and Hessian on a surface mesh.

Areas and their derivatives w.r.t. vertex positions for a triangle mesh that
may be embedded in 2D or 3D. The element tier holds the geometry formula; the
``*s`` global tier gathers per-face vertices and forwards to the element tier.

Dimension handling
    Inputs are normalized to 3D at the global entry points via ``_pad3``: a
    ``(n, 2)`` array is padded with a zero z-column so all element math runs in
    3D with ``cross``/``norm``. Areas are therefore unsigned, which is the only
    well-defined choice for a surface embedded in 3D.

Element tier (``*_element``)
    Per-face quantities from the three corner positions ``x0, x1, x2``. The
    only place the area formula lives.

Normal tier (``*_dnormal`` / ``*_d2normal``)
    Derivatives of the area w.r.t. the (unnormalized) area-normal vector,
    used as the chain-rule inner term when assembling the Hessian.

Global tier (``*s``)
    Takes vertex positions ``X`` and face connectivity ``F``, gathers the
    three corners per face, and calls the element tier.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Dimension normalization                                                     #
# --------------------------------------------------------------------------- #
def _pad3(X: np.ndarray) -> np.ndarray:
    """Pad 2D vertices to 3D; pass 3D through unchanged.

    Parameters
    ----------
    X : np.ndarray (n, 2) or (n, 3)
        Vertex positions, 2D or 3D.

    Returns
    -------
    X3 : np.ndarray (n, 3)
        Positions in 3D. A 2D input gains a zero z-column; a 3D input is
        returned as-is.
    """
    if X.shape[1] == 3:
        return X
    return np.concatenate([X, np.zeros((X.shape[0], 1), dtype=X.dtype)], axis=1)


# --------------------------------------------------------------------------- #
# Global tier: positions (X) and connectivity (F)                             #
# --------------------------------------------------------------------------- #
def triangle_areas(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face triangle areas.

    Parameters
    ----------
    X : np.ndarray (n, 2) or (n, 3)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangle vertex indices.

    Returns
    -------
    A : np.ndarray (m,)
        Per-face unsigned areas.
    """
    X = _pad3(X)
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_element(x0, x1, x2)


def triangle_areas_gradient(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face gradient of the area w.r.t. its three corner positions.

    Parameters
    ----------
    X : np.ndarray (n, 2) or (n, 3)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangle vertex indices.

    Returns
    -------
    dA_dx : np.ndarray (m, 9)
        Per-face area gradient, stacked as ``[dA/dx0, dA/dx1, dA/dx2]`` with
        three components each (z-components are zero for 2D input).
    """
    X = _pad3(X)
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_gradient_element(x0, x1, x2)


def triangle_areas_hessian(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face Hessian of the area w.r.t. its three corner positions.

    Parameters
    ----------
    X : np.ndarray (n, 2) or (n, 3)
        Vertex positions.
    F : np.ndarray (m, 3)
        Triangle vertex indices.

    Returns
    -------
    d2A_dx2 : np.ndarray (m, 9, 9)
        Per-face area Hessian blocks over the nine corner DOFs.
    """
    X = _pad3(X)
    x0 = X[F[:, 0], :]
    x1 = X[F[:, 1], :]
    x2 = X[F[:, 2], :]
    return triangle_area_hessian_element(x0, x1, x2)


# --------------------------------------------------------------------------- #
# Normal tier: derivatives w.r.t. the area-normal vector                      #
# --------------------------------------------------------------------------- #
def triangle_area_dnormal(n: np.ndarray) -> np.ndarray:
    """Area as a function of the (unnormalized) area-normal vector.

    Parameters
    ----------
    n : np.ndarray (m, 3)
        Per-face area-normal vectors (``cross`` of two edges).

    Returns
    -------
    area : np.ndarray (m, 1)
        Per-face areas, ``|n| / 2``.
    """
    area = np.linalg.norm(n, axis=1).reshape(-1, 1) / 2
    return area


def triangle_area_gradient_dnormal(an: np.ndarray) -> np.ndarray:
    """Gradient of the area w.r.t. the area-normal vector.

    Parameters
    ----------
    an : np.ndarray (m, 3)
        Per-face area-normal vectors.

    Returns
    -------
    darea_dnorm : np.ndarray (m, 3)
        Per-face derivative ``d(area)/d(an) = an / (2 |an|)``.
    """
    norm = np.linalg.norm(an, axis=1).reshape(-1, 1)
    darea_dnorm = an / (2 * norm)
    return darea_dnorm


def triangle_area_hessian_d2normal(an: np.ndarray) -> np.ndarray:
    """Hessian of the area w.r.t. the area-normal vector.

    Parameters
    ----------
    an : np.ndarray (m, 3)
        Per-face area-normal vectors.

    Returns
    -------
    combined_term : np.ndarray (m, 3, 3)
        Per-face second derivative ``0.5 * (I/|an| - an an^T / |an|^3)``.
    """
    an_norm = np.linalg.norm(an, axis=1).reshape(-1, 1)
    I = np.identity(3)[None, :, :]
    term_1 = I / (an_norm)
    term_2 = (an[:, :, None] @ an[:, None, :]) / (an_norm[:, :, None] ** 3)

    combined_term = 0.5 * (term_1 - term_2)
    return combined_term


# --------------------------------------------------------------------------- #
# Element tier: per-face corner positions (x0, x1, x2)                         #
# --------------------------------------------------------------------------- #
def triangle_area_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Per-face area from corner positions.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each face.

    Returns
    -------
    area : np.ndarray (m,)
        Per-face unsigned areas, ``|e1 x (-e2)| / 2``.
    """
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0

    n = np.cross(e1, -e2)
    area = np.linalg.norm(n, axis=1) / 2

    return area


def triangle_area_gradient_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Per-face area gradient from corner positions.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each face.

    Returns
    -------
    da_dx : np.ndarray (m, 9)
        Per-face gradient, stacked as ``[dA/dx0, dA/dx1, dA/dx2]``.
    """
    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x1 - x0

    n = np.cross(e1, -e2)
    n_hat = n / np.linalg.norm(n, axis=1).reshape(-1, 1)
    da_dx0 = -0.5 * np.cross(n_hat, e0)
    da_dx1 = -0.5 * np.cross(n_hat, e1)
    da_dx2 = -0.5 * np.cross(n_hat, e2)

    da_dx = np.concatenate([da_dx0, da_dx1, da_dx2], axis=1)
    return da_dx


def triangle_area_hessian_element(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Per-face area Hessian from corner positions.

    Assembles the chain rule through the area-normal vector ``an``: the
    Gauss-Newton-style term ``dan_dx^T (d2area/dan2) dan_dx`` plus the
    geometric term ``(darea/dan) . d2an_dx2``.

    Parameters
    ----------
    x0, x1, x2 : np.ndarray (m, 3)
        The three corner positions of each face.

    Returns
    -------
    d2a_dx2 : np.ndarray (m, 9, 9)
        Per-face area Hessian blocks over the nine corner DOFs.
    """
    an = area_normal_element(x0, x1, x2)
    an_norm = np.linalg.norm(an, axis=1).reshape(-1, 1)
    dan_dx = area_normal_gradient_element(x0, x1, x2)

    I = np.identity(3)[None, :, :]
    term_1 = I / (an_norm[:, :, None])
    term_2 = (an[:, :, None] @ an[:, None, :]) / (an_norm[:, :, None] ** 3)

    combined_term = 0.5 * (term_1 - term_2)
    big_term_1 = dan_dx.transpose(0, 2, 1) @ combined_term @ dan_dx

    da_dn = triangle_area_gradient_dnormal(an)[:, :, None, None]
    d2an_dx2 = area_normal_hessian_element(x0, x1, x2)
    big_term_2 = (da_dn * d2an_dx2).sum(axis=1)

    d2a_dx2 = big_term_1 + big_term_2
    return d2a_dx2