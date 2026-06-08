"""Winding number of query points against a closed curve (2D) or surface (3D).

The winding number measures how many times a closed, oriented boundary wraps
around a query point. For a watertight, consistently oriented boundary it is
(up to sign set by the orientation) ``1`` for points strictly inside and ``0``
for points strictly outside, and it varies smoothly in between -- the
*generalized* winding number of Jacobson et al. 2013, which stays meaningful
even for open or non-manifold boundaries.

In 2D the boundary is a set of oriented edges and each edge contributes the
signed angle it subtends at the query point. In 3D the boundary is a set of
oriented triangles and each triangle contributes its signed solid angle
(Van Oosterom & Strackee 1983). Both are summed and normalized by the measure
of the full sphere (``2*pi`` in 2D, ``4*pi`` in 3D).
"""

from __future__ import annotations

import numpy as np


def winding_number(Q: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Generalized winding number of query points against an oriented boundary.

    The dimension is inferred from ``V``: a 2-column ``V`` selects the planar
    (edge) formula and a 3-column ``V`` selects the spatial (triangle) formula.

    Parameters
    ----------
    Q : np.ndarray (q, d)
        Query points, ``d == 2`` or ``d == 3`` matching ``V``.
    V : np.ndarray (n, d)
        Boundary vertices.
    F : np.ndarray (m, 2) or (m, 3)
        Oriented boundary connectivity indexing into ``V``: edges
        (``[i, j]``, 2 columns) in 2D, triangles (``[i, j, k]``, 3 columns)
        in 3D.

    Returns
    -------
    w : np.ndarray (q,)
        Winding number at each query point. For a closed boundary wound
        counter-clockwise (2D) / with outward-facing normals (3D), interior
        points approach ``+1`` and exterior points approach ``0``; reversing
        the orientation flips the interior sign to ``-1``.

    Raises
    ------
    ValueError
        If ``V`` is not 2- or 3-dimensional, or if ``F``'s column count does
        not match the dimension (2 columns in 2D, 3 columns in 3D).

    Notes
    -----
    Fully vectorized over both query points and boundary elements; the
    intermediate arrays are ``(q, m, d)``, so memory scales with
    ``q * m``.

    Example
    -------
    ```python
    # Unit square wound counter-clockwise.
    V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    E = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    w = winding_number(np.array([[0.5, 0.5], [2.0, 2.0]]), V, E)
    # w approximately [1.0, 0.0]
    ```
    """
    Q = np.asarray(Q, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F)
    dim = V.shape[1]
    if dim == 2:
        return _winding_number_2d(Q, V, F)
    if dim == 3:
        return _winding_number_3d(Q, V, F)
    raise ValueError(f"winding_number expects 2D or 3D vertices, got d={dim}")


def _winding_number_2d(Q: np.ndarray, V: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Planar winding number via summed signed edge angles."""
    if E.shape[1] != 2:
        raise ValueError(f"2D winding_number expects edges with 2 columns, got {E.shape[1]}")

    # Edge endpoints, broadcast against every query point: (q, m, 2).
    a = V[E[:, 0]][None, :, :] - Q[:, None, :]
    b = V[E[:, 1]][None, :, :] - Q[:, None, :]

    # Signed angle ab subtends at each query point: atan2(cross, dot).
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    angle = np.arctan2(cross, dot)  # (q, m)
    return angle.sum(axis=1) / (2.0 * np.pi)


def _winding_number_3d(Q: np.ndarray, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Spatial winding number via summed signed triangle solid angles."""
    if F.shape[1] != 3:
        raise ValueError(f"3D winding_number expects triangles with 3 columns, got {F.shape[1]}")

    # Triangle corners as vectors from each query point: (q, m, 3).
    a = V[F[:, 0]][None, :, :] - Q[:, None, :]
    b = V[F[:, 1]][None, :, :] - Q[:, None, :]
    c = V[F[:, 2]][None, :, :] - Q[:, None, :]

    la = np.linalg.norm(a, axis=-1)
    lb = np.linalg.norm(b, axis=-1)
    lc = np.linalg.norm(c, axis=-1)

    # Van Oosterom & Strackee signed solid angle:
    #   tan(omega / 2) = [a . (b x c)] / [la lb lc + (a.b) lc + (b.c) la + (c.a) lb]
    triple = np.einsum("qmi,qmi->qm", a, np.cross(b, c))
    denom = (
        la * lb * lc
        + np.einsum("qmi,qmi->qm", a, b) * lc
        + np.einsum("qmi,qmi->qm", b, c) * la
        + np.einsum("qmi,qmi->qm", c, a) * lb
    )
    omega = 2.0 * np.arctan2(triple, denom)  # (q, m)
    return omega.sum(axis=1) / (4.0 * np.pi)
