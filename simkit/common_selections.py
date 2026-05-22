"""Geometric vertex selections for pinning boundary conditions.

Each selection returns a boolean mask over vertices plus the integer indices
where the mask is true. The threshold ``t`` is a fraction of the relevant
bounding-box extent, so ``t`` is dimensionless and scale-invariant.
"""

from typing import Callable, Tuple

import numpy as np


def create_selection(
    name: str, X: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to a named selection routine.

    Parameters
    ----------
    name : str
        Selection name. One of ``"center"`` or ``"back_z"``.
    X : np.ndarray (n, d)
        Vertex positions.
    t : float
        Threshold as a fraction of the bounding-box extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.

    Raises
    ------
    ValueError
        If ``name`` is not a known selection.
    """
    selections: dict[str, Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]] = {
        "center": center_indices,
        "back_z": back_z_indices,
    }
    if name not in selections:
        raise ValueError(f"Unknown pinning type: {name!r}")
    return selections[name](X, t)


def back_z_indices(X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select vertices within the lowest ``t`` fraction of the z-extent.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions; column 2 is the z-axis.
    t : float
        Threshold as a fraction of the z-extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.
    """
    diff = X[:, 2].max() - X[:, 2].min()                 # z-extent of the mesh
    pinned = X[:, 2] < X[:, 2].min() + diff * t          # near the low-z face
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI


def center_indices(X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select vertices within a radius ``t * extent`` of the centroid.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions.
    t : float
        Radius as a fraction of the largest bounding-box extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.
    """
    diff = np.max(X.max(axis=0) - X.min(axis=0))         # largest axis extent
    center = X.mean(axis=0)
    pinned = np.linalg.norm(X - center, axis=1) < diff * t
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI


def top_indices(X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select vertices within the highest ``t`` fraction of the y-extent.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions; column 1 is the y-axis.
    t : float
        Threshold as a fraction of the y-extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.
    """
    diff = X[:, 1].max() - X[:, 1].min()                 # y-extent of the mesh
    pinned = X[:, 1] > X[:, 1].max() - diff * t          # near the high-y face
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI


def bottom_indices(X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select vertices within the lowest ``t`` fraction of the y-extent.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions; column 1 is the y-axis.
    t : float
        Threshold as a fraction of the y-extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.
    """
    diff = X[:, 1].max() - X[:, 1].min()                 # y-extent of the mesh
    pinned = X[:, 1] < X[:, 1].min() + diff * t          # near the low-y face
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI


def center_top_indices(X: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select vertices near the x-center AND in the top ``t`` fraction of y.

    Parameters
    ----------
    X : np.ndarray (n, d)
        Vertex positions; column 0 is x, column 1 is y.
    t : float
        Threshold as a fraction of the largest bounding-box extent.

    Returns
    -------
    pinned : np.ndarray (n,) of bool
        Mask of selected vertices.
    pinnedI : np.ndarray (k,) of int
        Indices of selected vertices.
    """
    diff = np.max(X.max(axis=0) - X.min(axis=0))         # largest axis extent
    center = X.mean(axis=0)
    near_center_x = np.abs(X[:, 0] - center[0]) < diff * t
    in_top_y = X[:, 1] > (X[:, 1].max() - diff * t)
    pinned = np.logical_and(near_center_x, in_top_y)     # intersection of both
    pinnedI = np.where(pinned)[0]
    return pinned, pinnedI