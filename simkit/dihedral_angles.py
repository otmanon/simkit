"""Dihedral angles, unified across dimension.

Discrete bending lives on a **codimension-1 hinge** shared by two top-dimensional
flaps. The construction is the same one dimension apart, so it gets one name:

================  ==========================  ============================
                  2D (curve)                  3D (surface)
================  ==========================  ============================
positions ``X``   ``(n, 2)``                  ``(n, 3)``
flaps             edges                       triangles
shared hinge      a **vertex** (codim 1)      an **edge** (codim 1)
connectivity      vertex triples ``(A,B,C)``  vertex quads ``(x0,x1,x2,x3)``
implementation    :mod:`dihedral_angles_2d`   :mod:`dihedral_angles_3d`
================  ==========================  ============================

These front ends **infer the dimension from the data** (``X.shape[1]``) and
dispatch to the 2D or 3D implementation. Pass ``(n, 2)`` positions with triple
connectivity, or ``(n, 3)`` positions with quad connectivity.

Element derivatives are returned in the compact per-hinge layout (columns ordered
as the stacked hinge vertices ``[v0, v1, ...]``), matching
``kron(simkit.wedge_map(C, n), eye(dim))`` used to assemble the global operators.
"""

import numpy as np

from .dihedral_angles_2d import (
    dihedral_angles_2d,
    dihedral_angles_2d_gradient_element,
    dihedral_angles_2d_hessian_element,
)
from .dihedral_angles_3d import (
    dihedral_angles_3d,
    dihedral_angles_3d_gradient_element,
    dihedral_angles_3d_hessian_element,
)


def _dim(X: np.ndarray) -> int:
    dim = np.asarray(X).shape[1]
    if dim not in (2, 3):
        raise ValueError(f"dihedral_angles: unsupported ambient dimension {dim} (expected 2 or 3)")
    return dim


def dihedral_angles(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Signed dihedral angle per hinge, dispatched by ambient dimension.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions; ``dim`` (2 or 3) inferred from ``X.shape[1]``.
    C : np.ndarray (E, k)
        Hinge connectivity: vertex triples ``(A, B, C)`` in 2D (``k == 3``) or
        vertex quadruples ``(x0, x1, x2, x3)`` in 3D (``k == 4``).

    Returns
    -------
    theta : np.ndarray (E, 1)
        Signed hinge (2D) or dihedral (3D) angle per element.
    """
    return dihedral_angles_2d(X, C) if _dim(X) == 2 else dihedral_angles_3d(X, C)


def dihedral_angles_gradient_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge compact angle gradient ``(E, dim*k)``, dispatched by dimension.

    ``(E, 6)`` in 2D and ``(E, 12)`` in 3D, ordered as the stacked hinge vertices
    (matching ``kron(wedge_map(C, n), eye(dim))``).
    """
    return dihedral_angles_2d_gradient_element(X, C) if _dim(X) == 2 else dihedral_angles_3d_gradient_element(X, C)


def dihedral_angles_hessian_element(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-hinge compact angle Hessian ``(E, dim*k, dim*k)``, dispatched by dimension."""
    return dihedral_angles_2d_hessian_element(X, C) if _dim(X) == 2 else dihedral_angles_3d_hessian_element(X, C)
