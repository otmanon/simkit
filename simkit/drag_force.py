"""Quadratic drag force on a surface mesh immersed in a fluid.

Models drag proportional to the squared relative speed along the relative
velocity direction (implementation pending).
"""

from __future__ import annotations

import numpy as np


def drag_force(
    V_mesh: np.ndarray,
    N: np.ndarray,
    V_fluid: np.ndarray,
    C: float | None = None,
) -> None:
    """Drag force on surface elements from relative fluid velocity.

    Intended model (not yet implemented):

    ``V = V_mesh - V_fluid``,
    ``F = -C * ||V||^2 * V / ||V||``.

    Parameters
    ----------
    V_mesh : np.ndarray (t, d)
        Velocities of surface elements (typically interpolated from vertices).
    N : np.ndarray (t, d)
        Outward normals of surface elements.
    V_fluid : np.ndarray (t, d)
        Fluid velocity at each element.
    C : float, optional
        Drag coefficient.

    Returns
    -------
    None
        Not implemented.
    """
    return
