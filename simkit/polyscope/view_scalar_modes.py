"""Polyscope viewer that cycles scalar fields on a mesh.

Registers point cloud, curve, surface, or volume geometry from simplex width
``T.shape[1]``, then animates each column of ``W`` as a scalar quantity with
optional screenshots.
"""

from __future__ import annotations

import os
import time

import numpy as np
import polyscope as ps


def view_scalar_modes(
    X: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    period: float = 1 / 10,
    cmap: str = "coolwarm",
    name: str = "modes",
    dir: str | None = None,
    normalize: bool = True,
    vminmax: np.ndarray | list | None = None,
    eye_pos: np.ndarray | None = None,
    eye_target: np.ndarray | None = None,
) -> None:
    """Display each column of ``W`` as a scalar mode on a Polyscope mesh.

    Parameters
    ----------
    X : np.ndarray (n, dim)
        Vertex positions.
    T : np.ndarray (m, dt)
        Simplex connectivity; ``dt`` selects geometry type (1 point cloud,
        2 curve, 3 surface, 4 volume).
    W : np.ndarray (n, dw)
        Scalar modes; one column shown per frame.
    period : float, optional
        Seconds to pause between mode updates.
    cmap : str, optional
        Polyscope colormap name.
    name : str, optional
        Base name for scalar quantities.
    dir : str or None, optional
        If set, directory for PNG screenshots (one per mode).
    normalize : bool, optional
        If ``True``, set symmetric ``vminmax`` from each column's max abs value.
    vminmax : array-like or None, optional
        Fixed color range ``[vmin, vmax]``; if a 2D array with one row per
        mode, per-column ranges can be selected (see loop body).
    eye_pos, eye_target : np.ndarray or None, optional
        Camera look-at position and target passed to :func:`polyscope.look_at`.

    Returns
    -------
    None
        Side effect only: opens/updates Polyscope and optionally writes images.
    """
    ps.init()

    ps.remove_all_structures()
    ps.set_give_focus_on_show(True)
    if eye_pos is not None and eye_target is not None:
        ps.look_at(eye_pos, eye_target)
    dt = T.shape[1]

    if dt == 1:
        geo = ps.register_point_cloud("geo", X)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T)
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T)
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T)

    ps.set_ground_plane_mode("none")

    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    dw = W.shape[1]

    if vminmax is not None:
        if isinstance(vminmax, np.ndarray):
            if vminmax.shape[0] > 1:
                vminmax_mat = vminmax.copy()
    for i in range(dw):
        Wi = W[:, i]
        # if vminmax is None:
        if normalize:
            wmax = np.abs(Wi).max()
            vminmax = [-wmax, wmax]
        #     else:
        #         vminmax= None
        # else:
        #     if isinstance(vminmax, np.ndarray):
        #         if vminmax.shape[0] > 1:
        #             vminmax = vminmax_mat[i, :]

        geo.add_scalar_quantity(
            name + " " + str(i), Wi, enabled=True, cmap=cmap, vminmax=vminmax,
        )

        time.sleep(period)

        ps.frame_tick()

        if dir is not None:
            ps.screenshot(dir + "./" + str(i).zfill(4) + ".png")

    # ps.show()
    return
