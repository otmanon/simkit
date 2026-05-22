"""Assemble the backward-Euler trajectory operator over many timesteps."""

from typing import Tuple

import numpy as np
import scipy as sp


def backward_euler_rollout_matrix(
    num_timesteps: int,
) -> Tuple["sp.sparse.csc_matrix", "sp.sparse.csc_matrix"]:
    """Rolled-out backward-Euler operator and its initial-condition coupling.

    Backward Euler on a second-order system gives, per step, the stencil
    ``x_{k-2} - 2 x_{k-1} + x_k`` (here scaled by the symmetric weights below).
    Stacking this across a whole trajectory yields a banded system::

        T @ [x_0, x_1, ..., x_{num_timesteps-1}] = B @ [x_{-2}, x_{-1}]

    where ``x_{-2}`` and ``x_{-1}`` are the two states preceding the trajectory.
    ``T`` is the interior operator; ``B`` injects the two initial states into
    the first rows where the stencil reaches before ``x_0``.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps in the trajectory.

    Returns
    -------
    T : scipy.sparse.csc_matrix (num_timesteps, num_timesteps)
        Banded trajectory operator.
    B : scipy.sparse.csc_matrix (num_timesteps, 2)
        Coupling of the two pre-trajectory states into the first rows.
    """
    # Per-step 3x3 stencil over the offsets (-2, -1, 0) in both row and column.
    timesteps = np.arange(num_timesteps)
    inds = np.repeat(np.repeat(timesteps[:, None], 3, axis=1)[:, None], 3, axis=1)

    offset_i = np.array([[-2, -2, -2], [-1, -1, -1], [0, 0, 0]])
    offset_j = offset_i.T
    i = inds + offset_i                  # absolute row indices per stencil cell
    j = inds + offset_j                  # absolute col indices per stencil cell

    # Symmetric second-difference weights for the backward-Euler stencil.
    vals = np.array([[1 / 2, -1, 1 / 2],
                     [-1, 2, -1],
                     [1 / 2, -1, 1 / 2]])
    vals = np.repeat(vals[None, :, :], num_timesteps, axis=0)

    i_f = i.flatten()
    j_f = j.flatten()
    v_f = vals.flatten()

    # Keep only entries landing inside the trajectory; the rest reach into the
    # pre-trajectory states and are accounted for separately by B.
    valid_inds = (i_f >= 0) & (i_f < num_timesteps) & (j_f >= 0) & (j_f < num_timesteps)
    i = i_f[valid_inds]
    j = j_f[valid_inds]
    vals = v_f[valid_inds]

    T = sp.sparse.csc_matrix((vals, (i, j)), shape=(num_timesteps, num_timesteps))
    # B holds the stencil weights that couple x_{-2}, x_{-1} into rows 0 and 1.
    B = sp.sparse.csc_matrix(([2, -0.5, -0.5], ([0, 0, 1], [1, 0, 1])), shape=(num_timesteps, 2))
    return T, B