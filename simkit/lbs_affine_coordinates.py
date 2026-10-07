"""Reduced LBS coordinates that realise one global affine transform."""

from __future__ import annotations

import numpy as np


def lbs_affine_coordinates(n_nodes: int, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Reduced coordinates ``z`` that make every affine frame the SAME map.

    Because partition-of-unity skinning weights reproduce affine fields,
    setting every control frame to the identical transform ``x -> A x + t``
    makes the linear blend reproduce that transform *exactly* on the whole
    mesh: ``B z = A x + t``. This is the tool behind the linear-precision /
    patch test -- a subspace that reproduces every rigid and affine field is a
    necessary correctness property (rigid motions cost zero elastic energy,
    uniform stretch gives a uniform deformation gradient). ``A = I, t = 0``
    returns the rest-state ``z``.

    The DOF ordering matches :func:`simkit.lbs_jacobian` (dense or
    ``sparse=True``).

    Parameters
    ----------
    n_nodes : int
        Number of control frames ``k``.
    A : np.ndarray (dim, dim)
        Linear part of the affine transform.
    t : np.ndarray (dim,)
        Translation part.

    Returns
    -------
    z : np.ndarray (k * dim * (dim + 1), 1)
        Reduced coordinates realising ``x -> A x + t`` in the LBS subspace.
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    d = A.shape[0]
    p = d + 1
    M = np.hstack([A, t.reshape(d, 1)])  # (d, d+1): [A | t]
    # z[(node * p + poly) * d + a] = M[a, poly], identical for every node.
    return np.tile(M.T.reshape(-1), n_nodes).reshape(-1, 1)
