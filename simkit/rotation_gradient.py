"""Gradient of the rotation factor from polar/SVD decomposition of ``F``.

Uses rotationally invariant SVD (:func:`.svd_rv.svd_rv`) and closed-form
``dR/dF`` for 2D and 3D deformation gradients.
"""

import numpy as np

from .svd_rv import svd_rv


def rotation_gradient_F(F: np.ndarray) -> np.ndarray:
    """Rotation gradient ``dR/dF`` from the SVD of batched deformation gradients.

    Parameters
    ----------
    F : np.ndarray (n, dim, dim)
        Deformation gradients; ``dim`` must be 2 or 3.

    Returns
    -------
    K : np.ndarray (n, dim*dim, dim*dim)
        Per-element Jacobian of the rotation factor w.r.t. flattened ``F``.
    """
    dim = F.shape[-1]
    if dim == 2:
        d = F.shape[1]
        n = F.shape[0]
        [U, S, V] = svd_rv(F)

        T0 = np.array([[0, -1], [1, 0]])
        T0 = (1 / np.sqrt(2)) * U @ T0 @ V.transpose(0, 2, 1)

        t0 = np.reshape(T0, (n, d * d, 1))
        s0 = np.reshape(S[:, 0, 0], (n, 1, 1))
        s1 = np.reshape(S[:, 1, 1], (n, 1, 1))
        # gotta clamp these
        s01 = np.maximum(s0 + s1, 1e-12)
        dR_dF = (2 / s01) * (t0 @ t0.transpose(0, 2, 1))
        K = dR_dF

    elif dim == 3:
        d = F.shape[1]
        n = F.shape[0]
        [U, S, V] = svd_rv(F)

        T0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        T0 = (1 / np.sqrt(2)) * U @ T0 @ V.transpose(0, 2, 1)

        T1 = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
        T1 = (1 / np.sqrt(2)) * U @ T1 @ V.transpose(0, 2, 1)

        T2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        T2 = (1 / np.sqrt(2)) * U @ T2 @ V.transpose(0, 2, 1)

        t0 = np.reshape(T0, (n, d * d, 1))
        t1 = np.reshape(T1, (n, d * d, 1))
        t2 = np.reshape(T2, (n, d * d, 1))

        s0 = np.reshape(S[:, 0, 0], (n, 1, 1))
        s1 = np.reshape(S[:, 1, 1], (n, 1, 1))
        s2 = np.reshape(S[:, 2, 2], (n, 1, 1))

        # # gotta clamp these
        s01 = np.maximum(s0 + s1, 1e-8)
        s12 = np.maximum(s1 + s2, 1e-8)
        s02 = np.maximum(s0 + s2, 1e-8)

        dR_dF = (2 / s01) * (t0 @ t0.transpose(0, 2, 1)) \
            + (2 / s12) * (t1 @ t1.transpose(0, 2, 1)) \
            + (2 / s02) * (t2 @ t2.transpose(0, 2, 1))
        K = dR_dF
    else:
        ValueError("Only dim == 2 or 3 are supported")
    return K
