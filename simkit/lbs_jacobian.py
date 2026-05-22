"""Jacobian of linear blend skinning (LBS) w.r.t. skinning weights.

For rest positions ``V`` and weights ``W``, the deformed position of vertex
``i`` is ``sum_k W[i,k] * T_k(V_i)`` with affine transforms ``T_k``. This
module builds the matrix ``d x / d W`` in stacked form.
"""

import numpy as np


def lbs_jacobian(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Stacked Jacobian ``d x / d W`` for LBS with homogeneous transforms.

    Parameters
    ----------
    V : np.ndarray (n, d)
        Rest vertex positions.
    W : np.ndarray (n, k)
        Per-vertex skinning weights over ``k`` bones.

    Returns
    -------
    J : np.ndarray (n*d, n*k*(d+1))
        Jacobian of stacked deformed coordinates w.r.t. all weight DOFs.
    """
    n = V.shape[0]
    d = V.shape[1]
    k = W.shape[1]

    one_d1 = np.ones((d + 1, 1))
    one_k = np.ones((k, 1))

    # append 1s to V to make V1 , homogeneous
    V1 = np.hstack((V, np.ones((V.shape[0], 1))))

    Wexp = np.kron(W, one_d1.T)
    V1exp = np.kron(one_k.T, V1)
    J = Wexp * V1exp
    Jexp = np.kron(J, np.identity(d))

    return Jexp
