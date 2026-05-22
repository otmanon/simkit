"""Cluster-wise rotation factors from a reduced deformation subspace."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp

from .cluster_grouping_matrices import cluster_grouping_matrices
from .deformation_jacobian import deformation_jacobian
from .polar_svd import polar_svd
from .volume import volume


def subspace_rotation(
    z: np.ndarray,
    B: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    GAJB: Optional[sp.sparse.spmatrix] = None,
    return_GAJB: bool = False,
    labels: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, sp.sparse.spmatrix]]:
    """Rotation matrix for each cluster in the current subspace configuration.

    Applies a volume-weighted, cluster-averaged deformation Jacobian map to
    ``z``, then extracts the rotation factor of each cluster gradient via polar
    SVD.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced current coordinates.
    B : np.ndarray or scipy.sparse matrix (n * dim, r)
        Reduced displacement basis.
    X : np.ndarray (n, dim)
        Rest vertex positions.
    T : np.ndarray (t, s)
        Simplex connectivity.
    GAJB : scipy.sparse matrix, optional
        Precomputed grouped averaging Jacobian map in the subspace.
    return_GAJB : bool, optional
        If True, also return ``GAJB``. Default False.
    labels : np.ndarray (t,), optional
        Per-element cluster labels. A single cluster is assumed if None.

    Returns
    -------
    R : np.ndarray (k, dim, dim)
        Rotation matrix per cluster.
    GAJB : scipy.sparse matrix, optional
        Grouped averaging Jacobian map. Returned only if ``return_GAJB`` is
        True.
    """
    # Get the rotation matrix for the current state
    dim = X.shape[1]

    if GAJB is None:
        if labels is None:
            labels = np.zeros(T.shape[0]).astype(int)

        J = deformation_jacobian(X, T)
        A = sp.sparse.diags(volume(X, T).flatten())
        [G, Gm] = cluster_grouping_matrices(labels, X, T)
        GAe = sp.sparse.kron(G @ A, sp.sparse.identity(dim * dim))
        GAJB = GAe @ J @ B

    c = GAJB @ z
    C = c.reshape((dim, dim, -1)).transpose(2, 0, 1)  # covariances / deformation gradients
    R, Sf = polar_svd(C)  # this is the best fit rotation of the current state

    if return_GAJB:
        return R, GAJB
    else:
        return R
