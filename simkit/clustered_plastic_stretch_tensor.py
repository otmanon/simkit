"""Precomputed bilinear tensor exposing rotations in clustered actuation."""

from typing import Optional

import numpy as np

from .deformation_jacobian import deformation_jacobian
from .vectorized_transpose import vectorized_transpose


class clustered_plastic_stretch_tensor:
    r"""Precomputed tensor for the clustered plastic-stretch fit in Actuators
    A La Mode.

    The local-global solver minimizes, over actuation amplitudes ``a`` and a
    per-cluster rotation ``R``,

        argmin_{a, R}  sum_{i=1}^c || F(z) - R Y(a) ||^2

    where the configuration is ``x = B z`` and the actuation is ``y = D a``,
    both in low-dimensional subspaces. Solving for ``R`` with ``z`` fixed is a
    generalized Procrustes problem requiring a polar decomposition of
    ``F Y^T``. Since ``F Y^T`` is bilinear in ``z`` and ``a``, the bilinear map
    can be precomputed once per cluster; this class is that precomputed tensor.
    """

    def __init__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        l: np.ndarray,
        B: np.ndarray,
        D: np.ndarray,
        w: Optional[np.ndarray] = None,
    ) -> None:
        """Precompute the per-cluster bilinear tensor.

        Parameters
        ----------
        X : np.ndarray (n, d)
            Vertex positions.
        T : np.ndarray (t, s)
            Simplex indices.
        l : np.ndarray (t,)
            Per-element cluster label; rotations are evaluated per cluster.
        B : np.ndarray (n*d, m)
            Configuration-subspace basis.
        D : np.ndarray (n*d, m)
            Actuation-subspace basis.
        w : np.ndarray (t, 1), optional
            Per-element weights. Defaults to uniform weights.
        """
        dim = X.shape[1]
        if w is None:
            w = np.ones((T.shape[0], 1))

        # Lift each subspace into per-element deformation-gradient blocks.
        J = deformation_jacobian(X, T)
        t = T.shape[0]
        JB = (J @ B).reshape(t, dim, dim, -1)
        JD = (J @ D).reshape(t, dim, dim, -1)

        # Accumulate the weighted bilinear product per cluster. Indices:
        #   a, i = gradient rows; c = config mode; k = actuation mode.
        cBD = np.zeros(
            (l.max() + 1,) + (JB.shape[1], JB.shape[3]) + (JD.shape[1], JD.shape[3])
        )
        for i, c in enumerate(np.unique(l)):
            cI = np.where(l == c)[0]                  # elements in this cluster
            JBc = JB[cI, :, :, :]
            JDc = JD[cI, :, :, :]
            wJB = w[cI].reshape(-1, 1, 1, 1) * JBc    # weight the config blocks
            cBD[i] = np.einsum("pabc,pibk->acik", wJB, JDc)

        self.cBD = cBD

    def __call__(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Evaluate ``F Y^T`` per cluster for a configuration and actuation.

        Parameters
        ----------
        z : np.ndarray (m,) or (m, 1)
            Configuration coordinates in the ``B`` subspace.
        a : np.ndarray (m,) or (m, 1)
            Actuation amplitudes in the ``D`` subspace.

        Returns
        -------
        FYT : np.ndarray (c, dim, dim)
            Per-cluster ``F Y^T`` matrices, ready for polar decomposition.
        """
        a = a.reshape(-1)
        z = z.reshape(-1)
        # Contract out actuation (k) then configuration (c) to leave per-cluster
        # dim x dim matrices.
        BDa = np.einsum("...acik,k->...aci", self.cBD, a)
        FYT = np.einsum("...aci,c->...ai", BDa, z)
        return FYT