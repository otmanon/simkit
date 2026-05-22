"""Fast clustered evaluation of sandwich products ``A R B``.

Precomputes per-cluster tensors so that ``A R B`` can be evaluated quickly when
``R`` is a block-diagonal rotation matrix parameterized by few cluster-wise
parameters that change frequently, while ``A`` and ``B`` remain constant.
"""

import os
from typing import Optional

import numpy as np
import scipy as sp


class fast_sandwich_transform_clustered:
    """Precomputed clustered sandwich transform for block-diagonal rotations.

    Columns of ``A`` and rows of ``B`` are ordered in row-major ``xx yy zz``
    fashion for each element's ``dim x dim`` block:

    ``f11, f12, f13, f21, f22, f23, f31, f32, f33`` (for ``dim=3``), then the
    next element, and so on.

    Parameters
    ----------
    A : scipy.sparse matrix (m, dim^2 * n)
        Left factor with flattened per-element blocks.
    B : scipy.sparse matrix (dim^2 * n, m)
        Right factor with the same block ordering.
    l : np.ndarray (n,)
        Per-element cluster label.
    read_cache : bool, optional
        If ``True`` and ``cache_dir`` is set, attempt to load precomputed
        tensors from disk. Default ``False``.
    cache_dir : str, optional
        Directory for caching precomputed ``ARBs`` arrays.
    dim : int, optional
        Spatial dimension (size of each square block). Default 3.
    """

    def __init__(
        s,
        A: sp.sparse.spmatrix,
        B: sp.sparse.spmatrix,
        l: np.ndarray,
        read_cache: bool = False,
        cache_dir: Optional[str] = None,
        dim: int = 3,
    ) -> None:
        s.dim = dim
        n = A.shape[1] // (dim * dim)

        # this is not necessarily true for each tet

        A = A
        B = B

        num_clusters = l.max() + 1
        s.num_clusters = num_clusters

        Re = np.zeros((dim, dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                Re[i, j, i, j] = 1

        def clustered_ARBs(c: int) -> np.ndarray:
            dim = s.dim
            ARBs = np.zeros((A.shape[0], B.shape[1], dim, dim))
            ti = np.where(l == c)[0]  # tet indices that belong to cluster c
            num_t = ti.shape[0]
            v = np.ones(ti.shape[0])
            tie = (
                np.tile(ti[:, None], (1, dim * dim)) * dim * dim
                + np.arange(dim * dim)[None, :]
            ).flatten()

            Ati = A[:, tie]
            Bti = B[tie, :]

            off = np.arange(num_t)
            for c in range(dim * dim):
                i = c // dim
                j = c % dim
                II = dim * dim * off[:, None] + dim * i + np.arange(dim)
                JJ = dim * dim * off[:, None] + dim * j + np.arange(dim)

                VV = np.tile(v[:, None], (1, dim))
                S = sp.sparse.csc_matrix(
                    (VV.flatten(), (II.flatten(), JJ.flatten())),
                    shape=(dim * dim * num_t, dim * dim * num_t),
                )
                ARBs[:, :, i, j] = Ati @ (S @ Bti)
            return ARBs

        if cache_dir is not None and read_cache:
            try:
                s.ARBs = np.load(cache_dir + "/ARBs.npy")
                print("Loaded ARBs from cache " + cache_dir)
            except:
                vfunc = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
                o = vfunc(np.arange(num_clusters))
                s.ARBs = np.stack(o).transpose(1, 2, 0, 3, 4)
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_dir + "/ARBs.npy", s.ARBs)
        else:
            vfunc = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
            o = vfunc(np.arange(num_clusters))
            s.ARBs = np.stack(o).transpose(1, 2, 0, 3, 4)

            # vfunc1 = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
            # o1 = vfunc1(np.arange(num_clusters))
            # s.ARBs1 = np.stack(o1).transpose(1, 2, 0, 3, 4)
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_dir + "/ARBs.npy", s.ARBs)

    def __call__(s, r: np.ndarray) -> np.ndarray:
        """Evaluate the sandwich product for cluster rotations ``r``.

        Parameters
        ----------
        r : np.ndarray (num_clusters, dim, dim) or flattened equivalent
            Per-cluster rotation matrices.

        Returns
        -------
        ARB : np.ndarray
            Sandwich product ``A R B``.
        """
        ARB = s.eval(r)
        return ARB

    def eval(s, r: np.ndarray) -> np.ndarray:
        """Evaluate ``A R B`` from per-cluster rotation matrices.

        Parameters
        ----------
        r : np.ndarray (num_clusters, dim, dim)
            One ``dim x dim`` rotation per cluster.

        Returns
        -------
        ARB : np.ndarray (m, m)
            Weighted sum of precomputed cluster tensors contracted with ``r``.
        """
        r = r.reshape((-1, s.dim, s.dim))

        assert (
            r.shape[0] == s.num_clusters
            and "FST set up for "
            + str(s.num_clusters)
            + " rotation clusters, but only got "
            + str(r.shape[0])
        )

        prod = s.ARBs * r
        ARB = np.sum(prod, axis=(-3, -2, -1))
        return ARB
