"""Positive semidefinite projection for batched symmetric matrices.

Eigen-decomposes each matrix and clamps or projects eigenvalues before
reassembly via ``U @ diag(s) @ U^T``.
"""

import numpy as np

from .svd_rv import svd_rv


def psd_project(H: np.ndarray, method: str = 'proj') -> np.ndarray:
    """Project symmetric matrices to PSD by eigenvalue modification.

    Parameters
    ----------
    H : np.ndarray (n, d, d) or (d, d)
        Batch of symmetric matrices (or a single matrix, promoted to batch).
    method : {'proj', 'abs'}, optional
        ``'proj'`` floors eigenvalues below ``1e-6`` to ``1e-6``;
        ``'abs'`` replaces eigenvalues by their absolute values.

    Returns
    -------
    H_proj : np.ndarray (n, d, d) or (d, d)
        PSD-projected matrices with the same batch shape as the input.
    """
    if H.ndim == 2:
        H = H[None, :, :]
    [s, U] = np.linalg.eigh(H)

    dim = H.shape[-1]

    if method == 'abs':
        s = np.abs(s)
    elif method == 'proj':
        s[s < 1e-6] = 1e-6

    Id = np.identity(dim)[None, ...]
    S = Id * s.reshape(-1, dim, 1)

    # numpy's matmul can emit spurious divide/overflow/invalid FP warnings on
    # some batched-array layouts even though the result is correct.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        HbI = U @ S @ U.transpose(0, 2, 1)

    return HbI
