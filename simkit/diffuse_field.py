"""Heat-diffusion smoothing of values seeded at a set of mesh vertices."""

import numpy as np
import scipy as sp

from .edge_lengths import edge_lengths
from .dirichlet_laplacian import dirichlet_laplacian
from .massmatrix import massmatrix


def diffuse_field(
    Vv: np.ndarray,
    Tv: np.ndarray,
    bI: np.ndarray,
    phi: np.ndarray,
    dt: float = None,
    normalize: bool = True,
) -> np.ndarray:
    """Diffuse seeded values across a mesh by one implicit heat step.

    Solves one backward-Euler heat step ``(L*dt + M) W = M phi`` with the seed
    vertices ``bI`` clamped to ``phi``, producing a smooth field over the whole
    mesh. Useful for spreading handle weights or labels into surrounding tissue.

    Parameters
    ----------
    Vv : np.ndarray (n, 3)
        Mesh vertex positions.
    Tv : np.ndarray (t, 4)
        Tetrahedron indices.
    bI : np.ndarray (b,)
        Indices of the seed vertices.
    phi : np.ndarray (b, k)
        Values to diffuse, one row per seed vertex, ``k`` channels.
    dt : float, optional
        Diffusion time. Defaults to the squared mean edge length, which scales
        the smoothing radius with the mesh resolution.
    normalize : bool, optional
        If True, rescale each output channel to ``[0, 1]``. Default True.

    Returns
    -------
    W : np.ndarray (n, k)
        Diffused field over all vertices; rows at ``bI`` equal ``phi``.
    """
    if dt is None:
        dt = np.mean(edge_lengths(Vv, Tv)) ** 2

    L = dirichlet_laplacian(Vv, Tv)
    M = massmatrix(Vv, Tv)

    # Implicit (backward-Euler) heat operator for a single step of size dt.
    Q = L * dt + M

    # Split into free interior vertices (ii) and clamped seed vertices (bI).
    ii = np.setdiff1d(np.arange(Q.shape[0]), bI)
    Qii = Q[ii, :][:, ii]
    Qib = Q[ii, :][:, bI]

    # Solve for the interior with the seeds moved to the right-hand side.
    Wii = sp.sparse.linalg.spsolve(Qii, -Qib @ phi)
    Wii = Wii.reshape(-1, phi.shape[1])

    # Reassemble: solved interior values plus clamped seed values.
    W = np.zeros((L.shape[0], Wii.shape[1]))
    W[ii, :] = Wii
    W[bI, :] = phi

    if W.ndim == 1:
        W = W[:, None]
    if normalize:
        # Per-channel min-max rescale to [0, 1].
        W = (W - np.min(W, axis=0)) / (np.max(W, axis=0) - np.min(W, axis=0))
    return W