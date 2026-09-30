"""Cubic penalty ``E = k/3 sum_j a_j max(0, -phi(p_j))^3`` of points ``p = (S x).reshape(-1, dim)``
against a rigid object with signed distance ``sdf(P) -> (phi, grad)``. The Hessian is the
PSD Gauss-Newton part (SDF curvature dropped)."""
import numpy as np
import scipy as sp


def _depth(x, S, sdf, dim):
    phi, n = sdf((S @ np.ravel(x)).reshape(-1, dim))
    return np.maximum(-phi, 0.0), n


def sdf_contact_energy_x(x, S, sdf, k, a, dim=3):
    d, _ = _depth(x, S, sdf, dim)
    return float(k / 3 * (a * d ** 3).sum())


def sdf_contact_gradient_x(x, S, sdf, k, a, dim=3):
    d, n = _depth(x, S, sdf, dim)
    return S.T @ ((-k * a * d ** 2)[:, None] * n).reshape(-1, 1)


def sdf_contact_hessian_x(x, S, sdf, k, a, dim=3):
    d, n = _depth(x, S, sdf, dim)
    act = np.nonzero(d > 0)[0]
    Sa = sp.sparse.csr_matrix(S)[(dim * act[:, None] + np.arange(dim)).ravel()]
    blocks = (2 * k * a[act] * d[act])[:, None, None] * n[act][:, :, None] * n[act][:, None, :]
    return (Sa.T @ sp.sparse.block_diag(list(blocks), format="csr") @ Sa).tocsr() if len(act) else \
        sp.sparse.csr_matrix((S.shape[1], S.shape[1]))
