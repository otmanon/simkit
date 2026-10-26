"""Tests for ``simkit.force_dual_modes``, against the special cases the paper proves.

* Mass-weighted white force noise on every DOF is linear modal analysis: the
  modes span the lowest generalized eigenvectors of ``H phi = gamma M phi`` and
  the displacement variances are ``1 / gamma^2``.
* Variance on one DOF gives that DOF's Green's function ``H^{-1} e_i``.
* Without ``M``, the modes are the top eigenvectors of ``H^{-1} Sigma_F H^{-1}``.
* Modes are M-orthonormal and deterministic; impossible requests are refused.

The solid is the heterogeneous stiff-bar strip of ``conftest.py``, pinned along
``x = 0``, with its linear-elasticity Hessian: material-aware, so a mode that
ignored ``H`` would not pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from simkit.deformation_jacobian import deformation_jacobian
from simkit.energies.linear_elasticity import linear_elasticity_hessian_x
from simkit.force_dual_modes import force_dual_modes
from simkit.massmatrix import massmatrix
from simkit.volume import volume
from simkit.ympr_to_lame import ympr_to_lame


@pytest.fixture
def pinned_strip(small_stiff_bar_strip):
    """``(H, M)`` of the strip restricted to the DOFs off ``x = 0``."""
    X, T, ym = small_stiff_bar_strip
    mu, lam = ympr_to_lame(ym, 0.3)
    H = linear_elasticity_hessian_x(X, deformation_jacobian(X, T), mu.reshape(-1, 1), lam.reshape(-1, 1),
                                    volume(X, T), psd=False)
    mass = np.repeat(massmatrix(X, T).diagonal(), 2)
    free = np.flatnonzero(np.repeat(X[:, 0] > 0.0, 2))
    H = sp.sparse.csc_matrix(H)[free][:, free]
    M = sp.sparse.diags(mass[free])
    return H, M


def test_mass_variance_is_linear_modal_analysis(pinned_strip) -> None:
    H, M = pinned_strip
    k = 6
    D, B = force_dual_modes(H, M.diagonal(), k, M=M)
    gamma, phi = sp.linalg.eigh(H.toarray(), M.toarray())
    assert np.allclose(D, 1.0 / gamma[:k] ** 2, rtol=1e-8)
    residual = phi[:, :k] - B @ (B.T @ (M @ phi[:, :k]))
    assert np.sqrt(np.einsum("ij,ij->j", residual, M @ residual)).max() < 1e-8
    assert np.allclose(B.T @ (M @ B), np.eye(k), atol=1e-10)


def test_single_dof_variance_is_its_greens_function(pinned_strip) -> None:
    H, M = pinned_strip
    variance = np.zeros(H.shape[0])
    variance[11] = 2.0
    D, B = force_dual_modes(H, variance, 1, M=M)
    green = sp.sparse.linalg.spsolve(H, np.eye(H.shape[0])[:, 11])
    norm = np.sqrt(green @ (M @ green))
    assert np.allclose(np.abs(B[:, 0]), np.abs(green) / norm, atol=1e-12)
    assert np.isclose(D[0], 2.0 * norm**2)


def test_identity_metric_is_euclidean_pca(pinned_strip) -> None:
    H, _ = pinned_strip
    variance = np.where(np.arange(H.shape[0]) % 3 == 0, 1.0, 0.0)
    D, B = force_dual_modes(H, variance, 4)
    Hi = np.linalg.inv(H.toarray())
    eigenvalues, eigenvectors = np.linalg.eigh(Hi @ np.diag(variance) @ Hi)
    assert np.allclose(D, eigenvalues[::-1][:4], rtol=1e-8)
    assert np.allclose(np.abs(B.T @ eigenvectors[:, ::-1][:, :4]), np.eye(4), atol=1e-8)


def test_deterministic(pinned_strip) -> None:
    H, M = pinned_strip
    variance = np.where(np.arange(H.shape[0]) < 20, 1.0, 0.0)
    first = force_dual_modes(H, variance, 5, M=M)
    second = force_dual_modes(H, variance, 5, M=M)
    assert np.array_equal(first[1], second[1]) and np.array_equal(first[0], second[0])


def test_impossible_requests_are_refused(pinned_strip) -> None:
    H, M = pinned_strip
    d = H.shape[0]
    support = np.where(np.arange(d) < 20, 1.0, 0.0)
    with pytest.raises(ValueError):
        force_dual_modes(H, support, 21, M=M)  # more modes than the rank
    with pytest.raises(ValueError):
        force_dual_modes(H, -support, 2, M=M)  # not a covariance
    with pytest.raises(ValueError):
        force_dual_modes(H, support, 2, M=M + sp.sparse.eye(d, k=1))  # not lumped
    with pytest.raises(ValueError):
        force_dual_modes(H, support[:-1], 2, M=M)  # sizes disagree
