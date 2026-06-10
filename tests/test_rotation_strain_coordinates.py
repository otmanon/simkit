"""Tests for ``simkit.rotation_strain_coordinates``."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from simkit.rotation_strain_coordinates import RSPrecompute, rotation_strain_coordinates


def _unit_triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]], dtype=int)
    return X, T


def _unit_tet_mesh() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]], dtype=int)
    return X, T


def _affine_displacement(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Per-vertex displacement of the affine field ``u(x) = G x`` (grad u = G)."""
    return X @ G.T


def _reconstructed_F(
    X: np.ndarray, T: np.ndarray, G: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run RS coordinates on the affine field ``grad u = G`` for one element.

    Returns the reconstructed deformation gradient ``F_rs = I + grad(u_rs)`` for
    the (single) element together with the fitted displacement ``u_rs``.
    """
    dim = X.shape[1]
    u = _affine_displacement(X, G)
    u_rs, pre = rotation_strain_coordinates(
        X, T, u, pinned=np.array([0], dtype=int)
    )
    grad = (pre.J @ u_rs.reshape(-1, 1)).reshape(-1, dim, dim)
    F_rs = np.identity(dim)[None, ...] + grad
    return F_rs[0], u_rs


def _expected_F(G: np.ndarray) -> np.ndarray:
    """Reference RS target ``R @ S`` with ``R = exp(skew(G))`` and ``S = I + sym(G)``."""
    dim = G.shape[0]
    sym = (G + G.T) / 2.0
    skew = (G - G.T) / 2.0
    R = scipy.linalg.expm(skew)  # independent matrix-exponential oracle
    return R @ (np.identity(dim) + sym)


def test_rs_precompute_attributes_have_expected_shapes() -> None:
    X, T = _unit_triangle_mesh()
    n, dim = X.shape
    nt = T.shape[0]

    pre = RSPrecompute(X, T, pinned=np.array([0], dtype=int))

    assert pre.X.shape == (n, dim)
    assert pre.T.shape == (nt, 3)
    assert pre.J.shape[0] == nt * dim * dim
    assert pre.K.shape[0] == n * dim


def test_rotation_strain_coordinates_returns_displacement_shape() -> None:
    X, T = _unit_triangle_mesh()
    n, dim = X.shape
    u = np.zeros((n, dim))

    u_rs, pre = rotation_strain_coordinates(X, T, u, pinned=np.array([0], dtype=int))

    assert u_rs.shape == (n, dim)
    assert isinstance(pre, RSPrecompute)


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def test_strain_is_preserved_2d() -> None:
    # Pure stretch/shear (symmetric grad, no rotation): R = I, so F_rs = I + S - I.
    # The old "Y = R - I" behavior would have thrown this away (u_rs = 0).
    X, T = _unit_triangle_mesh()
    G = np.array([[0.2, 0.1], [0.1, -0.15]])

    F_rs, u_rs = _reconstructed_F(X, T, G)

    assert np.linalg.norm(u_rs) > 1e-3  # strain is NOT discarded
    np.testing.assert_allclose(F_rs, np.eye(2) + G, atol=1e-6)


def test_rotation_is_proper_2d() -> None:
    # Pure rotation generator (skew grad): F_rs must be a proper rotation.
    X, T = _unit_triangle_mesh()
    a = 0.7
    G = np.array([[0.0, -a], [a, 0.0]])

    F_rs, _ = _reconstructed_F(X, T, G)

    np.testing.assert_allclose(F_rs.T @ F_rs, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(F_rs), 1.0, atol=1e-6)
    np.testing.assert_allclose(F_rs, scipy.linalg.expm(G), atol=1e-6)


def test_reconstructs_rotation_times_strain_2d() -> None:
    # General gradient: F_rs == exp(skew) @ (I + sym).
    X, T = _unit_triangle_mesh()
    G = np.array([[0.15, -0.5], [0.3, 0.1]])

    F_rs, _ = _reconstructed_F(X, T, G)

    np.testing.assert_allclose(F_rs, _expected_F(G), atol=1e-6)


# ---------------------------------------------------------------------------
# 3D (axis-angle + Rodrigues exponential)
# ---------------------------------------------------------------------------

def test_strain_is_preserved_3d() -> None:
    X, T = _unit_tet_mesh()
    G = np.array([[0.2, 0.05, 0.0], [0.05, -0.1, 0.03], [0.0, 0.03, 0.15]])

    F_rs, u_rs = _reconstructed_F(X, T, G)

    assert np.linalg.norm(u_rs) > 1e-3
    np.testing.assert_allclose(F_rs, np.eye(3) + G, atol=1e-6)


def test_rotation_is_proper_3d() -> None:
    # Skew generator from axis-angle w = (0.3, -0.6, 0.4); F_rs must equal exp(skew).
    X, T = _unit_tet_mesh()
    a, b, c = 0.3, -0.6, 0.4
    G = np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])

    F_rs, _ = _reconstructed_F(X, T, G)

    np.testing.assert_allclose(F_rs.T @ F_rs, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(F_rs), 1.0, atol=1e-6)
    np.testing.assert_allclose(F_rs, scipy.linalg.expm(G), atol=1e-6)


def test_reconstructs_rotation_times_strain_3d() -> None:
    X, T = _unit_tet_mesh()
    G = np.array([[0.1, -0.4, 0.2], [0.5, -0.05, -0.3], [0.1, 0.25, 0.2]])

    F_rs, _ = _reconstructed_F(X, T, G)

    np.testing.assert_allclose(F_rs, _expected_F(G), atol=1e-6)
