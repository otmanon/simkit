"""Tests for ``simkit.diffuse_field``."""

from __future__ import annotations

import numpy as np

from simkit.diffuse_field import diffuse_field


def _unit_tet():
    V = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    T = np.array([[0, 1, 2, 3]])
    return V, T


def test_diffuse_field_clamps_seed_vertices() -> None:
    V, T = _unit_tet()
    bI = np.array([0, 2])
    phi = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = diffuse_field(V, T, bI, phi, dt=0.1, normalize=False)
    assert W.shape == (V.shape[0], 2)
    assert np.allclose(W[bI], phi, atol=1e-10)


def test_diffuse_field_normalized_to_unit_interval() -> None:
    V, T = _unit_tet()
    bI = np.array([0])
    phi = np.array([[1.0]])
    W = diffuse_field(V, T, bI, phi, dt=0.05, normalize=True)
    assert np.allclose(W.min(axis=0), 0.0, atol=1e-10)
    assert np.allclose(W.max(axis=0), 1.0, atol=1e-10)


def test_diffuse_field_default_dt_is_positive() -> None:
    V, T = _unit_tet()
    bI = np.array([1])
    phi = np.array([[0.5, 0.25]])
    W = diffuse_field(V, T, bI, phi, normalize=False)
    assert np.all(np.isfinite(W))
