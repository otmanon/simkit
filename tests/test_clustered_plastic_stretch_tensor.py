"""Tests for ``simkit.clustered_plastic_stretch_tensor``."""

from __future__ import annotations

import numpy as np

from simkit.clustered_plastic_stretch_tensor import clustered_plastic_stretch_tensor


def _setup():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    labels = np.array([0])
    n = X.size
    B = np.eye(n)
    D = np.eye(n)
    return X, T, labels, B, D


def test_clustered_plastic_stretch_tensor_output_shape() -> None:
    X, T, labels, B, D = _setup()
    tensor = clustered_plastic_stretch_tensor(X, T, labels, B, D)
    z = np.zeros(B.shape[1])
    a = np.zeros(D.shape[1])
    fyt = tensor(z, a)
    assert fyt.shape == (1, 2, 2)


def test_clustered_plastic_stretch_tensor_is_bilinear() -> None:
    X, T, labels, B, D = _setup()
    tensor = clustered_plastic_stretch_tensor(X, T, labels, B, D)
    z = np.array([1.0, 0.5, -0.25, 0.1, 0.0, 0.0])
    a = np.array([0.2, -0.3, 0.4, 0.0, 0.1, 0.0])
    f1 = tensor(z, a)
    f2 = tensor(2.0 * z, 2.0 * a)
    assert np.allclose(f2, 4.0 * f1, rtol=1e-10)
