"""Tests for ``simkit.hinge_hessian``."""

from __future__ import annotations

import numpy as np

from simkit.hinge_hessian import hinge_hessian, hinge_hessian_compact


def _right_angle_hinge() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    H = np.array([[0, 1, 2]], dtype=int)
    return X, H


def test_hinge_hessian_compact_blocks_are_symmetric() -> None:
    X, H = _right_angle_hinge()
    blocks = hinge_hessian_compact(X, H)
    assert blocks.shape == (H.shape[0], 6, 6)
    for i in range(H.shape[0]):
        assert np.allclose(blocks[i], blocks[i].T, atol=1e-10)


def test_hinge_hessian_assembled_is_symmetric() -> None:
    X, H = _right_angle_hinge()
    blocks = hinge_hessian_compact(X, H)
    Hglobal = hinge_hessian(X, H, blocks).toarray()
    assert Hglobal.shape == (2 * X.shape[0], 2 * X.shape[0])
    assert np.allclose(Hglobal, Hglobal.T, atol=1e-10)
