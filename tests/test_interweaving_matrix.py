"""Tests for ``simkit.interweaving_matrix``."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from simkit.interweaving_matrix import interweaving_matrix


def test_interweaving_matrix_permutes_vertex_to_component_order() -> None:
    t, d = 4, 3
    rng = np.random.default_rng(0)
    v = rng.standard_normal(t * d)

    M = interweaving_matrix(t, d)
    assert isinstance(M, sps.csc_matrix)
    assert M.shape == (t * d, t * d)

    expected = v.reshape(t, d, order="F").reshape(-1)
    assert np.allclose(M @ v, expected)


def test_interweaving_matrix_is_permutation() -> None:
    t, d = 5, 2
    M = interweaving_matrix(t, d).toarray()
    assert np.allclose(M @ M.T, np.eye(t * d))
    assert np.allclose(M.sum(axis=0), 1.0)
    assert np.allclose(M.sum(axis=1), 1.0)
