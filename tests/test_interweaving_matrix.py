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
    # numpy's matmul can emit spurious FP warnings on some array layouts even
    # though the result is correct.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        MMt = M @ M.T
    assert np.allclose(MMt, np.eye(t * d))
    assert np.allclose(M.sum(axis=0), 1.0)
    assert np.allclose(M.sum(axis=1), 1.0)
