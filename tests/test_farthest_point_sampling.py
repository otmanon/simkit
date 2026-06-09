"""Tests for ``simkit.farthest_point_sampling``."""

from __future__ import annotations

import numpy as np

from simkit.farthest_point_sampling import farthest_point_sampling


def test_farthest_point_sampling_returns_k_indices_with_seed() -> None:
    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    k = 3
    seed = np.array([2])

    I = farthest_point_sampling(V, k, sI=seed)

    assert I.shape == (k,)
    assert I[0] == seed[0]
    assert len(np.unique(I)) == k
    assert np.all(I >= 0)
    assert np.all(I < V.shape[0])
