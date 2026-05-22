"""Tests for ``simkit.normalize_and_center``."""

from __future__ import annotations

import numpy as np

from simkit.normalize_and_center import normalize_and_center


def test_normalize_and_center_places_centroid_at_origin() -> None:
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 1.0, 0.0],
        ]
    )
    normalize_and_center(X)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-12)
