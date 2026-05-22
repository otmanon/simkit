"""Tests for ``simkit.edge_lengths``."""

from __future__ import annotations

import numpy as np

from simkit.edge_lengths import edge_lengths


def test_unit_segment_length() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    E = np.array([[0, 1]])
    L = edge_lengths(X, E)
    assert L.shape == (1,)
    assert np.isclose(L[0], 1.0, rtol=1e-12)
