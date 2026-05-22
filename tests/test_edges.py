"""Tests for ``simkit.edges``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.edges import edges


def test_triangle_edge_count() -> None:
    T = np.array([[0, 1, 2]])
    E = edges(T)
    assert E.shape == (3, 2)
    assert np.all(E[:, 0] < E[:, 1])


def test_tet_edge_count() -> None:
    T = np.array([[0, 1, 2, 3]])
    E = edges(T)
    assert E.shape == (6, 2)


def test_bad_simplex_size_raises() -> None:
    with pytest.raises(ValueError, match="3 \\(tri\\) or 4 \\(tet\\)"):
        edges(np.array([[0, 1, 2, 3, 4]]))
