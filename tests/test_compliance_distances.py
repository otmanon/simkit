"""Tests for ``simkit.compliance_distances``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.compliance_distances import compliance_distances
from simkit.compliance_graph import compliance_graph


def _nearest(X, p):
    return int(np.argmin(np.linalg.norm(X - np.asarray(p), axis=1)))


def test_single_source_returns_2d_and_zero_at_source(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    D = compliance_distances(G, 5)
    assert D.shape == (1, X.shape[0])
    assert D[0, 5] == 0.0
    assert np.all(np.isfinite(D))


def test_multiple_sources_shape_and_symmetry(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    src = np.array([0, 40, 100])
    D = compliance_distances(G, src)
    assert D.shape == (3, X.shape[0])
    np.testing.assert_allclose(D[np.arange(3), src], 0.0)
    # metric symmetry: d(a, b) == d(b, a)
    np.testing.assert_allclose(D[0, 40], D[1, 0])
    np.testing.assert_allclose(D[1, 100], D[2, 40])


def test_stiff_material_is_cheap_to_cross(stiff_bar_strip) -> None:
    X, T, ym = stiff_bar_strip
    G = compliance_graph(X, T, ym)
    src = _nearest(X, [0.0, 0.0])
    D = compliance_distances(G, src)[0]
    before = _nearest(X, [1.0, 0.0])
    after = _nearest(X, [2.0, 0.0])
    # soft material: compliance distance along the bottom row equals arclength
    assert D[before] == pytest.approx(X[before, 0], rel=1e-6)
    # crossing the stiff bar (length ~1) costs ~1e-4 instead of ~1
    assert D[after] - D[before] < 1e-2
