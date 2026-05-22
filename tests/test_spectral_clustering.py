"""Tests for ``simkit.spectral_clustering``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytestmark = pytest.mark.learn

from simkit.spectral_clustering import spectral_clustering


def test_spectral_clustering_labels_match_basis_rows() -> None:
    rng = np.random.default_rng(0)
    n, p, k = 12, 5, 3
    W = rng.standard_normal((n, p))

    labels, centroids = spectral_clustering(W, k, seed=0)

    assert labels.shape == (n,)
    assert centroids.shape == (k, p)
    assert labels.min() >= 0
    assert labels.max() < k
