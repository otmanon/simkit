"""Tests for ``simkit.symmetric_stretch_map``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.symmetric_stretch_map import symmetric_stretch_map


@pytest.mark.parametrize("dim", [2, 3])
def test_symmetric_stretch_map_roundtrip(dim: int) -> None:
    rng = np.random.default_rng(7)
    t = 8
    S = rng.standard_normal((t, dim, dim))
    S = S + S.transpose(0, 2, 1)

    Se, Sei = symmetric_stretch_map(t, dim)
    n_sym = dim * (dim + 1) // 2

    assert Se.shape == (t * dim * dim, t * n_sym)
    assert Sei.shape == (t * n_sym, t * dim * dim)

    vec_s = S.reshape(-1, 1)
    sym = Sei @ vec_s
    vec_roundtrip = Se @ sym

    assert np.allclose(vec_s, vec_roundtrip, atol=1e-12)
