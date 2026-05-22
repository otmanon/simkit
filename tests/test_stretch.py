"""Tests for ``simkit.stretch``."""

from __future__ import annotations

import numpy as np

from simkit.stretch import stretch


def test_stretch_of_identity_deformation_is_identity() -> None:
    dim = 3
    t = 4
    F = np.tile(np.eye(dim), (t, 1, 1))

    s = stretch(F)

    assert s.shape == (t * dim * dim, 1)
    expected = np.tile(np.eye(dim).reshape(-1, 1), (t, 1))
    assert np.allclose(s, expected, atol=1e-10)
