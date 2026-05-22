"""Tests for ``simkit.ympr_to_lame``."""

from __future__ import annotations

import numpy as np

from simkit.ympr_to_lame import ympr_to_lame


def test_ympr_to_lame_broadcasts_over_arrays() -> None:
    ym = np.array([1.0e5, 2.0e5, 3.0e5])
    pr = np.array([0.30, 0.35, 0.40])

    mu, lam = ympr_to_lame(ym, pr)

    expected_mu = ym / (2.0 * (1.0 + pr))
    expected_lam = ym * pr / ((1.0 + pr) * (1.0 - 2.0 * pr))

    assert mu.shape == ym.shape
    assert lam.shape == ym.shape
    assert np.allclose(mu, expected_mu, rtol=1e-12)
    assert np.allclose(lam, expected_lam, rtol=1e-12)
