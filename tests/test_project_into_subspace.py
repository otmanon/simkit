"""Tests for ``simkit.project_into_subspace``."""

from __future__ import annotations

import numpy as np

from simkit.project_into_subspace import project_into_subspace


def test_project_into_subspace_minimizes_weighted_distance() -> None:
    rng = np.random.default_rng(4)
    n, r = 12, 4
    B = rng.standard_normal((n, r))
    y = rng.standard_normal(n)
    z = project_into_subspace(y, B, M=None)
    residual = B @ z.reshape(-1) - y
    assert np.linalg.norm(residual) <= np.linalg.norm(B @ rng.standard_normal(r) - y) + 1e-10

    z_alt = np.linalg.lstsq(B, y, rcond=None)[0]
    assert np.linalg.norm(B @ z.reshape(-1) - y) <= np.linalg.norm(B @ z_alt - y) + 1e-10
