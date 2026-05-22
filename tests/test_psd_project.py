"""Tests for ``simkit.psd_project``."""

from __future__ import annotations

import numpy as np

from simkit.psd_project import psd_project


def test_psd_project_floors_eigenvalues_with_proj_method() -> None:
    H = np.array([[1.0, 0.5], [0.5, -0.2]])
    H_proj = psd_project(H, method="proj")[0]
    evals = np.linalg.eigvalsh(H_proj)
    assert np.all(evals >= 1e-6 - 1e-12)


def test_psd_project_batch_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    H = rng.standard_normal((4, 3, 3))
    H = H + H.transpose(0, 2, 1)
    H_proj = psd_project(H, method="proj")
    assert H_proj.shape == (4, 3, 3)
    evals = np.linalg.eigvalsh(H_proj)
    assert np.all(evals >= 1e-6 - 1e-12)
