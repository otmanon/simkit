"""Shared pytest fixtures for the SimKit test suite."""

from __future__ import annotations

import numpy as np
import pytest


def _strip_mesh(nx: int, ny: int, length: float):
    """Regular triangulated strip ``[0, length] x [0, 1]`` with ``nx * ny`` vertices."""
    xs, ys = np.meshgrid(np.linspace(0, length, nx), np.linspace(0, 1, ny), indexing="ij")
    X = np.c_[xs.ravel(), ys.ravel()]
    idx = np.arange(nx * ny).reshape(nx, ny)
    T = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            T += [[a, b, c], [a, c, d]]
    return X, np.array(T)


@pytest.fixture
def stiff_bar_strip():
    """Soft strip ``[0, 3] x [0, 1]`` with a stiff bar on ``1 < x < 2``.

    Returns ``(X, T, ym)`` with ``ym = 1e4`` inside the bar and ``1`` outside.
    """
    X, T = _strip_mesh(21, 7, 3.0)
    cen = X[T].mean(axis=1)
    ym = np.where((cen[:, 0] > 1.0) & (cen[:, 0] < 2.0), 1e4, 1.0)
    return X, T, ym


@pytest.fixture
def small_stiff_bar_strip():
    """Coarser version of :func:`stiff_bar_strip` for cheaper tests."""
    X, T = _strip_mesh(11, 4, 3.0)
    cen = X[T].mean(axis=1)
    ym = np.where((cen[:, 0] > 1.0) & (cen[:, 0] < 2.0), 1e4, 1.0)
    return X, T, ym
