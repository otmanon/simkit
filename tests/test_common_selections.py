"""Tests for ``simkit.common_selections``."""

from __future__ import annotations

import numpy as np
import pytest

from simkit.common_selections import (
    back_z_indices,
    bottom_indices,
    center_indices,
    center_top_indices,
    create_selection,
    top_indices,
)


def _box_vertices() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )


def test_center_indices_selects_near_centroid() -> None:
    X = _box_vertices()
    pinned, pinnedI = center_indices(X, t=1.0)
    assert pinned.dtype == bool
    assert pinnedI.dtype.kind in "iu"
    assert pinned.sum() >= 1
    assert np.all(pinned[pinnedI])
    center = X.mean(axis=0)
    extent = np.max(X.max(axis=0) - X.min(axis=0))
    assert np.all(np.linalg.norm(X[pinnedI] - center, axis=1) < extent)


def test_back_z_indices_selects_low_z_vertices() -> None:
    X = _box_vertices()
    pinned, pinnedI = back_z_indices(X, t=0.25)
    assert np.all(X[pinnedI, 2] <= X[:, 2].min() + 0.25 * (X[:, 2].max() - X[:, 2].min()))


def test_top_and_bottom_indices() -> None:
    X = _box_vertices()
    top, topI = top_indices(X, t=0.25)
    bottom, bottomI = bottom_indices(X, t=0.25)
    assert top[4]
    assert bottom[0]
    assert not np.any(top & bottom)


def test_center_top_indices_requires_near_center_x_and_top_y() -> None:
    X = _box_vertices()
    t = 0.75
    pinned, pinnedI = center_top_indices(X, t=t)
    diff = np.max(X.max(axis=0) - X.min(axis=0))
    center = X.mean(axis=0)
    near_center_x = np.abs(X[:, 0] - center[0]) < diff * t
    in_top_y = X[:, 1] > (X[:, 1].max() - diff * t)
    expected = np.logical_and(near_center_x, in_top_y)
    assert np.array_equal(pinned, expected)


@pytest.mark.parametrize("name", ["center", "back_z"])
def test_create_selection_dispatches(name: str) -> None:
    X = _box_vertices()
    pinned, pinnedI = create_selection(name, X, t=0.5)
    assert pinned.shape == (X.shape[0],)
    assert pinnedI.shape[0] == pinned.sum()


def test_create_selection_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown pinning type"):
        create_selection("unknown", _box_vertices(), t=0.1)
