"""Tests for ``simkit.mat2py``."""

from __future__ import annotations

import numpy as np

import simkit.mat2py as mat2py


def test_ordering_constants_exist_and_are_integers() -> None:
    for name in (
        "_4vector_2D_ordering_",
        "_9vector_3D_ordering_",
        "_4x4matrix_2D_ordering_",
        "_9x9matrix_3D_ordering_",
    ):
        ordering = getattr(mat2py, name)
        assert isinstance(ordering, list)
        assert len(ordering) > 0
        assert all(isinstance(i, int) for i in ordering)


def test_y_index_map_is_integer_array() -> None:
    assert isinstance(mat2py.y, np.ndarray)
    assert mat2py.y.dtype.kind in "iu"
