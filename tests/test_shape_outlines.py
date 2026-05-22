"""Tests for ``simkit.shape_outlines``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("igl")
pytestmark = pytest.mark.mesh

from simkit.shape_outlines import (
    arrow_outline,
    circle_outline,
    ellipse_outline,
    plus_sign_outline,
    rectangle_outline,
    star_sign_outline,
)


def test_outline_functions_return_vertex_and_edge_arrays() -> None:
    cases = [
        (arrow_outline(), (7, 2), (7, 2)),
        (circle_outline(n=20), (20, 2), (20, 2)),
        (ellipse_outline(n=20), (20, 2), (20, 2)),
        (rectangle_outline(), (4, 2), (4, 2)),
        (plus_sign_outline(), (12, 2), (12, 2)),
        (star_sign_outline(legs=5), (15, 2), (15, 2)),
    ]

    for (X, E), x_shape, e_shape in cases:
        assert isinstance(X, np.ndarray)
        assert isinstance(E, np.ndarray)
        assert X.shape == x_shape
        assert E.shape == e_shape
