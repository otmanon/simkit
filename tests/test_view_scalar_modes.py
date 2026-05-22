"""Tests for ``simkit.view_scalar_modes``."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np


def test_view_scalar_modes_exists_with_expected_signature() -> None:
    from simkit.view_scalar_modes import view_scalar_modes

    sig = inspect.signature(view_scalar_modes)
    assert list(sig.parameters) == [
        "X",
        "T",
        "W",
        "period",
        "cmap",
        "name",
        "dir",
        "normalize",
        "vminmax",
        "eye_pos",
        "eye_target",
    ]


@patch("simkit.view_scalar_modes.time.sleep", return_value=None)
@patch("simkit.view_scalar_modes.ps")
def test_view_scalar_modes_runs_with_mocked_polyscope(
    mock_ps: MagicMock, _mock_sleep: MagicMock
) -> None:
    from simkit.view_scalar_modes import view_scalar_modes

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = np.array([[0, 1, 2]])
    W = np.array([[1.0, -1.0], [0.5, 0.5], [0.0, 0.0]])

    mock_geo = MagicMock()
    mock_ps.register_surface_mesh.return_value = mock_geo

    view_scalar_modes(X, T, W, period=0.0, normalize=False, vminmax=[-1.0, 1.0])

    mock_ps.init.assert_called_once()
    mock_ps.register_surface_mesh.assert_called_once()
    assert mock_geo.add_scalar_quantity.call_count == W.shape[1]
