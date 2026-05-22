"""Tests for ``simkit.joint_lengths``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("igl")
pytestmark = pytest.mark.mesh

from simkit.joint_lengths import joint_lengths


def _right_angle_joint() -> tuple[np.ndarray, np.ndarray]:
    V = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    E = np.array([[0, 1, 2]], dtype=int)
    return V, E


def test_joint_lengths_are_positive() -> None:
    V, E = _right_angle_joint()
    lengths = joint_lengths(V, E)
    assert lengths.shape == (E.shape[0],)
    assert np.all(lengths > 0.0)


def test_joint_lengths_match_mean_edge_length() -> None:
    V, E = _right_angle_joint()
    lengths = joint_lengths(V, E)
    # Both incident edges have unit length.
    assert np.allclose(lengths, 1.0, rtol=1e-12)
