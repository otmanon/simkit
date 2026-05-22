"""Tests for ``simkit.drag_force`` (stub)."""

from __future__ import annotations

from simkit.drag_force import drag_force


def test_drag_force_returns_none() -> None:
    assert drag_force() is None
    assert drag_force(object(), velocity=1.0) is None
