"""Tests for ``simkit.joint_edge_map``."""

from __future__ import annotations

import numpy as np

from simkit.joint_edge_map import joint_edge_map


def _line_mesh_with_joint() -> tuple[np.ndarray, np.ndarray]:
    """Three vertices on a line and one joint at the middle vertex."""
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    joints = np.array([[0, 1, 2]], dtype=int)
    return edges, joints


def test_joint_edge_map_shape() -> None:
    edges, joints = _line_mesh_with_joint()
    joint_edges = joint_edge_map(edges, joints)
    assert joint_edges.shape == (joints.shape[0], 2)
    assert joint_edges.dtype == int


def test_joint_edge_map_indices() -> None:
    edges, joints = _line_mesh_with_joint()
    joint_edges = joint_edge_map(edges, joints)
    assert joint_edges[0, 0] == 0
    assert joint_edges[0, 1] == 1
