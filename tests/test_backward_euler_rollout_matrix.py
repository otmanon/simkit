"""Tests for ``simkit.backward_euler_rollout_matrix``."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.backward_euler_rollout_matrix import backward_euler_rollout_matrix


@pytest.mark.parametrize("num_timesteps", [3, 8])
def test_shapes(num_timesteps: int) -> None:
    T, B = backward_euler_rollout_matrix(num_timesteps)
    assert T.shape == (num_timesteps, num_timesteps)
    assert B.shape == (num_timesteps, 2)
    assert isinstance(T, sps.csc_matrix)
    assert isinstance(B, sps.csc_matrix)


def test_single_timestep_raises() -> None:
    with pytest.raises(ValueError, match="exceeds matrix dimension"):
        backward_euler_rollout_matrix(1)


def test_T_is_symmetric() -> None:
    T, _ = backward_euler_rollout_matrix(5)
    dense = T.toarray()
    assert np.allclose(dense, dense.T, atol=1e-12)


def test_B_injects_pre_trajectory_states_into_first_rows() -> None:
    num_timesteps = 4
    T, B = backward_euler_rollout_matrix(num_timesteps)
    x = np.arange(num_timesteps, dtype=float)
    x_m2, x_m1 = 10.0, 20.0
    rhs = T @ x + B @ np.array([x_m2, x_m1])
    expected0 = (T @ x)[0] + B[0, 0] * x_m2 + B[0, 1] * x_m1
    assert rhs[0] == pytest.approx(expected0)
    assert B[0, 0] == pytest.approx(-0.5)
    assert B[0, 1] == pytest.approx(2.0)
    assert B[1, 1] == pytest.approx(-0.5)


def test_rollout_operator_dimensions_and_sparsity() -> None:
    num_timesteps = 6
    T, B = backward_euler_rollout_matrix(num_timesteps)
    assert T.nnz == 4 * num_timesteps
    assert B.nnz == 3
    assert np.count_nonzero(B.toarray()[:2]) == 3
