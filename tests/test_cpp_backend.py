"""Tests for the optional C++ backend (``simkit._cpp``).

Every accelerated function must agree with its NumPy implementation, so each
test evaluates the same call under ``set_backend('python')`` and
``set_backend('cpp')`` and compares. The numerical tests are skipped when the
extension is not built; the backend-selection helpers are tested regardless.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

import simkit
from simkit.energies import elastic as E
from simkit.polar_svd import polar_svd
from simkit.psd_project import psd_project
from simkit.svd_rv import svd_rv


@pytest.fixture
def both_backends():
    """Yield a runner that evaluates ``fn()`` under both backends and restores the default."""
    if not simkit.cpp_available():
        pytest.skip("simkit._cpp extension not built")
    original = simkit.backend()

    def run(fn):
        simkit.set_backend("python")
        try:
            py = fn()
        finally:
            simkit.set_backend("cpp")
        cpp = fn()
        return py, cpp

    yield run
    simkit.set_backend(original)


def _random_F(rng, t, dim, scale=0.15):
    F = np.tile(np.eye(dim)[None, :, :], (t, 1, 1)) + scale * rng.standard_normal((t, dim, dim))
    return F[np.linalg.det(F) > 0.3]


# --------------------------------------------------------------------------- #
# Backend selection                                                           #
# --------------------------------------------------------------------------- #
def test_backend_helpers_are_consistent():
    assert simkit.backend() in ("cpp", "python")
    if simkit.cpp_available():
        # The default is 'cpp' unless the environment forced the NumPy code.
        if os.environ.get("SIMKIT_BACKEND", "").lower() != "python":
            assert simkit.backend() == "cpp"
        assert simkit.cpp_num_threads() >= 1
    else:
        assert simkit.backend() == "python"
        with pytest.raises(RuntimeError):
            simkit.set_backend("cpp")
    with pytest.raises(ValueError):
        simkit.set_backend("fortran")


def test_set_backend_round_trip():
    original = simkit.backend()
    try:
        assert simkit.set_backend("python") == "python"
        assert simkit.backend() == "python"
        if simkit.cpp_available():
            assert simkit.set_backend("cpp") == "cpp"
    finally:
        simkit.set_backend(original)


def test_env_var_forces_python_backend():
    env = dict(os.environ, SIMKIT_BACKEND="python")
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", "import simkit; print(simkit.backend())"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "python"


def test_cpp_num_threads_setter():
    if not simkit.cpp_available():
        assert simkit.cpp_num_threads(3) == 1
        return
    default = simkit.cpp_num_threads()
    try:
        assert simkit.cpp_num_threads(2) == 2
    finally:
        simkit.cpp_num_threads(0)
    assert simkit.cpp_num_threads() == default


# --------------------------------------------------------------------------- #
# Decompositions                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_polar_svd_matches(both_backends, dim):
    F = _random_F(np.random.default_rng(1), 40, dim, scale=0.4)
    (R0, S0), (R1, S1) = both_backends(lambda: polar_svd(F))
    assert np.allclose(R0, R1, atol=1e-12)
    assert np.allclose(S0, S1, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_polar_svd_handles_reflections_identically(both_backends, dim):
    rng = np.random.default_rng(2)
    F = _random_F(rng, 20, dim, scale=0.4)
    F[::2, 0, :] *= -1.0  # inverted elements
    (R0, S0), (R1, S1) = both_backends(lambda: polar_svd(F))
    assert np.allclose(R0, R1, atol=1e-12)
    assert np.allclose(S0, S1, atol=1e-12)


def test_polar_svd_single_matrix(both_backends):
    F = np.array([[1.2, 0.1], [-0.3, 0.9]])
    (R0, S0), (R1, S1) = both_backends(lambda: polar_svd(F))
    assert R1.shape == (1, 2, 2)
    assert np.allclose(R0, R1) and np.allclose(S0, S1)


@pytest.mark.parametrize("dim", [2, 3])
def test_svd_rv_matches(both_backends, dim):
    F = _random_F(np.random.default_rng(3), 30, dim, scale=0.4)
    (U0, S0, V0), (U1, S1, V1) = both_backends(lambda: svd_rv(F))
    assert np.allclose(S0, S1, atol=1e-12)
    assert np.allclose(U0 @ V0.transpose(0, 2, 1), U1 @ V1.transpose(0, 2, 1), atol=1e-12)
    assert np.allclose(U0 @ S0 @ V0.transpose(0, 2, 1), F, atol=1e-12)
    assert np.allclose(U1 @ S1 @ V1.transpose(0, 2, 1), F, atol=1e-12)


@pytest.mark.parametrize("b", [2, 3, 4, 6, 9, 5])
@pytest.mark.parametrize("method", ["proj", "abs"])
def test_psd_project_matches(both_backends, b, method):
    rng = np.random.default_rng(5)
    H = rng.standard_normal((12, b, b))
    H = H + H.transpose(0, 2, 1)
    P0, P1 = both_backends(lambda: psd_project(H, method))
    assert P1.shape == H.shape
    assert np.allclose(P0, P1, atol=1e-11)


def test_psd_project_single_block(both_backends):
    H = np.array([[1.0, 0.5], [0.5, -0.2]])
    P0, P1 = both_backends(lambda: psd_project(H))
    assert P1.shape == (1, 2, 2)
    assert np.allclose(P0, P1)


# --------------------------------------------------------------------------- #
# Through the energy pipeline                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("material", list(E._MATERIALS))
def test_elastic_hessian_matches_through_pipeline(both_backends, material):
    """The energy tiers call polar_svd / psd_project; both backends must agree end to end."""
    rng = np.random.default_rng(6)
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    T = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    mu = np.full((2, 1), 1.5)
    lam = np.full((2, 1), 2.5)
    U = X + 0.05 * rng.standard_normal(X.shape)
    e0, e1 = both_backends(lambda: E.elastic_energy_x(U, J, mu, lam, vol, material))
    assert np.isclose(e0, e1, rtol=1e-10)
    g0, g1 = both_backends(lambda: E.elastic_gradient_x(U, J, mu, lam, vol, material))
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_x(U, J, mu, lam, vol, material))
    assert abs(H0 - H1).max() < 1e-9
