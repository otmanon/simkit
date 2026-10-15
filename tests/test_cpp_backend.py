"""Tests for the optional C++ backend (``simkit._cpp``).

Every accelerated function must agree with its NumPy implementation, so each
test evaluates the same call under ``set_backend('python')`` and
``set_backend('cpp')`` and compares. The whole module is skipped when the
extension is not built; the backend-selection helpers are tested regardless.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sps

import simkit
from simkit.energies import elastic as E
from simkit.energies import arap, bending_energy, mass_springs
from simkit.polar_svd import polar_svd
from simkit.psd_project import psd_project
from simkit.rotation_gradient import rotation_gradient_F
from simkit.svd_rv import svd_rv

MATERIALS = list(E._MATERIALS)


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


def _grid_2d(n):
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array(np.meshgrid(xs, xs, indexing="ij")).reshape(2, -1).T
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    c = idx[:-1, :-1].reshape(-1)
    T = np.concatenate([np.stack([c, c + (n + 1), c + 1], axis=1), np.stack([c + 1, c + (n + 1), c + (n + 2)], axis=1)])
    return X, T


def _grid_3d(n):
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    c = idx[:-1, :-1, :-1].reshape(-1)
    dx, dy, dz = (n + 1) ** 2, (n + 1), 1
    v = [c, c + dx, c + dy, c + dx + dy, c + dz, c + dx + dz, c + dy + dz, c + dx + dy + dz]
    tets = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7]]
    T = np.concatenate([np.stack([v[i] for i in tet], axis=1) for tet in tets], axis=0)
    return X, T


def _mesh(dim):
    return _grid_2d(4) if dim == 2 else _grid_3d(3)


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


@pytest.mark.parametrize("dim", [2, 3])
def test_rotation_gradient_matches(both_backends, dim):
    F = _random_F(np.random.default_rng(4), 30, dim)
    K0, K1 = both_backends(lambda: rotation_gradient_F(F))
    assert K1.shape == (F.shape[0], dim * dim, dim * dim)
    assert np.allclose(K0, K1, atol=1e-11)


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
# Element tier                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("kind", ["energy", "gradient", "hessian"])
def test_element_tier_matches(both_backends, dim, material, kind):
    rng = np.random.default_rng(6)
    F = _random_F(rng, 25, dim)
    t = F.shape[0]
    mu = rng.uniform(0.5, 2.0, (t, 1))
    lam = rng.uniform(0.5, 3.0, (t, 1))
    fn = getattr(E, f"elastic_{kind}_element_F")
    if kind == "hessian":
        a0, a1 = both_backends(lambda: fn(F, mu, lam, material, psd=False))
        p0, p1 = both_backends(lambda: fn(F, mu, lam, material, psd=True))
        assert np.allclose(p0, p1, atol=1e-10)
    else:
        a0, a1 = both_backends(lambda: fn(F, mu, lam, material))
    assert a0.shape == a1.shape
    assert np.allclose(a0, a1, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_element_tier_broadcasts_scalar_parameters(both_backends, dim):
    F = _random_F(np.random.default_rng(7), 10, dim)
    a0, a1 = both_backends(lambda: arap.arap_energy_element_F(F, 1.7))
    assert np.allclose(a0, a1)
    g0, g1 = both_backends(lambda: E.elastic_gradient_element_F(F, 1.7, np.array([[2.5]]), "fcr"))
    assert np.allclose(g0, g1)


def test_element_tier_rejects_wrong_parameter_length():
    if not simkit.cpp_available():
        pytest.skip("simkit._cpp extension not built")
    F = _random_F(np.random.default_rng(8), 10, 3)
    with pytest.raises(ValueError):
        simkit._cpp.arap_energy_element_F(F, np.ones(4))


# --------------------------------------------------------------------------- #
# Global tiers and assembly                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_deformation_jacobian_matches(both_backends, dim):
    X, T = _mesh(dim)
    J0, J1 = both_backends(lambda: simkit.deformation_jacobian(X, T))
    assert J0.shape == J1.shape
    assert sps.issparse(J1)
    assert abs(J0 - J1).max() < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_block_diag_sandwich_sparse_matches(both_backends, dim):
    rng = np.random.default_rng(9)
    X, T = _mesh(dim)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    b = dim * dim
    He = rng.standard_normal((T.shape[0], b, b))
    Q0, Q1 = both_backends(lambda: simkit.block_diag_sandwich(J, He))
    assert sps.issparse(Q1) and Q1.shape == Q0.shape
    assert abs(Q0 - Q1).max() < 1e-11
    # The dense (reduced) path too.
    B = rng.standard_normal((J.shape[1], 6))
    JB = J @ B
    D0, D1 = both_backends(lambda: simkit.block_diag_sandwich(JB, He))
    assert isinstance(D1, np.ndarray) and D1.shape == (6, 6)
    assert np.allclose(D0, D1, atol=1e-10)


def test_block_diag_sandwich_rejects_mismatched_rows():
    He = np.zeros((3, 4, 4))
    J = sps.identity(10)
    with pytest.raises(ValueError):
        simkit.block_diag_sandwich(J, He)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("material", MATERIALS)
def test_global_x_tier_matches(both_backends, dim, material):
    rng = np.random.default_rng(10)
    X, T = _mesh(dim)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, (t, 1))
    lam = rng.uniform(0.5, 3.0, (t, 1))
    U = X + 0.02 * rng.standard_normal(X.shape)
    e0, e1 = both_backends(lambda: E.elastic_energy_x(U, J, mu, lam, vol, material))
    assert np.isclose(e0, e1, rtol=1e-10, atol=1e-12)
    g0, g1 = both_backends(lambda: E.elastic_gradient_x(U, J, mu, lam, vol, material))
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_x(U, J, mu, lam, vol, material))
    assert sps.issparse(H1)
    assert abs(H0 - H1).max() < 1e-9


def test_mass_springs_hessian_matches(both_backends):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((12, 3))
    Edges = np.array([[i, (i + 1) % 12] for i in range(12)] + [[i, (i + 5) % 12] for i in range(12)])
    ym = rng.uniform(0.5, 2.0, (Edges.shape[0], 1))
    vol = np.ones((Edges.shape[0], 1))
    l0 = np.linalg.norm(x[Edges[:, 0]] - x[Edges[:, 1]], axis=1)[:, None] * 0.9
    H0, H1 = both_backends(lambda: mass_springs.mass_springs_hessian_x(x, Edges, ym, vol, l0))
    assert abs(H0 - H1).max() < 1e-10


def test_bending_hessian_matches(both_backends):
    rng = np.random.default_rng(12)
    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.3], [1.5, 1.0, -0.2]])
    C = np.array([[0, 1, 2, 3], [1, 2, 0, 4]])
    theta0 = np.zeros((2, 1))
    kappa = rng.uniform(0.5, 2.0, (2, 1))
    x = X + 0.05 * rng.standard_normal(X.shape)
    H0, H1 = both_backends(lambda: bending_energy.bending_hessian_x(x, C, theta0, kappa, psd=True))
    assert abs(H0 - H1).max() < 1e-10


@pytest.mark.parametrize("material", ["arap", "stable-neo-hookean"])
def test_reduced_z_tier_matches(both_backends, material):
    rng = np.random.default_rng(13)
    X, T = _mesh(3)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = np.full((t, 1), 1.0)
    lam = np.full((t, 1), 2.0)
    B = rng.standard_normal((J.shape[1], 8))
    G = sps.identity(J.shape[0])
    precomp = E.ElasticEnergyZPrecomp(B, X.reshape(-1, 1), G, J, 3)
    z = 0.01 * rng.standard_normal((8, 1))
    e0, e1 = both_backends(lambda: E.elastic_energy_z(z, mu, lam, vol, material, precomp))
    assert np.isclose(e0, e1, rtol=1e-10)
    g0, g1 = both_backends(lambda: E.elastic_gradient_z(z, mu, lam, vol, material, precomp))
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_z(z, mu, lam, vol, material, precomp))
    assert np.allclose(np.asarray(H0), np.asarray(H1), atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("material", MATERIALS)
def test_global_u_tier_matches(both_backends, dim, material):
    rng = np.random.default_rng(14)
    X, T = _mesh(dim)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, (t, 1))
    lam = rng.uniform(0.5, 3.0, (t, 1))
    Jx_bar = J @ X.reshape(-1, 1)
    u = 0.02 * rng.standard_normal(X.shape)
    e0, e1 = both_backends(lambda: E.elastic_energy_u(u, J, Jx_bar, mu, lam, vol, material))
    assert np.isclose(e0, e1, rtol=1e-10, atol=1e-12)
    g0, g1 = both_backends(lambda: E.elastic_gradient_u(u, J, Jx_bar, mu, lam, vol, material))
    assert g1.shape == g0.shape
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_u(u, J, Jx_bar, mu, lam, vol, material))
    assert sps.issparse(H1)
    assert abs(H0 - H1).max() < 1e-9


@pytest.mark.parametrize("material", MATERIALS)
def test_reduced_z_tier_matches_all_materials(both_backends, material):
    rng = np.random.default_rng(15)
    X, T = _mesh(3)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, (t, 1))
    lam = rng.uniform(0.5, 3.0, (t, 1))
    B = rng.standard_normal((J.shape[1], 7))
    precomp = E.ElasticEnergyZPrecomp(B, X.reshape(-1, 1), sps.identity(J.shape[0]), J, 3)
    z = 0.01 * rng.standard_normal((7, 1))
    e0, e1 = both_backends(lambda: E.elastic_energy_z(z, mu, lam, vol, material, precomp))
    assert np.isclose(e0, e1, rtol=1e-10, atol=1e-12)
    g0, g1 = both_backends(lambda: E.elastic_gradient_z(z, mu, lam, vol, material, precomp))
    assert g1.shape == (7, 1)
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_z(z, mu, lam, vol, material, precomp))
    assert np.asarray(H1).shape == (7, 7)
    assert np.allclose(np.asarray(H0), np.asarray(H1), atol=1e-9)
    # Precomputed F is honoured by both backends.
    F = (precomp.JB @ z + precomp.Jx0).reshape(-1, 3, 3)
    f0, f1 = both_backends(lambda: E.elastic_energy_z(z, mu, lam, vol, material, precomp, F=F))
    assert np.isclose(f0, f1, rtol=1e-10)


@pytest.mark.parametrize("material", MATERIALS)
def test_filtered_z_tier_matches(both_backends, material):
    rng = np.random.default_rng(16)
    X, T = _mesh(3)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = np.full((t, 1), 1.2)
    lam = np.full((t, 1), 2.0)
    B = rng.standard_normal((J.shape[1], 5))
    precomp = E.ElasticEnergyZFilteredPrecomp(B, X.reshape(-1, 1), sps.identity(J.shape[0]), J, 3, mu, vol)
    z = 0.01 * rng.standard_normal((5, 1))
    e0, e1 = both_backends(lambda: E.elastic_energy_filtered_z(z, mu, lam, vol, material, precomp))
    assert np.isclose(e0, e1, rtol=1e-10)
    g0, g1 = both_backends(lambda: E.elastic_gradient_filtered_z(z, mu, lam, vol, material, precomp))
    assert np.allclose(g0, g1, atol=1e-10)
    H0, H1 = both_backends(lambda: E.elastic_hessian_filtered_z(z, mu, lam, vol, material, precomp))
    assert np.allclose(np.asarray(H0), np.asarray(H1), atol=1e-9)


@pytest.mark.parametrize("dim", [2, 3])
def test_per_material_x_tiers_match(both_backends, dim):
    """Every per-material ``*_x`` / ``*_u`` function, not only the dispatcher."""
    import importlib
    rng = np.random.default_rng(17)
    X, T = _mesh(dim)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, (t, 1))
    lam = rng.uniform(0.5, 3.0, (t, 1))
    U = X + 0.02 * rng.standard_normal(X.shape)
    Jx_bar = J @ X.reshape(-1, 1)
    u = U - X
    for prefix in ["arap", "fcr", "neo_hookean", "stable_neo_hookean", "stvk", "linear_elasticity", "macklin_mueller_neo_hookean"]:
        mod = importlib.import_module("simkit.energies." + prefix)
        params = (mu,) if prefix == "arap" else (mu, lam)
        for kind in ["energy", "gradient", "hessian"]:
            fx = getattr(mod, f"{prefix}_{kind}_x")
            fu = getattr(mod, f"{prefix}_{kind}_u")
            a0, a1 = both_backends(lambda: fx(U, J, *params, vol))
            b0, b1 = both_backends(lambda: fu(u, J, Jx_bar, *params, vol))
            for p0, p1 in [(a0, a1), (b0, b1)]:
                if sps.issparse(p0) or sps.issparse(p1):
                    assert abs(p0 - p1).max() < 1e-9, (prefix, kind)
                else:
                    assert np.allclose(np.asarray(p0), np.asarray(p1), atol=1e-10), (prefix, kind)


def test_sparse_solve_matches_spsolve(both_backends):
    rng = np.random.default_rng(18)
    X, T = _mesh(3)
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = np.full((t, 1), 1.0)
    lam = np.full((t, 1), 2.0)
    H = E.elastic_hessian_x(X, J, mu, lam, vol, "arap") + sps.identity(J.shape[1])
    b = rng.standard_normal((J.shape[1], 1))
    x0, x1 = both_backends(lambda: simkit.sparse_solve(H, b))
    assert x1.shape == b.shape
    assert np.allclose(x0, x1, atol=1e-9)
    assert np.allclose(H @ x1, b, atol=1e-8)
    # Multiple right-hand sides and a 1D right-hand side.
    B = rng.standard_normal((J.shape[1], 3))
    y0, y1 = both_backends(lambda: simkit.sparse_solve(H, B))
    assert np.allclose(y0, y1, atol=1e-9)
    z0, z1 = both_backends(lambda: simkit.sparse_solve(H, b[:, 0]))
    assert z1.shape == (J.shape[1],) and np.allclose(z0, z1, atol=1e-9)
    # SPD systems take the Cholesky path; an indefinite one falls through to LDLT.
    simkit.set_backend("cpp")
    _, method = simkit.sparse_solve(H, b, return_method=True)
    assert method == "llt"
    K = H - 2.0 * sps.identity(J.shape[1])  # shifted below zero: indefinite but non-singular
    xk, method = simkit.sparse_solve(K, b, return_method=True)
    assert method in ("ldlt", "spsolve")
    assert np.allclose(K @ xk, b, atol=1e-7)
    simkit.set_backend("python")
    _, method = simkit.sparse_solve(H, b, return_method=True)
    assert method == "spsolve"


def test_sparse_solve_falls_back_on_singular_matrix():
    A = sps.csc_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))  # singular, symmetric
    b = np.array([[1.0], [2.0]])
    with np.errstate(all="ignore"):
        x = simkit.sparse_solve(A, b)
    assert x.shape == (2, 1)


def test_read_mesh_roundtrip(tmp_path):
    from simkit.read_mesh import read_mesh
    text = """MeshVersionFormatted 1
Dimension 3
Vertices
4
0 0 0 0
1 0 0 0
0 1 0 0
0 0 1 0
Triangles
1
1 2 3 0
Tetrahedra
1
1 2 3 4 0
End
"""
    p = tmp_path / "tet.mesh"
    p.write_text(text)
    X, T, F = read_mesh(str(p))
    assert X.shape == (4, 3) and T.shape == (1, 4) and F.shape == (1, 3)
    assert T.tolist() == [[0, 1, 2, 3]] and F.tolist() == [[0, 1, 2]]
    assert np.isclose(simkit.volume(X, T)[0, 0], 1.0 / 6.0)
