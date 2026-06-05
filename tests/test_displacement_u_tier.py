"""Tests for the displacement (``_u``) tier across all energies.

The ``_u`` tier expresses ``x = x_bar + u`` for an arbitrary reference
configuration ``x_bar`` (not necessarily the rest pose). Flavor A energies
(hyperelastic, membrane, EMU) take a precomputed ``Jx_bar = J @ x_bar.reshape(-1, 1)``
and construct ``F = (J @ u + Jx_bar).reshape(...)`` internally. Flavor B
energies (bending, kinetic) are thin substitution wrappers around the existing
``_x`` tier.

We check three equivalences for every Flavor A energy and the analogous parity
for Flavor B:

1. **Zero-offset parity** — ``_u(u=X, J, Jx_bar=0, ...) == _x(X, J, ...)``.
2. **Displacement-from-rest** — with ``u = X_def - X_rest`` and
   ``Jx_bar = J @ X_rest`` the ``_u`` value equals ``_x(X_def, ...)``.
3. **Arbitrary offset** — with a random non-rest ``x_bar`` the same equivalence
   holds (this is the case that motivated the change).

The Flavor B wrappers are checked against direct substitution
``_u(X - x_bar, ..., x_bar) == _x(X, ...)``.

The elastic dispatcher is exercised with every material string to confirm
``elastic_*_u`` routes to the per-material function.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

from simkit.deformation_jacobian import deformation_jacobian
from simkit.membrane_deformation_jacobian import membrane_deformation_jacobian
from simkit.volume import volume

from simkit.energies.arap import (
    arap_energy_u, arap_gradient_u, arap_hessian_u,
    arap_energy_x, arap_gradient_x, arap_hessian_x,
)
from simkit.energies.bending_energy import (
    bending_energy_u, bending_gradient_u, bending_hessian_u,
    bending_energy_x, bending_gradient_x, bending_hessian_x,
)
from simkit.energies.discrete_shells_bending import (
    discrete_shells_bending_energy_u, discrete_shells_bending_gradient_u,
    discrete_shells_bending_hessian_u,
    discrete_shells_bending_energy_x, discrete_shells_bending_gradient_x,
    discrete_shells_bending_hessian_x,
)
from simkit.energies.elastic import (
    elastic_energy_u, elastic_gradient_u, elastic_hessian_u,
    elastic_energy_x, elastic_gradient_x, elastic_hessian_x,
)
from simkit.energies.emu import (
    emu_energy_u, emu_gradient_u, emu_hessian_u,
    emu_energy_x, emu_gradient_x, emu_hessian_x,
)
from simkit.energies.fcr import (
    fcr_energy_u, fcr_gradient_u, fcr_hessian_u,
    fcr_energy_x, fcr_gradient_x, fcr_hessian_x,
)
from simkit.energies.kinetic import (
    kinetic_energy_be, kinetic_gradient_be,
    kinetic_energy_be_u, kinetic_gradient_be_u,
    kinetic_energy_bdf2, kinetic_gradient_bdf2,
    kinetic_energy_bdf2_u, kinetic_gradient_bdf2_u,
)
from simkit.energies.linear_elasticity import (
    linear_elasticity_energy_u, linear_elasticity_gradient_u, linear_elasticity_hessian_u,
    linear_elasticity_energy_x, linear_elasticity_gradient_x, linear_elasticity_hessian_x,
)
from simkit.energies.membrane_neo_hookean import (
    membrane_neo_hookean_energy_u, membrane_neo_hookean_gradient_u,
    membrane_neo_hookean_hessian_u,
    membrane_neo_hookean_energy_x, membrane_neo_hookean_gradient_x,
    membrane_neo_hookean_hessian_x,
)
from simkit.energies.macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_u, macklin_mueller_neo_hookean_gradient_u,
    macklin_mueller_neo_hookean_hessian_u,
    macklin_mueller_neo_hookean_energy_x, macklin_mueller_neo_hookean_gradient_x,
    macklin_mueller_neo_hookean_hessian_x,
)


TOL = 1e-10


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _two_tet_mesh_3d():
    """A two-tet mesh in 3D sharing a face. Returns (X, T)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    T = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ])
    return X, T


def _two_tri_mesh_2d():
    """Two triangles sharing an edge in 2D. Returns (X, T)."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    T = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return X, T


def _mesh(dim: int):
    return _two_tri_mesh_2d() if dim == 2 else _two_tet_mesh_3d()


def _membrane_mesh_3d():
    """Two triangles in 3D embedding (membrane). Returns (X, T)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [0.0, 1.0, -0.1],
        [1.0, 1.0, 0.0],
    ])
    T = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return X, T


def _rotation_3d(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotation_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _arbitrary_x_bar(rng: np.random.Generator, X_rest: np.ndarray) -> np.ndarray:
    """Return an arbitrary reference config (rotation + translation of rest)."""
    dim = X_rest.shape[1]
    if dim == 2:
        R = _rotation_2d(0.3)
    else:
        R = _rotation_3d(0.2, -0.15, 0.25)
    t = 0.5 * rng.standard_normal((1, dim))
    return X_rest @ R.T + t


def _setup_volumetric(dim: int, rng: np.random.Generator):
    """Build (X_rest, T, J, vol, mu, lam, X_def) for a volumetric mesh."""
    X_rest, T = _mesh(dim)
    J = deformation_jacobian(X_rest, T)
    vol = volume(X_rest, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    X_def = X_rest + 0.1 * rng.standard_normal(X_rest.shape)
    return X_rest, T, J, vol, mu, lam, X_def


def _hessian_to_dense(H):
    return H.toarray() if sps.issparse(H) else np.asarray(H)


# --------------------------------------------------------------------------- #
# Flavor A: Macklin-Mueller Neo-Hookean                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_macklin_mueller_neo_hookean_u_zero_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(0)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    Jx_bar = np.zeros((J.shape[0], 1))

    e_x = macklin_mueller_neo_hookean_energy_x(X_def, J, mu, lam, vol)
    e_u = macklin_mueller_neo_hookean_energy_u(X_def, J, Jx_bar, mu, lam, vol)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = macklin_mueller_neo_hookean_gradient_x(X_def, J, mu, lam, vol)
    g_u = macklin_mueller_neo_hookean_gradient_u(X_def, J, Jx_bar, mu, lam, vol)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_x(X_def, J, mu, lam, vol, psd=False))
    H_u = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_u(X_def, J, Jx_bar, mu, lam, vol, psd=False))
    assert np.allclose(H_u, H_x, atol=TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_macklin_mueller_neo_hookean_u_rest_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(1)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    u = X_def - X_rest
    Jx_bar = (J @ X_rest.reshape(-1, 1))

    e_x = macklin_mueller_neo_hookean_energy_x(X_def, J, mu, lam, vol)
    e_u = macklin_mueller_neo_hookean_energy_u(u, J, Jx_bar, mu, lam, vol)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = macklin_mueller_neo_hookean_gradient_x(X_def, J, mu, lam, vol)
    g_u = macklin_mueller_neo_hookean_gradient_u(u, J, Jx_bar, mu, lam, vol)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_x(X_def, J, mu, lam, vol, psd=False))
    H_u = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=False))
    assert np.allclose(H_u, H_x, atol=TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_macklin_mueller_neo_hookean_u_arbitrary_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(2)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    e_x = macklin_mueller_neo_hookean_energy_x(X_def, J, mu, lam, vol)
    e_u = macklin_mueller_neo_hookean_energy_u(u, J, Jx_bar, mu, lam, vol)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = macklin_mueller_neo_hookean_gradient_x(X_def, J, mu, lam, vol)
    g_u = macklin_mueller_neo_hookean_gradient_u(u, J, Jx_bar, mu, lam, vol)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_x(X_def, J, mu, lam, vol, psd=False))
    H_u = _hessian_to_dense(macklin_mueller_neo_hookean_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=False))
    assert np.allclose(H_u, H_x, atol=TOL)


# --------------------------------------------------------------------------- #
# Flavor A: ARAP (mu only, no lam)                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_arap_u_zero_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(10)
    X_rest, T, J, vol, mu, _, X_def = _setup_volumetric(dim, rng)
    Jx_bar = np.zeros((J.shape[0], 1))

    assert arap_energy_u(X_def, J, Jx_bar, mu, vol) == pytest.approx(
        arap_energy_x(X_def, J, mu, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        arap_gradient_u(X_def, J, Jx_bar, mu, vol),
        arap_gradient_x(X_def, J, mu, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(arap_hessian_u(X_def, J, Jx_bar, mu, vol, psd=False)),
        _hessian_to_dense(arap_hessian_x(X_def, J, mu, vol, psd=False)), atol=TOL,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_arap_u_arbitrary_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(11)
    X_rest, T, J, vol, mu, _, X_def = _setup_volumetric(dim, rng)
    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    assert arap_energy_u(u, J, Jx_bar, mu, vol) == pytest.approx(
        arap_energy_x(X_def, J, mu, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        arap_gradient_u(u, J, Jx_bar, mu, vol),
        arap_gradient_x(X_def, J, mu, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(arap_hessian_u(u, J, Jx_bar, mu, vol, psd=False)),
        _hessian_to_dense(arap_hessian_x(X_def, J, mu, vol, psd=False)), atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Flavor A: Linear elasticity                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_linear_elasticity_u_arbitrary_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(20)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    assert linear_elasticity_energy_u(u, J, Jx_bar, mu, lam, vol) == pytest.approx(
        linear_elasticity_energy_x(X_def, J, mu, lam, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        linear_elasticity_gradient_u(u, J, Jx_bar, mu, lam, vol),
        linear_elasticity_gradient_x(X_def, J, mu, lam, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(linear_elasticity_hessian_u(u, J, Jx_bar, mu, lam, vol)),
        _hessian_to_dense(linear_elasticity_hessian_x(X_def, J, mu, lam, vol)), atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Flavor A: FCR                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_fcr_u_arbitrary_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(30)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    assert fcr_energy_u(u, J, Jx_bar, mu, lam, vol) == pytest.approx(
        fcr_energy_x(X_def, J, mu, lam, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        fcr_gradient_u(u, J, Jx_bar, mu, lam, vol),
        fcr_gradient_x(X_def, J, mu, lam, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(fcr_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=False)),
        _hessian_to_dense(fcr_hessian_x(X_def, J, mu, lam, vol, psd=False)), atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Flavor A: EMU                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_emu_u_arbitrary_offset_matches_x(dim: int) -> None:
    rng = np.random.default_rng(40)
    X_rest, T, J, vol, _, _, X_def = _setup_volumetric(dim, rng)
    t = T.shape[0]
    d = rng.standard_normal((t, dim))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    a = rng.uniform(0.5, 2.0, size=(t, 1))

    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    assert emu_energy_u(u, J, Jx_bar, d, a, vol) == pytest.approx(
        emu_energy_x(X_def, J, d, a, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        emu_gradient_u(u, J, Jx_bar, d, a, vol),
        emu_gradient_x(X_def, J, d, a, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(emu_hessian_u(u, J, Jx_bar, d, a, vol)),
        _hessian_to_dense(emu_hessian_x(X_def, J, d, a, vol)), atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Flavor A: Membrane Neo-Hookean (3D-in-2D)                                   #
# --------------------------------------------------------------------------- #
def test_membrane_neo_hookean_u_arbitrary_offset_matches_x() -> None:
    rng = np.random.default_rng(50)
    X_rest, T = _membrane_mesh_3d()
    J = membrane_deformation_jacobian(X_rest, T)
    vol = volume(X_rest, T)
    t = T.shape[0]
    mu = rng.uniform(0.5, 2.0, size=(t, 1))
    lam = rng.uniform(0.5, 2.0, size=(t, 1))
    X_def = X_rest + 0.05 * rng.standard_normal(X_rest.shape)

    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    assert membrane_neo_hookean_energy_u(u, J, Jx_bar, mu, lam, vol) == pytest.approx(
        membrane_neo_hookean_energy_x(X_def, J, mu, lam, vol), abs=TOL, rel=TOL
    )
    assert np.allclose(
        membrane_neo_hookean_gradient_u(u, J, Jx_bar, mu, lam, vol),
        membrane_neo_hookean_gradient_x(X_def, J, mu, lam, vol), atol=TOL,
    )
    assert np.allclose(
        _hessian_to_dense(membrane_neo_hookean_hessian_u(u, J, Jx_bar, mu, lam, vol, psd=False)),
        _hessian_to_dense(membrane_neo_hookean_hessian_x(X_def, J, mu, lam, vol, psd=False)),
        atol=TOL,
    )


# --------------------------------------------------------------------------- #
# Elastic dispatcher                                                          #
# --------------------------------------------------------------------------- #
MATERIALS = ["linear-elasticity", "arap", "fcr", "macklin-mueller-neo-hookean"]


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("dim", [2, 3])
def test_elastic_dispatcher_u_matches_x(material: str, dim: int) -> None:
    rng = np.random.default_rng(60 + hash(material) % 1000)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(dim, rng)
    x_bar = _arbitrary_x_bar(rng, X_rest)
    u = X_def - x_bar
    Jx_bar = (J @ x_bar.reshape(-1, 1))

    e_x = elastic_energy_x(X_def, J, mu, lam, vol, material)
    e_u = elastic_energy_u(u, J, Jx_bar, mu, lam, vol, material)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = elastic_gradient_x(X_def, J, mu, lam, vol, material)
    g_u = elastic_gradient_u(u, J, Jx_bar, mu, lam, vol, material)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(elastic_hessian_x(X_def, J, mu, lam, vol, material, psd=False))
    H_u = _hessian_to_dense(elastic_hessian_u(u, J, Jx_bar, mu, lam, vol, material, psd=False))
    assert np.allclose(H_u, H_x, atol=TOL)


def test_elastic_dispatcher_u_unknown_material_raises() -> None:
    rng = np.random.default_rng(99)
    X_rest, T, J, vol, mu, lam, X_def = _setup_volumetric(3, rng)
    Jx_bar = np.zeros((J.shape[0], 1))
    with pytest.raises(ValueError):
        elastic_energy_u(X_def, J, Jx_bar, mu, lam, vol, "not-a-material")
    with pytest.raises(ValueError):
        elastic_gradient_u(X_def, J, Jx_bar, mu, lam, vol, "not-a-material")
    with pytest.raises(ValueError):
        elastic_hessian_u(X_def, J, Jx_bar, mu, lam, vol, "not-a-material")


# --------------------------------------------------------------------------- #
# Flavor B: Bending energy (2D beam)                                          #
# --------------------------------------------------------------------------- #
def _straight_beam(n_nodes: int = 5):
    X = np.zeros((n_nodes, 2))
    X[:, 0] = np.arange(n_nodes)
    H = np.array([[i, i + 1, i + 2] for i in range(n_nodes - 2)], dtype=np.int64)
    theta0 = np.zeros((H.shape[0], 1))
    ymI = np.ones((H.shape[0], 1))
    l = np.ones((H.shape[0], 1))
    return X, H, theta0, ymI, l


def test_bending_u_arbitrary_offset_matches_x() -> None:
    rng = np.random.default_rng(70)
    X, H, theta0, ymI, l = _straight_beam(n_nodes=5)
    X_def = X + 0.05 * rng.standard_normal(X.shape)
    x_def = X_def.flatten().reshape(-1, 1)
    x_bar_2d = _arbitrary_x_bar(rng, X)
    x_bar = x_bar_2d.flatten().reshape(-1, 1)
    u = x_def - x_bar

    e_x = bending_energy_x(x_def, H, theta0, ymI / l)
    e_u = bending_energy_u(u, x_bar, H, theta0, ymI / l)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = bending_gradient_x(x_def, H, theta0, ymI / l)
    g_u = bending_gradient_u(u, x_bar, H, theta0, ymI / l)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(bending_hessian_x(x_def, H, theta0, ymI / l))
    H_u = _hessian_to_dense(bending_hessian_u(u, x_bar, H, theta0, ymI / l))
    assert np.allclose(H_u, H_x, atol=TOL)


# --------------------------------------------------------------------------- #
# Flavor B: Discrete shells bending (3D)                                      #
# --------------------------------------------------------------------------- #
def _two_triangle_hinge_3d():
    """Two triangles sharing one edge forming a single dihedral in 3D."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ])
    # Dihedral connectivity: (i, j, k, l) where (i, j) is the hinge edge.
    D = np.array([[0, 1, 2, 3]], dtype=np.int64)
    theta0 = np.zeros((1, 1))
    ym_bending = np.ones((1, 1))
    he = np.ones((1, 1))
    le = np.ones((1, 3))
    return X, D, theta0, ym_bending, he, le


def test_discrete_shells_bending_u_arbitrary_offset_matches_x() -> None:
    rng = np.random.default_rng(80)
    X, D, theta0, ym_bending, he, le = _two_triangle_hinge_3d()
    X_def = X + 0.05 * rng.standard_normal(X.shape)
    x_bar = _arbitrary_x_bar(rng, X)
    u = X_def - x_bar

    e_x = discrete_shells_bending_energy_x(X_def, D, theta0, ym_bending, he, le)
    e_u = discrete_shells_bending_energy_u(u, x_bar, D, theta0, ym_bending, he, le)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = discrete_shells_bending_gradient_x(X_def, D, theta0, ym_bending, he, le)
    g_u = discrete_shells_bending_gradient_u(u, x_bar, D, theta0, ym_bending, he, le)
    assert np.allclose(g_u, g_x, atol=TOL)

    H_x = _hessian_to_dense(discrete_shells_bending_hessian_x(X_def, D, theta0, ym_bending, he, le))
    H_u = _hessian_to_dense(discrete_shells_bending_hessian_u(u, x_bar, D, theta0, ym_bending, he, le))
    assert np.allclose(H_u, H_x, atol=TOL)


# --------------------------------------------------------------------------- #
# Flavor B: Kinetic (BE and BDF2)                                             #
# --------------------------------------------------------------------------- #
def _spd_mass(rng: np.random.Generator, n: int) -> sps.csc_matrix:
    A = rng.standard_normal((n, n))
    return sps.csc_matrix(A.T @ A + n * np.eye(n))


def test_kinetic_be_u_arbitrary_offset_matches_x() -> None:
    rng = np.random.default_rng(90)
    n = 6
    M = _spd_mass(rng, n)
    x_curr = rng.standard_normal((n, 1))
    x_prev = rng.standard_normal((n, 1))
    h = 0.02
    x = rng.standard_normal((n, 1))
    x_bar = rng.standard_normal((n, 1))
    u = x - x_bar

    e_x = kinetic_energy_be(x, x_curr, x_prev, M, h)
    e_u = kinetic_energy_be_u(u, x_curr, x_prev, M, h, x_bar)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = kinetic_gradient_be(x, x_curr, x_prev, M, h)
    g_u = kinetic_gradient_be_u(u, x_curr, x_prev, M, h, x_bar)
    assert np.allclose(np.asarray(g_u), np.asarray(g_x), atol=TOL)


def test_kinetic_bdf2_u_arbitrary_offset_matches_x() -> None:
    rng = np.random.default_rng(91)
    n = 5
    M = _spd_mass(rng, n)
    x_curr = rng.standard_normal((n, 1))
    x_prev = rng.standard_normal((n, 1))
    x_prev2 = rng.standard_normal((n, 1))
    x_prev3 = rng.standard_normal((n, 1))
    h = 0.02
    x = rng.standard_normal((n, 1))
    x_bar = rng.standard_normal((n, 1))
    u = x - x_bar

    e_x = kinetic_energy_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h)
    e_u = kinetic_energy_bdf2_u(u, x_curr, x_prev, x_prev2, x_prev3, M, h, x_bar)
    assert e_u == pytest.approx(e_x, abs=TOL, rel=TOL)

    g_x = kinetic_gradient_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h)
    g_u = kinetic_gradient_bdf2_u(u, x_curr, x_prev, x_prev2, x_prev3, M, h, x_bar)
    assert np.allclose(np.asarray(g_u), np.asarray(g_x), atol=TOL)


if __name__ == "__main__":
    for d in (2, 3):
        test_macklin_mueller_neo_hookean_u_zero_offset_matches_x(d)
        test_macklin_mueller_neo_hookean_u_rest_offset_matches_x(d)
        test_macklin_mueller_neo_hookean_u_arbitrary_offset_matches_x(d)
        test_arap_u_zero_offset_matches_x(d)
        test_arap_u_arbitrary_offset_matches_x(d)
        test_linear_elasticity_u_arbitrary_offset_matches_x(d)
        test_fcr_u_arbitrary_offset_matches_x(d)
        test_emu_u_arbitrary_offset_matches_x(d)
        for m in MATERIALS:
            test_elastic_dispatcher_u_matches_x(m, d)
    test_elastic_dispatcher_u_unknown_material_raises()
    test_membrane_neo_hookean_u_arbitrary_offset_matches_x()
    test_bending_u_arbitrary_offset_matches_x()
    test_discrete_shells_bending_u_arbitrary_offset_matches_x()
    test_kinetic_be_u_arbitrary_offset_matches_x()
    test_kinetic_bdf2_u_arbitrary_offset_matches_x()
    print("All _u tier tests passed.")
