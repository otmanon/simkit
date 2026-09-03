"""Tests for the ``material=`` dispatcher in ``simkit.energies.elastic``.

Every material listed in ``_MATERIALS`` must be reachable through each tier
and must agree exactly with a direct call into its own module. The two
deliberately excluded materials (``emu``, ``membrane_neo_hookean``) must fail
loudly rather than silently producing wrong numbers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import simkit
from simkit.energies import elastic as E
from simkit.energies.elastic import _MATERIALS


# Materials whose modules mirror the dispatcher one-for-one. ARAP takes only
# ``mu``, so it is compared through the dispatcher's own signature handling.
DISPATCHED = [
    ("linear-elasticity", "linear_elasticity"),
    ("arap", "arap"),
    ("fcr", "fcr"),
    ("macklin-mueller-neo-hookean", "macklin_mueller_neo_hookean"),
    ("stvk", "stvk"),
    ("neo-hookean", "neo_hookean"),
    ("stable-neo-hookean", "stable_neo_hookean"),
]
ARAP_ONLY_MU = {"arap"}


@pytest.fixture(scope="module")
def mesh():
    """A small 2D two-triangle mesh with a mild random deformation."""
    rng = np.random.default_rng(0)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    T = np.array([[0, 1, 2], [1, 3, 2]])
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    t = T.shape[0]
    mu = np.full((t, 1), 1.5)
    lam = np.full((t, 1), 2.5)
    U = X + 0.08 * rng.standard_normal(X.shape)
    return dict(X=X, T=T, J=J, vol=vol, mu=mu, lam=lam, U=U,
                Jx_bar=J @ X.reshape(-1, 1), u=U - X)


def _direct(prefix, kind, tier):
    mod = importlib.import_module(f"simkit.energies.{prefix}")
    return getattr(mod, f"{prefix}_{kind}_{tier}")


def _args(prefix, m, tier):
    """Direct-call arguments, dropping ``lam`` for ARAP."""
    if prefix in ARAP_ONLY_MU:
        return (m["mu"],)
    return (m["mu"], m["lam"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_materials_registry_matches_the_dispatched_list():
    assert tuple(name for name, _ in DISPATCHED) == _MATERIALS


def test_every_dispatched_material_has_a_module():
    for _, prefix in DISPATCHED:
        importlib.import_module(f"simkit.energies.{prefix}")


# --------------------------------------------------------------------------- #
# Tier equivalence with direct calls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("material,prefix", DISPATCHED)
def test_element_F_tier_matches_direct_call(material, prefix, mesh):
    dim = mesh["X"].shape[1]
    F = (mesh["J"] @ mesh["U"].reshape(-1, 1)).reshape(-1, dim, dim)
    args = _args(prefix, mesh, "element_F")

    assert np.allclose(E.elastic_energy_element_F(F, mesh["mu"], mesh["lam"], material),
                       _direct(prefix, "energy", "element_F")(F, *args))
    assert np.allclose(E.elastic_gradient_element_F(F, mesh["mu"], mesh["lam"], material),
                       _direct(prefix, "gradient", "element_F")(F, *args))
    assert np.allclose(E.elastic_hessian_element_F(F, mesh["mu"], mesh["lam"], material, psd=False),
                       _direct(prefix, "hessian", "element_F")(F, *args))


@pytest.mark.parametrize("material,prefix", DISPATCHED)
def test_x_tier_matches_direct_call(material, prefix, mesh):
    m = mesh
    assert np.allclose(E.elastic_energy_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material),
                       _direct(prefix, "energy", "x")(m["U"], m["J"], *_args(prefix, m, "x"), m["vol"]))
    assert np.allclose(E.elastic_gradient_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material),
                       _direct(prefix, "gradient", "x")(m["U"], m["J"], *_args(prefix, m, "x"), m["vol"]))
    a = E.elastic_hessian_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material)
    b = _direct(prefix, "hessian", "x")(m["U"], m["J"], *_args(prefix, m, "x"), m["vol"])
    assert np.allclose(a.toarray(), b.toarray())


@pytest.mark.parametrize("material,prefix", DISPATCHED)
def test_u_tier_matches_direct_call(material, prefix, mesh):
    m = mesh
    args = _args(prefix, m, "u")
    assert np.allclose(E.elastic_energy_u(m["u"], m["J"], m["Jx_bar"], m["mu"], m["lam"], m["vol"], material),
                       _direct(prefix, "energy", "u")(m["u"], m["J"], m["Jx_bar"], *args, m["vol"]))
    assert np.allclose(E.elastic_gradient_u(m["u"], m["J"], m["Jx_bar"], m["mu"], m["lam"], m["vol"], material),
                       _direct(prefix, "gradient", "u")(m["u"], m["J"], m["Jx_bar"], *args, m["vol"]))
    a = E.elastic_hessian_u(m["u"], m["J"], m["Jx_bar"], m["mu"], m["lam"], m["vol"], material)
    b = _direct(prefix, "hessian", "u")(m["u"], m["J"], m["Jx_bar"], *args, m["vol"])
    assert np.allclose(a.toarray(), b.toarray())


@pytest.mark.parametrize("material,prefix", DISPATCHED)
def test_self_contained_tier_builds_J_and_vol(material, prefix, mesh):
    m = mesh
    got = E.elastic_energy(m["X"], m["T"], m["mu"], m["lam"], material, U=m["U"])
    want = E.elastic_energy_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material)
    assert np.allclose(got, want)


@pytest.mark.parametrize("material,_prefix", DISPATCHED)
def test_rest_state_is_a_stationary_point(material, _prefix, mesh):
    """At rest, the elastic gradient vanishes for every dispatched material."""
    m = mesh
    g = E.elastic_gradient_x(m["X"], m["J"], m["mu"], m["lam"], m["vol"], material)
    assert np.max(np.abs(g)) < 1e-8


# --------------------------------------------------------------------------- #
# Excluded materials must fail loudly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("material", ["emu", "membrane-neo-hookean", "not-a-material"])
def test_unknown_materials_raise(material, mesh):
    """``emu`` (fibre/activation) and membrane (non-square F) are documented
    exclusions -- they must raise, not silently compute something wrong."""
    m = mesh
    with pytest.raises(ValueError, match="Unknown material"):
        E.elastic_energy_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material)
    with pytest.raises(ValueError, match="Unknown material"):
        E.elastic_gradient_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material)
    with pytest.raises(ValueError, match="Unknown material"):
        E.elastic_hessian_x(m["U"], m["J"], m["mu"], m["lam"], m["vol"], material)


# --------------------------------------------------------------------------- #
# The _S tier is deliberately narrower
# --------------------------------------------------------------------------- #
def test_S_tier_supports_only_the_mixed_fem_materials():
    """Documented in the module docstring: ``_S`` needs a stretch-space energy,
    which only arap and macklin-mueller provide."""
    S = np.tile(np.array([1.0, 1.0, 0.0]), (2, 1))
    mu = np.full((2, 1), 1.0)
    lam = np.full((2, 1), 1.0)
    vol = np.full((2, 1), 0.5)

    for material in ("arap", "macklin-mueller-neo-hookean"):
        assert np.isfinite(E.elastic_energy_S(S, mu, lam, vol, material))

    for material in ("stvk", "neo-hookean", "fcr", "linear-elasticity"):
        with pytest.raises(ValueError, match="unsupported material type for S"):
            E.elastic_energy_S(S, mu, lam, vol, material)
