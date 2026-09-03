"""Guards on the shape of ``simkit``'s public API.

SimKit's convention is one public function per same-named module, re-exported
from ``simkit/__init__.py``. When that re-export is missing, the attribute
``simkit.<name>`` silently resolves to the *module* instead of the function, so
``simkit.triangle_areas(V, F)`` fails with ``'module' object is not callable``
only at the call site. These tests catch that at import time.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types

import pytest

import simkit


def _module_named_entry_points():
    """Yield ``(module_name, function)`` for every ``simkit/<name>.py`` that
    defines a public ``<name>``."""
    for info in pkgutil.iter_modules(simkit.__path__):
        if info.ispkg or info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"simkit.{info.name}")
        except ImportError:
            continue  # gated behind an optional extra; covered separately
        obj = getattr(mod, info.name, None)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            yield info.name, obj


def test_no_public_function_is_shadowed_by_its_module():
    """Every public function must be re-exported from ``simkit/__init__.py``.

    Skipped on a lean install: a name exported inside an optional bucket (say
    ``cluster_grouping_matrices``, behind ``[solvers]``) is legitimately
    unbound when its extra is missing, and then the module shadows it. The
    full-install CI job enforces the complete guarantee.
    """
    if simkit._missing:
        pytest.skip(
            "optional extras missing (%s); export completeness is checked on "
            "the full install" % ", ".join(sorted(simkit._missing))
        )

    shadowed = [
        name
        for name, fn in _module_named_entry_points()
        if isinstance(getattr(simkit, name, None), types.ModuleType)
    ]
    assert not shadowed, (
        "these resolve to a module, not a function -- add "
        "`from .<name> import <name>` to simkit/__init__.py: " + ", ".join(sorted(shadowed))
    )


def test_exported_names_are_the_module_level_definitions():
    """Where a name is exported, it must be the function from its own module."""
    mismatched = []
    for name, fn in _module_named_entry_points():
        exported = getattr(simkit, name, None)
        if callable(exported) and exported is not fn:
            mismatched.append(name)
    assert not mismatched, (
        "exported object differs from the same-named module's definition "
        "(duplicate implementation?): " + ", ".join(sorted(mismatched))
    )


def test_subpackages_are_reachable_from_the_top_level():
    """The base install is numpy + scipy, so these must always be present."""
    for name in ("energies", "integrators", "solvers", "filesystem"):
        assert isinstance(getattr(simkit, name, None), types.ModuleType), name


def test_version_is_reported():
    assert isinstance(simkit.__version__, str)
    assert simkit.__version__


# --------------------------------------------------------------------------- #
# membrane_deformation_jacobian is its own module, distinct from the
# volumetric operator.
# --------------------------------------------------------------------------- #
def test_membrane_jacobian_is_separate_from_the_volumetric_one():
    """``deformation_jacobian`` used to carry a broken copy of this function.

    The copy built the volumetric ``H`` and crashed on triangles-in-3D -- the
    only input it was documented for. The membrane operator now lives solely in
    ``simkit.membrane_deformation_jacobian``.
    """
    import numpy as np

    vol_mod = importlib.import_module("simkit.deformation_jacobian")
    assert not hasattr(vol_mod, "membrane_deformation_jacobian")

    mem_mod = importlib.import_module("simkit.membrane_deformation_jacobian")
    assert simkit.membrane_deformation_jacobian is mem_mod.membrane_deformation_jacobian


def test_membrane_jacobian_produces_a_3x2_deformation_gradient():
    """Membrane ``F`` is non-square: 3 rows (embedding) by 2 columns (surface)."""
    import numpy as np

    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    T = np.array([[0, 1, 2]])
    J = simkit.membrane_deformation_jacobian(X, T)

    assert J.shape[0] == 6, "6 = 3 x 2 entries per triangle"
    F = (J @ X.reshape(-1, 1)).reshape(-1, 3, 2)
    # The rest configuration maps the local frame to itself.
    assert np.allclose(F.transpose(0, 2, 1) @ F, np.eye(2), atol=1e-10)


def test_membrane_jacobian_feeds_the_membrane_energy():
    """End-to-end: zero at rest, positive under stretch."""
    import numpy as np
    from simkit.energies.membrane_neo_hookean import membrane_neo_hookean_energy_x

    X = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    T = np.array([[0, 1, 2]])
    J = simkit.membrane_deformation_jacobian(X, T)
    mu = np.ones((1, 1))
    lam = np.ones((1, 1))
    vol = np.ones((1, 1))

    assert np.isclose(membrane_neo_hookean_energy_x(X, J, mu, lam, vol), 0.0, atol=1e-12)
    stretched = X * np.array([2.0, 1.0, 1.0])
    assert membrane_neo_hookean_energy_x(stretched, J, mu, lam, vol) > 0.0
