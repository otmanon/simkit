"""Tests for ``simkit.polyscope``.

The viewers all call ``ps.init()`` and open an interactive window, so they
cannot be executed in CI. What *can* be checked -- and what actually breaks in
practice -- is that the package imports, exports every documented viewer, and
that each signature still matches how the demos call it. A rename in the
numerics layer surfaces here as an import error.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("polyscope")  # pip install 'simkit[viz]'

import simkit.polyscope as viz  # noqa: E402


VIEWERS = [
    "view_animation",
    "view_clusters",
    "view_cubature",
    "view_displacement_modes",
    "view_sample_points",
    "view_scalar_modes",
]


def test_all_viewers_are_exported():
    for name in VIEWERS:
        assert callable(getattr(viz, name, None)), name


def test_viewers_are_reachable_from_the_top_level_package():
    import simkit

    assert simkit.polyscope is viz


@pytest.mark.parametrize("name", VIEWERS)
def test_viewer_takes_geometry_first(name):
    """Every viewer follows the same ``(X, T, ...)`` calling convention."""
    params = list(inspect.signature(getattr(viz, name)).parameters)
    assert params[:2] == ["X", "T"], f"{name} takes {params[:2]}"


@pytest.mark.parametrize("name", VIEWERS)
def test_viewer_optional_arguments_have_defaults(name):
    """Anything past the geometry and the field being viewed must be optional,
    so demos can call the viewer with just the data."""
    sig = inspect.signature(getattr(viz, name))
    params = list(sig.parameters.values())
    required = [p for p in params if p.default is inspect.Parameter.empty]
    assert len(required) <= 3, (
        f"{name} requires {[p.name for p in required]}; viewers should take at "
        "most (X, T, field) positionally"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("view_animation", "U"),
        ("view_clusters", "l"),
        ("view_cubature", "cI"),
        ("view_displacement_modes", "W"),
        ("view_sample_points", "sample_points"),
    ],
)
def test_viewer_field_argument_name_is_stable(name, expected):
    """These names appear in the demos and tutorials as keyword arguments."""
    params = list(inspect.signature(getattr(viz, name)).parameters)
    assert expected in params, f"{name} lost its '{expected}' argument"


def test_importing_polyscope_subpackage_does_not_open_a_window():
    """Import must stay side-effect free: ``ps.init()`` belongs in the call."""
    import importlib

    mod = importlib.import_module("simkit.polyscope.view_cubature")
    src = inspect.getsource(mod)
    init_lines = [
        line for line in src.splitlines()
        if "ps.init()" in line and not line.startswith(("def ", "class "))
    ]
    # Every ps.init() must be indented, i.e. inside a function body.
    assert all(line.startswith((" ", "\t")) for line in init_lines), (
        "ps.init() at module scope would open a window on import"
    )
