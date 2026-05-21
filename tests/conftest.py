"""Shared pytest setup for the simkit test suite.

``simkit/energies/__init__.py`` is currently mid-refactor and references symbols
from sibling modules (notably ``arap``) that have been renamed. To keep the
per-energy test files independent of that work-in-progress aggregation, we
register an empty stub for ``simkit.energies`` in ``sys.modules`` *before* the
tests touch it. With ``__path__`` set to the on-disk directory, individual
submodules (``simkit.energies.arap``, ``simkit.energies.neo_hookean``, ...) can
still be loaded directly without running the broken package ``__init__``.
"""

from __future__ import annotations

import os
import sys
import types


def _install_energies_stub() -> None:
    if "simkit.energies" in sys.modules:
        return

    import simkit  # ensures the top-level package finishes loading first

    energies_dir = os.path.join(os.path.dirname(simkit.__file__), "energies")
    if not os.path.isdir(energies_dir):
        return

    stub = types.ModuleType("simkit.energies")
    stub.__path__ = [energies_dir]
    stub.__package__ = "simkit.energies"
    sys.modules["simkit.energies"] = stub
    setattr(simkit, "energies", stub)


_install_energies_stub()
