"""Smoke tests for the SimKit package.

These are intentionally minimal -- they exist to verify that the package is
installed correctly and importable. Add focused unit tests in sibling
``test_*.py`` files as functionality stabilizes.
"""

from __future__ import annotations

import math


def test_import_simkit() -> None:
    """SimKit imports cleanly from a fresh interpreter."""
    import simkit  # noqa: F401


def test_ympr_to_lame_matches_closed_form() -> None:
    """``ympr_to_lame`` produces the textbook Lame parameters."""
    from simkit import ympr_to_lame

    ym = 1.0e5
    pr = 0.45

    mu, lam = ympr_to_lame(ym, pr)

    expected_mu = ym / (2.0 * (1.0 + pr))
    expected_lam = ym * pr / ((1.0 + pr) * (1.0 - 2.0 * pr))

    assert math.isclose(mu, expected_mu, rel_tol=1e-12)
    assert math.isclose(lam, expected_lam, rel_tol=1e-12)
