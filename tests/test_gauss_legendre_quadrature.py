"""Tests for ``simkit.gauss_legendre_quadrature``.

The simplex rules are verified against the exact barycentric-monomial integral

    triangle:  ∫_T  L0^a L1^b L2^c       dA = 2A · a! b! c!       / (a+b+c+2)!
    tet:       ∫_K  L0^a L1^b L2^c L3^d  dV = 6V · a! b! c! d!    / (a+b+c+d+3)!
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from simkit.gauss_legendre_quadrature import gauss_legendre_quadrature
from simkit.volume import volume

TOL = 1e-12


def _skewed_simplex(dim: int):
    """A single non-degenerate, non-reference simplex."""
    if dim == 2:
        X = np.array([[0.2, 0.1], [1.3, 0.0], [0.4, 1.7]])
        T = np.array([[0, 1, 2]])
    elif dim == 3:
        X = np.array(
            [[0.1, 0.2, 0.0], [1.4, 0.1, 0.2], [0.3, 1.6, 0.1], [0.0, 0.2, 1.9]]
        )
        T = np.array([[0, 1, 2, 3]])
    else:
        raise ValueError(dim)
    return X, T


def _exact_bary_monomial_integral(exps, vol: float) -> float:
    """Analytic ∫ Π L_c^{exps_c} over a simplex of measure ``vol``."""
    s = len(exps)
    num = math.prod(math.factorial(int(a)) for a in exps)
    den = math.factorial(int(sum(exps)) + s - 1)
    # 2A/(.)! for triangles (s=3), 6V/(.)! for tets (s=4): the prefactor is
    # (s-1)! · vol because 2A = 2!·A and 6V = 3!·V.
    return math.factorial(s - 1) * vol * num / den


def _rule_integral(bary_e, weights_e, exps) -> float:
    """Quadrature estimate Σ_q w_q Π L_q,c^{exps_c} on one element."""
    total = 0.0
    for q in range(bary_e.shape[0]):
        term = weights_e[q]
        for c, a in enumerate(exps):
            term *= bary_e[q, c] ** a
        total += term
    return total


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_weights_partition_element_volume(dim: int, order: int) -> None:
    X, T = _skewed_simplex(dim)
    bary, weights = gauss_legendre_quadrature(X, T, order)

    s = dim + 1
    assert bary.shape == (T.shape[0], weights.shape[1], s)
    # Barycentric coordinates of every quadrature point sum to 1.
    assert np.allclose(bary.sum(axis=2), 1.0, atol=TOL)
    # Weights integrate the constant 1 -> the element measure.
    assert np.allclose(weights.sum(axis=1), volume(X, T).ravel(), atol=TOL)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_integrates_monomials_up_to_degree(dim: int, order: int) -> None:
    X, T = _skewed_simplex(dim)
    s = dim + 1
    bary, weights = gauss_legendre_quadrature(X, T, order)
    vol = float(volume(X, T).ravel()[0])

    # Every barycentric monomial of total degree <= order is integrated exactly.
    for exps in product(range(order + 1), repeat=s):
        if sum(exps) > order:
            continue
        approx = _rule_integral(bary[0], weights[0], exps)
        exact = _exact_bary_monomial_integral(exps, vol)
        assert approx == pytest.approx(exact, abs=1e-12, rel=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
def test_rule_is_not_exact_one_degree_higher(dim: int) -> None:
    # A degree-1 rule must NOT integrate every quadratic exactly (sanity check
    # that the rules really are order-limited rather than accidentally perfect).
    X, T = _skewed_simplex(dim)
    s = dim + 1
    bary, weights = gauss_legendre_quadrature(X, T, 1)
    vol = float(volume(X, T).ravel()[0])

    exps = [0] * s
    exps[0] = 2  # the monomial L0^2 (total degree 2)
    approx = _rule_integral(bary[0], weights[0], exps)
    exact = _exact_bary_monomial_integral(exps, vol)
    assert not np.isclose(approx, exact)


@pytest.mark.parametrize("dim", [2, 3])
def test_unsupported_order_raises(dim: int) -> None:
    X, T = _skewed_simplex(dim)
    with pytest.raises(ValueError):
        gauss_legendre_quadrature(X, T, 5)
