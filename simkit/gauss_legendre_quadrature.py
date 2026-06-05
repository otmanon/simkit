"""Symmetric Gaussian quadrature rules for simplices (triangles and tets).

The classic Gauss-Legendre rule is a *tensor-product* construction (natural on
quads/hexes). On a simplex the standard choice is instead a **symmetric
quadrature rule** tabulated by degree of exactness; that is what this function
returns. The name is kept for the P2 workflow it serves.

For a requested polynomial ``order`` the returned rule integrates any polynomial
of total degree ``<= order`` exactly over the reference simplex.
"""

import numpy as np

from .volume import volume

# Reference rules as (barycentric points, weights) with the weights normalized
# to sum to 1 (i.e. weights w.r.t. the *unit-measure* reference simplex). The
# physical per-element weights are obtained by scaling with the element volume.
#
# Triangle rules (barycentric L0,L1,L2):
_TRI_RULES = {
    1: (  # exact to degree 1: centroid
        [(1 / 3, 1 / 3, 1 / 3)],
        [1.0],
    ),
    2: (  # exact to degree 2: 3-point rule
        [(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)],
        [1 / 3, 1 / 3, 1 / 3],
    ),
    3: (  # exact to degree 3: Strang 4-point rule (one negative weight)
        [
            (1 / 3, 1 / 3, 1 / 3),
            (3 / 5, 1 / 5, 1 / 5),
            (1 / 5, 3 / 5, 1 / 5),
            (1 / 5, 1 / 5, 3 / 5),
        ],
        [-9 / 16, 25 / 48, 25 / 48, 25 / 48],
    ),
    4: (  # exact to degree 4: Dunavant 6-point rule (all positive weights)
        [
            (0.108103018168070, 0.445948490915965, 0.445948490915965),
            (0.445948490915965, 0.108103018168070, 0.445948490915965),
            (0.445948490915965, 0.445948490915965, 0.108103018168070),
            (0.816847572980459, 0.091576213509771, 0.091576213509771),
            (0.091576213509771, 0.816847572980459, 0.091576213509771),
            (0.091576213509771, 0.091576213509771, 0.816847572980459),
        ],
        [
            0.223381589678011,
            0.223381589678011,
            0.223381589678011,
            0.109951743655322,
            0.109951743655322,
            0.109951743655322,
        ],
    ),
}

# Tetrahedron rules (barycentric L0,L1,L2,L3):
_TET_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0  # ~0.5854102
_TET_B = (5.0 - np.sqrt(5.0)) / 20.0        # ~0.1381966
_TET_RULES = {
    1: (  # exact to degree 1: centroid
        [(0.25, 0.25, 0.25, 0.25)],
        [1.0],
    ),
    2: (  # exact to degree 2: 4-point rule
        [
            (_TET_A, _TET_B, _TET_B, _TET_B),
            (_TET_B, _TET_A, _TET_B, _TET_B),
            (_TET_B, _TET_B, _TET_A, _TET_B),
            (_TET_B, _TET_B, _TET_B, _TET_A),
        ],
        [0.25, 0.25, 0.25, 0.25],
    ),
    3: (  # exact to degree 3: 5-point rule (one negative weight)
        [
            (0.25, 0.25, 0.25, 0.25),
            (0.5, 1 / 6, 1 / 6, 1 / 6),
            (1 / 6, 0.5, 1 / 6, 1 / 6),
            (1 / 6, 1 / 6, 0.5, 1 / 6),
            (1 / 6, 1 / 6, 1 / 6, 0.5),
        ],
        [-4 / 5, 9 / 20, 9 / 20, 9 / 20, 9 / 20],
    ),
    4: (  # exact to degree 4: Keast 11-point rule (one negative weight)
        [
            (0.25, 0.25, 0.25, 0.25),
            # orbit b: permutations of (11/14, 1/14, 1/14, 1/14)
            (0.785714285714286, 0.071428571428571, 0.071428571428571, 0.071428571428571),
            (0.071428571428571, 0.785714285714286, 0.071428571428571, 0.071428571428571),
            (0.071428571428571, 0.071428571428571, 0.785714285714286, 0.071428571428571),
            (0.071428571428571, 0.071428571428571, 0.071428571428571, 0.785714285714286),
            # orbit c: permutations of (a, a, b, b), a=0.3994035762, b=0.1005964238
            (0.399403576166799, 0.399403576166799, 0.100596423833201, 0.100596423833201),
            (0.399403576166799, 0.100596423833201, 0.399403576166799, 0.100596423833201),
            (0.399403576166799, 0.100596423833201, 0.100596423833201, 0.399403576166799),
            (0.100596423833201, 0.399403576166799, 0.399403576166799, 0.100596423833201),
            (0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799),
            (0.100596423833201, 0.100596423833201, 0.399403576166799, 0.399403576166799),
        ],
        [
            -0.078933333333333,
            0.045733333333333, 0.045733333333333, 0.045733333333333, 0.045733333333333,
            0.149333333333333, 0.149333333333333, 0.149333333333333,
            0.149333333333333, 0.149333333333333, 0.149333333333333,
        ],
    ),
}


def gauss_legendre_quadrature(V: np.ndarray, T: np.ndarray, order: int):
    """Per-element quadrature points (barycentric) and physical weights.

    Written with explicit loops (no vectorization) for clarity.

    Parameters
    ----------
    V : np.ndarray (n, dim)
        Rest vertex positions (``dim`` is 2 or 3).
    T : np.ndarray (t, s)
        Connectivity: ``s = 3`` triangles or ``s = 4`` tetrahedra.
    order : int
        Degree of the polynomial integrand to integrate exactly (1 linear,
        2 quadratic, 3 cubic, 4 quartic). Supported range is 1..4. Order 4 is
        what the consistent P2 mass matrix needs, since the product ``N_i N_j``
        of two quadratic shape functions is quartic.

    Returns
    -------
    bary : np.ndarray (t, n_quad, s)
        Barycentric coordinates of the quadrature points in the rest
        configuration. The same reference rule is broadcast to every element, so
        ``bary[e]`` is identical across ``e``.
    weights : np.ndarray (t, n_quad)
        Physical cubature weights. They are the reference weights scaled by the
        element measure, so ``weights[e].sum()`` equals the element's area/volume
        and ``Σ_q weights[e, q] f(x_q)`` approximates ``∫_element f dX``.

    Notes
    -----
    These weights are exactly what an energy expects as its per-quadrature-point
    ``vol``: stacking the deformation gradients to ``(t * n_quad, dim, dim)`` and
    passing ``weights.reshape(-1, 1)`` reproduces ``∫ ψ(F) dX`` as a sum.
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T)
    t, s = T.shape

    if s == 3:
        rules = _TRI_RULES
    elif s == 4:
        rules = _TET_RULES
    else:
        raise ValueError(f"expected 3 (tri) or 4 (tet) columns in T, got {s}")

    if order not in rules:
        raise ValueError(
            f"order {order} not supported for this simplex; available: "
            f"{sorted(rules.keys())}"
        )

    ref_points, ref_weights = rules[order]
    n_quad = len(ref_points)

    # Per-element measure (triangle area / tet volume), shape (t, 1).
    vol = volume(V, T).reshape(-1)

    bary = np.zeros((t, n_quad, s))
    weights = np.zeros((t, n_quad))

    for e in range(t):
        for q in range(n_quad):
            for c in range(s):
                bary[e, q, c] = ref_points[q][c]
            # Physical weight = reference weight (sums to 1) times element size.
            weights[e, q] = ref_weights[q] * vol[e]

    return bary, weights
