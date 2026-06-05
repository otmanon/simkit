"""Quadratic (P2) Lagrange shape functions on simplices.

Shared low-level helper used by :func:`deformation_jacobian_p2`,
:func:`p2_massmatrix`, and :func:`p2_gravity_force` so that the P2 node ordering
is defined in exactly one place.

Node ordering (matches :func:`linear_to_quadratic_elements`)
------------------------------------------------------------
The first ``s`` nodes are the simplex corners (``s = 3`` for a triangle,
``s = 4`` for a tetrahedron). The remaining nodes are the edge midpoints, in the
same local-edge order as ``simkit.edges._SIMPLEX_EDGE_LOCAL``:

* triangle (6 nodes): corners ``0,1,2`` then midpoints of ``(0,1) (1,2) (2,0)``
* tet (10 nodes): corners ``0,1,2,3`` then midpoints of
  ``(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)``

Shape functions in barycentric coordinates ``L`` (``L.sum() == 1``)
-------------------------------------------------------------------
* corner ``i``:           ``N_i  = L_i (2 L_i - 1)``
* midpoint of edge (a,b): ``N    = 4 L_a L_b``

Gradients are returned with respect to the barycentric coordinates ``L`` (the
``L`` are treated as independent here; the not-all-independent constraint is
handled by callers when they multiply by ``dL/dxi``).
"""

import numpy as np

# Local edges (corner index pairs) within a single simplex, keyed by the number
# of corner vertices. Identical to ``simkit.edges._SIMPLEX_EDGE_LOCAL`` -- kept
# here too so the P2 node layout is self-documenting.
_SIMPLEX_EDGE_LOCAL = {
    3: [(0, 1), (1, 2), (2, 0)],
    4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


def p2_num_nodes(s: int) -> int:
    """Number of P2 nodes on a simplex with ``s`` corners (6 tri, 10 tet)."""
    return s + len(_SIMPLEX_EDGE_LOCAL[s])


def p2_shape_functions(L: np.ndarray, s: int):
    """Evaluate the P2 shape functions and their barycentric gradients.

    Written with explicit loops (no vectorization) for clarity.

    Parameters
    ----------
    L : np.ndarray (s,)
        Barycentric coordinates of a single evaluation point. ``s`` entries,
        one per simplex corner; they should sum to 1.
    s : int
        Number of simplex corners (3 for a triangle, 4 for a tetrahedron).

    Returns
    -------
    N : np.ndarray (n_nodes,)
        Shape-function values at the point, ordered corners-then-midpoints.
    dNdL : np.ndarray (n_nodes, s)
        Gradient of each shape function with respect to the barycentric
        coordinates ``L``.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    edges_local = _SIMPLEX_EDGE_LOCAL[s]
    n_nodes = s + len(edges_local)

    N = np.zeros(n_nodes)
    dNdL = np.zeros((n_nodes, s))

    # Corner nodes: N_i = L_i (2 L_i - 1), dN_i/dL_j = (4 L_i - 1) delta_ij.
    for i in range(s):
        N[i] = L[i] * (2.0 * L[i] - 1.0)
        dNdL[i, i] = 4.0 * L[i] - 1.0

    # Edge-midpoint nodes: N = 4 L_a L_b, dN/dL_a = 4 L_b, dN/dL_b = 4 L_a.
    for e, (a, b) in enumerate(edges_local):
        node = s + e
        N[node] = 4.0 * L[a] * L[b]
        dNdL[node, a] = 4.0 * L[b]
        dNdL[node, b] = 4.0 * L[a]

    return N, dNdL
