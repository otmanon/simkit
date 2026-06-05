"""Promote a linear (P1) simplicial mesh to a quadratic (P2) one.

Creates a new node at the midpoint of every unique mesh edge and rebuilds the
connectivity so each element indexes its corner nodes followed by its
edge-midpoint nodes. Midpoint nodes are *shared* between elements that share an
edge, so the result is a conforming P2 mesh.
"""

import numpy as np

# Local edges (corner index pairs) of one simplex, keyed by corner count. Same
# ordering as ``simkit.edges._SIMPLEX_EDGE_LOCAL`` and ``p2_shape_functions`` so
# the midpoint columns of ``T2`` line up with the P2 shape-function node order.
_SIMPLEX_EDGE_LOCAL = {
    3: [(0, 1), (1, 2), (2, 0)],
    4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


def linear_to_quadratic_elements(V: np.ndarray, T: np.ndarray):
    """Convert a P1 simplicial mesh into a P2 (midpoint-enriched) mesh.

    Written with explicit loops (no vectorization) for clarity.

    Parameters
    ----------
    V : np.ndarray (n, dim)
        Linear-element vertex positions (``dim`` is 2 or 3).
    T : np.ndarray (t, s)
        Linear connectivity: ``s = 3`` triangles or ``s = 4`` tetrahedra.

    Returns
    -------
    V2 : np.ndarray (n + n_edges, dim)
        Vertices of the quadratic mesh: the original corners followed by one new
        vertex per unique mesh edge, placed at that edge's midpoint.
    T2 : np.ndarray (t, n_nodes)
        Quadratic connectivity (``n_nodes = 6`` triangles, ``10`` tets). The
        first ``s`` columns are the corner nodes (copied from ``T``); the
        remaining columns are the shared edge-midpoint nodes, ordered by the
        local-edge convention above.

    Notes
    -----
    Because each new node sits exactly at an edge midpoint, the P2 *geometric*
    map from the reference simplex to a rest element is still affine -- the
    quadratic terms cancel. This keeps the reference-to-rest Jacobian constant
    per element (see :func:`deformation_jacobian_p2`).
    """
    V = np.asarray(V, dtype=float)
    T = np.asarray(T)
    n = V.shape[0]
    t, s = T.shape
    if s not in _SIMPLEX_EDGE_LOCAL:
        raise ValueError(f"expected 3 (tri) or 4 (tet) columns in T, got {s}")

    edges_local = _SIMPLEX_EDGE_LOCAL[s]
    n_nodes = s + len(edges_local)

    # Map each unique undirected edge (sorted corner pair) to its midpoint node
    # index. Midpoint indices start right after the original corner vertices.
    edge_to_node = {}
    new_vertices = []  # midpoint positions, appended after the original V rows

    T2 = np.zeros((t, n_nodes), dtype=np.int64)

    for e in range(t):
        # Corner nodes carry over unchanged into the first s columns.
        for c in range(s):
            T2[e, c] = T[e, c]

        # Edge-midpoint nodes fill the remaining columns. Look up (or create on
        # first sight) the shared node for each local edge.
        for k, (a, b) in enumerate(edges_local):
            va = int(T[e, a])
            vb = int(T[e, b])
            key = (va, vb) if va < vb else (vb, va)

            if key not in edge_to_node:
                node_idx = n + len(new_vertices)
                edge_to_node[key] = node_idx
                midpoint = 0.5 * (V[va] + V[vb])
                new_vertices.append(midpoint)

            T2[e, s + k] = edge_to_node[key]

    if new_vertices:
        V2 = np.vstack([V, np.array(new_vertices, dtype=float)])
    else:
        V2 = V.copy()

    return V2, T2
