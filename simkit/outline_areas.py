"""Signed areas of closed mesh outlines and their derivatives.

Each *outline* (or *chamber*) is a closed polyline given as an edge list
``E`` of shape ``(ne, 2)`` indexing into a shared 2D vertex array ``X``. The
signed area enclosed by an outline is the shoelace (Gauss) formula

.. math::

    A = \\tfrac{1}{2} \\sum_{(i, j) \\in E} \\big(x_i\\, y_j - x_j\\, y_i\\big),

which is positive for a counter-clockwise loop and negative for a clockwise
loop. Working with the *signed* area (rather than ``|A|``) keeps the measure a
smooth polynomial of the vertex positions, so its gradient and Hessian are
exact everywhere -- including the degenerate ``A = 0`` configuration where the
unsigned area is not differentiable.

The gradient and Hessian are returned in the *local* coordinate system of each
outline: only the vertices the outline actually touches are kept, ordered as
``np.unique(E)``, and their coordinates are interleaved ``[x0, y0, x1, y1,
...]``. Use :func:`outline_selection_matrices` to map global vertex positions
to (and derivatives back from) this local system.
"""

import numpy as np
import scipy.sparse as sp


def outline_selection_matrices(E_chambers, nv):
    """Selection matrices mapping global vertices to each outline's local ones.

    For every outline, builds a sparse matrix that picks out the rows of a
    global ``(nv, ...)`` array belonging to that outline, ordered to match the
    local indexing used by :func:`outline_areas_gradient` and
    :func:`outline_areas_hessian` (i.e. ``np.unique(E)`` order).

    Parameters
    ----------
    E_chambers : list of np.ndarray (ne, 2)
        One closed-outline edge list per chamber, indexing into the global
        vertex array.
    nv : int
        Total number of vertices in the global mesh.

    Returns
    -------
    S_list : list of scipy.sparse.csr_matrix (ne, nv)
        One selection matrix per outline. ``S @ X`` gathers the outline's
        vertices, and ``S.T @ g_local`` scatters a local quantity back to the
        global vertex layout.
    """
    S_list = []
    for E in E_chambers:
        verts, inv = np.unique(E.ravel(), return_inverse=True)
        ne = len(verts)
        S = sp.csr_matrix((np.ones(ne), (np.arange(ne), verts)), shape=(ne, nv))
        S_list.append(S)
    return S_list


def outline_areas(X, E_chambers):
    """Signed area enclosed by each closed mesh outline.

    Evaluates the shoelace formula per outline. The result is positive for a
    counter-clockwise loop, negative for a clockwise loop, and zero for a
    degenerate (self-cancelling) loop.

    Parameters
    ----------
    X : np.ndarray (nv, 2)
        Global 2D vertex positions.
    E_chambers : list of np.ndarray (ne, 2)
        One closed-outline edge list per chamber. Each edge ``(i, j)`` is
        oriented so that traversing the edges walks the loop in order.

    Returns
    -------
    areas : np.ndarray (n_chambers,)
        Signed area of each outline.
    """
    areas = np.zeros(len(E_chambers))
    for i, E in enumerate(E_chambers):
        xi, yi = X[E[:, 0], 0], X[E[:, 0], 1]
        xj, yj = X[E[:, 1], 0], X[E[:, 1], 1]
        areas[i] = 0.5 * np.sum(xi * yj - xj * yi)
    return areas


def outline_areas_gradient(X, E_chambers):
    """Gradient of each outline's signed area w.r.t. its local vertices.

    Differentiates the shoelace formula. Because the signed area is bilinear in
    the vertex coordinates, this gradient is exact and varies linearly with
    ``X``.

    Parameters
    ----------
    X : np.ndarray (nv, 2)
        Global 2D vertex positions.
    E_chambers : list of np.ndarray (ne, 2)
        One closed-outline edge list per chamber.

    Returns
    -------
    grads : list of np.ndarray (1, 2 * ne)
        One row-vector gradient per outline, where ``ne`` is the number of
        unique vertices in that outline. Entries are interleaved
        ``[dA/dx0, dA/dy0, dA/dx1, dA/dy1, ...]`` over the outline's vertices
        in ``np.unique(E)`` order (matching
        :func:`outline_selection_matrices`).
    """
    grads = []
    for E in E_chambers:
        verts, inv = np.unique(E.ravel(), return_inverse=True)
        ne = len(verts)
        local_i, local_j = inv[::2], inv[1::2]

        xi, yi = X[E[:, 0], 0], X[E[:, 0], 1]
        xj, yj = X[E[:, 1], 0], X[E[:, 1], 1]

        dAdx = np.zeros(ne)
        dAdy = np.zeros(ne)

        # d/dx_i (x_i y_j - x_j y_i) = y_j,  d/dx_j = -y_i
        np.add.at(dAdx, local_i,  0.5 * yj)
        np.add.at(dAdx, local_j, -0.5 * yi)
        # d/dy_i = -x_j,  d/dy_j = x_i
        np.add.at(dAdy, local_i, -0.5 * xj)
        np.add.at(dAdy, local_j,  0.5 * xi)

        # interleave into [dx0, dy0, dx1, dy1, ...]
        g = np.zeros(2 * ne)
        g[0::2] = dAdx
        g[1::2] = dAdy

        grads.append(g[np.newaxis, :])

    return grads


def outline_areas_hessian(X, E_chambers):
    """Hessian of each outline's signed area w.r.t. its local vertices.

    The signed area is bilinear in the vertex coordinates, so each Hessian is a
    constant, symmetric sparse coupling between the ``x`` of one endpoint and
    the ``y`` of the other (independent of ``X``).

    Parameters
    ----------
    X : np.ndarray (nv, 2)
        Global 2D vertex positions. Used only for shape information; the
        Hessian does not depend on the coordinate values.
    E_chambers : list of np.ndarray (ne, 2)
        One closed-outline edge list per chamber.

    Returns
    -------
    hessians : list of np.ndarray (2 * ne, 2 * ne)
        One symmetric Hessian per outline, in the same interleaved local
        ordering as :func:`outline_areas_gradient`.
    """
    hessians = []

    for E in E_chambers:
        verts, inv = np.unique(E.ravel(), return_inverse=True)
        ne = len(verts)
        local_i, local_j = inv[::2], inv[1::2]

        H = np.zeros((2 * ne, 2 * ne))

        ix_i = 2 * local_i
        iy_i = 2 * local_i + 1
        ix_j = 2 * local_j
        iy_j = 2 * local_j + 1

        # d^2 A / dx_i dy_j = 0.5,  d^2 A / dx_j dy_i = -0.5; fill one triangle
        np.add.at(H, (ix_i, iy_j),  0.5)
        np.add.at(H, (ix_j, iy_i), -0.5)

        H = H + H.T
        hessians.append(H)

    return hessians
