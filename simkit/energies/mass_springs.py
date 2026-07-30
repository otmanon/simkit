
"""Mass-spring elastic energy.

Follows the standardized layout (see :mod:`simkit.energies.arap`), adapted to
springs. The per-element variable is the edge displacement ``d`` (representation
suffix ``_d``), so the element tier is ``*_element_d`` rather than ``*_element_F``.

Element tier (``*_element_d``)
    Per-edge density and derivatives in the edge-displacement ``d``. Material
    params ``ym`` (stiffness) and ``l0`` (rest length) only; no quadrature
    weight ``vol``.

Global explicit tiers
    ``*_x`` takes vertex positions ``x`` and edge list ``E``, building the
    edge-difference operator internally; ``*_z`` takes a prebuilt operator ``J``
    and reduced coordinates ``z``. Both apply ``vol``. The ``_z`` tier is the
    one to call in a hot loop, since ``J`` is built once and reused.

Length-space helpers (``*_l``)
    Operate directly on edge lengths, for rest-length / sensitivity work.

Notes
-----
Density per edge: ``0.5 * ym * (||d|| - l0)^2``, i.e. ``ym`` is the spring
stiffness outright, with no rest-length normalization. The Hessian is the
standard spring stiffness block and may be indefinite under compression, so the
``_x`` / ``_z`` tiers expose a ``psd`` flag.
"""

from typing import Optional

import numpy as np
import scipy as sp

from ..edge_displacement_jacobian import edge_displacement_jacobian
from ..psd_project import psd_project


# --------------------------------------------------------------------------- #
# Element tier: edge displacement (d) representation                          #
# --------------------------------------------------------------------------- #
def mass_springs_energy_element_d(d: np.ndarray, ym: np.ndarray, l0: np.ndarray) -> np.ndarray:
    """Per-edge mass-spring energy density.

    Parameters
    ----------
    d : np.ndarray (num_edges, dim)
        Per-edge displacement vectors.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    psi : np.ndarray (num_edges, 1)
        Per-edge energy densities. No quadrature weighting applied.
    """
    ym = np.asarray(ym).reshape(-1, 1)
    l0 = np.asarray(l0).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    return 0.5 * ym * (l - l0) ** 2


def mass_springs_gradient_element_d(d: np.ndarray, ym: np.ndarray, l0: np.ndarray) -> np.ndarray:
    """Per-edge gradient of the density w.r.t. the displacement ``d``.

    Parameters
    ----------
    d : np.ndarray (num_edges, dim)
        Per-edge displacement vectors.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    g : np.ndarray (num_edges, dim)
        Per-edge gradient blocks. No quadrature weighting applied.
    """
    ym = np.asarray(ym).reshape(-1, 1)
    l0 = np.asarray(l0).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    return ym * (l - l0) * d / l


def mass_springs_hessian_element_d(d: np.ndarray, ym: np.ndarray, l0: np.ndarray) -> np.ndarray:
    """Per-edge Hessian of the density w.r.t. the displacement ``d``.

    Parameters
    ----------
    d : np.ndarray (num_edges, dim)
        Per-edge displacement vectors.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    H : np.ndarray (num_edges, dim, dim)
        Per-edge Hessian blocks. No quadrature weighting applied; not
        PSD-projected.
    """
    ym = np.asarray(ym).reshape(-1, 1)
    l0 = np.asarray(l0).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    l3 = l ** 3
    ddT = d[:, :, None] @ d[:, None, :]
    I = np.eye(d.shape[1])[None, :, :]
    term = I - l0[:, None, :] * (I / l[:, None, :] - ddT / l3[:, None, :])
    return ym[:, None, :] * term


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def _edge_displacements(x: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Per-edge displacement vectors ``x_i - x_j`` for each edge ``(i, j)``.

    Sign convention matches :func:`simkit.edge_displacement_jacobian`, so that
    ``d == edge_displacement_jacobian(x, E) @ x``.

    Parameters
    ----------
    x : np.ndarray (num_vertices, dim)
        Vertex positions.
    E : np.ndarray (num_edges, 2)
        Edges as pairs of vertex indices ``(i, j)``.

    Returns
    -------
    d : np.ndarray (num_edges, dim)
        Per-edge displacement vectors.
    """
    return x[E[:, 0]] - x[E[:, 1]]


def mass_springs_energy_x(x: np.ndarray, E: np.ndarray, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray) -> float:
    """Assembled mass-spring energy at vertex positions ``x``.

    Parameters
    ----------
    x : np.ndarray (num_vertices, dim)
        Vertex positions.
    E : np.ndarray (num_edges, 2)
        Edges as pairs of vertex indices ``(i, j)``.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    energy : float
        Total mass-spring energy.
    """
    d = _edge_displacements(x, E)
    psi = mass_springs_energy_element_d(d, ym, l0)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


def mass_springs_gradient_x(x: np.ndarray, E: np.ndarray, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray) -> np.ndarray:
    """Assembled mass-spring gradient w.r.t. vertex positions ``x``.

    Parameters
    ----------
    x : np.ndarray (num_vertices, dim)
        Vertex positions.
    E : np.ndarray (num_edges, 2)
        Edges as pairs of vertex indices ``(i, j)``.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    g : np.ndarray (num_vertices*dim, 1)
        Assembled gradient, flattened row-major to match ``x.reshape(-1, 1)``.
    """
    d = _edge_displacements(x, E)
    dedd = mass_springs_gradient_element_d(d, ym, l0) * np.asarray(vol).reshape(-1, 1)
    g = np.zeros_like(x, dtype=float)
    np.add.at(g, E[:, 0], dedd)
    np.add.at(g, E[:, 1], -dedd)
    return g.reshape(-1, 1)


def mass_springs_hessian_x(x: np.ndarray, E: np.ndarray, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled mass-spring Hessian w.r.t. vertex positions ``x``.

    Parameters
    ----------
    x : np.ndarray (num_vertices, dim)
        Vertex positions.
    E : np.ndarray (num_edges, 2)
        Edges as pairs of vertex indices ``(i, j)``.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.
    psd : bool, optional
        If ``True`` (default), project each per-edge block to PSD before assembly.

    Returns
    -------
    H : scipy.sparse matrix (num_vertices*dim, num_vertices*dim)
        Assembled Hessian, ordered to match ``x.reshape(-1, 1)``.
    """
    dim = x.shape[1]
    d = _edge_displacements(x, E)
    He = mass_springs_hessian_element_d(d, ym, l0) * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    Q = sp.sparse.block_diag(He)
    J = sp.sparse.kron(edge_displacement_jacobian(x, E), sp.sparse.identity(dim))
    return J.transpose() @ Q @ J


def mass_springs_energy_z(z: np.ndarray, J: sp.sparse.spmatrix, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray, Jx0: Optional[np.ndarray] = None) -> float:
    """Assembled mass-spring energy in reduced coordinates ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    J : scipy.sparse matrix (num_edges*dim, r)
        Edge-difference operator mapping ``z`` to stacked displacements.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.
    Jx0 : np.ndarray (num_edges*dim, 1), optional
        Constant displacement offset added before evaluation.

    Returns
    -------
    energy : float
        Total mass-spring energy.
    """
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    psi = mass_springs_energy_element_d(d, ym, l0)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


def mass_springs_gradient_z(z: np.ndarray, J: sp.sparse.spmatrix, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray, Jx0: Optional[np.ndarray] = None) -> np.ndarray:
    """Assembled mass-spring gradient in reduced coordinates ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    J : scipy.sparse matrix (num_edges*dim, r)
        Edge-difference operator.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.
    Jx0 : np.ndarray (num_edges*dim, 1), optional
        Constant displacement offset.

    Returns
    -------
    g : np.ndarray (r, 1)
        Assembled gradient.
    """
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    dedd = mass_springs_gradient_element_d(d, ym, l0) * np.asarray(vol).reshape(-1, 1)
    return J.transpose() @ dedd.reshape(-1, 1)


def mass_springs_hessian_z(z: np.ndarray, J: sp.sparse.spmatrix, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray, Jx0: Optional[np.ndarray] = None, psd: bool = True) -> sp.sparse.spmatrix:
    """Assembled mass-spring Hessian in reduced coordinates ``z``.

    Parameters
    ----------
    z : np.ndarray (r, 1)
        Reduced coordinates.
    J : scipy.sparse matrix (num_edges*dim, r)
        Edge-difference operator.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length.
    Jx0 : np.ndarray (num_edges*dim, 1), optional
        Constant displacement offset.
    psd : bool, optional
        If ``True`` (default), project each per-edge block to PSD before assembly.

    Returns
    -------
    H : scipy.sparse matrix (r, r)
        Assembled Hessian.
    """
    if Jx0 is not None:
        d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
    else:
        d = (J @ z).reshape(l0.shape[0], -1)
    He = mass_springs_hessian_element_d(d, ym, l0) * np.asarray(vol).reshape(-1, 1, 1)
    if psd:
        He = psd_project(He)
    Q = sp.sparse.block_diag(He)
    return J.transpose() @ Q @ J


# --------------------------------------------------------------------------- #
# Length-space helpers (l)                                                    #
# --------------------------------------------------------------------------- #
def mass_springs_energy_l(length: np.ndarray, ym: np.ndarray, vol: np.ndarray, length0: np.ndarray) -> float:
    """Mass-spring energy as a function of edge lengths.

    Parameters
    ----------
    length : np.ndarray (num_edges, 1)
        Current edge lengths.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    length0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    e : float
        Total mass-spring energy.
    """
    coeff = vol * ym
    e = 0.5 * np.sum(coeff * (length - length0) ** 2)
    return float(e)


def mass_springs_gradient_l(length: np.ndarray, ym: np.ndarray, vol: np.ndarray, length0: np.ndarray) -> np.ndarray:
    """Gradient of the length-space energy w.r.t. edge lengths.

    Parameters
    ----------
    length : np.ndarray (num_edges, dim)
        Current edge lengths.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    length0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    g : np.ndarray (num_edges, dim)
        Gradient w.r.t. lengths.
    """
    dim = length.shape[1]
    coeff = vol * ym
    g = coeff * (length - length0)
    return g.reshape(-1, dim)


def mass_springs_hessian_l(ym: np.ndarray, vol: np.ndarray, length0: np.ndarray) -> sp.sparse.spmatrix:
    """Hessian of the length-space energy w.r.t. edge lengths.

    Parameters
    ----------
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    length0 : np.ndarray (num_edges, 1)
        Per-edge rest length.

    Returns
    -------
    H : scipy.sparse matrix (num_edges, num_edges)
        Diagonal Hessian.
    """
    coeff = np.broadcast_to(vol * ym, np.shape(length0))
    return sp.sparse.diags(np.array(coeff).flatten(), 0)


def mass_springs_hessian_d_l0(d: np.ndarray, ym: np.ndarray, vol: np.ndarray, l0: np.ndarray) -> np.ndarray:
    """Mixed second derivative of the energy w.r.t. ``d`` and rest length ``l0``.

    Used for rest-length sensitivity / inverse-design.

    Parameters
    ----------
    d : np.ndarray (num_edges, dim)
        Per-edge displacement vectors.
    ym : np.ndarray (num_edges, 1)
        Per-edge stiffness.
    vol : np.ndarray (num_edges, 1)
        Per-edge quadrature weights.
    l0 : np.ndarray (num_edges, 1)
        Per-edge rest length. Unused: with the density ``0.5*ym*(||d||-l0)^2``
        the mixed derivative is ``-vol*ym*d/||d||``, independent of ``l0``. Kept
        in the signature to match the other element-tier entry points.

    Returns
    -------
    dg_dl0 : np.ndarray (num_edges, dim)
        Mixed derivative blocks.
    """
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    return -vol * ym * d / l