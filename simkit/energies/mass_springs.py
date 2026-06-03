
# import numpy as np
# import scipy as sp

# def mass_springs_energy_x(x, E, mu, vol, l0):
#     """
#     Mass-springs energy given vertex positions.

#     Parameters
#     ----------
#     x : (num_vertices, dim) np.ndarray
#         Vertex positions.
#     E : (num_edges, 2) np.ndarray
#         Undirected edges as pairs of vertex indices (i, j).
#     mu : (num_edges, 1) np.ndarray
#         Spring stiffness parameter per edge.
#     vol : (num_edges, 1) np.ndarray
#         Volume/weight per edge.
#     l0 : (num_edges, 1) np.ndarray
#         Rest length per edge.
#     """
#     d = x[E[:, 1]] - x[E[:, 0]]
#     return mass_springs_energy_d(d, mu, vol, l0)

# def mass_springs_energy_z(z, J, mu, vol, l0, Jx0=None):
#     if Jx0 is not None:
#         d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
#     else:
#         d = (J @ z).reshape(l0.shape[0], -1)
#     e = mass_springs_energy_d(d, mu, vol, l0)
#     return e

# def mass_springs_gradient_dz(z, J, mu, vol, l0, Jx0=None):
#     if Jx0 is not None:
#         d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
#     else:
#         d = (J @ z).reshape(l0.shape[0], -1)
#     dedd= mass_springs_gradient_dd(d, mu, vol, l0)
#     g = J.T @ (dedd.reshape(-1, 1))
#     return g

# def mass_springs_hessian_d2z(z, J, mu, vol, l0, Jx0=None):
#     if Jx0 is not None:
#         d = (Jx0 + J @ z).reshape(l0.shape[0], -1)
#     else:
#         d = (J @ z).reshape(l0.shape[0], -1)
#     de2dd2 = mass_springs_hessian_dd(d, mu, vol, l0)
#     Q = sp.sparse.block_diag(de2dd2)
#     H = J.T @ Q @ J
#     return H



# def mass_springs_energy_d(d, ym, vol, l0):
#     """
#     Parameters
#     ----------
#     d : (num_edges, dim) np.ndarray
#         per edge displacements
#     ym : (num_edges, 1) np.ndarray
#         The mu parameter of the system.
#     vol : (num_edges, 1) np.ndarray
#         The volume of the springs.
#     l0 : (num_edges, 1) np.ndarray
#         The rest length of the springs.
#     """
#     l0 = np.asarray(l0).reshape(-1, 1)
#     ym = np.asarray(ym).reshape(-1, 1)
#     vol = np.asarray(vol).reshape(-1, 1)
    
#     l = np.linalg.norm(d, axis=1)[:, None]
#     coeff = vol * ym / (l0**2)
#     e = 0.5 * np.sum(coeff * (l - l0)**2)
#     e = e.reshape(-1, 1)
#     return e


# def mass_springs_gradient_dd(d, ym, vol, l0):
#     l0 = np.asarray(l0).reshape(-1, 1)
#     ym = np.asarray(ym).reshape(-1, 1)
#     vol = np.asarray(vol).reshape(-1, 1)

#     l = np.linalg.norm(d, axis=1)[:, None]
    
#     coeff = vol * ym / (l0**2)
#     g = coeff * (l - l0) * d / l
#     return g

# def mass_springs_hessian_dddl0(d, ym, vol, l0):
#     l0 = np.asarray(l0).reshape(-1, 1)
#     ym = np.asarray(ym).reshape(-1, 1)
#     vol = np.asarray(vol).reshape(-1, 1)

#     l = np.linalg.norm(d, axis=1)[:, None]
   
#     coeff = vol * ym * d / l 
#     # g = coeff * (l/ (l0**2) - (1/l0))
    
#     dg_dl0 = coeff * (1/l0**2 - 2*l/ (l0**3) )
    
    
#     # import simkit as sk
#     # energy_func = lambda l: mass_springs_gradient_dd(d, ym, vol, l).reshape(-1, 1)
#     # g_fd =sk.gradient_cfd(energy_func, l0, 1e-6)[:, 0, :]
    
#     # Q = sp.sparse.block_diag(dg_dl0[:, None, :])
    
#     # error = np.linalg.norm(Q - g_fd.T)
#     # print(error)
#     return dg_dl0

# def mass_springs_hessian_dd(d, ym, vol, l0):
#     l0 = np.asarray(l0).reshape(-1, 1)
#     ym = np.asarray(ym).reshape(-1, 1)
#     vol = np.asarray(vol).reshape(-1, 1)
#     l = np.linalg.norm(d, axis=1)[:, None]
#     l3 = l**3
#     ddT = d[ :, :, None] @ d[:, None, :]
#     I = np.eye(d.shape[1])[None, :, :]
#     term = I - l0[:, None, :] * (I / l[ :, None, :] - ddT / l3[ :, None, :])
    
#     coeff = vol * ym / (l0**2)
#     hess = coeff[:, None, :] * (term) 
#     H = hess 
#     return H


# def mass_springs_energy_l(length, ym, vol, length0):
#     """
#     Parameters
#     ----------
#     z : np.ndarray
#         The current state of the system.
#     mu : np.ndarray
#         The mu parameter of the system.
#     vol : np.ndarray
#         The volume of the springs.
#     length : np.ndarray
#         The rest length of the springs.
#     """
#     coeff = vol * ym / (length0**2)
#     e = 0.5 * np.sum(coeff * (length - length0)**2)
#     e = e.reshape(-1, 1)
#     return e


# def mass_springs_gradient_dl(length, ym, vol, length0):
#     dim = length.shape[1]
#     coeff = vol * ym / (length0**2)
#     g = coeff * (length - length0) 
#     g = g.reshape(-1, dim)    
#     return g


# def mass_springs_hessian_d2l(ym, vol, length0):
#     coeff = vol * ym / (length0**2)
#     H = sp.sparse.diags(np.array(coeff).flatten(), 0)    
#     return H




"""Mass-spring elastic energy.

Follows the standardized layout (see :mod:`simkit.energies.arap`), adapted to
springs. The per-element variable is the edge displacement ``d`` (representation
suffix ``_d``), so the element tier is ``*_element_d`` rather than ``*_element_F``.

Element tier (``*_element_d``)
    Per-edge density and derivatives in the edge-displacement ``d``. Material
    params ``ym`` (stiffness) and ``l0`` (rest length) only; no quadrature
    weight ``vol``.

Global explicit tiers
    ``*_x`` takes vertex positions ``x`` and edge list ``E``; ``*_z`` takes a
    prebuilt edge-difference operator ``J`` and reduced coordinates ``z``. Both
    apply ``vol``.

Length-space helpers (``*_l``)
    Operate directly on edge lengths, for rest-length / sensitivity work.

Notes
-----
Density per edge: ``0.5 * (ym / l0^2) * (||d|| - l0)^2``. The Hessian is the
standard spring stiffness block and may be indefinite under compression, so the
``_x`` / ``_z`` tiers expose a ``psd`` flag.
"""

from typing import Optional

import numpy as np
import scipy as sp

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
    coeff = ym / (l0 ** 2)
    return 0.5 * coeff * (l - l0) ** 2


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
    coeff = ym / (l0 ** 2)
    return coeff * (l - l0) * d / l


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
    coeff = ym / (l0 ** 2)
    return coeff[:, None, :] * term


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
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
    d = x[E[:, 1]] - x[E[:, 0]]
    psi = mass_springs_energy_element_d(d, ym, l0)
    return float((np.asarray(vol).reshape(-1, 1) * psi).sum())


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
    coeff = vol * ym / (length0 ** 2)
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
    coeff = vol * ym / (length0 ** 2)
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
    coeff = vol * ym / (length0 ** 2)
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
        Per-edge rest length.

    Returns
    -------
    dg_dl0 : np.ndarray (num_edges, dim)
        Mixed derivative blocks.
    """
    l0 = np.asarray(l0).reshape(-1, 1)
    ym = np.asarray(ym).reshape(-1, 1)
    vol = np.asarray(vol).reshape(-1, 1)
    l = np.linalg.norm(d, axis=1)[:, None]
    coeff = vol * ym * d / l
    return coeff * (1 / l0 ** 2 - 2 * l / (l0 ** 3))