# # import numpy as np
# # import scipy as sp
# # from ..hinge_angles import hinge_angles
# # from ..hinge_jacobian import hinge_jacobian
# # from ..hinge_hessian import hinge_hessian_compact
# # from ..psd_project import psd_project


# # def bending_energy(x, H, theta0, ymI, l):
# #     """
# #     Bending energy for a beam.
# #     Parameters:
# #         x dim*n x 1  (np.ndarray): The positions of the nodes of the beam.
# #         H J x 3  (np.ndarray): The number of joints in the beam .
# #         theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
# #         ymI E x 1 (np.ndarray): The Young's modulus of the beam multiplied by the area moment of inertia (effectively the bending stiffness).
# #         l E x 1 (np.ndarray): The length of the beam.
# #     Returns:
# #         e float: The bending energy of the beam.
# #     """
# #     l = np.asarray(l).reshape(-1, 1)
# #     ymI = np.asarray(ymI).reshape(-1, 1)
# #     coeff = ymI / l
# #     theta = hinge_angles(x.reshape(-1, 2), H)
# #     dtheta = theta - theta0
# #     return 0.5 * np.sum( coeff * (dtheta**2))


# # def bending_gradient(x, H, theta0, ymI, l):
# #     """ Bending gradient for a beam.
# #     Parameters:
# #         x dim*n x 1 (np.ndarray): The positions of the nodes of the beam.
# #         H J x 3 (np.ndarray): The number of joints in the beam .
# #         theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
# #         ymI E x 1 (np.ndarray): The Young's modulus of the beam multiplied by the area moment of inertia (effectively the bending stiffness).
# #         l E x 1 (np.ndarray): The length of the beam.
# #         I E x 1 (np.ndarray): The area moment of inertia of the beam (width * thickness ** 3 / 12)
# #     Returns:
# #         g dim*n x 1 (np.ndarray): The bending gradient of the beam.
# #     """
# #     l = np.asarray(l).reshape(-1, 1)
# #     ymI = np.asarray(ymI).reshape(-1, 1)
# #     coeff = ymI / l
# #     theta = hinge_angles(x.reshape(-1, 2), H)
# #     dtheta = theta - theta0
# #     denergy_dtheta = coeff * dtheta
# #     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
# #     de = dtheta_dx.T @ denergy_dtheta
# #     return de.reshape(-1, 1)


# # def bending_hessian(x, H, theta0, ymI, l):
# #     """ Bending hessian for a beam.
# #     Parameters:
# #         x dim*n x 1 (np.ndarray): The positions of the nodes of the beam.
# #         H J x 3 (np.ndarray): The number of joints in the beam .
# #         theta0 J x 1 (np.ndarray): The initial hinge angles of the beam.
# #         ym E x 1 (np.ndarray): The Young's modulus of the beam.
# #         l E x 1 (np.ndarray): The length of the beam.
# #         I E x 1 (np.ndarray): The area moment of inertia of the beam (width * thickness ** 3 / 12)
# #     Returns:
# #         H dim*n x dim*n (sp.sparse.csc_matrix): The bending hessian of the beam.
# #     """
# #     l = np.asarray(l).reshape(-1, 1)
# #     ymI = np.asarray(ymI).reshape(-1, 1)
# #     coeff = ymI / l
# #     d2energy_dtheta2 = sp.sparse.diags(coeff.flatten())# * np.eye(dtheta.shape[0])
# #     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    
# #     term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)
        
# #     # need to dot third order term with derivative of d2theta_dx2.
# #     # Instead of building third order sparse tensor explicitly, just dot
    
# #     # don't even include this term because it can go indefinite. not worth the projection.
    
# #     # curvature = curvature_func(x.reshape(-1, 2), H)
# #     # dcurvature = curvature - curvature0
# #     # denergy_dcurvature =  dcurvature
# #     # d2curvature_dx2_vec = d2curvature_func(x.reshape(-1, 2), H)    
# #     # d2curvature_dx2_vec = psd_project(d2curvature_dx2_vec)
# #     # term_2_compact = (k.reshape(-1, 1, 1) * vol.reshape(-1, 1, 1)
# #     #                   * denergy_dcurvature[:, :, None]) * d2curvature_dx2_vec 

# #     # offset = np.array([0, 1, 0, 1, 0, 1])
# #     # cols   = np.repeat(H*2, 2, axis=1) + offset       # shape (|E|,6)
# #     # cols2  = np.repeat(cols[:, None, :], 6, axis=1)   # (|E|,6,6)
# #     # rows   = cols2.transpose(0, 2, 1) 

# #     # term_2 = sp.sparse.coo_matrix(
# #     #         (term_2_compact.ravel(), (rows.ravel(), cols2.ravel())),
# #     #         shape=(x.size, x.size))
    
# #     Q = term_1 #+ term_2 
# #     return Q.tocsc()


# # # def bending_energy(x, H, theta0, k):
# # #     theta = hinge_angles(x.reshape(-1, 2), H)
# # #     dtheta = theta - theta0
# # #     return 0.5 * np.sum( k * (dtheta**2))

# # # def bending_gradient(x, H, theta0, k):
# # #     theta = hinge_angles(x.reshape(-1, 2), H)
# # #     dtheta = theta - theta0
# # #     denergy_dtheta = k * dtheta
# # #     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
# # #     de = dtheta_dx.T @ denergy_dtheta
# # #     return de.reshape(-1, 1)


# # # def bending_hessian(x, H, theta0, k):
# # #     theta = hinge_angles(x.reshape(-1, 2), H)
# # #     dtheta = theta - theta0
# # #     denergy_dtheta = k * dtheta
# # #     d2energy_dtheta2 = k * np.eye(dtheta.shape[0])
# # #     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    
# # #     term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)
        
    
# # #     # need to dot third order term with derivative of d2theta_dx2.
# # #     # Instead of building third order sparse tensor explicitly, just dot
# # #     d2theta_dx2_vec = hinge_hessian_compact(x.reshape(-1, 2), H)    
# # #     term_2_compact = denergy_dtheta[:, :, None] * d2theta_dx2_vec 

# # #     offset = np.array([0, 1, 0, 1, 0, 1])
# # #     cols   = np.repeat(H*2, 2, axis=1) + offset       # shape (|E|,6)
# # #     cols2  = np.repeat(cols[:, None, :], 6, axis=1)   # (|E|,6,6)
# # #     rows   = cols2.transpose(0, 2, 1) 

# # #     term_2 = sp.sparse.coo_matrix(
# # #             (term_2_compact.ravel(), (rows.ravel(), cols2.ravel())),
# # #             shape=(x.size, x.size))
    
# # #     # for name, arr in {
# # #     #     "theta": theta,
# # #     #     "dtheta_dx": dtheta_dx,
# # #     #     "d2theta_dx2_vec": d2theta_dx2_vec
# # #     # }.items():
# # #     #     if has_nan_or_inf(arr):
# # #     #         raise ValueError(f"{name} contains NaN or inf")
            
# # #     Q = term_1 + term_2 
# # #     return Q

# """Bending energy for a 2D beam (hinge-angle representation).

# This is the 2D sibling of :mod:`simkit.energies.discrete_shells_bending`. Both
# penalize deviation of an angle from its rest value; the only difference is
# dimension and the angle/operator used:

# ================  ========================  ============================
#                   this module (2D)          discrete_shells_bending (3D)
# ================  ========================  ============================
# ambient space     2D (positions ``n, 2``)   3D (positions ``n, 3``)
# per-element angle  hinge angle ``theta``     dihedral angle ``theta``
# connectivity      ``H`` (vertex triples)    ``D`` (vertex quadruples)
# angle operator    hinge Jacobian/Hessian    dihedral Jacobian + wedge map
# ================  ========================  ============================

# Follows the standardized layout (see :mod:`simkit.energies.arap`). The
# per-element variable is the hinge angle ``theta`` (representation suffix
# ``_theta``), so the element tier is ``*_element_theta``.

# Element tier (``*_element_theta``)
#     Per-hinge density and derivatives in ``theta``. Parameters ``ymI`` (bending
#     stiffness) and ``l`` (segment length).

# Global explicit tier (``*_x``)
#     Takes vertex positions ``x`` and hinge connectivity ``H``; computes the
#     hinge angles and applies the hinge Jacobian / Hessian.

# Notes
# -----
# Unlike volumetric elasticity, bending has no separable quadrature weight: the
# coefficient ``ymI / l`` is the segment bending stiffness, so both ``ymI`` and
# ``l`` stay in the element tier. Density per hinge:
# ``0.5 * (ymI / l) * (theta - theta0)^2``.

# The ``_x`` Hessian assembles the true second derivative: the Gauss-Newton term
# ``dtheta_dx^T D dtheta_dx`` plus the geometric term
# ``denergy_dtheta * d2theta_dx2``. The geometric term can make the assembled
# matrix indefinite; pass ``psd=True`` to project per-hinge blocks if a definite
# matrix is required. The default is ``psd=False`` (true Hessian), consistent with
# the 3D module.
# """

# import numpy as np
# import scipy as sp

# from ..hinge_angles import hinge_angles
# from ..hinge_jacobian import hinge_jacobian
# from ..hinge_hessian import hinge_hessian_compact
# from ..psd_project import psd_project


# # --------------------------------------------------------------------------- #
# # Element tier: hinge angle (theta) representation                            #
# # --------------------------------------------------------------------------- #
# def bending_energy_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
#     """Per-hinge bending energy density.

#     Parameters
#     ----------
#     theta : np.ndarray (J, 1)
#         Current hinge angles.
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles.
#     ymI : np.ndarray (J, 1)
#         Bending stiffness (Young's modulus times area moment of inertia).
#     l : np.ndarray (J, 1)
#         Segment length.

#     Returns
#     -------
#     psi : np.ndarray (J, 1)
#         Per-hinge energy densities.
#     """
#     l = np.asarray(l).reshape(-1, 1)
#     ymI = np.asarray(ymI).reshape(-1, 1)
#     coeff = ymI / l
#     dtheta = theta - theta0
#     return 0.5 * coeff * (dtheta ** 2)


# def bending_gradient_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
#     """Per-hinge gradient of the density w.r.t. ``theta``.

#     Parameters
#     ----------
#     theta : np.ndarray (J, 1)
#         Current hinge angles.
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles.
#     ymI : np.ndarray (J, 1)
#         Bending stiffness.
#     l : np.ndarray (J, 1)
#         Segment length.

#     Returns
#     -------
#     g : np.ndarray (J, 1)
#         Per-hinge gradient w.r.t. ``theta``.
#     """
#     l = np.asarray(l).reshape(-1, 1)
#     ymI = np.asarray(ymI).reshape(-1, 1)
#     coeff = ymI / l
#     return coeff * (theta - theta0)


# def bending_hessian_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
#     """Per-hinge Hessian of the density w.r.t. ``theta``.

#     Parameters
#     ----------
#     theta : np.ndarray (J, 1)
#         Current hinge angles (unused; the Hessian is constant in ``theta``).
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles (unused).
#     ymI : np.ndarray (J, 1)
#         Bending stiffness.
#     l : np.ndarray (J, 1)
#         Segment length.

#     Returns
#     -------
#     h : np.ndarray (J, 1)
#         Per-hinge second derivative w.r.t. ``theta`` (scalar per hinge).
#     """
#     l = np.asarray(l).reshape(-1, 1)
#     ymI = np.asarray(ymI).reshape(-1, 1)
#     return ymI / l


# # --------------------------------------------------------------------------- #
# # Global explicit tier: position (x) variable                                 #
# # --------------------------------------------------------------------------- #
# def bending_energy_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> float:
#     """Assembled 2D bending energy at vertex positions ``x``.

#     Parameters
#     ----------
#     x : np.ndarray (dim*n, 1)
#         Flattened vertex positions (``dim == 2``).
#     H : np.ndarray (J, 3)
#         Hinge connectivity (vertex triples).
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles.
#     ymI : np.ndarray (J, 1)
#         Bending stiffness.
#     l : np.ndarray (J, 1)
#         Segment length.

#     Returns
#     -------
#     e : float
#         Total bending energy.
#     """
#     theta = hinge_angles(x.reshape(-1, 2), H)
#     psi = bending_energy_element_theta(theta, theta0, ymI, l)
#     return float(np.sum(psi))


# def bending_gradient_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
#     """Assembled 2D bending gradient w.r.t. positions ``x``.

#     Parameters
#     ----------
#     x : np.ndarray (dim*n, 1)
#         Flattened vertex positions (``dim == 2``).
#     H : np.ndarray (J, 3)
#         Hinge connectivity.
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles.
#     ymI : np.ndarray (J, 1)
#         Bending stiffness.
#     l : np.ndarray (J, 1)
#         Segment length.

#     Returns
#     -------
#     g : np.ndarray (dim*n, 1)
#         Assembled gradient.
#     """
#     theta = hinge_angles(x.reshape(-1, 2), H)
#     denergy_dtheta = bending_gradient_element_theta(theta, theta0, ymI, l)
#     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
#     return (dtheta_dx.T @ denergy_dtheta).reshape(-1, 1)


# def bending_hessian_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray, psd: bool = False) -> sp.sparse.spmatrix:
#     """Assembled 2D bending Hessian w.r.t. positions ``x``.

#     Assembles the true second derivative: the Gauss-Newton term
#     ``dtheta_dx^T D dtheta_dx`` plus the geometric term
#     ``denergy_dtheta * d2theta_dx2``.

#     Parameters
#     ----------
#     x : np.ndarray (dim*n, 1)
#         Flattened vertex positions (``dim == 2``).
#     H : np.ndarray (J, 3)
#         Hinge connectivity.
#     theta0 : np.ndarray (J, 1)
#         Rest hinge angles.
#     ymI : np.ndarray (J, 1)
#         Bending stiffness.
#     l : np.ndarray (J, 1)
#         Segment length.
#     psd : bool, optional
#         If ``True``, project the per-hinge ``(6, 6)`` blocks of the geometric
#         term to PSD before assembly. Default ``False`` (true Hessian).

#     Returns
#     -------
#     Q : scipy.sparse.csc_matrix (dim*n, dim*n)
#         Assembled Hessian.
#     """
#     theta = hinge_angles(x.reshape(-1, 2), H)

#     # Gauss-Newton term
#     h = bending_hessian_element_theta(theta, theta0, ymI, l)
#     d2energy_dtheta2 = sp.sparse.diags(h.flatten())
#     dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
#     term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)

#     # Geometric term: denergy_dtheta . d2theta_dx2 (per-hinge compact tensor)
#     denergy_dtheta = bending_gradient_element_theta(theta, theta0, ymI, l)
#     d2theta_dx2 = hinge_hessian_compact(x.reshape(-1, 2), H)
#     term_2_compact = denergy_dtheta[:, :, None] * d2theta_dx2
#     if psd:
#         term_2_compact = psd_project(term_2_compact)

#     # scatter per-hinge (6, 6) blocks to global (vertex triples, 2 dofs each)
#     offset = np.array([0, 1, 0, 1, 0, 1])
#     cols = np.repeat(H * 2, 2, axis=1) + offset       # (J, 6)
#     cols2 = np.repeat(cols[:, None, :], 6, axis=1)    # (J, 6, 6)
#     rows = cols2.transpose(0, 2, 1)
#     term_2 = sp.sparse.coo_matrix(
#         (term_2_compact.ravel(), (rows.ravel(), cols2.ravel())),
#         shape=(x.size, x.size),
#     )

#     Q = term_1 + term_2
#     return Q.tocsc()


"""Bending energy for a 2D beam (hinge-angle representation).

This is the 2D sibling of :mod:`simkit.energies.discrete_shells_bending`. Both
penalize deviation of an angle from its rest value; the only difference is
dimension and the angle/operator used:

================  ========================  ============================
                  this module (2D)          discrete_shells_bending (3D)
================  ========================  ============================
ambient space     2D (positions ``n, 2``)   3D (positions ``n, 3``)
per-element angle  hinge angle ``theta``     dihedral angle ``theta``
connectivity      ``H`` (vertex triples)    ``D`` (vertex quadruples)
angle operator    hinge Jacobian/Hessian    dihedral Jacobian + wedge map
================  ========================  ============================

Follows the standardized layout (see :mod:`simkit.energies.arap`). The
per-element variable is the hinge angle ``theta`` (representation suffix
``_theta``), so the element tier is ``*_element_theta``.

Element tier (``*_element_theta``)
    Per-hinge density and derivatives in ``theta``. Parameters ``ymI`` (bending
    stiffness) and ``l`` (segment length).

Global explicit tier (``*_x``)
    Takes vertex positions ``x`` and hinge connectivity ``H``; computes the
    hinge angles and applies the hinge Jacobian / Hessian.

Notes
-----
Unlike volumetric elasticity, bending has no separable quadrature weight: the
coefficient ``ymI / l`` is the segment bending stiffness, so both ``ymI`` and
``l`` stay in the element tier. Density per hinge:
``0.5 * (ymI / l) * (theta - theta0)^2``.

The ``_x`` Hessian assembles the true second derivative: the Gauss-Newton term
``dtheta_dx^T D dtheta_dx`` plus the geometric term
``denergy_dtheta * d2theta_dx2``. The geometric term can make the assembled
matrix indefinite; pass ``psd=True`` to project per-hinge blocks if a definite
matrix is required. The default is ``psd=False`` (true Hessian), consistent with
the 3D module.
"""

import numpy as np
import scipy as sp

from ..hinge_angles import hinge_angles
from ..hinge_jacobian import hinge_jacobian
from ..hinge_hessian import hinge_hessian_compact, hinge_hessian
from ..psd_project import psd_project


# --------------------------------------------------------------------------- #
# Element tier: hinge angle (theta) representation                            #
# --------------------------------------------------------------------------- #
def bending_energy_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Per-hinge bending energy density.

    Parameters
    ----------
    theta : np.ndarray (J, 1)
        Current hinge angles.
    theta0 : np.ndarray (J, 1)
        Rest hinge angles.
    ymI : np.ndarray (J, 1)
        Bending stiffness (Young's modulus times area moment of inertia).
    l : np.ndarray (J, 1)
        Segment length.

    Returns
    -------
    psi : np.ndarray (J, 1)
        Per-hinge energy densities.
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    coeff = ymI / l
    dtheta = theta - theta0
    return 0.5 * coeff * (dtheta ** 2)


def bending_gradient_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Per-hinge gradient of the density w.r.t. ``theta``.

    Parameters
    ----------
    theta : np.ndarray (J, 1)
        Current hinge angles.
    theta0 : np.ndarray (J, 1)
        Rest hinge angles.
    ymI : np.ndarray (J, 1)
        Bending stiffness.
    l : np.ndarray (J, 1)
        Segment length.

    Returns
    -------
    g : np.ndarray (J, 1)
        Per-hinge gradient w.r.t. ``theta``.
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    coeff = ymI / l
    return coeff * (theta - theta0)


def bending_hessian_element_theta(theta: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Per-hinge Hessian of the density w.r.t. ``theta``.

    Parameters
    ----------
    theta : np.ndarray (J, 1)
        Current hinge angles (unused; the Hessian is constant in ``theta``).
    theta0 : np.ndarray (J, 1)
        Rest hinge angles (unused).
    ymI : np.ndarray (J, 1)
        Bending stiffness.
    l : np.ndarray (J, 1)
        Segment length.

    Returns
    -------
    h : np.ndarray (J, 1)
        Per-hinge second derivative w.r.t. ``theta`` (scalar per hinge).
    """
    l = np.asarray(l).reshape(-1, 1)
    ymI = np.asarray(ymI).reshape(-1, 1)
    return ymI / l


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def bending_energy_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> float:
    """Assembled 2D bending energy at vertex positions ``x``.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1)
        Flattened vertex positions (``dim == 2``).
    H : np.ndarray (J, 3)
        Hinge connectivity (vertex triples).
    theta0 : np.ndarray (J, 1)
        Rest hinge angles.
    ymI : np.ndarray (J, 1)
        Bending stiffness.
    l : np.ndarray (J, 1)
        Segment length.

    Returns
    -------
    e : float
        Total bending energy.
    """
    theta = hinge_angles(x.reshape(-1, 2), H)
    psi = bending_energy_element_theta(theta, theta0, ymI, l)
    return float(np.sum(psi))


def bending_gradient_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Assembled 2D bending gradient w.r.t. positions ``x``.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1)
        Flattened vertex positions (``dim == 2``).
    H : np.ndarray (J, 3)
        Hinge connectivity.
    theta0 : np.ndarray (J, 1)
        Rest hinge angles.
    ymI : np.ndarray (J, 1)
        Bending stiffness.
    l : np.ndarray (J, 1)
        Segment length.

    Returns
    -------
    g : np.ndarray (dim*n, 1)
        Assembled gradient.
    """
    theta = hinge_angles(x.reshape(-1, 2), H)
    denergy_dtheta = bending_gradient_element_theta(theta, theta0, ymI, l)
    dtheta_dx = hinge_jacobian(x.reshape(-1, 2), H)
    return (dtheta_dx.T @ denergy_dtheta).reshape(-1, 1)


def bending_hessian_x(x: np.ndarray, H: np.ndarray, theta0: np.ndarray, ymI: np.ndarray, l: np.ndarray, psd: bool = False) -> sp.sparse.spmatrix:
    """Assembled 2D bending Hessian w.r.t. positions ``x``.

    Assembles the true second derivative: the Gauss-Newton term
    ``dtheta_dx^T D dtheta_dx`` plus the geometric term
    ``denergy_dtheta * d2theta_dx2``.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1)
        Flattened vertex positions (``dim == 2``).
    H : np.ndarray (J, 3)
        Hinge connectivity.
    theta0 : np.ndarray (J, 1)
        Rest hinge angles.
    ymI : np.ndarray (J, 1)
        Bending stiffness.
    l : np.ndarray (J, 1)
        Segment length.
    psd : bool, optional
        If ``True``, project the per-hinge ``(6, 6)`` blocks of the geometric
        term to PSD before assembly. Default ``False`` (true Hessian).

    Returns
    -------
    Q : scipy.sparse.csc_matrix (dim*n, dim*n)
        Assembled Hessian.
    """
    X = x.reshape(-1, 2)
    theta = hinge_angles(X, H)

    # Gauss-Newton term
    h = bending_hessian_element_theta(theta, theta0, ymI, l)
    d2energy_dtheta2 = sp.sparse.diags(h.flatten())
    dtheta_dx = hinge_jacobian(X, H)
    term_1 = dtheta_dx.T @ (d2energy_dtheta2 @ dtheta_dx)

    # Geometric term: denergy_dtheta . d2theta_dx2, scattered by hinge_hessian.
    denergy_dtheta = bending_gradient_element_theta(theta, theta0, ymI, l)
    blocks = denergy_dtheta[:, :, None] * hinge_hessian_compact(X, H)
    if psd:
        blocks = psd_project(blocks)
    term_2 = hinge_hessian(X, H, blocks)

    Q = term_1 + term_2
    return Q.tocsc()