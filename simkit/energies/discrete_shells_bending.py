
"""Discrete-shells bending energy (dihedral-angle representation).

This is the 3D sibling of :mod:`simkit.energies.bending_energy`. Both penalize
deviation of an angle from its rest value; the only difference is dimension and
the angle/operator used:

================  ============================  ========================
                  this module (3D)              bending_energy (2D)
================  ============================  ========================
ambient space     3D (positions ``n, 3``)       2D (positions ``n, 2``)
per-element angle  dihedral angle ``theta``      hinge angle ``theta``
connectivity      ``D`` (vertex quadruples)     ``H`` (vertex triples)
angle operator    dihedral Jacobian + wedge map hinge Jacobian/Hessian
================  ============================  ========================

Follows the standardized layout (see :mod:`simkit.energies.arap`), adapted to
shells. The per-element variable is the dihedral angle ``theta`` (representation
suffix ``_theta``), so the element tier is ``*_element_theta``.

Element tier (``*_element_theta``)
    Per-edge density and derivatives in ``theta``. Parameters ``ym_bending``
    (bending stiffness), ``he`` (hinge height) and ``le`` (edge length).

Global explicit tier (``*_x``)
    Takes vertex positions ``X`` and dihedral connectivity ``D``; computes the
    dihedral angles and applies the per-element angle Jacobian/Hessian and the
    dihedral wedge map.

Notes
-----
As with the 2D beam, shells have no separable quadrature weight; the geometric
factor ``w = ||le|| / he`` combines with ``ym_bending`` in the element tier.
Density per edge: ``ym_bending * w * (theta - theta0)^2``. The ``_x`` Hessian
assembles the true second derivative (Gauss-Newton plus the geometric
second-order angle term); pass ``psd=True`` to project per-edge blocks if a
definite matrix is required.
"""

import numpy as np
import scipy as sp

from ..dihedral_angles import (
    dihedral_angles,
    dihedral_angles_gradient_element,
    dihedral_angles_hessian_element,
)
from ..dihedral_wedge_map import dihedral_wedge_map
from ..psd_project import psd_project


def _edge_weight(he: np.ndarray, le: np.ndarray) -> np.ndarray:
    """Geometric bending weight ``w = ||le|| / he`` as a column vector.

    Parameters
    ----------
    he : np.ndarray (E, 1)
        Per-edge hinge heights.
    le : np.ndarray (E, dim) or (E, 1)
        Per-edge edge vectors (or precomputed lengths).

    Returns
    -------
    w : np.ndarray (E, 1)
        Geometric weight per edge.
    """
    le_len = np.linalg.norm(np.atleast_2d(le), axis=1).reshape(-1, 1)
    return le_len / np.asarray(he).reshape(-1, 1)


# --------------------------------------------------------------------------- #
# Element tier: dihedral angle (theta) representation                         #
# --------------------------------------------------------------------------- #
def discrete_shells_bending_energy_element_theta(theta: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray) -> np.ndarray:
    """Per-edge discrete-shells bending energy density.

    Parameters
    ----------
    theta : np.ndarray (E, 1)
        Current dihedral angles.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.

    Returns
    -------
    psi : np.ndarray (E, 1)
        Per-edge energy densities.
    """
    w = _edge_weight(he, le)
    dtheta = theta - theta0
    return ym_bending * w * dtheta ** 2


def discrete_shells_bending_gradient_element_theta(theta: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray) -> np.ndarray:
    """Per-edge gradient of the density w.r.t. ``theta``.

    Parameters
    ----------
    theta : np.ndarray (E, 1)
        Current dihedral angles.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.

    Returns
    -------
    g : np.ndarray (E, 1)
        Per-edge gradient w.r.t. ``theta``.
    """
    w = _edge_weight(he, le)
    dtheta = theta - theta0
    return 2 * ym_bending * w * dtheta


def discrete_shells_bending_hessian_element_theta(theta: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray) -> np.ndarray:
    """Per-edge Hessian of the density w.r.t. ``theta``.

    Parameters
    ----------
    theta : np.ndarray (E, 1)
        Current dihedral angles (unused; constant in ``theta``).
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles (unused).
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.

    Returns
    -------
    h : np.ndarray (E, 1)
        Per-edge second derivative w.r.t. ``theta``.
    """
    w = _edge_weight(he, le)
    return 2 * ym_bending * w


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def discrete_shells_bending_energy_x(X: np.ndarray, D: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray) -> float:
    """Assembled discrete-shells bending energy at positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity (four vertices per interior edge).
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.

    Returns
    -------
    energy : float
        Total bending energy.
    """
    theta = dihedral_angles(X, D)
    psi = discrete_shells_bending_energy_element_theta(theta, theta0, ym_bending, he, le)
    return float(np.sum(psi))


def discrete_shells_bending_gradient_x(X: np.ndarray, D: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray, W : sp.sparse.spmatrix = None) -> np.ndarray:
    """Assembled discrete-shells bending gradient w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.
    We : scipy.sparse matrix, optional
        The dihedral wedge map. If provided,a voids recomputation.
        
    Returns
    -------
    de_dx : np.ndarray (n*3, 1)
        Assembled gradient.
    """
    theta = dihedral_angles(X, D)
    de_dtheta = discrete_shells_bending_gradient_element_theta(theta, theta0, ym_bending, he, le)

    x0, x1, x2, x3 = X[D[:, 0]], X[D[:, 1]], X[D[:, 2]], X[D[:, 3]]
    dtheta_dx = dihedral_angles_gradient_element(x0, x1, x2, x3)

    
    if W is None:
        W = dihedral_wedge_map(D, X.shape[0])
        W = sp.sparse.kron(W, sp.sparse.identity(3))
    
    return W.T @ (dtheta_dx * de_dtheta).reshape(-1, 1)


def discrete_shells_bending_hessian_x(X: np.ndarray, D: np.ndarray,
                                      theta0: np.ndarray,
                                      ym_bending: np.ndarray, 
                                      he: np.ndarray, 
                                      le: np.ndarray,
                                      psd: bool = False, W:sp.sparse.spmatrix = None) -> sp.sparse.spmatrix:
    """Assembled discrete-shells bending Hessian w.r.t. positions ``X``.

    Parameters
    ----------
    X : np.ndarray (n, 3)
        Vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.
    psd : bool, optional
        If ``True``, project each per-edge block to PSD before assembly.
        Default ``False`` (true Hessian). The full Hessian includes the
        indefinite second-order angle term, so enable this when a definite
        matrix is required.
    W : scipy.sparse matrix, optional
        The dihedral wedge map. If provided,a voids recomputation.

    Returns
    -------
    H : scipy.sparse matrix (n*3, n*3)
        Assembled Hessian.
    """
    theta = dihedral_angles(X.reshape(-1, 3), D)
    x0, x1, x2, x3 = X[D[:, 0]], X[D[:, 1]], X[D[:, 2]], X[D[:, 3]]

    dtheta_dx = dihedral_angles_gradient_element(x0, x1, x2, x3)
    d2theta_dx2 = dihedral_angles_hessian_element(x0, x1, x2, x3)

    de_dtheta = discrete_shells_bending_gradient_element_theta(theta, theta0, ym_bending, he, le)
    d2e_dtheta2 = discrete_shells_bending_hessian_element_theta(theta, theta0, ym_bending, he, le)

    term_1 = de_dtheta[:, :, None] * d2theta_dx2
    term_2 = d2e_dtheta2[:, :, None] * (dtheta_dx[:, :, None] @ dtheta_dx[:, None, :])
    Q = term_1 + term_2

    if psd:
        Q = psd_project(Q)

    Q2 = sp.sparse.block_diag(Q)
    
    
    if W is None:
        W = dihedral_wedge_map(D, X.shape[0])
        W = sp.sparse.kron(W, sp.sparse.identity(3))
    return W.T @ Q2 @ W


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def discrete_shells_bending_energy_u(u: np.ndarray, x_bar: np.ndarray, D: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray) -> float:
    """Assembled discrete-shells bending energy at displacement ``u`` from a reference ``x_bar``.

    Thin wrapper around :func:`discrete_shells_bending_energy_x`: dihedral
    angles are nonlinear in vertex positions, so no offset precompute helps.
    The wrapper exists for a uniform displacement entrypoint across the energy
    library. The reference ``x_bar`` is arbitrary (not required to be the rest
    pose).

    Parameters
    ----------
    u : np.ndarray (n, 3)
        Displacement from the reference configuration.
    x_bar : np.ndarray (n, 3)
        Reference vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.

    Returns
    -------
    energy : float
        Total bending energy.
    """
    return discrete_shells_bending_energy_x(x_bar + u, D, theta0, ym_bending, he, le)


def discrete_shells_bending_gradient_u(u: np.ndarray, x_bar: np.ndarray, D: np.ndarray, theta0: np.ndarray, ym_bending: np.ndarray, he: np.ndarray, le: np.ndarray, W: sp.sparse.spmatrix = None) -> np.ndarray:
    """Assembled discrete-shells bending gradient w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, 3)
        Displacement from the reference configuration.
    x_bar : np.ndarray (n, 3)
        Reference vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.
    W : scipy.sparse matrix, optional
        The dihedral wedge map. If provided, avoids recomputation.

    Returns
    -------
    de_du : np.ndarray (n*3, 1)
        Assembled gradient.
    """
    return discrete_shells_bending_gradient_x(x_bar + u, D, theta0, ym_bending, he, le, W=W)


def discrete_shells_bending_hessian_u(u: np.ndarray, x_bar: np.ndarray, D: np.ndarray,
                                      theta0: np.ndarray,
                                      ym_bending: np.ndarray,
                                      he: np.ndarray,
                                      le: np.ndarray,
                                      psd: bool = False, W: sp.sparse.spmatrix = None) -> sp.sparse.spmatrix:
    """Assembled discrete-shells bending Hessian w.r.t. displacement ``u``.

    Parameters
    ----------
    u : np.ndarray (n, 3)
        Displacement from the reference configuration.
    x_bar : np.ndarray (n, 3)
        Reference vertex positions.
    D : np.ndarray (E, 4)
        Dihedral connectivity.
    theta0 : np.ndarray (E, 1)
        Rest dihedral angles.
    ym_bending : np.ndarray (E, 1)
        Per-edge bending stiffness.
    he : np.ndarray (E, 1)
        Per-edge hinge height.
    le : np.ndarray (E, dim)
        Per-edge edge vectors.
    psd : bool, optional
        If ``True``, project each per-edge block to PSD before assembly.
    W : scipy.sparse matrix, optional
        The dihedral wedge map. If provided, avoids recomputation.

    Returns
    -------
    H : scipy.sparse matrix (n*3, n*3)
        Assembled Hessian.
    """
    return discrete_shells_bending_hessian_x(x_bar + u, D, theta0, ym_bending, he, le, psd=psd, W=W)