"""Discrete bending energy, unified across dimension (hinge/dihedral).

A single bending energy that works for both a 2D curve (folds at a **vertex**
hinge, triple connectivity) and a 3D surface (folds across an **edge** hinge,
quadruple connectivity). The two are the same construction one codimension apart
(see :mod:`simkit.dihedral_angles`); only the angle ``theta(x)``, its element
Jacobian/Hessian, and the gather map differ, and those are dispatched by
:mod:`simkit.dihedral_angles`.

The ambient dimension is **inferred from the data**, so there is no ``dim``
argument: the ``_x`` tier reads it from the hinge connectivity arity
(``C.shape[1] == 3`` -> 2D, ``== 4`` -> 3D) and reshapes the flattened
positions accordingly.

Follows the standardized layout (see :mod:`simkit.energies.arap`). The
per-element variable is the bending angle ``theta`` (representation suffix
``_theta``), so the element tier is ``*_element_theta``.

Element tier (``*_element_theta``)
    Per-hinge density and derivatives in ``theta``. The single stiffness is the
    effective coefficient ``kappa``; density per hinge is
    ``0.5 * kappa * (theta - theta0)^2``. (For a 2D beam ``kappa = ymI / l``;
    the 3D discrete-shells module supplies ``kappa = 2 * ym_bending * ||le||/he``
    via its back-compat wrappers.)

Global explicit tier (``*_x``)
    Takes flattened vertex positions ``x`` and hinge connectivity ``C``; computes
    the bending angles and applies the per-element angle Jacobian/Hessian and the
    wedge gather map ``W = kron(wedge_map(C, n), eye(dim))``.

Notes
-----
The ``_x`` Hessian assembles the true second derivative: the Gauss-Newton term
``kappa * dtheta_dx (x) dtheta_dx`` plus the geometric term
``kappa*dtheta * d2theta_dx2``. The geometric term can make the assembled matrix
indefinite; pass ``psd=True`` to project the per-hinge blocks. ``psd=True``
projects the *combined* per-hinge block (Gauss-Newton + geometric), consistent
with the 3D discrete-shells module. The default ``psd=False`` is the exact
Hessian.
"""

import numpy as np
import scipy as sp

from ..dihedral_angles import (
    dihedral_angles,
    dihedral_angles_gradient_element,
    dihedral_angles_hessian_element,
)
from ..wedge_map import wedge_map
from ..psd_project import psd_project


def _infer_dim(C: np.ndarray) -> int:
    """Infer ambient dimension from the hinge connectivity arity."""
    k = C.shape[1]
    if k == 3:
        return 2
    if k == 4:
        return 3
    raise ValueError(
        f"bending_energy: cannot infer dim from connectivity arity {k} "
        "(expected 3 for a 2D vertex hinge or 4 for a 3D edge hinge)"
    )


# --------------------------------------------------------------------------- #
# Element tier: bending angle (theta) representation                          #
# --------------------------------------------------------------------------- #
def bending_energy_element_theta(theta: np.ndarray, theta0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Per-hinge bending energy density ``0.5 * kappa * (theta - theta0)^2``.

    Parameters
    ----------
    theta : np.ndarray (J, 1)
        Current bending angles.
    theta0 : np.ndarray (J, 1)
        Rest bending angles.
    kappa : np.ndarray (J, 1)
        Per-hinge effective bending stiffness.

    Returns
    -------
    psi : np.ndarray (J, 1)
        Per-hinge energy densities.
    """
    kappa = np.asarray(kappa).reshape(-1, 1)
    dtheta = theta - theta0
    return 0.5 * kappa * (dtheta ** 2)


def bending_gradient_element_theta(theta: np.ndarray, theta0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Per-hinge gradient of the density w.r.t. ``theta``: ``kappa * (theta - theta0)``."""
    kappa = np.asarray(kappa).reshape(-1, 1)
    return kappa * (theta - theta0)


def bending_hessian_element_theta(theta: np.ndarray, theta0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Per-hinge Hessian of the density w.r.t. ``theta``: ``kappa`` (constant)."""
    kappa = np.asarray(kappa).reshape(-1, 1)
    return kappa * np.ones_like(np.asarray(theta).reshape(-1, 1))


# --------------------------------------------------------------------------- #
# Global explicit tier: position (x) variable                                 #
# --------------------------------------------------------------------------- #
def bending_energy_x(x: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray) -> float:
    """Assembled bending energy at flattened vertex positions ``x``.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1) or (n, dim)
        Vertex positions. ``dim`` is inferred from ``C``'s arity.
    C : np.ndarray (E, k)
        Hinge connectivity: vertex triples in 2D (``k == 3``) or quadruples in
        3D (``k == 4``).
    theta0 : np.ndarray (E, 1)
        Rest bending angles.
    kappa : np.ndarray (E, 1)
        Per-hinge effective bending stiffness.

    Returns
    -------
    e : float
        Total bending energy.
    """
    dim = _infer_dim(C)
    X = np.asarray(x).reshape(-1, dim)
    theta = dihedral_angles(X, C)
    psi = bending_energy_element_theta(theta, theta0, kappa)
    return float(np.sum(psi))


def bending_gradient_x(x: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray, W: sp.sparse.spmatrix = None) -> np.ndarray:
    """Assembled bending gradient w.r.t. positions ``x``.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1) or (n, dim)
        Vertex positions; ``dim`` inferred from ``C``.
    C : np.ndarray (E, k)
        Hinge connectivity.
    theta0 : np.ndarray (E, 1)
        Rest bending angles.
    kappa : np.ndarray (E, 1)
        Per-hinge effective bending stiffness.
    W : scipy.sparse matrix, optional
        Precomputed wedge gather ``kron(wedge_map(C, n), eye(dim))``. Avoids
        recomputation when calling repeatedly.

    Returns
    -------
    g : np.ndarray (dim*n, 1)
        Assembled gradient.
    """
    dim = _infer_dim(C)
    X = np.asarray(x).reshape(-1, dim)
    nv = X.shape[0]
    theta = dihedral_angles(X, C)
    de_dtheta = bending_gradient_element_theta(theta, theta0, kappa)   # (E, 1)
    Je = dihedral_angles_gradient_element(X, C)                             # (E, dim*k)
    if W is None:
        W = sp.sparse.kron(wedge_map(C, nv), sp.sparse.identity(dim))
    return W.T @ (Je * de_dtheta).reshape(-1, 1)


def bending_hessian_x(x: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray, psd: bool = False, W: sp.sparse.spmatrix = None) -> sp.sparse.spmatrix:
    """Assembled bending Hessian w.r.t. positions ``x``.

    Assembles the true second derivative: the Gauss-Newton term plus the
    geometric ``d2theta/dx2`` term, gathered to global DOFs by the wedge map.

    Parameters
    ----------
    x : np.ndarray (dim*n, 1) or (n, dim)
        Vertex positions; ``dim`` inferred from ``C``.
    C : np.ndarray (E, k)
        Hinge connectivity.
    theta0 : np.ndarray (E, 1)
        Rest bending angles.
    kappa : np.ndarray (E, 1)
        Per-hinge effective bending stiffness.
    psd : bool, optional
        If ``True``, project each per-hinge block (Gauss-Newton + geometric) to
        PSD before assembly. Default ``False`` (true Hessian).
    W : scipy.sparse matrix, optional
        Precomputed wedge gather ``kron(wedge_map(C, n), eye(dim))``.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (dim*n, dim*n)
        Assembled Hessian.
    """
    dim = _infer_dim(C)
    X = np.asarray(x).reshape(-1, dim)
    nv = X.shape[0]
    theta = dihedral_angles(X, C)
    de_dtheta = bending_gradient_element_theta(theta, theta0, kappa)   # (E, 1)
    d2e_dtheta2 = bending_hessian_element_theta(theta, theta0, kappa)  # (E, 1)
    Je = dihedral_angles_gradient_element(X, C)                             # (E, m)
    He = dihedral_angles_hessian_element(X, C)                              # (E, m, m)

    # Per-hinge block: geometric (de * d2theta/dx2) + Gauss-Newton (d2e * J (x) J)
    Q = (de_dtheta[:, :, None] * He
         + d2e_dtheta2[:, :, None] * (Je[:, :, None] * Je[:, None, :]))
    if psd:
        Q = psd_project(Q)

    Q2 = sp.sparse.block_diag(Q)
    if W is None:
        W = sp.sparse.kron(wedge_map(C, nv), sp.sparse.identity(dim))
    return (W.T @ Q2 @ W).tocsc()


# --------------------------------------------------------------------------- #
# Global explicit tier: displacement (u) variable                             #
# --------------------------------------------------------------------------- #
def bending_energy_u(u: np.ndarray, x_bar: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray) -> float:
    """Bending energy at displacement ``u`` from reference ``x_bar``.

    Thin wrapper around :func:`bending_energy_x`: bending angles are nonlinear in
    positions, so no offset precompute helps. The reference ``x_bar`` is
    arbitrary (not required to be the rest pose).
    """
    return bending_energy_x(np.asarray(x_bar).reshape(-1, 1) + np.asarray(u).reshape(-1, 1), C, theta0, kappa)


def bending_gradient_u(u: np.ndarray, x_bar: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray, W: sp.sparse.spmatrix = None) -> np.ndarray:
    """Bending gradient w.r.t. displacement ``u`` (equals the ``_x`` gradient at ``x_bar + u``)."""
    return bending_gradient_x(np.asarray(x_bar).reshape(-1, 1) + np.asarray(u).reshape(-1, 1), C, theta0, kappa, W=W)


def bending_hessian_u(u: np.ndarray, x_bar: np.ndarray, C: np.ndarray, theta0: np.ndarray, kappa: np.ndarray, psd: bool = False, W: sp.sparse.spmatrix = None) -> sp.sparse.spmatrix:
    """Bending Hessian w.r.t. displacement ``u`` (equals the ``_x`` Hessian at ``x_bar + u``)."""
    return bending_hessian_x(np.asarray(x_bar).reshape(-1, 1) + np.asarray(u).reshape(-1, 1), C, theta0, kappa, psd=psd, W=W)
