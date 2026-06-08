"""Flat-function modal-muscle simulator.

This is the flat-function replacement for the deleted
``simkit.sims.elastic.ModalMuscleSim`` class. The library exposes only flat
functions; here we compose them into a small *local* helper that the modal
muscle examples / tutorials / interactive demos all reuse.

The method (a.k.a. *Modal Muscles* / *Actuators a la Mode*):

* A reduced subspace ``B`` (skinning eigenmodes) carries the deformation, with
  reduced state ``z`` so that ``x = B @ z``.
* A small set of displacement modes ``D`` act as **muscles**. Activating them
  with a per-mode amplitude vector ``a`` drives a *clustered plastic stretch
  tensor* that the active local/global steps try to match.
* Passive ARAP elasticity + inertia resist the actuation.
* Everything is solved by alternating a **local** step (clustered ``polar_svd``
  of the passive Jacobian and of the active plastic stretch tensor) and a
  **global** step (a single Cholesky solve), driven by ``block_coord``.
* Optional ground-plane contact is projected inside the global step.

Build the system once with :func:`build_modal_muscle`; get the rest state with
:func:`rest_state`; advance one timestep with :func:`step`.
"""
import numpy as np
import scipy as sp

import simkit as sk
from simkit.clustered_plastic_stretch_tensor import clustered_plastic_stretch_tensor
from simkit.fast_sandwich_transform_clustered import fast_sandwich_transform_clustered
from simkit.solvers import block_coord


def build_modal_muscle(X, T, B, D, l=None, d=None, *, mu=1e6, gamma=1e6,
                       rho=1e3, h=1e-2, b0=None, Q0=None,
                       contact=False, cI=None, plane_pos=None, plane_normal=None,
                       alpha=1.0, max_iter=10, tolerance=1e-6):
    """Precompute every operator the modal-muscle simulation needs.

    Parameters
    ----------
    X, T : mesh vertices / elements.
    B : (dim*n, r) skinning-eigenmode subspace; reduced state is ``z`` (r,1).
    D : (dim*n, m) displacement "muscle" modes; activation is ``a`` (m,1).
    l : (|T|,) passive cubature cluster labels (defaults to one cluster/elem).
    d : (|T|,) active cubature cluster labels (defaults to a single cluster).
    mu, gamma : passive / active stiffness (scalar or per-element).
    rho, h : density and timestep.
    b0 : (r,1) constant reduced force (e.g. gravity ``-B.T g``).
    Q0 : (r,r) extra reduced quadratic term (e.g. soft constraints).
    contact : enable ground-plane contact projection.
    cI : boundary vertex indices used as contact points.
    plane_pos, plane_normal : ground-plane position / normal (dim,1).
    alpha : contact restitution (1 = inelastic tangential slip preserved).

    Returns
    -------
    s : dict holding every precomputed quantity + config (passed to
        :func:`rest_state` / :func:`step`).
    """
    dim = X.shape[1]
    nT = T.shape[0]
    if l is None:
        l = np.arange(nT)
    if d is None:
        d = np.zeros(nT)
    l = l.astype(int)
    d = d.astype(int)

    mu_v = np.ones((nT, 1)) * mu
    gamma_v = np.ones((nT, 1)) * gamma
    vol = sk.volume(X, T)
    J = sk.deformation_jacobian(X, T)

    # ---- passive ARAP (clustered) ----------------------------------------
    P, _ = sk.cluster_grouping_matrices(l, X, T)
    A = sp.sparse.diags(vol.flatten())
    Mu = sp.sparse.diags(mu_v.flatten())
    AMue = sp.sparse.kron(A @ Mu, sp.sparse.identity(dim * dim))
    PAMue = sp.sparse.kron(P @ (A @ Mu), sp.sparse.identity(dim * dim))
    AMuPJB = (PAMue @ J) @ B
    L_passive = J.T @ AMue @ J

    Mv = sp.sparse.kron(sk.massmatrix(X, T, rho=rho), sp.sparse.identity(dim))
    BLB_passive = B.T @ L_passive @ B
    BMB = B.T @ Mv @ B
    BMy = B.T @ Mv @ X.reshape(-1, 1)

    # ---- active muscle force (clustered plastic stretch tensor) ----------
    Gamma = sp.sparse.diags(gamma_v.flatten())
    AGammae = sp.sparse.kron(A @ Gamma, sp.sparse.identity(dim * dim))
    L_active = J.T @ (AGammae @ J)
    JD = J @ D
    BLB_active = B.T @ L_active @ B
    BJAgamma = B.T @ (J.T @ AGammae)
    DMD = D.T @ Mv @ D
    DMy = D.T @ Mv @ X.reshape(-1, 1)

    num_passive_clusters = int(l.max()) + 1
    num_active_clusters = int(d.max()) + 1
    K = clustered_plastic_stretch_tensor(X, T, d, B, D, w=(vol * gamma_v).reshape(-1, 1))
    fst = fast_sandwich_transform_clustered(BJAgamma, JD, d, dim=dim)

    if Q0 is None:
        Q0 = np.zeros((B.shape[1], B.shape[1]))
    if b0 is None:
        b0 = np.zeros((B.shape[1], 1))
    b0 = b0.reshape(-1, 1)

    H = BLB_passive + BLB_active + BMB / h**2 + Q0
    chol_H = sp.linalg.cho_factor(H)

    s = dict(
        dim=dim, h=h, alpha=alpha, max_iter=max_iter, tolerance=tolerance,
        B=B, D=D, Mv=Mv, BMB=BMB, BMy=BMy, DMD=DMD, DMy=DMy,
        AMuPJB=AMuPJB, fst=fst, K=K, chol_H=chol_H, b=b0,
        num_passive_clusters=num_passive_clusters,
        num_active_clusters=num_active_clusters,
        contact=contact,
    )

    # ---- contact ---------------------------------------------------------
    if contact:
        if plane_normal is None:
            plane_normal = np.zeros((dim, 1)); plane_normal[1] = 1.0
        if plane_pos is None:
            plane_pos = np.zeros((dim, 1))
        tangent = np.random.rand(dim, 1)
        tangent = tangent - (tangent.T @ plane_normal) * plane_normal
        tangent = tangent / np.linalg.norm(tangent)
        if dim == 3:
            t2 = np.cross(plane_normal.flatten(), tangent.flatten()).reshape(-1, 1)
            tangent = np.hstack([tangent, t2])
        contact_frame = np.vstack([plane_normal.T, tangent.T])

        cI = np.unique(cI)
        S = sk.selection_matrix(cI, X.shape[0])
        Se = sp.sparse.kron(S, sp.sparse.identity(dim))
        Je = Se @ B
        JeQi = sp.linalg.cho_solve(chol_H, Je.T).T
        s.update(cI=cI, plane_pos=plane_pos, plane_normal=plane_normal,
                 Je=Je, JeQi=JeQi)
    return s


def rest_state(s, X):
    """Reduced rest state ``(z, z_dot, a)`` projected from the rest pose ``X``."""
    z = sk.project_into_subspace(X.reshape(-1, 1), s['B'], M=s['Mv'],
                                 BMB=s['BMB'], BMy=s['BMy'])
    z_dot = np.zeros_like(z)
    a = sk.project_into_subspace(X.reshape(-1, 1), s['D'], M=s['Mv'],
                                 BMB=s['DMD'], BMy=s['DMy'])
    return z, z_dot, a


def _contact_projection(s, z, f, z_curr):
    """Velocity-level ground-plane contact, projected onto the global solve."""
    dim, h, alpha = s['dim'], s['h'], s['alpha']
    Je, JeQi, plane_pos = s['Je'], s['JeQi'], s['plane_pos']
    z_dot_tent = (sp.linalg.cho_solve(s['chol_H'], f) - z_curr) / h
    Pp = (Je @ z).reshape(-1, dim)
    under = (Pp[:, 1] < plane_pos[1]).flatten()
    local_vel = (Je @ z_dot_tent).reshape(-1, dim)
    closer = local_vel[:, 1] < 0
    ci = np.where(under * closer)[0]
    if ci.shape[0] == 0:
        return np.zeros_like(f)
    idx = (np.repeat(ci[:, None], dim, axis=1) * dim + np.arange(dim)).flatten()
    JeI = Je[idx, :]
    L = JeQi[idx, :]
    vt = local_vel[ci, 0]
    v = np.zeros((ci.shape[0], dim)); v[:, 0] = (1.0 - alpha) * vt
    p = (JeI @ z_curr).reshape(-1, dim)
    local_f = (L @ f).reshape(-1, dim)
    bb = (v * h + p - local_f).reshape(-1, 1)
    if L.shape[0] >= L.shape[1]:
        c = np.linalg.solve(L.T @ L, L.T @ bb)
    else:
        c = L.T @ np.linalg.solve(L @ L.T, bb)
    return c


def step(s, z, z_dot, a, b_ext=None):
    """Advance one timestep. Returns the next reduced state ``z_next``."""
    dim, h = s['dim'], s['h']
    y = z + h * z_dot
    k = s['BMB'] @ y / h**2
    b = s['b'] if b_ext is None else (s['b'] + b_ext)
    z_curr = z.copy()

    def local_step(zc):
        c = s['AMuPJB'] @ zc
        R_p = sk.polar_svd(c.reshape(-1, dim, dim))[0]
        C = s['K'](zc, a)
        R_a = sk.polar_svd(C)[0]
        return np.vstack([R_p.reshape(-1, 1), R_a.reshape(-1, 1)])

    def global_step(zc, r):
        pd = s['num_passive_clusters'] * dim**2
        e_p = s['AMuPJB'].T @ r[:pd]
        ad = s['num_active_clusters'] * dim**2
        e_a = s['fst'](r[pd:pd + ad].reshape(-1, dim, dim)) @ a
        f = k + e_p + e_a - b
        if s['contact']:
            f = f + _contact_projection(s, zc, f, z_curr)
        return sp.linalg.cho_solve(s['chol_H'], f)

    return block_coord(z.copy(), global_step, local_step,
                       tolerance=s['tolerance'], max_iter=s['max_iter'])
