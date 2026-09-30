"""The compliant hand as ``system(x, a) -> (E, g, H)`` in a coarse subspace ``x_fine = B x``,
``B = kron(P, I3)`` (``P = I``: full space). Neo-Hookean on the coarse tets; actuator springs,
hinge pins and wrist on embedded fine points; contact + lagged friction on the fine surface."""
import os
import time

import igl
import numpy as np
import scipy as sp

import simkit
from simkit.energies import (stable_neo_hookean_energy_x, stable_neo_hookean_gradient_x,
                             stable_neo_hookean_hessian_x, mass_springs_energy_z, mass_springs_gradient_z,
                             mass_springs_hessian_z, sdf_contact_energy_x, sdf_contact_gradient_x,
                             sdf_contact_hessian_x, tangential_friction_energy_x, tangential_friction_gradient_x,
                             tangential_friction_hessian_x)
from simkit.filesystem import get_data_directory

DATA = os.path.join(get_data_directory(), "sdm_hand")
K_SPRING, K_BASE, K_CONTACT, MAX_STEP = 1e5, 1e8, 1e12, 1e-3
try:
    from sksparse.cholmod import cho_factor
    solve_spd = lambda A, b: cho_factor(A).solve(b)
except ImportError:
    solve_spd = lambda A, b: sp.sparse.linalg.spsolve(A, b)


def load(name):
    d = dict(np.load(os.path.join(DATA, name)))
    if "P_data" in d:
        d["P"] = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]), shape=tuple(d["P_shape"]))
    return d


def part_vertices(T, part, pid):
    """Vertices whose tets all belong to part ``pid`` (all its vertices if fewer than 4)."""
    touch = np.zeros(T.max() + 1, bool)
    touch[np.unique(T[part == pid])] = True
    inside = touch.copy()
    inside[np.unique(T[part != pid])] = False
    return np.nonzero(inside if inside.sum() >= 4 else touch)[0]


def build_hand_system(fine, rig, coarse=None, k_pin=1e7, sdf=None, friction=0.0):
    Xf, Tf, nf = fine["X"], fine["T"], len(fine["X"])
    mesh = fine if coarse is None else coarse
    P = sp.sparse.identity(nf, format="csr") if coarse is None else coarse["P"].tocsr()
    B = sp.sparse.kron(P, sp.sparse.identity(3), format="csr")
    E, nu = mesh["E"], mesh["nu"]
    mu, lam = (E / (2 * (1 + nu)))[:, None], (E * nu / ((1 + nu) * (1 - 2 * nu)))[:, None]
    J, vol = simkit.deformation_jacobian(mesh["X"], mesh["T"]), simkit.volume(mesh["X"], mesh["T"])
    x0 = mesh["X"].reshape(-1).copy()
    E0 = stable_neo_hookean_energy_x(mesh["X"], J, mu, lam, vol)
    names = list(fine["part_names"])
    embed = lambda pts, parts: simkit.affine_embedding_matrix(
        Xf, pts, [part_vertices(Tf, fine["part"], names.index(str(n))) for n in parts])
    Js = ((embed(rig["spring_pb"], rig["spring_above"]) - embed(rig["spring_pa"], rig["spring_below"])) @ B).tocsr()
    q = rig["hinge_q"].reshape(-1, 3)
    Gp = (embed(q, np.repeat(rig["hinge_below"], 2)) - embed(q, np.repeat(rig["hinge_above"], 2))) @ B
    Sb = simkit.selection_matrix((3 * rig["base"][:, None] + np.arange(3)).ravel(), 3 * nf).tocsr() @ B
    xb = Xf[rig["base"]].reshape(-1)
    Q = (k_pin * (Gp.T @ Gp) + K_BASE * (Sb.T @ Sb)).tocsr()
    b = -K_BASE * (Sb.T @ xb)
    ks = K_SPRING * np.ones((len(rig["c"]), 1))
    l0 = lambda a: ((1 - rig["c"] * a) * rig["l_rest"])[:, None]
    Fb = igl.boundary_facets(Tf)[0]
    sv = np.unique(Fb)
    area = np.bincount(Fb.ravel(), np.repeat(igl.doublearea(Xf, Fb) / 6, 3), nf)[sv]
    Bs = sp.sparse.kron(P[sv], sp.sparse.identity(3), format="csr")
    fr = {}

    def set_anchor(x):
        """Anchor the fine surface vertices inside the object for the next step's friction."""
        fr.clear()
        if sdf is not None and friction > 0:
            Ps = (Bs @ x).reshape(-1, 3)
            phi, n = sdf(Ps)
            act = np.nonzero(phi < 0)[0]
            if len(act):
                fr.update(S=Bs[(3 * act[:, None] + np.arange(3)).ravel()], p0=Ps[act], n=n[act], w=friction * area[act])

    def system(x, a, energy_only=False):
        X3 = x.reshape(-1, 3)
        e = stable_neo_hookean_energy_x(X3, J, mu, lam, vol) - E0 + mass_springs_energy_z(x, Js, ks, ks * 0 + 1, l0(a)) \
            + 0.5 * x @ (Q @ x) + b @ x + 0.5 * K_BASE * xb @ xb
        e += sdf_contact_energy_x(x, Bs, sdf, K_CONTACT, area) if sdf else 0
        e += tangential_friction_energy_x(x, **fr) if fr else 0
        if energy_only:
            return e, None, None
        g = stable_neo_hookean_gradient_x(X3, J, mu, lam, vol).ravel() + \
            mass_springs_gradient_z(x, Js, ks, ks * 0 + 1, l0(a)).ravel() + Q @ x + b
        H = stable_neo_hookean_hessian_x(X3, J, mu, lam, vol, psd=True) + \
            mass_springs_hessian_z(x, Js, ks, ks * 0 + 1, l0(a)) + Q
        if sdf:
            g, H = g + sdf_contact_gradient_x(x, Bs, sdf, K_CONTACT, area).ravel(), H + sdf_contact_hessian_x(
                x, Bs, sdf, K_CONTACT, area)
        if fr:
            g, H = g + tangential_friction_gradient_x(x, **fr).ravel(), H + tangential_friction_hessian_x(x, **fr)
        return e, g, H.tocsr()

    def contact(x):
        d = np.maximum(-sdf((Bs @ x).reshape(-1, 3))[0], 0)
        return dict(contact_vertices=int((d > 0).sum()), force_N=float((K_CONTACT * area * d ** 2).sum()))

    return dict(system=system, x0=x0, P=P, set_anchor=set_anchor, contact=contact if sdf else None,
                tips=np.split(rig["tip_ids"], rig["tip_ptr"][1:-1]))


def newton(system, x, a, tol=1e-9, max_iters=60, max_step=None):
    """Projected Newton; stops on half the Newton decrement < ``tol``; ``max_step`` caps any
    vertex's move per iteration (no tunnelling through thin contact walls)."""
    for it in range(1, max_iters + 1):
        _, g, H = system(x, a)
        dx = solve_spd(H.tocsc(), -g)
        if -0.5 * g @ dx < tol:
            break
        if max_step:
            dx *= min(1.0, max_step / np.linalg.norm(dx.reshape(-1, 3), axis=1).max())
        alpha, x, _ = simkit.backtracking_line_search(lambda y: system(y, a, True)[0], x, g, dx)
        if alpha == 0:
            break
    return x, it


def close_hand(sysd, n_steps=24):
    """Quasi-static sweep a = 0..1 (with contact: no extrapolated start, capped steps, friction
    re-anchored each step)."""
    x, xs, its, c = sysd["x0"].copy(), [], [], sysd["contact"] is not None
    t0 = time.perf_counter()
    for a in np.linspace(0, 1, n_steps + 1):
        sysd["set_anchor"](x)
        x, it = newton(sysd["system"], x if c or len(xs) < 2 else 2 * xs[-1] - xs[-2], a,
                       max_iters=200 if c else 60, max_step=MAX_STEP if c else None)
        xs.append(x)
        its.append(it)
    return dict(a=np.linspace(0, 1, n_steps + 1), x=np.array(xs), its=np.array(its), seconds=time.perf_counter() - t0)


def fingertip_travel(sysd, fine, x):
    xf = sysd["P"] @ x.reshape(-1, 3)
    return np.array([np.linalg.norm(xf[t].mean(0) - fine["X"][t].mean(0)) for t in sysd["tips"]])
