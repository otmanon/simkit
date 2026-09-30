"""The compliant hand as an energy ``system(x, a) -> (E, g, H)``, full or hyper-reduced.

The unknowns ``x`` are the positions of a coarse mesh's vertices; the fine hand
follows through the coarse mesh's prolongation ``x_fine = B x``, ``B = kron(P, I3)``
(``P = I`` gives the full space). Each term is integrated as cheaply as it can be:

* elastic     stable Neo-Hookean on the COARSE mesh (its tets are the cubature)
* actuators   one spring per joint between two embedded anchor points, rest length
              ``(1 - c a) l_rest`` -- the actuation ``a`` is the only control
* hinge pins  two points on each joint's hinge axis carried by both blocks, tied by
              a quadratic penalty (leaves only flexion soft)
* wrist       the fine base vertices held by a quadratic penalty
* contact     cubic penalty against an analytic SDF, on the FINE surface vertices
              ``B_s x`` (the one term integrated on the fine mesh)
* friction    lagged tangential penalty on the fine surface vertices in contact
"""
import os
import time

import numpy as np
import scipy as sp
import igl

import simkit
from simkit.energies import (stable_neo_hookean_energy_x, stable_neo_hookean_gradient_x,
                             stable_neo_hookean_hessian_x, mass_springs_energy_z, mass_springs_gradient_z,
                             mass_springs_hessian_z, sdf_contact_energy_x, sdf_contact_gradient_x,
                             sdf_contact_hessian_x, tangential_friction_energy_x, tangential_friction_gradient_x,
                             tangential_friction_hessian_x)
from simkit.filesystem import get_data_directory

DATA = os.path.join(get_data_directory(), "sdm_hand")
K_SPRING = 1e5            # N/m, actuator springs
K_BASE = 1e8              # N/m, wrist base penalty
K_CONTACT = 1e12          # N/m^4, cubic contact penalty
MAX_STEP = 1e-3           # m, largest vertex move per Newton iteration when in contact


def load_fine():
    return dict(np.load(os.path.join(DATA, "hand_tets.npz"))), dict(np.load(os.path.join(DATA, "hand_rig.npz")))


def load_coarse(name="hand_coarse_1200.npz"):
    d = dict(np.load(os.path.join(DATA, name)))
    d["P"] = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]), shape=tuple(d["P_shape"]))
    return d


def part_vertices(T, part, pid):
    """Vertices whose incident tets all belong to part ``pid`` (all its vertices if fewer than 4)."""
    inside = np.zeros(T.max() + 1, bool)
    inside[np.unique(T[part == pid])] = True
    touch = inside.copy()
    inside[np.unique(T[part != pid])] = False
    return np.nonzero(inside if inside.sum() >= 4 else touch)[0]


def build_hand_system(fine, rig, coarse=None, k_pin=1e7, sdf=None, friction=0.0):
    """Energy, gradient and Hessian of the hand as a function of the coarse state and
    the actuation, plus what the solver and renderer need.

    ``coarse=None`` is the full space. ``sdf(P) -> (phi, grad)`` is a rigid object
    in contact with the fine surface; ``friction`` is ``k_f`` (N/m^3) of the lagged
    tangential friction on the fine surface vertices inside it.
    """
    Xf, Tf = fine["X"], fine["T"]
    n_f = len(Xf)
    mesh = fine if coarse is None else coarse
    P = sp.sparse.identity(n_f, format="csr") if coarse is None else coarse["P"].tocsr()
    B = sp.sparse.kron(P, sp.sparse.identity(3), format="csr")
    Xc, Tc = mesh["X"], mesh["T"]
    E, nu = mesh["E"], mesh["nu"]
    mu = (E / (2 * (1 + nu)))[:, None]
    lam = (E * nu / ((1 + nu) * (1 - 2 * nu)))[:, None]
    J = simkit.deformation_jacobian(Xc, Tc)
    vol = simkit.volume(Xc, Tc)
    x0 = Xc.reshape(-1).copy()
    E_rest = stable_neo_hookean_energy_x(Xc, J, mu, lam, vol)

    # fine-side linear maps (springs, hinge pins, wrist), pulled back into the subspace
    names = list(fine["part_names"])
    cand = lambda n: part_vertices(Tf, fine["part"], names.index(str(n)))
    Ga = simkit.affine_embedding_matrix(Xf, rig["spring_pa"], [cand(n) for n in rig["spring_below"]])
    Gb = simkit.affine_embedding_matrix(Xf, rig["spring_pb"], [cand(n) for n in rig["spring_above"]])
    Js = ((Gb - Ga) @ B).tocsr()                                  # spring vectors d = Js x
    q = rig["hinge_q"].reshape(-1, 3)
    Gp = (simkit.affine_embedding_matrix(Xf, q, [cand(n) for n in np.repeat(rig["hinge_below"], 2)]) -
          simkit.affine_embedding_matrix(Xf, q, [cand(n) for n in np.repeat(rig["hinge_above"], 2)])) @ B
    base = rig["base"]
    Sb = simkit.selection_matrix((3 * base[:, None] + np.arange(3)).ravel(), 3 * n_f).tocsr() @ B
    rb0 = Xf[base].reshape(-1)
    Q = (k_pin * (Gp.T @ Gp) + K_BASE * (Sb.T @ Sb)).tocsr()       # pins: Gp X = 0 at rest
    ones = np.ones((len(rig["l_rest"]), 1))
    k_s = K_SPRING * ones
    l0 = lambda a: ((1 - rig["c"] * a) * rig["l_rest"])[:, None]

    # contact + lagged friction on the fine surface
    Fb = igl.boundary_facets(Tf)[0]
    sv = np.unique(Fb)
    area = np.zeros(n_f)
    for j in range(3):
        np.add.at(area, Fb[:, j], 0.5 * igl.doublearea(Xf, Fb) / 3)
    Bs = sp.sparse.kron(P[sv], sp.sparse.identity(3), format="csr")
    a_s = area[sv]
    anchor = dict(S=None)

    def set_anchor(x):
        """Anchor the fine surface vertices inside the object for the next step."""
        anchor["S"] = None
        if sdf is None or friction <= 0:
            return
        Ps = (Bs @ x).reshape(-1, 3)
        phi, nrm = sdf(Ps)
        act = np.nonzero(phi < 0)[0]
        if len(act):
            anchor.update(S=Bs[(3 * act[:, None] + np.arange(3)).ravel()], p0=Ps[act], n=nrm[act],
                          w=friction * a_s[act])

    def system(x, a, energy_only=False):
        e = stable_neo_hookean_energy_x(x.reshape(-1, 3), J, mu, lam, vol) - E_rest
        rb = Sb @ x - rb0
        e += mass_springs_energy_z(x, Js, k_s, ones, l0(a)) + 0.5 * x @ (Q @ x) - K_BASE * (rb0 @ (Sb @ x)) \
            + 0.5 * K_BASE * rb0 @ rb0
        fr = anchor["S"] is not None
        if sdf is not None:
            e += sdf_contact_energy_x(x, Bs, sdf, K_CONTACT, a_s)
        if fr:
            e += tangential_friction_energy_x(x, anchor["S"], anchor["p0"], anchor["n"], anchor["w"])
        if energy_only:
            return float(e), None, None
        g = stable_neo_hookean_gradient_x(x.reshape(-1, 3), J, mu, lam, vol).reshape(-1) \
            + mass_springs_gradient_z(x, Js, k_s, ones, l0(a)).ravel() + Q @ x - K_BASE * (Sb.T @ rb0)
        H = stable_neo_hookean_hessian_x(x.reshape(-1, 3), J, mu, lam, vol, psd=True) \
            + mass_springs_hessian_z(x, Js, k_s, ones, l0(a), psd=True) + Q
        if sdf is not None:
            g = g + sdf_contact_gradient_x(x, Bs, sdf, K_CONTACT, a_s).ravel()
            H = H + sdf_contact_hessian_x(x, Bs, sdf, K_CONTACT, a_s)
        if fr:
            g = g + tangential_friction_gradient_x(x, anchor["S"], anchor["p0"], anchor["n"], anchor["w"]).ravel()
            H = H + tangential_friction_hessian_x(x, anchor["S"], anchor["p0"], anchor["n"], anchor["w"])
        return float(e), g, H.tocsr()

    def contact_report(x):
        phi = sdf((Bs @ x).reshape(-1, 3))[0]
        d = np.maximum(-phi, 0)
        return dict(n=int((d > 0).sum()), force=float((K_CONTACT * a_s * d ** 2).sum()), pen=float(d.max()))

    tips = np.split(rig["tip_ids"], rig["tip_ptr"][1:-1])
    return dict(system=system, x0=x0, P=P, n_dof=len(x0), set_anchor=set_anchor, contact=sdf is not None,
                contact_report=contact_report if sdf is not None else None, tips=tips, fingers=list(rig["fingers"]))


try:                                                      # CHOLMOD when available
    from sksparse.cholmod import cho_factor
    solve_spd = lambda A, b: cho_factor(A).solve(b)
except ImportError:
    solve_spd = lambda A, b: sp.sparse.linalg.spsolve(A, b)


def newton(system, x0, a, tol=1e-9, max_iters=60, max_step=None):
    """Projected Newton with backtracking; stops when half the Newton decrement drops
    below ``tol`` (J). ``max_step`` caps any vertex's move per iteration (no tunnelling
    through thin contact walls)."""
    f = lambda y: system(y, a, True)[0]
    x = x0.copy()
    for it in range(1, max_iters + 1):
        _, g, H = system(x, a)
        dx = solve_spd(H.tocsc(), -g)
        if -0.5 * g @ dx < tol:
            break
        if max_step is not None:
            m = np.linalg.norm(dx.reshape(-1, 3), axis=1).max()
            dx *= min(1.0, max_step / m)
        alpha, x, _ = simkit.backtracking_line_search(f, x, g, dx)
        if alpha == 0:
            break
    return x, it


def close_hand(sysd, n_steps=24):
    """Quasi-static actuation sweep a = 0 .. 1; returns states, Newton counts and
    wall time. With contact, each step starts from the last state (no extrapolated
    predictor), caps the Newton step, and re-anchors the friction."""
    system, x = sysd["system"], sysd["x0"].copy()
    contact = sysd["contact"]
    t0 = time.perf_counter()
    xs, its = [], []
    for a in np.linspace(0, 1, n_steps + 1):
        sysd["set_anchor"](x)
        start = x if contact or len(xs) < 2 else 2 * xs[-1] - xs[-2]
        x, it = newton(system, start, a, max_iters=200 if contact else 60, max_step=MAX_STEP if contact else None)
        xs.append(x.copy())
        its.append(it)
    return dict(a=np.linspace(0, 1, n_steps + 1), x=np.array(xs), its=np.array(its),
                seconds=time.perf_counter() - t0)


def fingertip_travel(sysd, fine, x):
    """Displacement (m) of each fingertip (mean of its far-face fine vertices)."""
    xf = sysd["P"] @ x.reshape(-1, 3)
    return np.array([np.linalg.norm(xf[t].mean(0) - fine["X"][t].mean(0)) for t in sysd["tips"]])
