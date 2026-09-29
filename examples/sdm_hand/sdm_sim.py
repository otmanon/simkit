"""Tendon-driven closing of the SDM hand: SimKit statics + backward Euler.

Model
-----
* **Elasticity** -- SimKit stable Neo-Hookean (``stable_neo_hookean_*_x``) with
  per-tet ``mu, lam`` from the winding-number material labels
  (stiff polyurethane links/palm, soft elastomer flexures, softer pads). The
  rest energy of stable NH is not zero, so it is subtracted.
* **Base** -- every vertex on the palm's bottom face (``z = 0``) is pinned;
  its DOFs are eliminated (hard Dirichlet).
* **Gravity** -- along ``+z`` (the hand is mounted facing down, fingers
  hanging, as on the robot arm the SDM hand was designed for).
* **Tendons** -- SimKit mass springs (``mass_springs_*_x``, energy
  ``0.5 k (|d| - l0)^2``). Each finger has two tendon spans on its palmar side:
  palm -> proximal link across the proximal flexure, and proximal link -> distal
  link across the distal flexure, three parallel springs per span (27 mm wide
  flexure gap is bridged at y offsets -6, 0, +6 mm). All attach to stiff
  (polyurethane) surface vertices. ONE actuation parameter ``a in [0, 1]``
  sets every rest length ``l0 = (1 - c a) l_rest`` -- the single actuator
  closes all four fingers together, the softer proximal flexure bends first.
* **Statics** -- continuation in ``a`` (0 -> 1 in 30 steps), Newton with
  CHOLMOD and SimKit's backtracking line search.
* **Dynamics** -- backward Euler as incremental-potential minimisation
  ``V(x) + 1/(2h^2) (x - x~)^T M (x - x~)``, lumped SimKit mass matrix,
  ``a(t)`` a smoothstep ramp over 1.2 s, then held for 0.8 s.

    python sdm_sim.py                 # statics + dynamics -> output/sdm_sim.npz
    python sdm_sim.py --no-dynamics
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy as sp
import igl

import simkit
import simkit.energies as energies
from simkit.backtracking_line_search import backtracking_line_search

from sdm_geometry import OUT, HandParams, z_levels

try:
    from sksparse.cholmod import cho_factor

    def solve_spd(A, b):
        return cho_factor(A.tocsc()).solve(b)
except ImportError:  # pragma: no cover
    def solve_spd(A, b):
        return sp.sparse.linalg.spsolve(A.tocsc(), b)

K_TENDON = 4.0e3        # N/m per spring (3 springs per span -> 12 kN/m per span)
CONTRACTION = 0.30      # l0 = (1 - CONTRACTION * a) * l_rest
GRAVITY = np.array([0.0, 0.0, 9.81])
TENDON_Y = (-0.006, 0.0, 0.006)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class SDMHand:
    def __init__(self, scene, k_tendon=K_TENDON, contraction=CONTRACTION):
        self.X = X = scene["X"].astype(float)
        self.T = T = scene["T"].astype(np.int64)
        self.n = n = len(X)
        self.part, self.names = scene["part"], [str(s) for s in scene["part_names"]]
        E, nu = scene["E"], scene["nu"]
        self.mu = (E / (2 * (1 + nu))).reshape(-1, 1)
        self.lam = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1)
        self.J = simkit.deformation_jacobian(X, T)
        self.vol = simkit.volume(X, T)
        self.E_rest = energies.stable_neo_hookean_energy_x(X, self.J, self.mu, self.lam, self.vol)
        self.Mv = simkit.massmatrix(X, T, rho=scene["rho"].reshape(-1, 1)).diagonal()
        self.m = np.repeat(self.Mv, 3)
        self.f_g = (self.Mv[:, None] * GRAVITY[None]).ravel()
        # pins: bottom face of the palm
        self.pinned = np.where(X[:, 2] < 1e-9)[0]
        fixed = np.zeros(3 * n, bool)
        fixed[(3 * self.pinned[:, None] + np.arange(3)).ravel()] = True
        self.free = np.where(~fixed)[0]
        self.contraction = contraction
        self._tendons(k_tendon)

    # -------------------------------------------------------------- tendons
    def part_vertices(self, name, surface=True):
        """Vertices whose incident tets all belong to part ``name`` (optionally on the boundary)."""
        pid = self.names.index(name)
        inside = np.zeros(self.n, bool)
        inside[np.unique(self.T[self.part == pid])] = True
        inside[np.unique(self.T[self.part != pid])] = False
        if surface:
            on = np.zeros(self.n, bool)
            on[np.unique(igl.boundary_facets(self.T)[0])] = True
            inside &= on
        return np.where(inside)[0]

    def _tendons(self, k):
        p = HandParams()
        z0, zp0, zp1, zd0, zd1 = z_levels(p)
        w = p.link_w / 2
        edges, finger_of, span_of = [], [], []
        self.fingers = []
        for side, s in (("L", -1.0), ("R", 1.0)):
            for j, yc in enumerate(p.finger_y):
                fn = f"{side}{j}"
                self.fingers.append(fn)
                xin = s * (p.finger_x - w)                 # palmar face
                anchors = [("palm", (xin - s * 0.002, z0)), (f"prox_{fn}", (xin, zp0 + 0.008)),
                           (f"prox_{fn}", (xin, zp1 - 0.008)), (f"dist_{fn}", (xin, zd0 + 0.005))]
                ids = []
                for part, (x, z) in anchors:
                    cand = self.part_vertices(part)
                    row = []
                    for dy in TENDON_Y:
                        q = np.array([x, yc + dy, z])
                        row.append(cand[np.argmin(np.linalg.norm(self.X[cand] - q, axis=1))])
                    ids.append(row)
                for span, (a, b) in enumerate(((0, 1), (2, 3))):
                    for t in range(len(TENDON_Y)):
                        edges.append((ids[a][t], ids[b][t]))
                        finger_of.append(len(self.fingers) - 1)
                        span_of.append(span)
        self.E_t = np.array(edges, np.int64)
        self.tendon_finger, self.tendon_span = np.array(finger_of), np.array(span_of)
        self.l_rest = np.linalg.norm(self.X[self.E_t[:, 0]] - self.X[self.E_t[:, 1]], axis=1)
        self.ym = np.full((len(self.E_t), 1), float(k))
        self.svol = np.ones((len(self.E_t), 1))

    def l0(self, a):
        return ((1.0 - self.contraction * a) * self.l_rest).reshape(-1, 1)

    # -------------------------------------------------------------- energy (full x, flat)
    def energy(self, x, a):
        Xm = x.reshape(-1, 3)
        e = energies.stable_neo_hookean_energy_x(Xm, self.J, self.mu, self.lam, self.vol) - self.E_rest
        e += energies.mass_springs_energy_x(Xm, self.E_t, self.ym, self.svol, self.l0(a))
        return float(e - self.f_g @ x)

    def gradient(self, x, a):
        Xm = x.reshape(-1, 3)
        g = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        g = g + energies.mass_springs_gradient_x(Xm, self.E_t, self.ym, self.svol, self.l0(a)).ravel()
        return g - self.f_g

    def hessian(self, x, a):
        Xm = x.reshape(-1, 3)
        H = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        H = H + energies.mass_springs_hessian_x(Xm, self.E_t, self.ym, self.svol, self.l0(a), psd=True)
        return H.tocsr()

    # -------------------------------------------------------------- solvers
    def minimize(self, x0, a, x_tilde=None, h=None, iters=40, tol=1e-8):
        """Newton over the free DOFs of V(x) [+ inertia if ``x_tilde`` given]."""
        fr = self.free
        x = x0.copy()
        inert = x_tilde is not None
        w = self.m / h ** 2 if inert else None

        def full(xf):
            y = x.copy()
            y[fr] = xf
            return y

        def f(xf):
            y = full(xf)
            e = self.energy(y, a)
            if inert:
                d = y - x_tilde
                e += 0.5 * d @ (w * d)
            return e
        it = 0
        for it in range(1, iters + 1):
            g = self.gradient(x, a)
            H = self.hessian(x, a)
            if inert:
                g = g + w * (x - x_tilde)
                H = H + sp.sparse.diags(w)
            gf = g[fr]
            Hf = H[fr][:, fr]
            dx = solve_spd(Hf, -gf)
            alpha, xf, _ = backtracking_line_search(f, x[fr], gf, dx)
            x[fr] = xf
            if alpha == 0 or np.abs(alpha * dx).max() < tol:
                break
        return x, it

    # -------------------------------------------------------------- measurements
    def link_rotation(self, x, name):
        v = self.part_vertices(name, surface=False)
        P0, P = self.X[v], x.reshape(-1, 3)[v]
        A = (P - P.mean(0)).T @ (P0 - P0.mean(0))
        U, _, Vt = np.linalg.svd(A)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    def joint_angles(self, x):
        """(proximal, distal) flexion angles per finger, degrees."""
        out = []
        for fn in self.fingers:
            R1, R2 = self.link_rotation(x, f"prox_{fn}"), self.link_rotation(x, f"dist_{fn}")
            ang = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            out.append((ang(R1), ang(R1.T @ R2)))
        return np.array(out)

    def tip_vertices(self, fn):
        zd1 = z_levels()[4]
        v = self.part_vertices(f"dist_{fn}")
        return v[self.X[v, 2] > zd1 - 1e-9]

    def fingertip_positions(self, x):
        Xm = x.reshape(-1, 3)
        return np.array([Xm[self.tip_vertices(fn)].mean(0) for fn in self.fingers])

    def finger_tets(self, fn):
        ids = [self.names.index(f"{k}_{fn}") for k in ("prox", "dist", "flexdist", "pad")]
        return self.T[np.isin(self.part, ids)]

    def opposing_clearance(self, x):
        """For each opposing finger pair (L_j, R_j): min distance between their
        surface vertices and #vertices of one inside the other (winding number)."""
        Xm = x.reshape(-1, 3)
        res = []
        for j in range(len(HandParams().finger_y)):
            TL, TR = self.finger_tets(f"L{j}"), self.finger_tets(f"R{j}")
            FL, FR = igl.boundary_facets(TL)[0], igl.boundary_facets(TR)[0]
            vL, vR = np.unique(FL), np.unique(FR)
            tree = sp.spatial.cKDTree(Xm[vR])
            dmin = tree.query(Xm[vL])[0].min()
            inside = (igl.winding_number(Xm, FR, Xm[vL]) > 0.5).sum() + \
                     (igl.winding_number(Xm, FL, Xm[vR]) > 0.5).sum()
            res.append((float(dmin), int(inside)))
        return res


def von_mises(hand, x):
    """Per-tet von Mises Cauchy stress of a compressible neo-Hookean solid (Pa)."""
    F = (hand.J @ x).reshape(-1, 3, 3)
    Jd = np.linalg.det(F)
    B = F @ np.transpose(F, (0, 2, 1))
    mu, lam = hand.mu.ravel(), hand.lam.ravel()
    sig = (mu[:, None, None] * (B - np.eye(3)) + (lam * np.log(np.abs(Jd)))[:, None, None]
           * np.eye(3)) / Jd[:, None, None]
    s = sig - np.trace(sig, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * (s * s).sum((1, 2)))


def run(dynamics=True, n_static=30, h=1 / 60, t_ramp=1.2, t_end=2.0):
    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = SDMHand(scene)
    print(f"{hand.n} vertices, {len(hand.T)} tets, {len(hand.pinned)} pinned vertices, "
          f"{len(hand.E_t)} tendon springs (k={K_TENDON:g} N/m, c={CONTRACTION})")
    x0 = hand.X.ravel().copy()
    out = dict(E_t=hand.E_t, l_rest=hand.l_rest, pinned=hand.pinned,
               tendon_finger=hand.tendon_finger, tendon_span=hand.tendon_span)
    info = dict(n_vertices=hand.n, n_tets=len(hand.T), n_pinned=len(hand.pinned),
                n_springs=len(hand.E_t), k_tendon=K_TENDON, contraction=CONTRACTION,
                fingers=hand.fingers, l_rest_mm=(hand.l_rest * 1e3).round(2).tolist())

    # ---------------------------------------------------------- statics
    t0 = time.time()
    a_vals = np.linspace(0, 1, n_static + 1)
    xs, iters = [], []
    x = x0.copy()
    for a in a_vals:
        x, it = hand.minimize(x, a)
        xs.append(x.copy())
        iters.append(it)
    t_static = time.time() - t0
    xs = np.array(xs)
    tips0 = hand.fingertip_positions(x0)
    stat = []
    for a, xa in zip(a_vals, xs):
        ang = hand.joint_angles(xa)
        tip = hand.fingertip_positions(xa)
        stat.append(dict(a=float(a), prox_deg=ang[:, 0].mean(), dist_deg=ang[:, 1].mean(),
                         tip_disp_mm=float(np.linalg.norm(tip - tips0, axis=1).mean() * 1e3),
                         tip_x_mm=(tip[:, 0] * 1e3).round(2).tolist(),
                         clearance=hand.opposing_clearance(xa)))
    for s in stat[::5]:
        print(f"  a={s['a']:.2f}  prox {s['prox_deg']:5.1f} deg  dist {s['dist_deg']:5.1f} deg  "
              f"tip disp {s['tip_disp_mm']:5.1f} mm  clearance {s['clearance']}")
    print(f"statics: {len(a_vals)} steps, {sum(iters)} Newton iterations, {t_static:.1f} s")
    ang1 = hand.joint_angles(xs[-1])
    info.update(t_static_s=t_static, static_newton_iters=int(sum(iters)), statics=stat,
                joint_angles_a1_deg=ang1.round(2).tolist(),
                tendon_force_a1_N=float(np.mean(K_TENDON * (np.linalg.norm(
                    xs[-1].reshape(-1, 3)[hand.E_t[:, 0]] - xs[-1].reshape(-1, 3)[hand.E_t[:, 1]],
                    axis=1) - hand.l0(1.0).ravel())) * len(TENDON_Y)))
    out.update(static_a=a_vals, static_x=xs.astype(np.float32),
               static_vm=np.array([von_mises(hand, xa) for xa in xs[::10]]).astype(np.float32))

    # ---------------------------------------------------------- dynamics
    if dynamics:
        t0 = time.time()
        n_steps = int(round(t_end / h))
        x, v = x0.copy(), np.zeros_like(x0)
        frames, a_t, times, its = [x.copy()], [0.0], [0.0], []
        for k in range(1, n_steps + 1):
            t = k * h
            a = smoothstep(t / t_ramp)
            x_tilde = x + h * v
            x_new, it = hand.minimize(x, a, x_tilde=x_tilde, h=h, tol=1e-7)
            v = (x_new - x) / h
            x = x_new
            frames.append(x.copy())
            a_t.append(a)
            times.append(t)
            its.append(it)
            if k % 20 == 0:
                tip = hand.fingertip_positions(x)
                print(f"  t={t:.3f}s a={a:.3f} newton={it} tip disp "
                      f"{np.linalg.norm(tip - tips0, axis=1).mean() * 1e3:.1f} mm "
                      f"({time.time() - t0:.0f}s)")
        t_dyn = time.time() - t0
        frames = np.array(frames)
        vm = np.array([von_mises(hand, f) for f in frames])
        tipd = np.array([np.linalg.norm(hand.fingertip_positions(f) - tips0, axis=1).mean()
                         for f in frames])
        print(f"dynamics: {n_steps} steps (h={h:.4f}s), {sum(its)} Newton iterations, {t_dyn:.1f} s")
        out.update(dyn_x=frames.astype(np.float32), dyn_a=np.array(a_t), dyn_t=np.array(times),
                   dyn_vm=vm.astype(np.float32), dyn_tip_disp=tipd)
        info.update(t_dynamic_s=t_dyn, dyn_steps=n_steps, h=h, t_ramp=t_ramp, t_end=t_end,
                    dyn_newton_iters=int(sum(its)),
                    dyn_final_tip_disp_mm=float(tipd[-1] * 1e3),
                    dyn_max_tip_disp_mm=float(tipd.max() * 1e3),
                    dyn_final_clearance=hand.opposing_clearance(frames[-1]),
                    dyn_min_clearance_mm=float(min(min(c[0] for c in hand.opposing_clearance(f))
                                                   for f in frames[::5]) * 1e3))
    np.savez_compressed(os.path.join(OUT, "sdm_sim.npz"), **out)
    with open(os.path.join(OUT, "sim_report.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dynamics", action="store_true")
    args = ap.parse_args()
    run(dynamics=not args.no_dynamics)
