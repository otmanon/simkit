"""Allegro hand closes on the cup and lifts it -- heterogeneous FEM in SimKit.

* **Actuation (joint space)** -- the palm and every steel joint shaft (found by
  winding number) are driven by forward kinematics of the joint trajectory
  ``q(t)`` (flat hand -> fitted grasp pose), plus a wrist lift. That is how the
  Allegro's motors act: through the joint shafts. Aluminium phalanges, rubber
  tips, silicone palmar pads and the cup are free elastic bodies.
* **Elasticity** -- SimKit stable Neo-Hookean, per-tet ``mu, lam``.
* **Time integration** -- backward Euler as incremental-potential minimisation
  (Newton + SimKit's backtracking line search) over the free DOFs.
* **Contact** -- fingertip rubber vs cup outer wall (two-way node-to-plane
  penalty with the cup's radial normal) + lagged smoothed Coulomb friction;
  cup vs table penalty.

    python simulate_hand.py            # full run -> output/hand_frames.npz
    python simulate_hand.py --steps 2  # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy as sp
from scipy.spatial import cKDTree
import igl

import simkit.energies as energies
from simkit.deformation_jacobian import deformation_jacobian
from simkit.massmatrix import massmatrix
from simkit.volume import volume
from simkit.backtracking_line_search import backtracking_line_search

from allegro_kinematics import AllegroHand
from hand_geometry import XML, CUP_Y_BOTTOM
from hand_grasp import body_transforms, vertex_bodies
from materials import lame

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from simulate_grasp import smoothstep, solve_spd  # noqa: E402


class HandSchedule:
    def __init__(self, q_grasp, t_close=0.6, t_hold=0.1, t_lift=0.6, t_end=1.5, lift=0.04):
        self.q_grasp, self.lift = q_grasp, lift
        self.t_close, self.t_lift0 = t_close, t_close + t_hold
        self.t_lift1, self.t_end = t_close + t_hold + t_lift, t_end

    def __call__(self, t):
        s = smoothstep(t / self.t_close)
        q = {j: s * v for j, v in self.q_grasp.items()}
        y = self.lift * smoothstep((t - self.t_lift0) / (self.t_lift1 - self.t_lift0))
        return q, y


class HandGraspSim:
    def __init__(self, scene, schedule, h=0.01, k_contact=2e4, k_floor=2e5,
                 friction=1.0, eps_v=1e-3, gravity=-9.81, newton_iters=25, newton_tol=1e-6):
        self.hand = AllegroHand(XML)
        self.X = scene["X"].astype(float)
        T, part = scene["T"], scene["part"]
        names = [str(n) for n in scene["part_names"]]
        self.body_names = [str(b) for b in scene["body_names"]]
        self.n = n = len(self.X)
        self.h, self.sched = h, schedule
        self.k_c, self.k_f, self.mu_f, self.eps = k_contact, k_floor, friction, eps_v * h
        self.iters, self.tol = newton_iters, newton_tol
        self.vb = vertex_bodies(scene)

        is_part = lambda pred: np.array([pred(nm) for nm in names])[part]
        verts = lambda mask: np.unique(T[mask])
        driven = verts(is_part(lambda nm: nm in ("palm", "wrist") or nm.endswith("_shaft")))
        fixed = np.zeros(n, bool)
        fixed[driven] = True
        self.fixed_v = fixed
        dof_fixed = np.repeat(fixed, 3)
        self.free, self.fix = np.where(~dof_fixed)[0], np.where(dof_fixed)[0]

        active = ~fixed[T].all(axis=1)
        self.T, self.part = T[active], part[active]
        mu, lam = lame(scene["E"][active], scene["nu"][active])
        self.mu, self.lam = mu.reshape(-1, 1), lam.reshape(-1, 1)
        self.J = deformation_jacobian(self.X, self.T)
        self.vol = volume(self.X, self.T)
        self.E_rest = energies.stable_neo_hookean_energy_x(self.X, self.J, self.mu, self.lam, self.vol)
        Mv = massmatrix(self.X, self.T, rho=scene["rho"][active].reshape(-1, 1)).diagonal()
        self.m = np.repeat(Mv, 3)
        self.f_g = np.zeros(3 * n)
        self.f_g[1::3] = gravity * Mv

        # contact surfaces: rubber skin of the tips, outer wall of the cup
        F_b, J_b, _ = igl.boundary_facets(T)
        self.cup_pid = names.index("cup")
        rubber_pids = [i for i, nm in enumerate(names) if nm.endswith("_tip_rubber")]
        self.rub_v = np.unique(F_b[np.isin(part[J_b], rubber_pids)])
        cup_f = F_b[part[J_b] == self.cup_pid]
        self.cup_all = verts(part == self.cup_pid)
        c0 = self.X[self.cup_all].mean(0)
        N = igl.per_face_normals(self.X, cup_f, np.array([0, 0, 1.0]))
        C = self.X[cup_f].mean(1) - c0
        radial = C[:, [0, 2]] / np.linalg.norm(C[:, [0, 2]], axis=1, keepdims=True)
        self.cup_v = np.unique(cup_f[(N[:, [0, 2]] * radial).sum(1) > 0.3])
        self.cup_h = float(np.median(igl.edge_lengths(self.X, cup_f)))
        self.floor_y = CUP_Y_BOTTOM

        self.t = 0.0
        self.x = self.X.reshape(-1).copy()
        self.x_prev = self.x.copy()
        self.pairs, self.pair_n, self.lam_n = np.zeros((0, 2), int), np.zeros((0, 3)), np.zeros(0)

    def prescribed(self, t):
        q, y = self.sched(t)
        A = body_transforms(self.hand, self.body_names, q)
        P = np.einsum("nij,nj->ni", A[self.vb, :3, :3], self.X) + A[self.vb, :3, 3]
        P[self.vb == self.body_names.index("cup")] = self.X[self.vb == self.body_names.index("cup")]
        P[:, 1] += y
        return P.reshape(-1)

    def find_pairs(self, x):
        Xc = x.reshape(-1, 3)
        tree = cKDTree(Xc[self.cup_v])
        dist, j = tree.query(Xc[self.rub_v], distance_upper_bound=4e-3)
        ok = np.isfinite(dist)
        p, c = self.rub_v[ok], self.cup_v[j[ok]]
        axis = Xc[self.cup_all].mean(0)
        n = Xc[c] - axis
        n[:, 1] = 0.0
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        r = Xc[p] - Xc[c]
        tang = np.linalg.norm(r - (r * n).sum(1, keepdims=True) * n, axis=1)
        keep = tang < 0.75 * self.cup_h
        return np.stack([p[keep], c[keep]], 1), n[keep]

    def _contact(self, x, need_hess):
        Xc = x.reshape(-1, 3)
        E, g, rows, cols, vals = 0.0, np.zeros_like(x), [], [], []
        if len(self.pairs):
            p, c, n = self.pairs[:, 0], self.pairs[:, 1], self.pair_n
            d = ((Xc[p] - Xc[c]) * n).sum(1)          # < 0: rubber inside the cup wall
            pen = np.minimum(d, 0.0)
            E += 0.5 * self.k_c * (pen ** 2).sum()
            fc = self.k_c * pen[:, None] * n
            Xn = self.x.reshape(-1, 3)
            du = (Xc[p] - Xn[p]) - (Xc[c] - Xn[c])
            u = du - (du * n).sum(1, keepdims=True) * n
            yv = np.linalg.norm(u, axis=1)
            e = self.eps
            f0 = np.where(yv < e, -yv ** 3 / (3 * e * e) + yv * yv / e + e / 3, yv)
            f1_y = np.where(yv < e, -yv / (e * e) + 2.0 / e, 1.0 / np.maximum(yv, 1e-30))
            w = self.mu_f * self.lam_n
            E += (w * f0).sum()
            ff = (w * f1_y)[:, None] * u
            for k in range(3):
                np.add.at(g, 3 * p + k, fc[:, k] + ff[:, k])
                np.add.at(g, 3 * c + k, -fc[:, k] - ff[:, k])
            if need_hess:
                nn = np.einsum("mi,mj->mij", n, n)
                B = nn * (self.k_c * (pen < 0))[:, None, None] + \
                    (np.eye(3)[None] - nn) * (w * f1_y)[:, None, None]
                for a, sa in ((p, 1.0), (c, -1.0)):
                    for b, sb in ((p, 1.0), (c, -1.0)):
                        ii = (3 * a[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
                        jj = (3 * b[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
                        rows.append(ii.ravel()); cols.append(jj.ravel())
                        vals.append((sa * sb * B).ravel())
        cy = Xc[self.cup_all, 1] - self.floor_y
        pen = np.minimum(cy, 0.0)
        E += 0.5 * self.k_f * (pen ** 2).sum()
        g[3 * self.cup_all + 1] += self.k_f * pen
        H = None
        if need_hess:
            act = self.cup_all[pen < 0]
            rows.append(3 * act + 1); cols.append(3 * act + 1)
            vals.append(np.full(len(act), self.k_f))
            N = x.shape[0]
            H = sp.sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                     np.concatenate(cols))), shape=(N, N)).tocsr()
        return E, g, H

    def _full(self, xf, xb):
        x = np.empty(3 * self.n)
        x[self.free], x[self.fix] = xf, xb
        return x

    def energy(self, xf, xb, x_tilde):
        x = self._full(xf, xb)
        E = energies.stable_neo_hookean_energy_x(x.reshape(-1, 3), self.J, self.mu, self.lam,
                                                 self.vol) - self.E_rest
        E += self._contact(x, False)[0] - self.f_g @ x
        r = xf - x_tilde
        return E + 0.5 / self.h ** 2 * (self.m[self.free] * r) @ r

    def grad_hess(self, xf, xb, x_tilde):
        x = self._full(xf, xb)
        Xm = x.reshape(-1, 3)
        g = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        H = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        _, gc, Hc = self._contact(x, True)
        g = g + gc - self.f_g
        mf = self.m[self.free] / self.h ** 2
        H = (H.tocsr() + Hc)[self.free][:, self.free] + sp.sparse.diags(mf)
        return g[self.free] + mf * (xf - x_tilde), H

    def step(self):
        t1 = self.t + self.h
        P1, P0 = self.prescribed(t1), self.prescribed(self.t)
        xb = P1[self.fix]
        v = self.x - self.x_prev
        x_tilde = (self.x + v)[self.free]
        self.pairs, self.pair_n = self.find_pairs(self.x)
        Xc = self.x.reshape(-1, 3)
        d = ((Xc[self.pairs[:, 0]] - Xc[self.pairs[:, 1]]) * self.pair_n).sum(1)
        self.lam_n = self.k_c * np.maximum(-d, 0.0)
        # warm start: hand vertices follow their link's FK increment
        x0 = self.x + v
        hand = self.vb != self.body_names.index("cup")
        x0.reshape(-1, 3)[hand] = (self.x + (P1 - P0)).reshape(-1, 3)[hand]
        xf = x0[self.free]
        f = lambda z: self.energy(z, xb, x_tilde)
        for it in range(self.iters):
            g, H = self.grad_hess(xf, xb, x_tilde)
            dx = solve_spd(H, -g)
            alpha, xf, _ = backtracking_line_search(f, xf, g, dx)
            if alpha == 0.0 or np.abs(alpha * dx).max() < self.tol:
                break
        self.x_prev, self.x, self.t = self.x, self._full(xf, xb), t1
        return it + 1

    def grip_force(self):
        if not len(self.pairs):
            return 0.0
        Xc = self.x.reshape(-1, 3)
        d = ((Xc[self.pairs[:, 0]] - Xc[self.pairs[:, 1]]) * self.pair_n).sum(1)
        return float(self.k_c * np.maximum(-d, 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    out = os.path.join(HERE, "output")
    scene = dict(np.load(os.path.join(out, "scene_tets.npz")))
    meta = json.load(open(os.path.join(out, "hand_meta.json")))
    sim = HandGraspSim(scene, HandSchedule(meta["grasp_q"]))
    print(f"free DOFs {len(sim.free)}, active tets {len(sim.T)}, rubber verts {len(sim.rub_v)}, "
          f"cup wall verts {len(sim.cup_v)}", flush=True)
    n_steps = args.steps or int(round(sim.sched.t_end / sim.h))
    frames, t0 = [sim.x.reshape(-1, 3).astype(np.float32)], time.time()
    for k in range(n_steps):
        its = sim.step()
        cup_y = sim.x.reshape(-1, 3)[sim.cup_all, 1].min() - sim.floor_y
        print(f"t={sim.t:5.2f}s newton={its:2d} pairs={len(sim.pairs):4d} "
              f"grip={sim.grip_force():7.2f} N cup lift={cup_y*1e3:6.2f} mm "
              f"[{time.time()-t0:6.1f}s]", flush=True)
        if (k + 1) % 2 == 0:
            frames.append(sim.x.reshape(-1, 3).astype(np.float32))
    np.savez_compressed(os.path.join(out, "hand_frames.npz"), frames=np.array(frames))


if __name__ == "__main__":
    main()
