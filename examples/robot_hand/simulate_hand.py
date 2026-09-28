"""Allegro hand closes on the cup and lifts it -- heterogeneous FEM in SimKit.

* **Actuation (joint space)** -- the palm and every steel joint shaft (found by
  winding number) are driven by forward kinematics of the joint trajectory
  ``q(t)`` (flat hand -> fitted grasp pose), plus a wrist lift. That is how the
  Allegro's motors act: through the joint shafts. Aluminium phalanges, rubber
  tips, silicone palmar pads and the cup are free elastic bodies.
* **Elasticity** -- SimKit stable Neo-Hookean, per-tet ``mu, lam``.
* **Time integration** -- backward Euler as incremental-potential minimisation
  (Newton + SimKit's backtracking line search) over the free DOFs.
* **Cup** -- a 6-DoF rigid body (mass/inertia from its tets) with an exact
  analytic SDF (``cup_sdf.py``: capped cones + rim torus). Its centre of mass
  and incremental rotation vector are unknowns of the same Newton solve.
* **Contact** -- every surface vertex of the rubber tips and silicone pads
  (``--contact tips`` for tips only) vs the cup SDF: penalty
  ``k/2 min(phi, 0)^2`` coupling the vertex with the cup's translation and
  rotation, + lagged smoothed Coulomb friction on the slip relative to the
  cup's material point; cup base ring vs table penalty.

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
import igl

import simkit.energies as energies
from simkit.deformation_jacobian import deformation_jacobian
from simkit.massmatrix import massmatrix
from simkit.volume import volume
from simkit.backtracking_line_search import backtracking_line_search

from allegro_kinematics import AllegroHand
from hand_geometry import XML, CUP_Y_BOTTOM, CUP_AXIS_XZ, CupParamsTall
from cup_sdf import CupSDF
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


def rotvec_to_R(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * v / (2 * np.sin(th))


def skew(r):
    """(m,3) -> (m,3,3) cross-product matrices, skew(r) @ w = r x w."""
    Z = np.zeros(len(r))
    return np.stack([np.stack([Z, -r[:, 2], r[:, 1]], -1),
                     np.stack([r[:, 2], Z, -r[:, 0]], -1),
                     np.stack([-r[:, 1], r[:, 0], Z], -1)], 1)


class RigidCup:
    """The cup as a 6-DoF rigid body with an analytic SDF.

    Pose: centre of mass ``c`` and rotation ``R`` (world = R @ (q - q_com) + c,
    ``q`` in the SDF's cup-local frame). Mass and inertia come from its tets.
    """

    def __init__(self, scene, cup_pid, cp=CupParamsTall()):
        T, X = scene["T"], scene["X"]
        cup_t = T[scene["part"] == cup_pid]
        rho = scene["rho"][scene["part"] == cup_pid]
        Xt = X[cup_t]
        vol = np.abs(np.einsum("ij,ij->i", Xt[:, 1] - Xt[:, 0],
                               np.cross(Xt[:, 2] - Xt[:, 0], Xt[:, 3] - Xt[:, 0]))) / 6
        mt = rho * vol
        self.m = mt.sum()
        self.c = (mt[:, None] * Xt.mean(1)).sum(0) / self.m
        # inertia about the COM, lumped to tet corners (a quarter each)
        r = (Xt - self.c).reshape(-1, 3)
        w = np.repeat(mt / 4, 4)
        self.I_body = (w[:, None, None] * ((r * r).sum(1)[:, None, None] * np.eye(3)
                                            - np.einsum("ni,nj->nij", r, r))).sum(0)
        self.origin0 = np.array([CUP_AXIS_XZ[0], CUP_Y_BOTTOM, CUP_AXIS_XZ[1]])
        self.q_com = self.c - self.origin0           # COM in cup-local coordinates
        self.R = np.eye(3)
        self.c_prev, self.R_prev = self.c.copy(), self.R.copy()
        self.sdf = CupSDF(cp)
        self.verts = np.unique(cup_t)
        self.X0 = X[self.verts] - self.c
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        self.base_ring = np.stack([cp.radius_bottom * np.cos(th), np.zeros_like(th),
                                   cp.radius_bottom * np.sin(th)], 1) - self.q_com

    def local(self, P, c, R):
        return (P - c) @ R + self.q_com                # R^T (p - c) + q_com

    def world_vertices(self):
        return self.X0 @ self.R.T + self.c


class HandGraspSim:
    def __init__(self, scene, schedule, h=0.01, k_contact=2e4, k_floor=2e5,
                 friction=1.0, eps_v=1e-3, gravity=-9.81, newton_iters=25, newton_tol=1e-6,
                 contact_parts=("_tip_rubber", "_pad")):
        self.hand = AllegroHand(XML)
        self.X = scene["X"].astype(float)
        T, part = scene["T"], scene["part"]
        names = [str(n) for n in scene["part_names"]]
        self.body_names = [str(b) for b in scene["body_names"]]
        self.n = n = len(self.X)
        self.h, self.sched = h, schedule
        self.k_c, self.k_f, self.mu_f, self.eps = k_contact, k_floor, friction, eps_v * h
        self.g = gravity
        self.iters, self.tol = newton_iters, newton_tol
        self.vb = vertex_bodies(scene)
        self.cup_pid = names.index("cup")
        self.cup = RigidCup(scene, self.cup_pid)

        is_part = lambda pred: np.array([pred(nm) for nm in names])[part]
        verts = lambda mask: np.unique(T[mask])
        # palm, wrist and steel shafts are driven by FK; the cup's FEM vertices
        # are replaced by the rigid body (they only follow its pose for output)
        driven = verts(is_part(lambda nm: nm in ("palm", "wrist", "cup") or nm.endswith("_shaft")))
        fixed = np.zeros(n, bool)
        fixed[driven] = True
        self.fixed_v = fixed
        dof_fixed = np.repeat(fixed, 3)
        self.free, self.fix = np.where(~dof_fixed)[0], np.where(dof_fixed)[0]
        self.nf = len(self.free)

        active = ~fixed[T].all(axis=1)
        self.T, self.part = T[active], part[active]
        mu, lam = lame(scene["E"][active], scene["nu"][active])
        self.mu, self.lam = mu.reshape(-1, 1), lam.reshape(-1, 1)
        # (J is sized by the largest vertex index used; the cup's vertices come
        # last and are not FEM, so pad it to all 3n DOFs)
        self.J = deformation_jacobian(self.X, self.T).tocsc()
        self.J.resize((self.J.shape[0], 3 * n))
        self.vol = volume(self.X, self.T)
        self.E_rest = energies.stable_neo_hookean_energy_x(self.X, self.J, self.mu, self.lam, self.vol)
        Mv = massmatrix(self.X, self.T, rho=scene["rho"][active].reshape(-1, 1)).diagonal()
        Mv = np.pad(Mv, (0, n - len(Mv)))
        self.m = np.repeat(Mv, 3)
        self.f_g = np.zeros(3 * n)
        self.f_g[1::3] = gravity * Mv

        # contact vertices: free surface vertices of the soft parts
        F_b, J_b, _ = igl.boundary_facets(T)
        pids = [i for i, nm in enumerate(names) if nm.endswith(tuple(contact_parts))]
        cv = np.unique(F_b[np.isin(part[J_b], pids)])
        self.contact_v = cv[~fixed[cv]]
        self.floor_y = CUP_Y_BOTTOM

        self.t = 0.0
        self.x = self.X.reshape(-1).copy()
        self.x_prev = self.x.copy()
        self.cand = np.zeros(0, int)

    # ------------------------------------------------------------------ #
    def prescribed(self, t):
        q, y = self.sched(t)
        A = body_transforms(self.hand, self.body_names, q)
        P = np.einsum("nij,nj->ni", A[self.vb, :3, :3], self.X) + A[self.vb, :3, 3]
        P[:, 1] += y
        P[self.cup.verts] = self.cup.world_vertices()
        return P.reshape(-1)

    def _begin_step(self):
        """Lag the contact candidates, normals, lever arms and normal forces."""
        Xc = self.x.reshape(-1, 3)
        P = Xc[self.contact_v]
        d, gl = self.cup.sdf.sdf_and_grad(self.cup.local(P, self.cup.c, self.cup.R))
        keep = d < 5e-3                                      # broad phase
        self.cand = self.contact_v[keep]
        self.n0 = gl[keep] @ self.cup.R.T                    # world normals
        self.r0 = P[keep] - self.cup.c
        self.lam_n = self.k_c * np.maximum(-d[keep], 0.0)
        self.xn = Xc[self.cand].copy()
        self.cn = self.cup.c.copy()

    def _contact(self, x, c, w, need_hess):
        """Soft-vertex vs rigid-cup SDF penalty + lagged friction + cup/floor.

        Returns energy, gradient over [x (3n), c (3), w (3)] and (optionally)
        a Gauss-Newton Hessian over the same vector.
        """
        N = 3 * self.n + 6
        ic, iw = 3 * self.n, 3 * self.n + 3
        g = np.zeros(N)
        E, rows, cols, vals = 0.0, [], [], []
        R = rotvec_to_R(w) @ self.cup.R

        def scatter_block(idx, B):
            # idx (m, 9) global dof indices, B (m, 9, 9)
            rows.append(np.repeat(idx, 9, axis=1).ravel())
            cols.append(np.tile(idx, (1, 9)).ravel())
            vals.append(B.ravel())

        p_ids = self.cand
        if len(p_ids):
            P = x.reshape(-1, 3)[p_ids]
            d, gl = self.cup.sdf.sdf_and_grad(self.cup.local(P, c, R))
            n = gl @ R.T
            r = P - c
            pen = np.minimum(d, 0.0)
            E += 0.5 * self.k_c * (pen ** 2).sum()
            Jn = np.concatenate([n, -n, np.cross(n, r)], 1)            # (m, 9)
            gn = (self.k_c * pen)[:, None] * Jn
            # friction: tangential slip of the vertex relative to the cup
            # material point under it, over the step (lagged n, r, lambda)
            n0, r0 = self.n0, self.r0
            Pt = np.eye(3)[None] - np.einsum("mi,mj->mij", n0, n0)
            du = (P - self.xn) - (c - self.cn) + np.einsum("mij,j->mi", skew(r0), w)
            u = np.einsum("mij,mj->mi", Pt, du)
            yv = np.linalg.norm(u, axis=1)
            e = self.eps
            f0 = np.where(yv < e, -yv ** 3 / (3 * e * e) + yv * yv / e + e / 3, yv)
            f1_y = np.where(yv < e, -yv / (e * e) + 2.0 / e, 1.0 / np.maximum(yv, 1e-30))
            wf = self.mu_f * self.lam_n
            E += (wf * f0).sum()
            Ju = np.concatenate([Pt, -Pt, Pt @ skew(r0)], 2)            # (m, 3, 9)
            gf = np.einsum("mki,mk->mi", Ju, (wf * f1_y)[:, None] * u)
            idx = np.concatenate([3 * p_ids[:, None] + np.arange(3), np.tile(np.arange(ic, ic + 6), (len(p_ids), 1))], 1)
            np.add.at(g, idx.ravel(), (gn + gf).ravel())
            if need_hess:
                B = (self.k_c * (pen < 0))[:, None, None] * np.einsum("mi,mj->mij", Jn, Jn) \
                    + (wf * f1_y)[:, None, None] * np.einsum("mki,mkj->mij", Ju, Ju)
                scatter_block(idx, B)
        # cup base ring vs floor
        A = self.cup.base_ring @ R.T
        dfl = A[:, 1] + c[1] - self.floor_y
        pen = np.minimum(dfl, 0.0)
        E += 0.5 * self.k_f * (pen ** 2).sum()
        Jf = np.concatenate([np.tile([0, 1.0, 0], (len(A), 1)), np.cross(A, [0, 1.0, 0])], 1)
        g[ic:ic + 6] += ((self.k_f * pen)[:, None] * Jf).sum(0)
        H = None
        if need_hess:
            Hf = self.k_f * np.einsum("mi,mj->ij", Jf[pen < 0], Jf[pen < 0])
            ii, jj = np.meshgrid(np.arange(ic, ic + 6), np.arange(ic, ic + 6), indexing="ij")
            rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(Hf.ravel())
            H = sp.sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                     np.concatenate(cols))), shape=(N, N)).tocsr()
        return E, g, H

    # ------------------------------------------------------------------ #
    def _unpack(self, z, xb):
        x = np.empty(3 * self.n)
        x[self.free], x[self.fix] = z[:self.nf], xb
        return x, z[self.nf:self.nf + 3], z[self.nf + 3:]

    def _cup_kinetic(self, c, w):
        cup, h = self.cup, self.h
        dc, dw = c - self.c_tilde, w - self.w_tilde
        E = 0.5 / h ** 2 * (cup.m * dc @ dc + dw @ self.I_w @ dw) - cup.m * self.g * c[1]
        gc = cup.m / h ** 2 * dc - np.array([0, cup.m * self.g, 0])
        gw = self.I_w @ dw / h ** 2
        return E, np.concatenate([gc, gw])

    def energy(self, z, xb, x_tilde):
        x, c, w = self._unpack(z, xb)
        E = energies.stable_neo_hookean_energy_x(x.reshape(-1, 3), self.J, self.mu, self.lam,
                                                 self.vol) - self.E_rest
        E += self._contact(x, c, w, False)[0] - self.f_g @ x + self._cup_kinetic(c, w)[0]
        r = z[:self.nf] - x_tilde
        return E + 0.5 / self.h ** 2 * (self.m[self.free] * r) @ r

    def grad_hess(self, z, xb, x_tilde):
        x, c, w = self._unpack(z, xb)
        Xm = x.reshape(-1, 3)
        ge = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        He = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        _, gc, Hc = self._contact(x, c, w, True)
        sel = np.concatenate([self.free, 3 * self.n + np.arange(6)])
        mf = self.m[self.free] / self.h ** 2
        g = gc[sel]
        g[:self.nf] += (ge - self.f_g)[self.free] + mf * (z[:self.nf] - x_tilde)
        g[self.nf:] += self._cup_kinetic(c, w)[1]
        H = Hc[sel][:, sel]
        He = He.tocsr()[self.free][:, self.free] + sp.sparse.diags(mf)
        Mc = sp.sparse.block_diag([sp.sparse.eye(3) * self.cup.m / self.h ** 2,
                                   sp.sparse.csr_matrix(self.I_w / self.h ** 2)])
        H = H + sp.sparse.block_diag([He, Mc])
        return g, H.tocsc()

    def step(self):
        t1 = self.t + self.h
        P1, P0 = self.prescribed(t1), self.prescribed(self.t)
        xb = P1[self.fix]
        v = self.x - self.x_prev
        x_tilde = (self.x + v)[self.free]
        cup = self.cup
        self.c_tilde = 2 * cup.c - cup.c_prev
        self.w_tilde = R_to_rotvec(cup.R @ cup.R_prev.T)
        self.I_w = cup.R @ cup.I_body @ cup.R.T
        self._begin_step()
        # warm start: hand vertices follow their link's FK increment
        x0 = self.x + v
        x0.reshape(-1, 3)[self.vb != self.body_names.index("cup")] = \
            (self.x + (P1 - P0)).reshape(-1, 3)[self.vb != self.body_names.index("cup")]
        z = np.concatenate([x0[self.free], self.c_tilde, self.w_tilde])
        f = lambda zz: self.energy(zz, xb, x_tilde)
        for it in range(self.iters):
            g, H = self.grad_hess(z, xb, x_tilde)
            dz = solve_spd(H, -g)
            alpha, z, _ = backtracking_line_search(f, z, g, dz)
            if alpha == 0.0 or np.abs(alpha * dz).max() < self.tol:
                break
        x, c, w = self._unpack(z, xb)
        cup.c_prev, cup.R_prev = cup.c.copy(), cup.R.copy()
        cup.c, cup.R = c.copy(), rotvec_to_R(w) @ cup.R
        x.reshape(-1, 3)[cup.verts] = cup.world_vertices()
        self.x_prev, self.x, self.t = self.x, x, t1
        return it + 1

    def contact_report(self):
        """(number of penetrating vertices, total normal force [N])."""
        P = self.x.reshape(-1, 3)[self.contact_v]
        d = self.cup.sdf(self.cup.local(P, self.cup.c, self.cup.R))
        pen = np.maximum(-d, 0.0)
        return int((pen > 0).sum()), float(self.k_c * pen.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--contact", choices=["soft", "tips"], default="soft",
                    help="vertices that touch the cup: rubber tips + silicone pads, or tips only")
    args = ap.parse_args()
    out = os.path.join(HERE, "output")
    scene = dict(np.load(os.path.join(out, "scene_tets.npz")))
    meta = json.load(open(os.path.join(out, "hand_meta.json")))
    parts = ("_tip_rubber",) if args.contact == "tips" else ("_tip_rubber", "_pad")
    sim = HandGraspSim(scene, HandSchedule(meta["grasp_q"]), contact_parts=parts)
    print(f"free DOFs {len(sim.free)} + 6 (rigid cup, m = {sim.cup.m*1e3:.1f} g), "
          f"active tets {len(sim.T)}, contact vertices {len(sim.contact_v)}", flush=True)
    n_steps = args.steps or int(round(sim.sched.t_end / sim.h))
    frames, t0 = [sim.x.reshape(-1, 3).astype(np.float32)], time.time()
    for k in range(n_steps):
        its = sim.step()
        n_c, f_c = sim.contact_report()
        cup_y = sim.cup.c[1] - (sim.cup.origin0[1] + sim.cup.q_com[1])   # COM rise
        print(f"t={sim.t:5.2f}s newton={its:2d} contacts={n_c:4d} "
              f"normal force={f_c:7.2f} N cup lift={cup_y*1e3:6.2f} mm "
              f"[{time.time()-t0:6.1f}s]", flush=True)
        if (k + 1) % 2 == 0:
            frames.append(sim.x.reshape(-1, 3).astype(np.float32))
    np.savez_compressed(os.path.join(out, "hand_frames.npz"), frames=np.array(frames))


if __name__ == "__main__":
    main()
