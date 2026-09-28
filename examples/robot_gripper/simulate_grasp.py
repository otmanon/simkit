"""Close the gripper on the cup, then lift it -- a heterogeneous FEM sim in SimKit.

Everything lives in one tet mesh (``scene_tets.npz``) with per-tet materials
from the winding-number labelling. The physics:

* **Elasticity** -- SimKit's stable Neo-Hookean energy with per-tet ``mu, lam``
  (aluminium, steel, silicone and polystyrene in the same solve).
* **Time integration** -- backward Euler written as incremental-potential
  minimisation, solved with Newton + backtracking line search, exactly as
  :func:`simkit.integrators.backward_euler` does, but over the *free* DOFs only
  so actuated DOFs can be prescribed exactly.
* **Actuation** -- the housing and rail follow the robot flange (a vertical lift
  trajectory). Each jaw carriage (found by winding number) is driven along the
  rail by a prescribed stroke ``delta(t)``, like the lead-screw / rack in an
  electric parallel gripper. Fingers and pads are free and deform elastically.
* **Contact** -- pad-surface vs cup-surface node-to-plane penalty (pad face
  normal), two-way coupled so the silicone compresses; cup vs floor penalty.
* **Friction** -- lagged, smoothed Coulomb friction (IPC-style ``f0`` with lagged
  normal force and tangent basis) between pads and cup; without it the cup
  would slip out when lifted.

Output: ``grasp_frames.npz`` (vertex positions per saved frame + diagnostics).
"""
from __future__ import annotations

import os
import time
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from scipy.spatial import cKDTree
import igl

import simkit.energies as energies
from simkit.deformation_jacobian import deformation_jacobian
from simkit.massmatrix import massmatrix
from simkit.volume import volume
from simkit.backtracking_line_search import backtracking_line_search

from materials import lame

try:  # CHOLMOD is ~10x faster than SuperLU on these systems
    from sksparse.cholmod import cho_factor as _cho_factor

    def solve_spd(A, b):
        return _cho_factor(A.tocsc()).solve(b)
except ImportError:  # pragma: no cover - SuperLU fallback
    def solve_spd(A, b):
        return sp.sparse.linalg.spsolve(A.tocsc(), b)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class GraspSchedule:
    """Stroke ``delta(t)`` per jaw (towards the centre) and flange lift ``y(t)``."""

    def __init__(self, stroke=0.0072, lift=0.04, t_close=0.5, t_hold=0.1,
                 t_lift=0.6, t_end=1.4):
        self.stroke, self.lift = stroke, lift
        self.t_close, self.t_lift0 = t_close, t_close + t_hold
        self.t_lift1, self.t_end = t_close + t_hold + t_lift, t_end

    def __call__(self, t):
        d = self.stroke * smoothstep(t / self.t_close)
        y = self.lift * smoothstep((t - self.t_lift0) / (self.t_lift1 - self.t_lift0))
        return d, y


class GripperGraspSim:
    def __init__(self, scene, schedule, h=0.01, k_contact=2e5, k_floor=2e5,
                 friction=0.9, eps_v=1e-3, gravity=-9.81, newton_iters=25,
                 newton_tol=1e-6):
        self.X = scene["X"].astype(float)
        T, part = scene["T"], scene["part"]
        names = list(scene["part_names"])
        self.n = n = self.X.shape[0]
        self.h, self.sched = h, schedule
        self.k_c, self.k_f, self.mu_f, self.eps = k_contact, k_floor, friction, eps_v * h
        self.iters, self.tol = newton_iters, newton_tol

        pid = {nm: i for i, nm in enumerate(names)}
        tet_is = lambda *nms: np.isin(part, [pid[x] for x in nms])

        # --- vertex roles (from winding-number part labels) ------------------
        def verts(mask):
            return np.unique(T[mask])
        mount = verts(tet_is("housing", "rail"))
        self.side = np.sign(self.X[:, 0])
        carriage = verts(tet_is("carriage_left", "carriage_right"))
        jaw = verts(tet_is("carriage_left", "carriage_right", "finger_left",
                           "finger_right", "pad_left", "pad_right"))
        self.mount, self.carriage, self.jaw = mount, carriage, jaw
        fixed = np.zeros(n, bool)
        fixed[mount] = True
        fixed[carriage] = True
        self.fixed_v = fixed
        dof_fixed = np.repeat(fixed, 3)
        self.free = np.where(~dof_fixed)[0]
        self.fix = np.where(dof_fixed)[0]

        # --- active elements: any tet with a free vertex ---------------------
        active = ~fixed[T].all(axis=1)
        self.T = T[active]
        self.part = part[active]
        E, nu, rho = scene["E"][active], scene["nu"][active], scene["rho"][active]
        mu, lam = lame(E, nu)
        self.mu, self.lam = mu.reshape(-1, 1), lam.reshape(-1, 1)
        self.J = deformation_jacobian(self.X, self.T)
        self.vol = volume(self.X, self.T)
        self.strength = scene["strength"][active]
        # Stable Neo-Hookean has a non-zero rest energy; with GPa moduli it is
        # ~1e5 J, which would swamp the Armijo test in floating point. Remove it.
        self.E_rest = energies.stable_neo_hookean_energy_x(
            self.X, self.J, self.mu, self.lam, self.vol)

        Mv = massmatrix(self.X, self.T, rho=rho.reshape(-1, 1)).diagonal()
        self.m = np.repeat(Mv, 3)                      # lumped, per DOF
        self.f_g = np.zeros(3 * n)
        self.f_g[1::3] = gravity * Mv

        # --- contact surfaces ------------------------------------------------
        F_b, J_b, _ = igl.boundary_facets(T)          # boundary tris, owning tet
        N_b = igl.per_face_normals(self.X, F_b, np.array([0.0, 0.0, 1.0]))
        C_b = self.X[F_b].mean(axis=1)
        pad_face = np.isin(part[J_b], [pid["pad_left"], pid["pad_right"]]) & \
            (N_b[:, 0] * -np.sign(C_b[:, 0]) > 0.7)
        self.pad_v = np.unique(F_b[pad_face])
        self.pad_n = np.zeros((n, 3))
        self.pad_n[self.pad_v, 0] = -np.sign(self.X[self.pad_v, 0])
        cup_tet = part[J_b] == pid["cup"]
        radial = C_b[:, [0, 2]] / np.linalg.norm(C_b[:, [0, 2]], axis=1, keepdims=True)
        cup_out = cup_tet & ((N_b[:, [0, 2]] * radial).sum(1) > 0.3)
        self.cup_v = np.unique(F_b[cup_out])
        self.cup_all = verts(part == pid["cup"])
        self.cup_pid = pid["cup"]
        L = igl.edge_lengths(self.X, F_b[pad_face])
        self.pad_h = float(np.median(L))

        # --- state ------------------------------------------------------------
        self.t = 0.0
        self.x = self.X.reshape(-1).copy()
        self.x_prev = self.x.copy()
        self.pairs = np.zeros((0, 2), int)
        self.lam_n = np.zeros(0)

    # ------------------------------------------------------------------ #
    def prescribed(self, t):
        d, y = self.sched(t)
        P = self.X.copy()
        P[:, 1] += y
        P[self.carriage, 0] -= self.side[self.carriage] * d
        return P.reshape(-1)

    def find_pairs(self, x):
        """Pair each cup surface vertex with the nearest pad-face vertex whose
        face it projects onto. Returns (m,2) [cup_v, pad_v]."""
        Xc = x.reshape(-1, 3)
        tree = cKDTree(Xc[self.pad_v])
        dist, j = tree.query(Xc[self.cup_v], distance_upper_bound=4e-3)
        ok = np.isfinite(dist)
        c, p = self.cup_v[ok], self.pad_v[j[ok]]
        n = self.pad_n[p]
        r = Xc[c] - Xc[p]
        tang = np.linalg.norm(r - (r * n).sum(1, keepdims=True) * n, axis=1)
        keep = tang < 0.75 * self.pad_h
        return np.stack([c[keep], p[keep]], 1)

    # ------------------------------------------------------------------ #
    def _contact(self, x, need_hess):
        Xc = x.reshape(-1, 3)
        E, g, rows, cols, vals = 0.0, np.zeros_like(x), [], [], []
        # pad / cup normal penalty + lagged friction
        if len(self.pairs):
            c, p = self.pairs[:, 0], self.pairs[:, 1]
            n = self.pad_n[p]
            d = ((Xc[c] - Xc[p]) * n).sum(1)
            pen = np.minimum(d, 0.0)
            E += 0.5 * self.k_c * (pen ** 2).sum()
            fc = self.k_c * pen[:, None] * n
            # friction on relative tangential displacement over the step
            Xn = self.x.reshape(-1, 3)
            du = (Xc[c] - Xn[c]) - (Xc[p] - Xn[p])
            u = du - (du * n).sum(1, keepdims=True) * n
            y = np.linalg.norm(u, axis=1)
            e = self.eps
            f0 = np.where(y < e, -y ** 3 / (3 * e * e) + y * y / e + e / 3, y)
            f1_y = np.where(y < e, -y / (e * e) + 2.0 / e, 1.0 / np.maximum(y, 1e-30))
            w = self.mu_f * self.lam_n
            E += (w * f0).sum()
            ff = (w * f1_y)[:, None] * u
            for k in range(3):
                np.add.at(g, 3 * c + k, fc[:, k] + ff[:, k])
                np.add.at(g, 3 * p + k, -fc[:, k] - ff[:, k])
            if need_hess:
                act = pen < 0
                nn = np.einsum("mi,mj->mij", n, n) * (self.k_c * act)[:, None, None]
                Pt = (np.eye(3)[None] - np.einsum("mi,mj->mij", n, n)) * (w * f1_y)[:, None, None]
                B = nn + Pt
                for a, sa in ((c, 1.0), (p, -1.0)):
                    for b, sb in ((c, 1.0), (p, -1.0)):
                        ii = (3 * a[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
                        jj = (3 * b[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
                        rows.append(ii.ravel()); cols.append(jj.ravel())
                        vals.append((sa * sb * B).ravel())
        # cup vs floor (y = 0)
        cy = Xc[self.cup_all, 1]
        pen = np.minimum(cy, 0.0)
        E += 0.5 * self.k_f * (pen ** 2).sum()
        g[3 * self.cup_all + 1] += self.k_f * pen
        if need_hess:
            act = self.cup_all[pen < 0]
            rows.append(3 * act + 1); cols.append(3 * act + 1)
            vals.append(np.full(len(act), self.k_f))
        H = None
        if need_hess:
            N = x.shape[0]
            H = sp.sparse.coo_matrix((np.concatenate(vals) if vals else [],
                                      (np.concatenate(rows) if rows else [],
                                       np.concatenate(cols) if cols else [])),
                                     shape=(N, N)).tocsr()
        return E, g, H

    def _full(self, xf, xb):
        x = np.empty(3 * self.n)
        x[self.free] = xf
        x[self.fix] = xb
        return x

    def energy(self, xf, xb, x_tilde):
        x = self._full(xf, xb)
        Xm = x.reshape(-1, 3)
        E = energies.stable_neo_hookean_energy_x(Xm, self.J, self.mu, self.lam, self.vol) - self.E_rest
        E += self._contact(x, False)[0]
        E -= self.f_g @ x
        r = xf - x_tilde
        E += 0.5 / self.h ** 2 * (self.m[self.free] * r) @ r
        return E

    def grad_hess(self, xf, xb, x_tilde):
        x = self._full(xf, xb)
        Xm = x.reshape(-1, 3)
        g = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        H = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        _, gc, Hc = self._contact(x, True)
        g = g + gc - self.f_g
        H = (H.tocsr() + Hc)[self.free][:, self.free]
        mf = self.m[self.free] / self.h ** 2
        gf = g[self.free] + mf * (xf - x_tilde)
        H = H + sp.sparse.diags(mf)
        return gf, H

    # ------------------------------------------------------------------ #
    def step(self):
        t1 = self.t + self.h
        xb = self.prescribed(t1)[self.fix]
        v = self.x - self.x_prev
        x_tilde = (self.x + v)[self.free]
        # lag contact pairs + normal forces at the start of the step
        self.pairs = self.find_pairs(self.x)
        if len(self.pairs):
            Xc = self.x.reshape(-1, 3)
            c, p = self.pairs[:, 0], self.pairs[:, 1]
            d = ((Xc[c] - Xc[p]) * self.pad_n[p]).sum(1)
            self.lam_n = self.k_c * np.maximum(-d, 0.0)
        else:
            self.lam_n = np.zeros(0)

        # warm start: jaws follow their carriage rigidly, everything else inertial
        x0 = self.x + v
        dP = (self.prescribed(t1) - self.prescribed(self.t)).reshape(-1, 3)
        x0 = x0.reshape(-1, 3)
        x0[self.jaw] = self.x.reshape(-1, 3)[self.jaw] + dP[self.jaw]
        xf = x0.reshape(-1)[self.free]

        f = lambda z: self.energy(z, xb, x_tilde)
        for it in range(self.iters):
            g, H = self.grad_hess(xf, xb, x_tilde)
            dx = solve_spd(H, -g)
            alpha, xf, _ = backtracking_line_search(f, xf, g, dx)
            if alpha == 0.0 or np.abs(alpha * dx).max() < self.tol:
                break
        x_new = self._full(xf, xb)
        self.x_prev, self.x, self.t = self.x, x_new, t1
        return it + 1

    # ------------------------------------------------------------------ #
    def diagnostics(self):
        Xc = self.x.reshape(-1, 3)
        out = {}
        if len(self.pairs):
            c, p = self.pairs[:, 0], self.pairs[:, 1]
            d = ((Xc[c] - Xc[p]) * self.pad_n[p]).sum(1)
            f = self.k_c * np.maximum(-d, 0)
            s = self.side[p]
            out["grip_force_left"] = float(f[s < 0].sum())
            out["grip_force_right"] = float(f[s > 0].sum())
        else:
            out["grip_force_left"] = out["grip_force_right"] = 0.0
        out["cup_min_y"] = float(Xc[self.cup_all, 1].min())
        vm = self.von_mises()
        cup = self.part == self.cup_pid
        out["cup_max_von_mises"] = float(vm[cup].max())
        return out

    def von_mises(self):
        """Per-active-tet von Mises stress from the St. Venant-Kirchhoff
        second Piola-Kirchhoff stress pushed forward to Cauchy."""
        F = (self.J @ self.x.reshape(-1, 1)).reshape(-1, 3, 3)
        Ft = np.transpose(F, (0, 2, 1))
        Eg = 0.5 * (Ft @ F - np.eye(3))
        tr = np.trace(Eg, axis1=1, axis2=2)[:, None, None]
        S = self.lam[:, :, None] * tr * np.eye(3) + 2 * self.mu[:, :, None] * Eg
        detF = np.linalg.det(F)[:, None, None]
        sig = F @ S @ Ft / detF
        dev = sig - np.trace(sig, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
        return np.sqrt(1.5 * (dev * dev).sum(axis=(1, 2)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "output")
    scene = dict(np.load(os.path.join(data_dir, "scene_tets.npz")))
    sched = GraspSchedule()
    sim = GripperGraspSim(scene, sched)
    print(f"free DOFs: {len(sim.free)}, active tets: {len(sim.T)}, "
          f"pad verts: {len(sim.pad_v)}, cup surface verts: {len(sim.cup_v)}")
    frames, times, diags = [sim.x.reshape(-1, 3).astype(np.float32)], [0.0], []
    n_steps = int(round(sched.t_end / sim.h))
    t0 = time.time()
    for k in range(n_steps):
        its = sim.step()
        dg = sim.diagnostics()
        diags.append(dg)
        if (k + 1) % 2 == 0:
            frames.append(sim.x.reshape(-1, 3).astype(np.float32))
            times.append(sim.t)
        print(f"t={sim.t:5.2f}s newton={its:2d} pairs={len(sim.pairs):4d} "
              f"grip L/R={dg['grip_force_left']:6.2f}/{dg['grip_force_right']:6.2f} N "
              f"cup_y={dg['cup_min_y']*1e3:6.2f} mm "
              f"cup vM={dg['cup_max_von_mises']/1e6:6.3f} MPa  [{time.time()-t0:6.1f}s]",
              flush=True)
    vm = sim.von_mises()
    vm_full = np.zeros(len(scene["T"]))
    active = ~sim.fixed_v[scene["T"]].all(axis=1)
    vm_full[active] = vm
    np.savez_compressed(os.path.join(data_dir, "grasp_frames.npz"),
                        frames=np.array(frames), times=np.array(times),
                        von_mises=vm_full,
                        diagnostics=np.array([[d[k] for k in sorted(d)] for d in diags]),
                        diagnostic_names=np.array(sorted(diags[0])))


if __name__ == "__main__":
    main()
