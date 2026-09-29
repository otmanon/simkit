"""Tendon-driven closing of the compliant hand: SimKit statics + backward Euler.

Model
-----
* **Elasticity** -- SimKit stable Neo-Hookean (``stable_neo_hookean_*_x``) with
  per-tet ``mu, lam`` from the winding-number material labels (stiff
  polyurethane palm/phalanges, 6 MPa elastomer flexures, 0.2 MPa pads). Stable NH
  has a nonzero rest energy; it is subtracted.
* **Base** -- every vertex on the bottom face of the wrist (``z = -30 mm``) is
  pinned; those DOFs are eliminated (hard Dirichlet).
* **Gravity** -- off by default (``--gravity``: ``-z``, hand upright); the
  soft thumb root flexure twists a lot under the thumb's own weight.
* **Tendons** -- mass springs (energy
  ``0.5 k (|d| - l0)^2``, vol = 1): one spring on the PALMAR side across every
  one of the 15 flexure joints (4 fingers x 3 + thumb x 3), from the palmar
  face of the block below the joint to the palmar face of the block above it
  (4 mm from the joint on each side; stiff vertices only). The flexures sit at
  the dorsal side, so shortening a palmar spring flexes its joint. Each end is
  embedded at its exact anchor point as an affine combination of its block's
  vertices, so coarse meshes keep the same tendon geometry.
* **One actuator** -- a single parameter ``a in [0, 1]`` sets every rest
  length ``l0_j = (1 - c_j a) l_rest_j``. The per-joint contraction ratio
  ``c_j = r_j theta_j / l_rest_j`` is a fixed routing constant (like the pulley
  radii of a real tendon hand): ``r_j`` is the tendon's moment arm about the
  flexure and ``theta_j`` the flexion wanted at ``a = 1``
  (``TARGET_DEG``: base joint of the fingers, middle, distal; thumb joints).
* **Statics** -- continuation in ``a`` (0 -> 1 in 12 steps), Newton over the
  free DOFs with CHOLMOD and SimKit's backtracking line search.
* **Dynamics** -- backward Euler as incremental-potential minimisation
  ``V(x) + 1/(2h^2) (x - x~)^T M (x - x~)``, ``x~ = x_n + h v_n``, SimKit lumped
  mass matrix; ``a(t)`` = smoothstep ramp over 1.2 s, then held to 2.0 s.

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

from sdm_geometry import OUT, HandParams, joints, tendon_anchors

try:
    from sksparse.cholmod import cho_factor

    def solve_spd(A, b):
        return cho_factor(A.tocsc()).solve(b)
except ImportError:  # pragma: no cover
    def solve_spd(A, b):
        return sp.sparse.linalg.spsolve(A.tocsc(), b)

K_TENDON = 1.0e5                     # N/m, every tendon spring
# Hinge pins: at both ends of every flexure's bending axis, a zero-length spring
# ties a point carried by the block below to the same point carried by the block
# above. Points on the axis stay put under pure flexion, so only flexion is left
# soft; abduction, twist, shear and stretch of the joint are penalised.
MAX_STEP_CONTACT = 1e-3              # m, largest vertex move per Newton iteration with contact on
K_PIN = 1.0e7                        # N/m, per pin (~150x the flexure's shear stiffness)
# flexion wanted at a = 1 (deg): fingers (base, middle, distal); thumb (0, 1, 2)
TARGET_DEG = {"finger": (32.0, 34.0, 26.0), "thumb": (30.0, 26.0, 26.0)}
# Gravity is off by default (before the hinge pins, the thumb's first flexure was
# soft in torsion and the 36 g thumb twisted ~1 rad under its own weight);
# ``--gravity`` turns it on, along -z with the hand upright.
GRAVITY = np.array([0.0, 0.0, 0.0])


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Hand:
    def __init__(self, scene, k_tendon=K_TENDON, target=TARGET_DEG):
        self.X = X = scene["X"].astype(float)
        self.T = T = scene["T"].astype(np.int64)
        self.n = n = len(X)
        self.part = scene["part"]
        self.names = [str(s) for s in scene["part_names"]]
        self.kinds = [str(s) for s in scene["part_kind"]]
        E, nu = scene["E"], scene["nu"]
        self.mu = (E / (2 * (1 + nu))).reshape(-1, 1)
        self.lam = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1)
        self.J = simkit.deformation_jacobian(X, T)
        self.vol = simkit.volume(X, T)
        self.E_rest = energies.stable_neo_hookean_energy_x(X, self.J, self.mu, self.lam, self.vol)
        self.Mv = simkit.massmatrix(X, T, rho=scene["rho"].reshape(-1, 1)).diagonal()
        self.m = np.repeat(self.Mv, 3)
        self.f_g = (self.Mv[:, None] * GRAVITY[None]).ravel()
        # pinned: the wrist's base plane; a coarse scene carries its own list (the coarse
        # vertices the fine base vertices are interpolated from, via mesh4PDE's P)
        self.pinned = (np.asarray(scene["pinned"], np.int64) if "pinned" in scene
                       else np.where(X[:, 2] < X[:, 2].min() + 1e-7)[0])
        fixed = np.zeros(3 * n, bool)
        fixed[(3 * self.pinned[:, None] + np.arange(3)).ravel()] = True
        self.free = np.where(~fixed)[0]
        self.on_surface = np.zeros(n, bool)
        self.Fb = igl.boundary_facets(T)[0]
        self.on_surface[np.unique(self.Fb)] = True
        self._pv = {}
        self._tendons(k_tendon, target)
        self._pins(K_PIN)
        self.contact = None                  # optional sdm_cup.Contact (rigid obstacle)

    def part_vertices(self, name, surface=True):
        """Vertices whose incident tets all belong to part ``name``. On a coarse
        mesh a thin part can have fewer than 4 such vertices; then every vertex
        of the part's tets is used (dropping the surface filter if needed)."""
        key = (name, surface)
        if key not in self._pv:
            pid = self.names.index(name)
            touch = np.zeros(self.n, bool)
            touch[np.unique(self.T[self.part == pid])] = True
            inside = touch.copy()
            inside[np.unique(self.T[self.part != pid])] = False
            cands = [inside & self.on_surface, touch & self.on_surface, touch] if surface \
                else [inside, touch]
            v = next((np.where(c)[0] for c in cands if c.sum() >= 4), np.where(cands[-1])[0])
            self._pv[key] = v
        return self._pv[key]

    # -------------------------------------------------------------- tendons
    def _tendons(self, k, target):
        edges, info, ends = [], [], []
        self._pin_specs = []
        p = HandParams()
        anchors = {j[0]: j for j in tendon_anchors(p)}
        for jt in joints(p):
            A = jt["A"]
            to_world = lambda q: A[:3, :3] @ q + A[:3, 3]
            ids, gaps = [], []
            _, below, pa, above, pb = anchors[jt["name"]]
            for part, qw in ((below, pa), (above, pb)):
                cand = self.part_vertices(part)
                d = np.linalg.norm(self.X[cand] - qw, axis=1)
                ids.append(cand[d.argmin()])
                gaps.append(d.min())
            # moment arm: distance of the tendon line from the flexure's
            # mid-plane axis (local x through (y_flex_mid, z_mid))
            name = jt["name"]
            finger = name.split("_")[0]
            idx = int(name[-1])
            if p.flex_side == "palmar":
                y_mid = (-p.thumb_half_t + p.thumb_flex_t / 2 if finger == "thumb"
                         else (p.palm_t - p.phal_t) / 2 + p.flex_t / 2)
            else:
                y_mid = (p.thumb_half_t - p.thumb_flex_t / 2 if finger == "thumb"
                         else p.palm_t - p.flex_dorsal_gap - p.flex_t / 2)
            hinge = to_world(np.array([0.0, y_mid, 0.5 * (jt["z0"] + jt["z1"])]))
            axis = A[:3, 0]
            a_, b_ = pa, pb                   # the exact anchor points (embedded below)
            u = (b_ - a_) / np.linalg.norm(b_ - a_)
            r = abs(np.dot(np.cross(u, axis), hinge - a_)) / np.linalg.norm(np.cross(u, axis))
            l_rest = np.linalg.norm(b_ - a_)
            th = np.radians(target["thumb" if finger == "thumb" else "finger"][idx])
            c = min(0.85, r * th / l_rest)
            if p.flex_side == "palmar":         # dorsal actuator: lengthens by r*theta
                c = -r * th / l_rest
            fv = self.part_vertices(name, surface=False)
            if len(fv) == 0:                    # flexure lost on a coarse mesh: span the block above
                fv = self.part_vertices(jt["above"], surface=False)
            xs = (self.X[fv] - hinge) @ axis
            self._pin_specs.append((jt["below"], jt["above"], hinge + xs.min() * axis,
                                    hinge + xs.max() * axis))
            edges.append(ids)
            ends.append(((below, pa), (above, pb)))
            info.append(dict(joint=name, finger=finger, index=idx, below=jt["below"],
                             above=jt["above"], l_rest=l_rest, arm=r, contraction=c,
                             target_deg=float(np.degrees(th)),
                             anchor_snap_mm=[g * 1e3 for g in gaps]))
        self.E_t = np.array(edges, np.int64)
        self.tendon_info = info
        self.l_rest = np.array([t["l_rest"] for t in info])
        self.c = np.array([t["contraction"] for t in info])
        self.ym = np.full((len(edges), 1), float(k))
        self.svol = np.ones((len(edges), 1))
        # each tendon end is embedded at its exact anchor point: an affine
        # combination of its block's vertices (``_affine_weights``), so a coarse
        # mesh without a vertex at the anchor keeps the same moment arm
        rows, cols, vals = [], [], []
        for t, ((pa_, qa), (pb_, qb)) in enumerate(ends):
            for part, q, sgn in ((pb_, qb, 1.0), (pa_, qa, -1.0)):
                vid, w = self._affine_weights(part, q)
                for dd in range(3):
                    rows += [3 * t + dd] * len(vid)
                    cols += list(3 * vid + dd)
                    vals += list(sgn * w)
        self.Gt = sp.sparse.csr_matrix((vals, (rows, cols)), shape=(3 * len(ends), 3 * self.n))
        self.fingers = list(dict.fromkeys(t["finger"] for t in info))

    def _affine_weights(self, part, q, k=64):
        """Weights w over the k vertices of ``part`` nearest to q with
        sum w_i [x_i, 1] = [q, 1]: q moves exactly with any affine (so any rigid)
        motion of the block."""
        cand = self.part_vertices(part, surface=False)
        ids = cand[np.argsort(np.linalg.norm(self.X[cand] - q, axis=1))[:k]]
        c = self.X[ids].mean(0)
        A = np.vstack([(self.X[ids] - c).T, np.ones(len(ids))])
        w = A.T @ np.linalg.solve(A @ A.T, np.append(q - c, 1.0))
        return ids, w

    def _pins(self, k):
        rows, cols, vals = [], [], []
        pts = []
        for below, above, *qs in self._pin_specs:
            for q in qs:
                r = len(pts)
                for part, sgn in ((below, 1.0), (above, -1.0)):
                    ids, w = self._affine_weights(part, q)
                    for d in range(3):
                        rows += [3 * r + d] * len(ids)
                        cols += list(3 * ids + d)
                        vals += list(sgn * w)
                pts.append(q)
        self.pin_points = np.array(pts)
        self.G = sp.sparse.csr_matrix((vals, (rows, cols)), shape=(3 * len(pts), 3 * self.n))
        self.g0 = self.G @ self.X.reshape(-1)
        self.k_pin = k
        self.H_pin = (k * (self.G.T @ self.G)).tocsr()

    def pin_gap(self, x):
        """Per-pin mismatch |p_below - p_above| (m)."""
        return np.linalg.norm((self.G @ x - self.g0).reshape(-1, 3), axis=1)

    def l0(self, a):
        return ((1.0 - self.c * a) * self.l_rest).reshape(-1, 1)

    def _tendon_d(self, x):
        d = (self.Gt @ x).reshape(-1, 3)
        return d, np.linalg.norm(d, axis=1)

    def tendon_forces(self, x, a):
        _, l = self._tendon_d(x)
        return self.ym.ravel() * (l - self.l0(a).ravel())

    def tendon_energy(self, x, a):
        _, l = self._tendon_d(x)
        return 0.5 * float((self.ym.ravel() * (l - self.l0(a).ravel()) ** 2).sum())

    def tendon_gradient(self, x, a):
        d, l = self._tendon_d(x)
        f = (self.ym.ravel() * (l - self.l0(a).ravel()) / l)[:, None] * d
        return self.Gt.T @ f.ravel()

    def tendon_hessian(self, x, a):
        """k [ (1 - l0/l)_+ I + (l0/l) u u^T ] per tendon (PSD), pulled back through Gt."""
        d, l = self._tendon_d(x)
        u = d / l[:, None]
        r = self.l0(a).ravel() / l
        k = self.ym.ravel()
        blk = (k * np.maximum(1 - r, 0))[:, None, None] * np.eye(3) + (k * r)[:, None, None] * \
            u[:, :, None] * u[:, None, :]
        m = len(l)
        Hb = sp.sparse.block_diag(list(blk), format="csr") if m else sp.sparse.csr_matrix((0, 0))
        return (self.Gt.T @ Hb @ self.Gt).tocsr()

    # -------------------------------------------------------------- energy (flat x)
    def energy(self, x, a):
        Xm = x.reshape(-1, 3)
        e = energies.stable_neo_hookean_energy_x(Xm, self.J, self.mu, self.lam, self.vol) - self.E_rest
        e += self.tendon_energy(x, a)
        r = self.G @ x - self.g0
        e += 0.5 * self.k_pin * (r @ r)
        if self.contact is not None:
            e += self.contact.energy(x)
        return float(e - self.f_g @ x)

    def gradient(self, x, a):
        Xm = x.reshape(-1, 3)
        g = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        g = g + self.tendon_gradient(x, a)
        g = g + self.k_pin * (self.G.T @ (self.G @ x - self.g0))
        if self.contact is not None:
            g = g + self.contact.gradient(x)
        return g - self.f_g

    def hessian(self, x, a):
        Xm = x.reshape(-1, 3)
        H = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        H = H + self.tendon_hessian(x, a)
        H = H + self.H_pin
        if self.contact is not None:
            H = H + self.contact.hessian(x)
        return H.tocsr()

    def minimize(self, x0, a, x_tilde=None, h=None, iters=60, tol=1e-5):
        """Newton over the free DOFs of V(x) [+ inertia when ``x_tilde`` is given],
        starting from ``x0``. Stops when the Newton step is below ``tol`` (m)
        everywhere. (With 7500:1 stiffness ratios a cold start converges slowly
        -- linearised rotations of the stiff blocks stretch them -- so callers
        pass an extrapolated start: the secant predictor in statics, x_n + h v_n
        in dynamics.)"""
        fr = self.free
        x = x0.copy()
        inert = x_tilde is not None
        w = self.m / h ** 2 if inert else None

        def f(xf):
            y = x.copy()
            y[fr] = xf
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
            dx = solve_spd(H[fr][:, fr], -gf)
            if np.abs(dx).max() < tol:
                break
            if self.contact is not None:        # no tunnelling through the cup wall in one step
                step = np.zeros_like(x)
                step[fr] = dx
                m = np.linalg.norm(step.reshape(-1, 3), axis=1).max()
                if m > MAX_STEP_CONTACT:
                    dx = dx * (MAX_STEP_CONTACT / m)
            alpha, xf, _ = backtracking_line_search(f, x[fr], gf, dx)
            x[fr] = xf
            if alpha == 0:
                break
        return x, it

    # -------------------------------------------------------------- measurements
    def rotation(self, x, name):
        v = self.part_vertices(name, surface=False)
        P0, P = self.X[v], x.reshape(-1, 3)[v]
        U, _, Vt = np.linalg.svd((P - P.mean(0)).T @ (P0 - P0.mean(0)))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    def joint_angles(self, x):
        """Flexion of every joint (deg): rotation of the block above relative to below."""
        ang = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return np.array([ang(self.rotation(x, t["below"]).T @ self.rotation(x, t["above"]))
                         for t in self.tendon_info])

    def tip_vertices(self, finger):
        """Surface vertices at the far end of the distal block (+ its pad)."""
        name = f"{finger}_ph2"
        v = self.part_vertices(name, surface=False)
        A = [j["A"] for j in joints() if j["name"] == f"{finger}_flex2"][0]
        s = (self.X[v] - A[:3, 3]) @ A[:3, 2]
        return v[s > s.max() - 1e-6]

    def fingertips(self, x):
        Xm = x.reshape(-1, 3)
        return np.array([Xm[self.tip_vertices(f)].mean(0) for f in self.fingers])

    def interpenetration(self, x, depth_tol=2e-4):
        """For every pair of bodies -- palm+wrist, the 15 phalanges and the 6
        pads -- the vertices of one inside the other's deformed surface (winding
        number) and the deepest one (distance to that surface). Pairs joined by
        a flexure, and pads vs the block they are fused to, are flagged
        'adjacent' (their overlap is the corner pinch at a bent flexure)."""
        Xm = x.reshape(-1, 3)
        blocks = [n for n, k in zip(self.names, self.kinds) if k in ("link", "pad")] + ["palm+wrist"]
        pid = {n: i for i, n in enumerate(self.names)}
        surf = {}
        for b in blocks:
            ids = [pid["palm"], pid["wrist"]] if b == "palm+wrist" else [pid[b]]
            F = igl.boundary_facets(self.T[np.isin(self.part, ids)])[0]
            surf[b] = (F, np.unique(F))
        host = lambda n: "palm+wrist" if n in ("palm", "palm_pad") else (
            n.replace("_pad", "_ph2") if n.endswith("_pad") else n)
        adjacent = {frozenset((host(t["below"]), t["above"])) for t in self.tendon_info}
        adjacent |= {frozenset((n, host(n))) for n in blocks if n.endswith("_pad")}
        # a pad also touches whatever its host block's neighbours are
        adjacent |= {frozenset((n, o)) for n in blocks if n.endswith("_pad")
                     for pair in list(adjacent) if host(n) in pair and n not in pair
                     for o in pair if o != host(n)}
        box = {b: (Xm[v].min(0), Xm[v].max(0)) for b, (F, v) in surf.items()}
        out = []
        for i, A in enumerate(blocks):
            for B in blocks[i + 1:]:
                (la, ha), (lb, hb) = box[A], box[B]
                if (np.maximum(la, lb) > np.minimum(ha, hb)).any():
                    continue
                depth, n_in = 0.0, 0
                for P, Q in ((A, B), (B, A)):
                    vP = surf[P][1]
                    vP = vP[~np.isin(vP, surf[Q][1])]          # skip shared (fused) vertices
                    if not len(vP):
                        continue
                    w = igl.winding_number(Xm, surf[Q][0], Xm[vP])
                    inside = vP[w > 0.5]
                    if len(inside):
                        d = igl.point_mesh_squared_distance(Xm[inside], Xm, surf[Q][0])[0]
                        depth = max(depth, float(np.sqrt(d.max())))
                        n_in += len(inside)
                if n_in and depth > depth_tol:
                    out.append(dict(pair=f"{A} / {B}", adjacent=frozenset((A, B)) in adjacent,
                                    n_inside=n_in, depth_mm=depth * 1e3))
        return out


def von_mises(hand, x):
    """Per-tet von Mises Cauchy stress (compressible neo-Hookean), Pa."""
    F = (hand.J @ x).reshape(-1, 3, 3)
    Jd = np.linalg.det(F)
    B = F @ np.transpose(F, (0, 2, 1))
    mu, lam = hand.mu.ravel(), hand.lam.ravel()
    sig = (mu[:, None, None] * (B - np.eye(3))
           + (lam * np.log(np.abs(Jd)))[:, None, None] * np.eye(3)) / Jd[:, None, None]
    s = sig - np.trace(sig, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * (s * s).sum((1, 2)))


def summarize_pen(pen):
    adj = [q for q in pen if q["adjacent"]]
    non = [q for q in pen if not q["adjacent"]]
    mx = lambda L: max((q["depth_mm"] for q in L), default=0.0)
    return dict(adjacent_max_mm=mx(adj), nonadjacent_max_mm=mx(non),
                nonadjacent_pairs=[f"{q['pair']} ({q['depth_mm']:.1f} mm)" for q in non])


def run(dynamics=True, n_static=12, h=1 / 60, t_ramp=1.2, t_end=2.0, scene_file="sdm_tets.npz",
        tag="", cup=None):
    """Statics + dynamics on ``output/<scene_file>``; writes ``sdm_sim<tag>.npz`` and
    ``sim_report<tag>.json`` (tag "" is the fine hand)."""
    scene = dict(np.load(os.path.join(OUT, scene_file)))
    hand = Hand(scene)
    if cup is not None:
        from sdm_cup import Contact
        hand.contact = Contact(hand, cup)
        print(f"rigid cup: R {cup.R*1e3:.0f} mm, centre {np.round(cup.c*1e3, 1)} mm, cubic penalty "
              f"k = {cup.k:g}, {len(hand.contact.v)} contact vertices")
    print(f"{hand.n} vertices, {len(hand.T)} tets, {len(hand.pinned)} pinned vertices, "
          f"{len(hand.E_t)} tendon springs (k = {K_TENDON:g} N/m)")
    for t in hand.tendon_info:
        print(f"  {t['joint']:13s} l_rest {t['l_rest']*1e3:5.1f} mm  arm {t['arm']*1e3:5.1f} mm  "
              f"c {t['contraction']:.3f}  (target {t['target_deg']:.0f} deg; anchor snap "
              f"{t['anchor_snap_mm'][0]:.1f}/{t['anchor_snap_mm'][1]:.1f} mm)")
    x0 = hand.X.ravel().copy()
    out = dict(E_t=hand.E_t, l_rest=hand.l_rest, contraction=hand.c, pinned=hand.pinned)
    info = dict(n_vertices=hand.n, n_tets=len(hand.T), n_pinned=len(hand.pinned),
                n_springs=len(hand.E_t), k_tendon=K_TENDON, target_deg=TARGET_DEG,
                tendons=hand.tendon_info, fingers=hand.fingers)

    # ---------------------------------------------------------- statics
    t0 = time.time()
    a_vals = np.linspace(0, 1, n_static + 1)
    xs, iters = [], []
    x = x0.copy()
    for a in a_vals:
        start = 2 * xs[-1] - xs[-2] if len(xs) > 1 else x    # secant predictor
        x, it = hand.minimize(start, a)
        xs.append(x.copy())
        iters.append(it)
        print(f"  a = {a:.3f}: {it} Newton iterations ({time.time() - t0:.0f} s)")
    t_static = time.time() - t0
    xs = np.array(xs)
    tips0 = hand.fingertips(x0)
    stat = []
    for a, xa in zip(a_vals, xs):
        ang = hand.joint_angles(xa)
        tip = hand.fingertips(xa)
        pen = summarize_pen(hand.interpenetration(xa))
        stat.append(dict(a=float(a), joint_deg=ang.round(1).tolist(),
                         tip_disp_mm=dict(zip(hand.fingers, (np.linalg.norm(tip - tips0, axis=1)
                                                             * 1e3).round(1).tolist())),
                         tendon_force_N=hand.tendon_forces(xa, a).round(2).tolist(), **pen))
    for s in stat[::4]:
        print(f"  a={s['a']:.2f} tip disp {s['tip_disp_mm']}  penetration adjacent "
              f"{s['adjacent_max_mm']:.1f} mm, non-adjacent {s['nonadjacent_pairs']}")
    print(f"statics: {len(a_vals)} steps, {sum(iters)} Newton iterations, {t_static:.1f} s")
    info.update(t_static_s=t_static, static_newton_iters=int(sum(iters)), statics=stat)
    out.update(static_a=a_vals, static_x=xs.astype(np.float32),
               static_vm=np.array([von_mises(hand, xa) for xa in xs]).astype(np.float32))

    # ---------------------------------------------------------- dynamics
    if dynamics:
        t0 = time.time()
        n_steps = int(round(t_end / h))
        x, v = x0.copy(), np.zeros_like(x0)
        frames, a_t, times, its = [x.copy()], [0.0], [0.0], []
        contact_t = []
        for k in range(1, n_steps + 1):
            t = k * h
            a = float(smoothstep(t / t_ramp))
            x_new, it = hand.minimize(x + h * v, a, x_tilde=x + h * v, h=h)
            v = (x_new - x) / h
            x = x_new
            frames.append(x.copy())
            a_t.append(a)
            times.append(t)
            its.append(it)
            if hand.contact is not None:
                contact_t.append(hand.contact.report(x))
            if k % 10 == 0:
                d = np.linalg.norm(hand.fingertips(x) - tips0, axis=1).mean()
                c = "" if hand.contact is None else (f" contact {contact_t[-1]['contact_vertices']} v, "
                                                     f"pen {contact_t[-1]['max_penetration_mm']:.2f} mm, "
                                                     f"F {contact_t[-1]['contact_force_N']:.2f} N")
                print(f"  t={t:.3f}s a={a:.3f} newton={it} mean tip disp {d*1e3:.1f} mm{c} "
                      f"({time.time() - t0:.0f}s)")
        t_dyn = time.time() - t0
        frames = np.array(frames)
        tipd = np.array([np.linalg.norm(hand.fingertips(f) - tips0, axis=1) for f in frames])
        print(f"dynamics: {n_steps} steps (h = {h:.4f} s), {sum(its)} Newton iterations, "
              f"{t_dyn:.1f} s")
        out.update(dyn_x=frames.astype(np.float32), dyn_a=np.array(a_t), dyn_t=np.array(times),
                   dyn_vm=np.array([von_mises(hand, f) for f in frames]).astype(np.float32),
                   dyn_tip_disp=tipd)
        info.update(t_dynamic_s=t_dyn, dyn_steps=n_steps, h=h, t_ramp=t_ramp, t_end=t_end,
                    dyn_newton_iters=int(sum(its)),
                    dyn_final_tip_disp_mm=dict(zip(hand.fingers, (tipd[-1] * 1e3).round(1).tolist())),
                    dyn_peak_tip_disp_mm=dict(zip(hand.fingers, (tipd.max(0) * 1e3).round(1).tolist())),
                    dyn_final_joint_deg=hand.joint_angles(frames[-1]).round(1).tolist(),
                    dyn_final_penetration=summarize_pen(hand.interpenetration(frames[-1])))
        if hand.contact is not None:
            info.update(cup=dict(center=hand.contact.cup.c.tolist(), R=hand.contact.cup.R,
                                 wall=hand.contact.cup.w, base=hand.contact.cup.b, length=hand.contact.cup.L,
                                 k=hand.contact.cup.k), contact=contact_t)
            out.update(dyn_contact_force=np.array([c["contact_force_N"] for c in contact_t]),
                       dyn_contact_pen=np.array([c["max_penetration_mm"] for c in contact_t]))
    np.savez_compressed(os.path.join(OUT, f"sdm_sim{tag}.npz"), **out)
    with open(os.path.join(OUT, f"sim_report{tag}.json"), "w") as f:
        json.dump(info, f, indent=1, default=float)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--gravity", action="store_true")
    args = ap.parse_args()
    if args.gravity:
        GRAVITY[:] = (0.0, 0.0, -9.81)
    run(dynamics=not args.no_dynamics)
