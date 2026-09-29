"""Two soft fingers with stiff bones, closed by mass-spring tendons -- in 2D.

Each finger is a soft silicone body (rounded rectangle) containing two stiff
bones (proximal and distal phalanx); the soft gaps between bones are the
joints. The fingers hang from a pinned top edge. Two SimKit mass-spring
"tendons" per finger run along the inner side, across the base joint and the
middle joint. A single actuation parameter ``a in [0, 1]`` shortens every
tendon's rest length, ``l0(a) = (1 - c a) l_rest``, which curls both fingers
towards each other.

The script then

1. shows the fingers (materials, tendons, pins) open and statically closed,
2. shows the lowest eigenfunctions of the operator
   ``A = H_elastic + H_springs + Q_pin`` (generalized with the mass matrix),
3. coarsens the mesh with mesh4PDE (scored through those eigenfunctions) and
   with a geometry-only shortest-edge collapse to the same vertex count,
4. simulates the closing motion (backward Euler) on the fine mesh and in both
   coarse subspaces ``x = X + (P kron I) z`` and writes a side-by-side video.

    MESH4PDE=~/mesh4pde python two_finger_2d.py

Outputs go to ``output/``.
"""
from __future__ import annotations

import heapq
import os
import sys
import time
from collections import defaultdict

import numpy as np
import scipy as sp
import triangle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.path import Path

import simkit
import simkit.energies as energies
from simkit.backtracking_line_search import backtracking_line_search

sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import coarsen  # noqa: E402  (mesh4PDE)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- parameters
FINGER_X = 0.024          # finger centre lines at x = +-FINGER_X  [m]
WIDTH = 0.020
LENGTH = 0.090            # top (y = 0) to fingertip
BONE_W = 0.008
BONES = [(-0.012, -0.042), (-0.050, -0.076)]   # (top, bottom) of proximal, distal
MAX_AREA = 1.2e-6         # triangle area bound -> ~2000 vertices
SOFT = dict(E=3e5, nu=0.45, rho=1100.0)        # silicone
BONE = dict(E=5e9, nu=0.30, rho=1900.0)        # bone
K_TENDON = 1e5            # tendon stiffness [N/m] (2D: per unit thickness)
CONTRACTION = 0.30        # l0 = (1 - CONTRACTION * a) l_rest: fingertips just meet
GAMMA_PIN = 1e11
N_MODES = 16
TARGET = 150              # coarse vertex count
H = 1.0 / 60.0            # time step (one video frame)
T_CLOSE, T_END = 1.2, 2.0

MAT_COLOURS = {"soft": "#f2b8a2", "bone": "#f4f1e8"}
BONE_EDGE = "#6b5e4a"


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------- geometry
def finger_outline(xc, n_arc=24):
    r = WIDTH / 2
    pts = [(xc - r, 0.0)]
    yb = -LENGTH + r
    pts.append((xc - r, yb))
    for t in np.linspace(np.pi, 2 * np.pi, n_arc)[1:-1]:
        pts.append((xc + r * np.cos(t), yb + r * np.sin(t)))
    pts += [(xc + r, yb), (xc + r, 0.0)]
    return np.array(pts)


def bone_rects(xc):
    return [np.array([(xc - BONE_W / 2, y0), (xc - BONE_W / 2, y1),
                      (xc + BONE_W / 2, y1), (xc + BONE_W / 2, y0)]) for y0, y1 in BONES]


def build_mesh():
    V, S, holes = [], [], []

    def add_loop(P):
        o = len(V)
        V.extend(P.tolist())
        S.extend([(o + i, o + (i + 1) % len(P)) for i in range(len(P))])
    bones = []
    for s in (-1, 1):
        xc = s * FINGER_X
        add_loop(finger_outline(xc))
        for b in bone_rects(xc):
            add_loop(b)            # bone boundaries are mesh edges
            bones.append(b)
    m = triangle.triangulate(dict(vertices=np.array(V), segments=np.array(S)),
                             f"pq32a{MAX_AREA:.12f}")
    X, T = m["vertices"].astype(float), m["triangles"].astype(np.int64)
    # material labels by point-in-polygon of the centroid (2D winding number)
    C = X[T].mean(1)
    is_bone = np.zeros(len(T), bool)
    bone_id = -np.ones(len(T), int)
    for i, b in enumerate(bones):
        inside = Path(b).contains_points(C)
        is_bone |= inside
        bone_id[inside] = i
    return X, T, is_bone, bone_id, bones


def label_tris(X, T, bones):
    C = X[T].mean(1)
    lab = np.zeros(len(T), bool)
    for b in bones:
        lab |= Path(b).contains_points(C)
    return lab


# ----------------------------------------------------------------- physics
class Fingers:
    def __init__(self, X, T, is_bone):
        self.X, self.T = X, T
        self.n = len(X)
        E = np.where(is_bone, BONE["E"], SOFT["E"])
        nu = np.where(is_bone, BONE["nu"], SOFT["nu"])
        rho = np.where(is_bone, BONE["rho"], SOFT["rho"])
        self.mu = (E / (2 * (1 + nu))).reshape(-1, 1)
        self.lam = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1)
        self.J = simkit.deformation_jacobian(X, T)
        self.vol = simkit.volume(X, T)
        self.E_rest = energies.stable_neo_hookean_energy_x(X, self.J, self.mu, self.lam, self.vol)
        Mv = simkit.massmatrix(X, T, rho=rho.reshape(-1, 1)).diagonal()
        self.M = sp.sparse.diags(np.repeat(Mv, 2)).tocsc()
        self.f_g = np.zeros(2 * self.n)
        self.f_g[1::2] = -9.81 * Mv
        # pins: the top edge of both fingers
        self.pinned = np.where(X[:, 1] > -1e-9)[0]
        self.Q, self.b_pin = simkit.dirichlet_penalty(self.pinned, X[self.pinned], self.n, GAMMA_PIN)
        self.b_pin = np.asarray(self.b_pin).ravel()
        self.c_pin = 0.5 * GAMMA_PIN * (X[self.pinned] ** 2).sum()
        # tendons: inner side of each finger, across the base and middle joints
        E_t = []
        for s in (-1, 1):
            xc, inner = s * FINGER_X, -s
            xi = xc + inner * (BONE_W / 2 - 0.0005)       # on the bone's inner edge
            xo = xc + inner * (WIDTH / 2 - 0.0015)        # near the finger's inner skin
            (p0, p1), (d0, d1) = BONES
            span = [((xo, -0.0005), (xi, p0 - 0.004)),     # base joint
                    ((xi, p1 + 0.004), (xi, d0 - 0.004))]  # middle joint
            for a_, b_ in span:
                ia = np.argmin(np.linalg.norm(X - a_, axis=1))
                ib = np.argmin(np.linalg.norm(X - b_, axis=1))
                E_t.append((ia, ib))
        # route the middle-joint tendon off the neutral axis: attach it to the
        # finger's inner skin rather than the bone edge for a longer moment arm
        self.E_t = np.array(E_t)
        self.l_rest = np.linalg.norm(X[self.E_t[:, 0]] - X[self.E_t[:, 1]], axis=1)
        self.ym = np.full((len(self.E_t), 1), K_TENDON)
        self.svol = np.ones((len(self.E_t), 1))

    def l0(self, a):
        return ((1.0 - CONTRACTION * a) * self.l_rest).reshape(-1, 1)

    # potential energy V(x) at actuation a (x flattened, interleaved)
    def energy(self, x, a):
        Xm = x.reshape(-1, 2)
        E = energies.stable_neo_hookean_energy_x(Xm, self.J, self.mu, self.lam, self.vol) - self.E_rest
        E += energies.mass_springs_energy_x(Xm, self.E_t, self.ym, self.svol, self.l0(a))
        E += 0.5 * x @ (self.Q @ x) + self.b_pin @ x + self.c_pin - self.f_g @ x
        return E

    def gradient(self, x, a):
        Xm = x.reshape(-1, 2)
        g = energies.stable_neo_hookean_gradient_x(Xm, self.J, self.mu, self.lam, self.vol).ravel()
        g += energies.mass_springs_gradient_x(Xm, self.E_t, self.ym, self.svol, self.l0(a)).ravel()
        return g + self.Q @ x + self.b_pin - self.f_g

    def hessian(self, x, a):
        Xm = x.reshape(-1, 2)
        H = energies.stable_neo_hookean_hessian_x(Xm, self.J, self.mu, self.lam, self.vol, psd=True)
        H = H + energies.mass_springs_hessian_x(Xm, self.E_t, self.ym, self.svol, self.l0(a), psd=True)
        return (H + self.Q).tocsc()

    def rest_operator(self):
        return self.hessian(self.X.reshape(-1), 0.0)


def solve(f, x0, B=None, x_ref=None, iters=40, tol=1e-10):
    """Minimise f(x) (energy, gradient, hessian callables) with Newton + line
    search, either in full space or over x = x_ref + B z."""
    E, g, Hf = f
    if B is None:
        x = x0.copy()
        for _ in range(iters):
            gx = g(x)
            dx = sp.sparse.linalg.spsolve(Hf(x), -gx)
            alpha, x, _ = backtracking_line_search(E, x, gx, dx)
            if alpha == 0 or np.abs(alpha * dx).max() < tol:
                break
        return x
    z = np.linalg.lstsq(B.toarray() if sp.sparse.issparse(B) else B, x0 - x_ref, rcond=None)[0] \
        if x0 is not None else np.zeros(B.shape[1])
    Ez = lambda zz: E(x_ref + B @ zz)
    for _ in range(iters):
        xz = x_ref + B @ z
        gz = B.T @ g(xz)
        Hz = (B.T @ Hf(xz) @ B)
        Hz = Hz.toarray() if sp.sparse.issparse(Hz) else Hz
        dz = np.linalg.solve(Hz, -gz)
        alpha, z, _ = backtracking_line_search(Ez, z, gz, dz)
        if alpha == 0 or np.abs(B @ (alpha * dz)).max() < tol:
            break
    return x_ref + B @ z


# ----------------------------------------------------------------- 2D shortest-edge collapse
def shortest_edge_collapse(X, T, target):
    """Greedy shortest-edge collapse of a triangle mesh (geometry only).

    Interior edges collapse to their midpoint; an edge with one boundary
    endpoint collapses onto that endpoint; a boundary edge collapses along the
    boundary onto its endpoint of sharper turning angle, and only if the other
    endpoint is not a corner. Collapses must satisfy the link condition and may
    not flip or degenerate any triangle. Returns (Xc, Tc) with coarse vertices
    re-indexed."""
    X = X.copy()
    tris = {i: list(t) for i, t in enumerate(T)}
    v2t = defaultdict(set)
    for i, t in tris.items():
        for v in t:
            v2t[v].add(i)
    alive = np.ones(len(X), bool)

    def boundary_nbrs(v):
        cnt = defaultdict(int)
        for ti in v2t[v]:
            t = tris[ti]
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                if v in (a, b):
                    cnt[tuple(sorted((a, b)))] += 1
        return [a if b == v else b for (a, b), c in cnt.items() if c == 1]

    def turning(v):
        nb = boundary_nbrs(v)
        if len(nb) != 2:
            return np.pi
        e1, e2 = X[nb[0]] - X[v], X[nb[1]] - X[v]
        c = e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))
        return np.pi - np.arccos(np.clip(c, -1, 1))      # 0 = straight

    def signed_area(t, P):
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    heap, stamp = [], defaultdict(int)

    def push(a, b):
        a, b = min(a, b), max(a, b)
        heapq.heappush(heap, (np.linalg.norm(X[a] - X[b]), a, b, stamp[a] + stamp[b]))
    for t in T:
        for k in range(3):
            push(t[k], t[(k + 1) % 3])
    n_alive = len(X)
    corner = 0.35  # rad: boundary vertices turning more than this are features
    while n_alive > target and heap:
        _, a, b, st = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or st != stamp[a] + stamp[b]:
            continue
        shared = v2t[a] & v2t[b]
        if not shared:
            continue
        na = {v for ti in v2t[a] for v in tris[ti]} - {a}
        nb = {v for ti in v2t[b] for v in tris[ti]} - {b}
        opp = {v for ti in shared for v in tris[ti]} - {a, b}
        if (na & nb) != opp:
            continue                                   # link condition
        ba, bb = bool(boundary_nbrs(a)), bool(boundary_nbrs(b))
        is_bedge = len(shared) == 1
        if ba and bb:
            if not is_bedge:
                continue                               # would pinch the domain
            ta, tb = turning(a), turning(b)
            if min(ta, tb) > corner:
                continue                               # both corners
            keep, drop = (a, b) if ta >= tb else (b, a)
            p = X[keep].copy()
        elif ba or bb:
            keep, drop = (a, b) if ba else (b, a)
            p = X[keep].copy()
        else:
            keep, drop = a, b
            p = 0.5 * (X[a] + X[b])
        # validity: no flipped / degenerate triangles
        P = X.copy()
        P[keep] = p
        ok = True
        for ti in (v2t[a] | v2t[b]) - shared:
            t = [keep if v == drop else v for v in tris[ti]]
            a0 = signed_area(tris[ti], X)
            a1 = signed_area(t, P)
            if a1 * a0 <= 0 or abs(a1) < 1e-3 * abs(a0):
                ok = False
                break
        if not ok:
            continue
        for ti in shared:
            for v in tris[ti]:
                v2t[v].discard(ti)
            del tris[ti]
        for ti in list(v2t[drop]):
            tris[ti] = [keep if v == drop else v for v in tris[ti]]
            v2t[keep].add(ti)
        v2t[drop].clear()
        alive[drop] = False
        X[keep] = p
        n_alive -= 1
        stamp[keep] += 1
        for ti in v2t[keep]:
            for v in tris[ti]:
                if v != keep:
                    push(keep, v)
    used = np.where(alive)[0]
    remap = -np.ones(len(X), int)
    remap[used] = np.arange(len(used))
    Tc = np.array([[remap[v] for v in t] for t in tris.values()])
    return X[used], Tc


def barycentric_prolongation(Xf, Xc, Tc):
    """P (n_fine x n_coarse): each fine vertex in barycentric coordinates of the
    coarse triangle that contains it (or, failing that, the one it is least
    outside of) -- the same construction mesh4PDE uses."""
    A, B, C = Xc[Tc[:, 0]], Xc[Tc[:, 1]], Xc[Tc[:, 2]]
    v0, v1 = B - A, C - A
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    rows, cols, vals = [], [], []
    for i, p in enumerate(Xf):
        w = p - A
        l1 = (w[:, 0] * v1[:, 1] - w[:, 1] * v1[:, 0]) / den
        l2 = (v0[:, 0] * w[:, 1] - v0[:, 1] * w[:, 0]) / den
        l0 = 1 - l1 - l2
        worst = np.minimum(np.minimum(l0, l1), l2)
        t = np.argmax(worst)
        rows += [i] * 3
        cols += list(Tc[t])
        vals += [l0[t], l1[t], l2[t]]
    return sp.sparse.csc_matrix((vals, (rows, cols)), shape=(len(Xf), len(Xc)))


def boundary_is_manifold(T):
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    be = u[c == 1]
    deg = np.bincount(be.ravel(), minlength=T.max() + 1)
    return bool((c <= 2).all() and np.all((deg == 0) | (deg == 2)))


# ----------------------------------------------------------------- drawing
def draw_mesh(ax, X, T, is_bone, edges=True, lw=0.25, alpha=1.0):
    cols = [MAT_COLOURS["bone"] if b else MAT_COLOURS["soft"] for b in is_bone]
    pc = PolyCollection(X[T], facecolors=cols, edgecolors="#3d3d3a" if edges else "none",
                        linewidths=lw, alpha=alpha)
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.098, 0.006)
    ax.axis("off")


def main():
    t_all = time.time()
    X, T, is_bone, _, bones = build_mesh()
    fin = Fingers(X, T, is_bone)
    n = len(X)
    print(f"fine mesh: {n} vertices, {len(T)} triangles; {len(fin.pinned)} pinned; "
          f"{len(fin.E_t)} tendons")

    # ---- static closing (a = 1) on the fine mesh
    x_open = X.reshape(-1)
    closed = x_open
    for a in np.linspace(0.1, 1.0, 10):                 # continuation in a
        closed = solve((lambda x, a=a: fin.energy(x, a), lambda x, a=a: fin.gradient(x, a),
                        lambda x, a=a: fin.hessian(x, a)), closed)
    Xc_static = closed.reshape(-1, 2)
    tip = [np.argmin(np.linalg.norm(X - (s * FINGER_X, -LENGTH), axis=1)) for s in (-1, 1)]
    print("closed fingertips:", np.round(Xc_static[tip] * 1e3, 1), "mm")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, P, title in [(axes[0], X, "open (a = 0)"), (axes[1], Xc_static, "closed (a = 1), static")]:
        draw_mesh(ax, P, T, is_bone, lw=0.2)
        segs = [[P[i], P[j]] for i, j in fin.E_t]
        ax.add_collection(LineCollection(segs, colors="#2a78d6", linewidths=2.5))
        ax.plot(P[fin.pinned, 0], P[fin.pinned, 1], "s", ms=3, color="#3d3d3a")
        ax.set_title(title, fontsize=12)
    fig.suptitle(f"Two soft fingers with stiff bones ({n} vertices)\nsilicone E = "
                 f"{SOFT['E']/1e3:.0f} kPa, bone E = {BONE['E']/1e9:.0f} GPa; blue = "
                 "mass-spring tendons, squares = pinned", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "fingers.png"), dpi=140)
    plt.close(fig)

    # ---- eigenfunctions of the rest operator
    A = fin.rest_operator()
    lam, Bm = simkit.eigs(A, k=N_MODES, M=fin.M)
    order = np.argsort(lam)
    lam, Bm = lam[order], Bm[:, order]
    print("lowest eigenvalues:", np.round(lam[:6], 1))
    nshow = 12
    fig, axes = plt.subplots(2, 6, figsize=(16, 6.4))
    for i, ax in enumerate(axes.ravel()[:nshow]):
        u = Bm[:, i].reshape(-1, 2)
        mag = np.linalg.norm(u, axis=1)
        P = X + u * (0.008 / mag.max())
        tpc = ax.tripcolor(P[:, 0], P[:, 1], T, mag / mag.max(), cmap="magma",
                           shading="gouraud", vmin=0, vmax=1)
        ax.triplot(P[:, 0], P[:, 1], T[is_bone], color="#ffffff", lw=0.15, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.1, 0.006)
        ax.axis("off")
        ax.set_title(f"mode {i+1}\n$\\lambda$ = {lam[i]:.3g}, f = {np.sqrt(lam[i])/(2*np.pi):.1f} Hz",
                     fontsize=10)
    fig.colorbar(tpc, ax=axes, fraction=0.015, pad=0.01, label="|displacement| (normalised)")
    fig.suptitle("Lowest eigenfunctions of A = H_elastic + H_tendons + Q_pin "
                 "(A phi = lambda M phi; bones outlined in white)", fontsize=12)
    fig.savefig(os.path.join(OUT, "eigenfunctions.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- coarsening
    t0 = time.time()
    Xm, Tm, Pm = coarsen(X=X, T=T, B=Bm, eigenvalues=lam, target_vertices=TARGET)
    print(f"mesh4PDE: {len(Xm)} vertices [{time.time()-t0:.1f}s]")
    t0 = time.time()
    Xs, Ts = shortest_edge_collapse(X, T, len(Xm))
    Ps = barycentric_prolongation(X, Xs, Ts)
    print(f"shortest-edge: {len(Xs)} vertices [{time.time()-t0:.1f}s]")
    coarse = {"mesh4PDE (PDE-aware)": (Xm, Tm, sp.sparse.csc_matrix(Pm)),
              "shortest-edge (geometry only)": (Xs, Ts, Ps)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    draw_mesh(axes[0], X, T, is_bone, lw=0.15)
    axes[0].set_title(f"fine\n{n} vertices, {len(T)} triangles", fontsize=12)
    for ax, (name, (Xc, Tc, _)) in zip(axes[1:], coarse.items()):
        lab = label_tris(Xc, Tc, bones)
        draw_mesh(ax, Xc, Tc, lab, lw=0.6)
        nb = int(lab.sum())
        ax.set_title(f"{name}\n{len(Xc)} vertices, {len(Tc)} triangles ({nb} in bone), "
                     f"{'manifold' if boundary_is_manifold(Tc) else 'NON-manifold'}", fontsize=12)
    fig.suptitle(f"Coarsened to {len(Xm)} vertices; mesh4PDE scores collapses through the "
                 f"lowest {N_MODES} eigenfunctions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, "coarse_meshes.png"), dpi=140)
    plt.close(fig)

    # ---- simulation: fine, and in both coarse subspaces
    steps = int(round(T_END / H))
    subsp = {k: sp.sparse.kron(P, sp.sparse.identity(2)).tocsc() for k, (_, _, P) in coarse.items()}
    runs = {"fine": None, **subsp}
    traj = {k: [x_open.copy()] for k in runs}
    prev = {k: x_open.copy() for k in runs}
    for k, B in runs.items():
        t0 = time.time()
        x, xp = x_open.copy(), x_open.copy()
        for s in range(steps):
            a = smoothstep((s + 1) * H / T_CLOSE)
            xt = 2 * x - xp
            E_ = lambda y: fin.energy(y, a) + 0.5 / H ** 2 * (y - xt) @ (fin.M @ (y - xt))
            g_ = lambda y: fin.gradient(y, a) + fin.M @ (y - xt) / H ** 2
            H_ = lambda y: fin.hessian(y, a) + fin.M / H ** 2
            xn = solve((E_, g_, H_), xt if B is None else x, B=B, x_ref=None if B is None else x_open,
                       iters=12, tol=1e-9)
            xp, x = x, xn
            traj[k].append(x.copy())
        print(f"simulated {k}: {steps} steps [{time.time()-t0:.1f}s]")
    tips_err = {}
    fine_tips = np.array([t.reshape(-1, 2)[tip] for t in traj["fine"]])
    for k in subsp:
        tt = np.array([t.reshape(-1, 2)[tip] for t in traj[k]])
        tips_err[k] = np.linalg.norm(tt - fine_tips, axis=2).max(1)
        print(f"  {k}: max fingertip error over the motion {tips_err[k].max()*1e3:.2f} mm, "
              f"at the end {tips_err[k][-1]*1e3:.2f} mm")

    # ---- video
    import imageio.v2 as imageio
    writer = imageio.get_writer(os.path.join(OUT, "two_fingers.mp4"), fps=int(round(1 / H)),
                                codec="libx264", quality=8, macro_block_size=1)
    for f in range(len(traj["fine"])):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), dpi=100)
        a = smoothstep(f * H / T_CLOSE)
        xf = traj["fine"][f].reshape(-1, 2)
        draw_mesh(axes[0], xf, T, is_bone, lw=0.1)
        axes[0].set_title(f"fine ({n} vertices)", fontsize=11)
        for ax, (k, (Xc, Tc, P)) in zip(axes[1:], coarse.items()):
            B = subsp[k]
            z = sp.sparse.linalg.lsqr(B, traj[k][f] - x_open)[0] if f else np.zeros(B.shape[1])
            xc = Xc + z.reshape(-1, 2)
            draw_mesh(ax, xc, Tc, label_tris(Xc, Tc, bones), lw=0.5)
            ax.plot(xf[tip, 0], xf[tip, 1], "o", ms=5, mfc="none", mec="#2a78d6", mew=1.5)
            ax.set_title(f"{k} ({len(Xc)} vertices)\nfingertip error {tips_err[k][f]*1e3:.2f} mm "
                         "(blue ring = fine)", fontsize=10)
        fig.suptitle(f"t = {f*H:.2f} s   actuation a = {a:.2f}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    writer.close()
    np.savez_compressed(os.path.join(OUT, "results.npz"), X=X, T=T, is_bone=is_bone,
                        eigenvalues=lam, B=Bm, fine_traj=np.array(traj["fine"]),
                        tips_err_m4p=tips_err["mesh4PDE (PDE-aware)"],
                        tips_err_se=tips_err["shortest-edge (geometry only)"])
    print(f"done [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
