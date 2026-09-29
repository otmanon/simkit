"""A rigid cup with an analytic signed distance, and cubic-penalty contact against it.

The cup is an open cylinder (outer radius R, wall w, base b, length L) lying across the
palm with its axis along x, so the curling fingers wrap its circumference. It is rigid
and fixed in space (a kinematic obstacle). Its signed distance is exact for each
capped cylinder and combined as ``max(phi_outer, -phi_cavity)`` (solid minus cavity).

Contact energy, summed over the hand's surface vertices (lumped surface area a_v):

    E_c = k / 3 * sum_v a_v * max(0, -phi(x_v))^3

gradient ``-k a_v d^2 grad(phi)`` (d = penetration depth), Hessian
``2 k a_v d grad(phi) grad(phi)^T`` (Gauss-Newton: PSD, drops the SDF curvature term).

    python sdm_cup.py --target 300          # air vs cup on output/coarse/sdm_tets_300.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy as sp
import igl

mm = 1e-3


class Cup:
    def __init__(self, center=(-5 * mm, -40 * mm, 125 * mm), R=35 * mm, wall=3 * mm,
                 base=4 * mm, length=100 * mm, k=1e12, yaw=0.0, pitch=0.0):
        self.c = np.asarray(center, float)
        self.R, self.w, self.b, self.L, self.k = R, wall, base, length, k
        # orientation: the local axis is x, turned by `yaw` (deg, about z) then `pitch`
        # (deg, about y); world = Rm @ local
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp_ = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        Ry = np.array([[cp, 0, sp_], [0, 1.0, 0], [-sp_, 0, cp]])
        self.Rm = Ry @ Rz
        self.yaw, self.pitch = yaw, pitch

    @staticmethod
    def _capped(p, r, h0, h1):
        """Exact SDF of the solid cylinder radius r, axis x, x in [h0, h1] (local coords)."""
        q = np.stack([np.hypot(p[:, 1], p[:, 2]) - r, np.abs(p[:, 0] - 0.5 * (h0 + h1)) - 0.5 * (h1 - h0)], 1)
        return np.minimum(q.max(1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)

    def sdf(self, P):
        p = (np.atleast_2d(P) - self.c) @ self.Rm           # local coordinates
        h0, h1 = -self.L / 2, self.L / 2               # base at -x, open end at +x
        outer = self._capped(p, self.R, h0, h1)
        cavity = self._capped(p, self.R - self.w, h0 + self.b, h1 + 1.0)
        return np.maximum(outer, -cavity)

    def grad(self, P, eps=1e-7):
        """Gradient of the analytic SDF by central differences (unit length a.e.)."""
        g = np.empty_like(P)
        for i in range(3):
            e = np.zeros(3)
            e[i] = eps
            g[:, i] = (self.sdf(P + e) - self.sdf(P - e)) / (2 * eps)
        return g

    def mesh(self, n=96):
        """Triangle mesh of the cup for rendering (outer wall, rim, inner wall, base)."""
        import pyvista as pv
        ax = self.Rm[:, 0]
        outer = pv.Cylinder(center=self.c, direction=ax, radius=self.R, height=self.L,
                            resolution=n, capping=True).triangulate()
        cav = pv.Cylinder(center=self.c + ax * (self.b / 2 + 0.5 * mm), direction=ax,
                          radius=self.R - self.w, height=self.L - self.b + 1 * mm, resolution=n,
                          capping=True).triangulate()
        try:
            return outer.boolean_difference(cav)
        except Exception:
            return outer


class Contact:
    """Cubic-penalty contact of a Hand's surface vertices with a Cup."""

    def __init__(self, hand, cup):
        self.cup = cup
        self.v = np.unique(hand.Fb)
        A = 0.5 * igl.doublearea(hand.X, hand.Fb)
        a = np.zeros(hand.n)
        for j in range(3):
            np.add.at(a, hand.Fb[:, j], A / 3)
        self.a = a[self.v]

    def _eval(self, x):
        P = x.reshape(-1, 3)[self.v]
        phi = self.cup.sdf(P)
        act = phi < 0
        return P, phi, act

    def energy(self, x):
        _, phi, act = self._eval(x)
        return self.cup.k / 3 * float((self.a[act] * (-phi[act]) ** 3).sum())

    def gradient(self, x):
        P, phi, act = self._eval(x)
        g = np.zeros_like(x)
        if act.any():
            d, n = -phi[act], self.cup.grad(P[act])
            f = (-self.cup.k * self.a[act] * d ** 2)[:, None] * n
            idx = 3 * self.v[act][:, None] + np.arange(3)
            g[idx.ravel()] = f.ravel()
        return g

    def hessian(self, x):
        P, phi, act = self._eval(x)
        n3 = len(x)
        if not act.any():
            return sp.sparse.csr_matrix((n3, n3))
        d, n = -phi[act], self.cup.grad(P[act])
        w = 2 * self.cup.k * self.a[act] * d
        blocks = w[:, None, None] * n[:, :, None] * n[:, None, :]
        idx = 3 * self.v[act][:, None] + np.arange(3)
        rows = np.repeat(idx, 3, axis=1).ravel()
        cols = np.tile(idx, (1, 3)).ravel()
        return sp.sparse.csr_matrix((blocks.ravel(), (rows, cols)), shape=(n3, n3))

    def report(self, x):
        _, phi, act = self._eval(x)
        return dict(contact_vertices=int(act.sum()),
                    max_penetration_mm=float(-phi.min() * 1e3) if act.any() else 0.0,
                    contact_force_N=float(np.linalg.norm(self.gradient(x).reshape(-1, 3), axis=1).sum()))


def render(target=300):
    """Air vs cup: closed poses and a side-by-side video (renders/17_*, cup_<n>.mp4)."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from sdm_sim import OUT
    from sdm_coarse_render import kind_surface, kind_opts, legend
    pv.OFF_SCREEN = True
    R = os.path.join(OUT, "renders")
    sc = dict(np.load(os.path.join(OUT, "coarse", f"sdm_tets_{target}.npz")))
    kinds = [str(k) for k in sc["part_kind"]]
    air = dict(np.load(os.path.join(OUT, f"sdm_sim_{target}.npz")))
    cupd = dict(np.load(os.path.join(OUT, f"sdm_sim_{target}_cup.npz")))
    rep = json.load(open(os.path.join(OUT, f"sim_report_{target}_cup.json")))
    cup = Cup(center=rep["cup"]["center"], R=rep["cup"]["R"], wall=rep["cup"]["wall"],
              base=rep["cup"]["base"], length=rep["cup"]["length"], k=rep["cup"]["k"])
    cm = cup.mesh()
    views = {"three-quarter (palm side)": ((0.75, -1.0, 0.35), (0, 0, 1)),
             "side (thumb side)": ((-1.0, -0.25, 0.1), (0, 0, 1))}
    frame_bounds = kind_surface(sc["X"], sc["T"], sc["part"], kinds).merge(cm)

    def cam(pl, v, zoom=1.2):
        d, up = views[v]
        d = np.asarray(d, float) / np.linalg.norm(d)
        c = np.array(frame_bounds.bounds).reshape(3, 2).mean(1)
        pl.camera_position = [tuple(c + d), tuple(c), up]
        pl.reset_camera(bounds=frame_bounds.bounds)
        pl.camera.zoom(zoom)

    def still(x, with_cup, v):
        pl = pv.Plotter(window_size=(800, 860), off_screen=True)
        pl.set_background("white")
        pl.add_mesh(kind_surface(x.reshape(-1, 3).astype(float), sc["T"], sc["part"], kinds), **kind_opts(0.6))
        if with_cup:
            pl.add_mesh(cm, color="#d9d4c7", opacity=0.55, smooth_shading=True)
        cam(pl, v)
        pl.enable_anti_aliasing("ssaa")
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 10.4))
    cols = [("open (a = 0)", air["dyn_x"][0], False), ("closed in air (t = 2 s)", air["dyn_x"][-1], False),
            ("closed on the rigid cup (t = 2 s)", cupd["dyn_x"][-1], True)]
    for j, (t, x, wc) in enumerate(cols):
        for i, v in enumerate(views):
            ax[i, j].imshow(still(x, wc or j == 0, v))
            ax[i, j].axis("off")
        ax[0, j].set_title(t, fontsize=13)
    f = cupd["dyn_contact_force"]
    legend(fig)
    fig.suptitle(f"mesh4PDE hand, {len(sc['X'])} vertices / {len(sc['T'])} tets: closing in air vs on a rigid "
                 f"cup (analytic SDF, cubic penalty)  --  final contact force {f[-1]:.1f} N, max penetration "
                 f"{cupd['dyn_contact_pen'].max():.2f} mm", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(R, f"17_cup_{target}.png"), dpi=100)
    print(f"wrote renders/17_cup_{target}.png")

    surf0 = kind_surface(sc["X"], sc["T"], sc["part"], kinds)
    pid = surf0.point_data["vtkOriginalPointIds"]
    pls = []
    for lab, sim, wc in (("in air", air, False), ("rigid cup", cupd, True)):
        pl = pv.Plotter(window_size=(640, 700), off_screen=True)
        pl.set_background("white")
        s = surf0.copy()
        pl.add_mesh(s, **kind_opts(0.6))
        if wc:
            pl.add_mesh(cm, color="#d9d4c7", opacity=0.55, smooth_shading=True)
        cam(pl, "three-quarter (palm side)")
        txt = pl.add_text("", position="upper_left", font_size=11, color="black")
        pls.append((pl, s, sim, txt, lab))
    w = imageio.get_writer(os.path.join(R, f"cup_{target}.mp4"), fps=30, codec="libx264",
                           macro_block_size=1, quality=8)
    for k in range(len(air["dyn_x"])):
        imgs = []
        for pl, s, sim, txt, lab in pls:
            s.points = sim["dyn_x"][k].reshape(-1, 3).astype(float)[pid]
            extra = f"   F = {sim['dyn_contact_force'][k - 1]:.1f} N" if "dyn_contact_force" in sim and k else ""
            txt.SetText(2, f"{len(sc['X'])} v, {lab}   t = {sim['dyn_t'][k]:.2f} s   a = {sim['dyn_a'][k]:.2f}{extra}")
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for p in pls:
        p[0].close()
    print(f"wrote renders/cup_{target}.mp4")


if __name__ == "__main__":
    from sdm_sim import run, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--no-air", action="store_true")
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    if args.render_only:
        render(args.target)
        raise SystemExit
    scene = os.path.join("coarse", f"sdm_tets_{args.target}.npz")
    if not args.no_air:
        print(f"===== {args.target}: closing in air =====", flush=True)
        run(scene_file=scene, tag=f"_{args.target}")
    print(f"===== {args.target}: closing on the rigid cup =====", flush=True)
    run(scene_file=scene, tag=f"_{args.target}_cup", cup=Cup())
    render(args.target)

