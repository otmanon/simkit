"""Every intermediate step of the hyper-reduced hand, rendered.

    python sdm_reduced_render.py [--setup] [--modes] [--statics] [--compare] [--video]

18_reduced_setup.png   per level: the coarse integration mesh (materials); the fine
                       mesh coloured by how it sits in the coarse tets (min P weight);
                       the coarse DOFs the tendons / hinge pins / wrist base act on;
                       the reduced (Galerkin) mass B^T M B lumped per coarse vertex
19_reduced_modes.png   lowest eigenmodes of each reduced system at a = 0, prolonged
                       onto the fine hand (x_fine = B x), next to the full space
20_reduced_statics.png the actuation sweep a = 0 .. 1 per level (prolonged fine hand)
21_reduced_compare.png fingertip curves, error vs the full-space solve, cost
reduced_closing.mp4    backward-Euler closing, every level side by side
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy as sp
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import simkit

import sdm_reduced as R
from sdm_render import tet_grid, shot, KIND_COL, KIND_LAB
from sdm_coarse_render import kind_surface, kind_opts, ORDER

pv.OFF_SCREEN = True
OUT = R.OUT
REN = os.path.join(OUT, "renders")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # coarse -> finer (ordinal blue)
VIEW = "three-quarter (palm side)"


def levels():
    s = json.load(open(os.path.join(R.CO, "coarse_summary.json")))
    return [r["target"] for r in s["levels"]]


class Fine:
    """The fine hand's surface, drawn at prolonged positions."""

    def __init__(self):
        sc = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
        self.sc = sc
        self.kinds = [str(k) for k in sc["part_kind"]]
        self.X = sc["X"]
        self.surf = kind_surface(sc["X"], sc["T"], sc["part"], self.kinds)
        self.pid = self.surf.point_data["vtkOriginalPointIds"]
        self.bounds = self.surf.bounds

    def at(self, xf):
        s = self.surf.copy()
        s.points = np.asarray(xf, float)[self.pid]
        return s


def fig_setup(F):
    lv = levels()
    hf = R.fine_hand()
    cols = ["coarse integration mesh\n(Neo-Hookean cubature, winding-number materials)",
            "fine mesh inside the coarse tets\n(min barycentric weight of its P row)",
            "coarse DOFs the springs act on\n(tendons red, hinge pins orange, wrist green)",
            "reduced mass  B^T M B\n(lumped, per coarse vertex)"]
    fig, ax = plt.subplots(len(lv), 4, figsize=(19, 4.9 * len(lv)))
    for i, L in enumerate(lv):
        sysd = R.build_hand_system(L)
        sc, P, Xc = sysd["scene"], sysd["P"], sysd["X"]
        csurf = kind_surface(Xc, sysd["T"], sc["part"], F.kinds)
        ax[i, 0].imshow(shot(lambda pl: pl.add_mesh(csurf, **kind_opts(0.5)), VIEW, size=(700, 780),
                             bounds_mesh=F.surf))
        Pr = P.tocsr()
        wmin = np.minimum.reduceat(Pr.data, Pr.indptr[:-1])
        fs = F.surf.copy()
        fs.point_data["w"] = np.clip(wmin[F.pid], -1.0, 0.0)

        def add_w(pl, fs=fs):
            pl.add_mesh(fs, scalars="w", cmap="magma", clim=(-1, 0), show_scalar_bar=False)
        ax[i, 1].imshow(shot(add_w, VIEW, size=(700, 780), bounds_mesh=F.surf))
        supp = {}
        for name, G in (("tendons", sysd["Gt"]), ("hinges", sysd["Gp"]), ("wrist", sysd["Sb"])):
            supp[name] = np.unique(G.tocsc().indices[:0]) if G.nnz == 0 else \
                np.unique(np.nonzero(np.abs(G).sum(0).A1)[0] // 3)

        def add_s(pl, supp=supp, Xc=Xc):
            pl.add_mesh(F.surf, color="#d8d2c4", opacity=0.25)
            pl.add_mesh(csurf, color="#8a8d91", opacity=0.25, show_edges=True, edge_color="#555555",
                        line_width=0.3)
            for name, col, r in (("hinges", "#f28e2b", 9), ("tendons", "#d62728", 12), ("wrist", "#1baf7a", 14)):
                if len(supp[name]):
                    pl.add_mesh(pv.PolyData(Xc[supp[name]]), color=col, point_size=r,
                                render_points_as_spheres=True)
            for (a_, pa), (b_, pb) in [((0, t["below"]), (0, t["above"])) for t in hf.tendon_info][:0]:
                pass
        ax[i, 2].imshow(shot(add_s, VIEW, size=(700, 780), bounds_mesh=F.surf))
        m = np.asarray(sysd["M"].sum(1)).ravel()[::3]

        def add_m(pl, m=m, Xc=Xc):
            pl.add_mesh(csurf, color="#d8d2c4", opacity=0.3)
            pc = pv.PolyData(Xc)
            pc.point_data["m"] = np.log10(np.maximum(m, 1e-9))
            pl.add_mesh(pc, scalars="m", cmap="viridis", point_size=9, render_points_as_spheres=True,
                        scalar_bar_args=dict(title="log10 mass [kg]", vertical=True, position_x=0.83,
                                             position_y=0.2, height=0.6, width=0.07, color="black",
                                             title_font_size=14, label_font_size=12, fmt="%.1f"))
        ax[i, 3].imshow(shot(add_m, VIEW, size=(700, 780), bounds_mesh=F.surf))
        for j in range(4):
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=12)
        ax[i, 0].text(-0.04, 0.5, f"level {L}\n{len(Xc):,} vertices\n{sysd['n_dof']:,} DOFs\n"
                      f"{sysd['n_tets']:,} tets", transform=ax[i, 0].transAxes, ha="right", va="center",
                      fontsize=13)
        ax[i, 1].text(0.5, -0.02, f"{(wmin < -1e-9).mean() * 100:.0f}% of fine vertices extrapolated, "
                      f"min weight {wmin.min():.0f}", transform=ax[i, 1].transAxes, ha="center",
                      va="top", fontsize=10, color=INK2)
        ax[i, 2].text(0.5, -0.02, f"tendons {len(supp['tendons'])} coarse v, hinge pins "
                      f"{len(supp['hinges'])}, wrist {len(supp['wrist'])}",
                      transform=ax[i, 2].transAxes, ha="center", va="top", fontsize=10, color=INK2)
        ax[i, 3].text(0.5, -0.02, f"total {m.sum() * 1e3:.0f} g (fine {hf.m.sum() / 3 * 1e3:.0f} g)",
                      transform=ax[i, 3].transAxes, ha="center", va="top", fontsize=10, color=INK2)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER] +
               [Patch(color="black", label="magma: P-row min weight, black = -1 (far outside), "
                                           "light = 0 (inside)")],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Building the hyper-reduced hand: x_fine = B x, B = kron(P, I3) from mesh4PDE", fontsize=15)
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.975))
    fig.savefig(os.path.join(REN, "18_reduced_setup.png"), dpi=80)
    print("wrote renders/18_reduced_setup.png")


def fig_modes(F, k=5):
    rows = levels() + [None]
    fig, ax = plt.subplots(len(rows), k, figsize=(3.6 * k, 3.9 * len(rows)))
    for i, L in enumerate(rows):
        sysd = R.build_hand_system(L)
        H = sysd["system"](sysd["x0"], 0.0)[2].tocsc()
        lam, V = simkit.eigs(H, k=k, M=sysd["M"])
        o = np.argsort(lam)
        lam, V = lam[o], V[:, o]
        for j in range(k):
            u = sysd["P"] @ V[:, j].reshape(-1, 3)
            mag = np.linalg.norm(u, axis=1)
            s = F.at(F.X + u * (0.02 / mag.max()))
            s.point_data["m"] = (mag / mag.max())[F.pid]
            ax[i, j].imshow(shot(lambda pl, s=s: pl.add_mesh(s, scalars="m", cmap="magma", clim=(0, 1),
                                                             show_scalar_bar=False), VIEW, size=(600, 650),
                                 bounds_mesh=F.surf))
            ax[i, j].axis("off")
            ax[i, j].set_title(f"mode {j + 1}: {np.sqrt(lam[j]) / (2 * np.pi):.0f} Hz", fontsize=11)
        ax[i, 0].text(-0.05, 0.5, "full space" if L is None else f"level {L}\n{sysd['n_dof']:,} DOFs",
                      transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=12)
        print(f"modes {L}: {np.round(np.sqrt(lam) / 2 / np.pi, 1)}", flush=True)
    fig.suptitle("Lowest eigenmodes of each reduced system (a = 0), prolonged onto the fine hand "
                 "(colour |u|, exaggerated to 20 mm)", fontsize=14)
    fig.tight_layout(rect=(0.05, 0, 1, 0.97))
    fig.savefig(os.path.join(REN, "19_reduced_modes.png"), dpi=80)
    print("wrote renders/19_reduced_modes.png")


def load_runs():
    out = []
    for L in levels() + [None]:
        f = os.path.join(OUT, f"reduced_sim_{'fine' if L is None else L}.npz")
        if os.path.exists(f):
            d = dict(np.load(f))
            _, P = (None, sp.sparse.identity(len(R.fine_hand().X), format="csc")) if L is None else R.load_level(L)
            out.append((L, d, P))
    return out


def fig_statics(F, runs, name="20_reduced_statics.png", labels=None):
    ks = [0, 4, 8, 12]
    rows = []                                  # (run, "coarse" | "fine")
    for r in runs:
        if r[0] is not None:
            rows.append((r, "coarse"))
        rows.append((r, "fine"))
    fig, ax = plt.subplots(len(rows), len(ks), figsize=(4.0 * len(ks), 4.3 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, ((L, d, P), what) in enumerate(rows):
        if what == "coarse":
            sc, _ = R.load_level(L)
        for j, k in enumerate(ks):
            xk = d["static_x"][k].reshape(-1, 3).astype(float)
            if what == "coarse":
                s = kind_surface(xk, sc["T"], sc["part"], F.kinds)
                opts = kind_opts(0.6)
            else:
                s = F.at(P @ xk)
                opts = kind_opts(0.0) | dict(show_edges=False)
            ax[i, j].imshow(shot(lambda pl, s=s, o=opts: pl.add_mesh(s, **o), VIEW, size=(620, 680),
                                 bounds_mesh=F.surf))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"a = {d['static_a'][k]:.2f}", fontsize=13)
        lab = (labels or {}).get(L) or ("full space\n(P = I)" if L is None else f"level {L}\n{3 * P.shape[1]:,} DOFs")
        if what == "coarse":
            lab = f"coarse mesh\n{len(sc['X']):,} v, {len(sc['T']):,} tets\n(the integration mesh)"
        else:
            lab += f"\nstatics {d['t_static']:.1f} s" + ("" if L is None else "\n(fine hand, x_f = B x)")
        ax[i, 0].text(-0.05, 0.5, lab, transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=12)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER], loc="lower center",
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Actuation sweep in each subspace (quasi-static Newton per a), drawn on the fine hand "
                 "x_fine = B x", fontsize=14)
    fig.tight_layout(rect=(0.06, 0.03, 1, 0.97))
    fig.savefig(os.path.join(REN, name), dpi=85)
    print("wrote renders/" + name)


def fig_compare(runs):
    ref = [r for r in runs if r[0] is None]
    cols = {L: RAMP[i] for i, L in enumerate(levels())}
    cols[None] = INK
    lab = lambda L, P: "full space" if L is None else f"{3 * P.shape[1]:,} DOFs"
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.0))
    for L, d, P in runs:
        ls = "--" if L is None else "-"
        ax[0].plot(d["static_a"], 1e3 * d["static_tip"].mean(1), ls, color=cols[L], lw=2.2, label=lab(L, P))
        ax[1].plot(d["dyn_t"], 1e3 * d["dyn_tip"].mean(1), ls, color=cols[L], lw=2.2, label=lab(L, P))
    ax[0].set(xlabel="actuation a", ylabel="mean fingertip displacement [mm]", title="Statics")
    ax[1].set(xlabel="time [s]", ylabel="mean fingertip displacement [mm]", title="Dynamics (backward Euler)")
    ax[0].legend(frameon=False, fontsize=10)
    co = [r for r in runs if r[0] is not None]
    if ref:
        Xf = R.fine_hand().X
        xf_ref = ref[0][1]["static_x"][-1].reshape(-1, 3).astype(float)
        yref = ref[0][1]["dyn_x"].reshape(len(ref[0][1]["dyn_x"]), -1, 3).astype(float)
        errs, derrs = [], []
        for L, d, P in co:
            xf = P @ d["static_x"][-1].reshape(-1, 3).astype(float)
            errs.append(np.linalg.norm(xf - xf_ref) / np.linalg.norm(xf_ref - Xf))
            yd = np.array([P @ f.reshape(-1, 3).astype(float) for f in d["dyn_x"]])
            derrs.append(np.linalg.norm(yd - yref) / np.linalg.norm(yref - Xf[None]))
        n = [3 * P.shape[1] for _, _, P in co]
        ax[2].plot(n, 100 * np.array(errs), "-o", color=RAMP[2], lw=2, ms=8, mec=SURF, mew=2, label="statics, a = 1")
        ax[2].plot(n, 100 * np.array(derrs), "-s", color=RAMP[4], lw=2, ms=8, mec=SURF, mew=2,
                   label="dynamics, whole trajectory")
        for x_, y_ in zip(n, errs):
            ax[2].annotate(f"{100 * y_:.0f}%", (x_, 100 * y_), textcoords="offset points", xytext=(0, 9),
                           ha="center", fontsize=10, color=INK2)
        ax[2].set(xscale="log", xlabel="subspace DOFs", ylabel="relative error vs full space [%]",
                  title="Error |B x - x_full| / |x_full - X|")
        ax[2].legend(frameon=False, fontsize=10)
    for L, d, P in runs:
        n_ = 3 * P.shape[1]
        ax[3].plot([n_], [d["t_static"] + d["t_dynamic"]], "s" if L is None else "o", color=cols[L], ms=9,
                   mec=SURF, mew=2)
        ax[3].annotate(f"{d['t_static'] + d['t_dynamic']:.0f} s", (n_, d["t_static"] + d["t_dynamic"]),
                       textcoords="offset points", xytext=(0, 9), ha="center", fontsize=10, color=INK2)
    ax[3].set(xscale="log", yscale="log", xlabel="DOFs", ylabel="wall time, statics + dynamics [s]",
              title="Cost")
    for a in ax:
        a.grid(True, color=GRID, lw=0.8)
        a.set_facecolor(SURF)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(REN, "21_reduced_compare.png"), dpi=110, facecolor=SURF)
    print("wrote renders/21_reduced_compare.png")


def video(F, runs, fps=30, name="reduced_closing.mp4", labels=None):
    import imageio.v2 as imageio
    pls = []
    for L, d, P in runs:
        if L is not None:                     # the coarse integration mesh itself, at state x
            sc, _ = R.load_level(L)
            cs = kind_surface(sc["X"], sc["T"], sc["part"], F.kinds)
            cpid = cs.point_data["vtkOriginalPointIds"]
            pl = pv.Plotter(window_size=(460, 540), off_screen=True)
            pl.set_background("white")
            pl.add_mesh(cs, **kind_opts(0.6))
            dd = np.array([0.75, -1.0, 0.35]); dd /= np.linalg.norm(dd)
            c = np.array(F.bounds).reshape(3, 2).mean(1)
            pl.camera_position = [tuple(c + dd), tuple(c), (0, 0, 1)]
            pl.reset_camera(bounds=F.bounds)
            pl.camera.zoom(1.25)
            pl.add_text(f"coarse mesh ({len(sc['X']):,} v)", position="upper_left", font_size=11, color="black")
            pls.append((pl, cs, d, ("coarse", cpid)))
        pl = pv.Plotter(window_size=(460, 540), off_screen=True)
        pl.set_background("white")
        s = F.at(F.X)
        pl.add_mesh(s, **kind_opts(0.0) | dict(show_edges=False))
        dd, up = (np.array([0.75, -1.0, 0.35]), (0, 0, 1))
        dd /= np.linalg.norm(dd)
        c = np.array(F.bounds).reshape(3, 2).mean(1)
        pl.camera_position = [tuple(c + dd), tuple(c), up]
        pl.reset_camera(bounds=F.bounds)
        pl.camera.zoom(1.25)
        pl.add_text((labels or {}).get(L) or ("full space" if L is None else f"{3 * P.shape[1]:,} DOFs"), position="upper_left",
                    font_size=11, color="black")
        pls.append((pl, s, d, P))
    w = imageio.get_writer(os.path.join(REN, name), fps=fps, codec="libx264",
                           macro_block_size=1, quality=8)
    for k in range(len(runs[0][1]["dyn_x"])):
        imgs = []
        for pl, s, d, P in pls:
            if isinstance(P, tuple):
                s.points = d["dyn_x"][k].reshape(-1, 3).astype(float)[P[1]]
            else:
                s.points = (P @ d["dyn_x"][k].reshape(-1, 3).astype(float))[F.pid]
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for p in pls:
        p[0].close()
    print("wrote renders/" + name)


def one_level(tag, file_tag, title):
    """``tag`` (e.g. 1200_q30) run from reduced_sim_<file_tag>.npz next to the full space."""
    F = Fine()
    _, P = R.load_level(tag)
    d = dict(np.load(os.path.join(OUT, f"reduced_sim_{file_tag}.npz")))
    ref = dict(np.load(os.path.join(OUT, "reduced_sim_fine.npz")))
    Pf = sp.sparse.identity(len(F.X), format="csc")
    runs = [(tag, d, P), (None, ref, Pf)]
    labels = {tag: title}
    fig_statics(F, runs, name=f"23_reduced_statics_{file_tag}.png", labels={tag: title.replace(", ", "\n")})
    video(F, runs, name=f"reduced_closing_{file_tag}.mp4", labels=labels)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one_level(sys.argv[2], sys.argv[3], sys.argv[4])
        raise SystemExit
    ap = argparse.ArgumentParser()
    for f in ("setup", "modes", "statics", "compare", "video"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    F = Fine()
    everything = not any(vars(a).values())
    if a.setup or everything:
        fig_setup(F)
    if a.modes or everything:
        fig_modes(F)
    if a.statics or a.compare or a.video or everything:
        runs = load_runs()
        if a.statics or everything:
            fig_statics(F, runs)
        if a.compare or everything:
            fig_compare(runs)
        if a.video or everything:
            video(F, runs)
