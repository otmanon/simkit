"""PCA-driven coarsening vs mixed (PCA + Hessian eigenmodes), rendered.

    python sdm_pca_render.py

25_pca_coarsening.png  PCA spectrum, the PCA components on the hand, and the coarse meshes
                       (elastic eigenmodes, PCA only, mixed) with fingertip close-ups
26_pca_statics.png     actuation sweep: coarse mesh + fine hand (x_f = B x) per basis, full space
pca_closing.mp4        the same sweep as a video (quasi-static, interpolated in a)
"""
import os
import sys

import numpy as np
import scipy as sp
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sdm_reduced as R
from sdm_render import shot, KIND_COL, KIND_LAB
from sdm_coarse_render import kind_surface, kind_opts, ORDER
from sdm_reduced_render import Fine, VIEW, INK, INK2, GRID, SURF

sys.path.insert(0, "/home/user/mesh4pde")
pv.OFF_SCREEN = True
REN = os.path.join(R.OUT, "renders")
RUNS = [("1200_q30", "elastic eigenmodes (92)", None),
        ("1200_q30pca", "PCA of the closing only (3)", "1200_q30pca_kpin100000"),
        ("1200_q30mixed", "PCA + eigenmodes, Rayleigh-Ritz (95)", "1200_q30mixed_kpin100000")]


def tip_err(F, P, x):
    hf = R.fine_hand()
    ref = np.load(os.path.join(R.OUT, "reduced_sim_fine.npz"))["static_x"][-1].reshape(-1, 3).astype(float)
    xf = P @ x
    return np.linalg.norm(xf - ref) / np.linalg.norm(ref - hf.X)


def fig_modes_meshes(F):
    pb = np.load(os.path.join(R.CO, "pca_basis.npz"))
    Phi, sig, sall = pb["Phi"], pb["sigma"], pb["sigma_all"]
    fig = plt.figure(figsize=(20, 12.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.05, 0.95])
    ax = fig.add_subplot(gs[0, 0])
    k = np.arange(1, 21)
    ax.semilogy(k, sall[:20] / sall[0], "-o", color="#2a78d6", lw=2, ms=6, mec=SURF, mew=1.5)
    ax.axvline(len(sig) + 0.5, color=INK2, lw=1, ls="--")
    ax.annotate(f"{len(sig)} kept", (len(sig) + 0.7, 0.3), fontsize=10, color=INK2)
    ax.set(xlabel="component", ylabel="singular value / first", title="PCA of 134 full-space closing snapshots",
           ylim=(1e-7, 2))
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_facecolor(SURF)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for j in range(3):
        a = fig.add_subplot(gs[0, j + 1])
        u = Phi[:, j].reshape(-1, 3)
        mag = np.linalg.norm(u, axis=1)
        s = F.at(F.X + u * (0.03 / mag.max()))
        s.point_data["m"] = (mag / mag.max())[F.pid]
        a.imshow(shot(lambda pl, s=s: pl.add_mesh(s, scalars="m", cmap="magma", clim=(0, 1), show_scalar_bar=False),
                      VIEW, size=(640, 620), bounds_mesh=F.surf))
        a.axis("off")
        a.set_title(f"PCA component {j + 1}  (sigma {sig[j] / sig[0]:.1e} of first)", fontsize=11)
    sel = np.isin(F.sc["part"], [list(F.sc["part_names"]).index(n) for n in ("index_pad", "middle_pad", "ring_pad")])
    c = F.X[np.unique(F.sc["T"][sel])].mean(0) + [0.012, 0, -0.03]
    box = pv.Box(bounds=(c[0] - 0.04, c[0] + 0.04, c[1] - 0.04, c[1] + 0.04, c[2] - 0.045, c[2] + 0.04))
    for j, (tag, title, _) in enumerate(RUNS):
        d, P = R.load_level(tag)
        cs = kind_surface(d["X"], d["T"], d["part"], F.kinds)
        a = fig.add_subplot(gs[1, j])
        a.imshow(shot(lambda pl, cs=cs: pl.add_mesh(cs, **kind_opts(0.5)), VIEW, size=(640, 700), bounds_mesh=F.surf))
        a.axis("off")
        kinds = np.array(F.kinds)[d["part"]]
        a.set_title(f"{title}\n{len(d['X']):,} v, {len(d['T']):,} tets; pad tets {(kinds == 'pad').sum():,}, "
                    f"flexure {(kinds == 'flexure').sum()}", fontsize=11)
        a = fig.add_subplot(gs[2, j])
        a.imshow(shot(lambda pl, cs=cs: pl.add_mesh(cs, **kind_opts(0.8)), ((0.55, -1.0, 0.45), (0, 0, 1)),
                      size=(640, 600), zoom=1.0, focus=c, bounds_mesh=box))
        a.axis("off")
    a = fig.add_subplot(gs[1:, 3])
    a.axis("off")
    a.text(0.02, 0.95, "What each basis asks the coarsener to keep\n\n"
           "elastic: small vibrations about rest,\n  dominated by the pads (87 of 92 modes)\n\n"
           "PCA only: exactly the observed closing;\n  blocks rotate rigidly (free to collapse),\n"
           "  joints bend (kept); pads never deform\n  in the data, so 4 of 5 lose all pad tets\n\n"
           "mixed: both sets, energies from a\n  Rayleigh-Ritz step on the Hessian",
           va="top", fontsize=12, color=INK)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER], loc="lower center",
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Coarsening from data: PCA of the full-space closing, and PCA mixed with Hessian eigenmodes "
                 "(1,200 vertices, tet quality >= 0.3)", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(os.path.join(REN, "25_pca_coarsening.png"), dpi=85)
    print("wrote renders/25_pca_coarsening.png")


def load_runs(F):
    out = []
    for tag, title, f in RUNS:
        if f is None:
            continue
        _, P = R.load_level(tag)
        d = dict(np.load(os.path.join(R.OUT, f"reduced_sim_{f}.npz")))
        out.append((tag, title, d, P))
    ref = dict(np.load(os.path.join(R.OUT, "reduced_sim_fine.npz")))
    return out, ref


def interp(d, a):
    A = d["static_a"]
    i = int(np.clip(np.searchsorted(A, a) - 1, 0, len(A) - 2))
    t = (a - A[i]) / (A[i + 1] - A[i])
    return ((1 - t) * d["static_x"][i] + t * d["static_x"][i + 1]).reshape(-1, 3).astype(float)


def fig_statics(F, runs, ref, name="26_pca_statics.png"):
    avals = [0.0, 1 / 3, 2 / 3, 1.0]
    rows = []
    for tag, title, d, P in runs:
        rows += [("coarse", tag, title, d, P), ("fine", tag, title, d, P)]
    rows.append(("full", None, "full space", ref, None))
    fig, ax = plt.subplots(len(rows), 4, figsize=(15.5, 4.1 * len(rows)))
    for i, (what, tag, title, d, P) in enumerate(rows):
        if what == "coarse":
            sc, _ = R.load_level(tag)
        for j, a in enumerate(avals):
            x = interp(d, a)
            if what == "coarse":
                s, o = kind_surface(x, sc["T"], sc["part"], F.kinds), kind_opts(0.6)
            else:
                s, o = F.at(x if P is None else P @ x), kind_opts(0.0) | dict(show_edges=False)
            ax[i, j].imshow(shot(lambda pl, s=s, o=o: pl.add_mesh(s, **o), VIEW, size=(600, 650), bounds_mesh=F.surf))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"a = {a:.2f}", fontsize=13)
        if what == "coarse":
            lab = f"{title}\ncoarse mesh\n{len(sc['X']):,} v, {len(sc['T']):,} tets"
        elif what == "fine":
            lab = (f"{title}\nfine hand (x_f = B x)\nstatics {d['t_static']:.1f} s\n"
                   f"error {100 * tip_err(F, P, d['static_x'][-1].reshape(-1, 3).astype(float)):.1f}%")
        else:
            lab = f"full space\n34,776 DOFs\nstatics {d['t_static']:.0f} s"
        ax[i, 0].text(-0.05, 0.5, lab, transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=11.5)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER], loc="lower center",
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle("Quasi-static closing, 3,600-DOF subspaces (hinge pins 1e5) vs the full space; "
                 "error = |B x - x_full| / |x_full - X| at a = 1", fontsize=13)
    fig.tight_layout(rect=(0.1, 0.02, 1, 0.975))
    fig.savefig(os.path.join(REN, name), dpi=80)
    print("wrote renders/" + name)


def video(F, runs, ref, n=73, fps=24, name="pca_closing.mp4"):
    import imageio.v2 as imageio
    panels = []

    def plotter(title):
        pl = pv.Plotter(window_size=(430, 520), off_screen=True)
        pl.set_background("white")
        dd = np.array([0.75, -1.0, 0.35]); dd /= np.linalg.norm(dd)
        c = np.array(F.bounds).reshape(3, 2).mean(1)
        pl.camera_position = [tuple(c + dd), tuple(c), (0, 0, 1)]
        pl.reset_camera(bounds=F.bounds)
        pl.camera.zoom(1.25)
        pl.add_text(title, position="upper_left", font_size=10, color="black")
        return pl
    for tag, title, d, P in runs:
        sc, _ = R.load_level(tag)
        cs = kind_surface(sc["X"], sc["T"], sc["part"], F.kinds)
        pl = plotter(title.split(",")[0].split("(")[0].strip() + ": coarse mesh")
        pl.add_mesh(cs, **kind_opts(0.6))
        panels.append((pl, cs, cs.point_data["vtkOriginalPointIds"], d, None))
        fs = F.at(F.X)
        pl = plotter(title.split(",")[0].split("(")[0].strip() + ": fine hand")
        pl.add_mesh(fs, **kind_opts(0.0) | dict(show_edges=False))
        panels.append((pl, fs, F.pid, d, P))
    fs = F.at(F.X)
    pl = plotter("full space")
    pl.add_mesh(fs, **kind_opts(0.0) | dict(show_edges=False))
    panels.append((pl, fs, F.pid, ref, "full"))
    w = imageio.get_writer(os.path.join(REN, name), fps=fps, codec="libx264", macro_block_size=1,
                           quality=8)
    for a in np.concatenate([np.linspace(0, 1, n), np.ones(18)]):
        imgs = []
        for pl, s, pid, d, P in panels:
            x = interp(d, a)
            if isinstance(P, str) or P is None:
                s.points = x[pid]
            else:
                s.points = (P @ x)[pid]
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for p in panels:
        p[0].close()
    print("wrote renders/" + name)


if __name__ == "__main__":
    F = Fine()
    if "--bounded" in sys.argv:
        RUNS[:] = [("1200_q30pca", "PCA, extrapolation unbounded", "1200_q30pca_kpin100000"),
                   ("1200_q30pcaE1", "PCA, extrapolation <= 1", "1200_q30pcaE1_kpin100000")]
        runs, ref = load_runs(F)
        fig_statics(F, runs, ref, name="27_pca_bounded_statics.png")
        video(F, runs, ref, name="pca_bounded_closing.mp4")
        raise SystemExit
    fig_modes_meshes(F)
    runs, ref = load_runs(F)
    fig_statics(F, runs, ref)
    video(F, runs, ref)
