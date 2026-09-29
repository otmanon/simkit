"""Eigenmodes only -> PCA only: the mix sweep at 1,200 vertices, rendered.

    python sdm_mix_sweep_render.py     # 30_mix_sweep.png, mix_sweep_closing.mp4
"""
import json
import os

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
from sdm_pca_render import video

pv.OFF_SCREEN = True
REN = os.path.join(R.OUT, "renders")


def main():
    F = Fine()
    rows = json.load(open(os.path.join(R.CO, "mix_sweep.json")))
    ref = dict(np.load(os.path.join(R.OUT, "reduced_sim_fine.npz")))
    n = len(rows) + 1
    fig = plt.figure(figsize=(3.3 * n, 15))
    gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 1, 0.9])
    for j, r in enumerate(rows + [dict(tag=None, label="full space")]):
        if r["tag"] is None:
            xs = ref["static_x"]
            x1 = xs[-1].reshape(-1, 3).astype(float)
            for i in (0, 1):
                a = fig.add_subplot(gs[i, j]); a.axis("off")
            a = fig.add_subplot(gs[2, j])
            a.imshow(shot(lambda pl: pl.add_mesh(F.at(x1), **kind_opts(0.0) | dict(show_edges=False)), VIEW,
                          size=(560, 620), bounds_mesh=F.surf))
            a.axis("off")
            a.set_title("full space\n34,776 DOFs, 254 s", fontsize=11)
            continue
        sc, P = R.load_level(r["tag"])
        d = np.load(os.path.join(R.OUT, f"reduced_sim_{r['tag']}_kpin100000.npz"))
        x1 = d["static_x"][-1].reshape(-1, 3).astype(float)
        kinds = np.array(F.kinds)[sc["part"]]
        for i, (X, fine) in enumerate([(sc["X"], False), (x1, False), (P @ x1, True)]):
            s = F.at(X) if fine else kind_surface(X, sc["T"], sc["part"], F.kinds)
            o = kind_opts(0.0) | dict(show_edges=False) if fine else kind_opts(0.5)
            a = fig.add_subplot(gs[i, j])
            a.imshow(shot(lambda pl, s=s, o=o: pl.add_mesh(s, **o), VIEW, size=(560, 620), bounds_mesh=F.surf))
            a.axis("off")
            if i == 0:
                a.set_title(f"{r['label']}\npad tets {(kinds == 'pad').sum():,}, flexure {(kinds == 'flexure').sum():,}",
                            fontsize=11)
            if i == 2:
                a.set_title(f"error {100 * r['error']:.1f}%, sideways {r['side_max']:.1f} mm\nstatics "
                            f"{float(d['t_static']):.0f} s", fontsize=10.5)
    for i, t in enumerate(["coarse mesh\n(rest)", "coarse mesh\n(a = 1)", "fine hand\n(a = 1)"]):
        fig.text(0.004, 0.86 - i * 0.235, t, fontsize=12, va="center")
    ax = fig.add_subplot(gs[3, 1:n - 1])
    labs = [r["label"] for r in rows]
    x = np.arange(len(rows))
    ax.plot(x, [100 * r["error"] for r in rows], "-o", color="#2a78d6", lw=2, ms=8, mec=SURF, mew=2,
            label="error vs full space [%]")
    ax.plot(x, [r["side_max"] for r in rows], "-s", color="#eb6834", lw=2, ms=8, mec=SURF, mew=2,
            label="max sideways drift [mm]")
    for xi, r in zip(x, rows):
        ax.annotate(f"{100 * r['error']:.1f}%", (xi, 100 * r["error"]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_xlabel("coarsening basis: more eigenmode weight  <-->  more PCA weight")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_facecolor(SURF)
    ax.legend(frameon=False, fontsize=10)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER], loc="lower center", ncol=4,
               frameon=False, fontsize=11)
    fig.suptitle("Eigenmodes only to PCA only: mixing ratio rho (PCA weight / eigenmode weight) at 1,200 vertices "
                 "(quality >= 0.3, extrapolation <= 1, pins 1e5, statics)", fontsize=14)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.965))
    fig.savefig(os.path.join(REN, "30_mix_sweep.png"), dpi=80)
    print("wrote renders/30_mix_sweep.png")
    runs = []
    for r in rows[:3]:
        _, P = R.load_level(r["tag"])
        runs.append((r["tag"], r["label"], dict(np.load(os.path.join(R.OUT, f"reduced_sim_{r['tag']}_kpin100000.npz"))), P))
    video(F, runs, ref, name="mix_sweep_closing.mp4")


if __name__ == "__main__":
    main()
