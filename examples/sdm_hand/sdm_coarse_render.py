"""Render the mesh4PDE-coarsened hands and, once simulated, compare them with the fine one.

    python sdm_coarse_render.py --meshes     # output/renders/14_coarse_meshes.png
    python sdm_coarse_render.py --sims       # 15_coarse_closing.png, 16_coarse_error.png,
                                             # coarse_closing.mp4
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

from sdm_render import tet_grid, shot, KIND_COL, KIND_LAB
from sdm_sim import OUT

pv.OFF_SCREEN = True
CO = os.path.join(OUT, "coarse")
R = os.path.join(OUT, "renders")
ORDER = ["palm", "link", "flexure", "pad"]


def levels():
    s = json.load(open(os.path.join(CO, "coarse_summary.json")))
    return s, [r["target"] for r in s["levels"]]


def load(target=None):
    f = os.path.join(OUT, "sdm_tets.npz") if target is None else os.path.join(CO, f"sdm_tets_{target}.npz")
    d = dict(np.load(f))
    if target is not None:
        d["P"] = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]),
                                      shape=tuple(d["P_shape"]))
    return d


def kind_surface(X, T, part, kinds):
    k = np.array([ORDER.index(kinds[p]) for p in part], float)
    return tet_grid(X, T, k=k).extract_surface(algorithm="dataset_surface")


def kind_opts(lw=0.6):
    return dict(scalars="k", cmap=[KIND_COL[k] for k in ORDER], clim=(-0.5, 3.5), n_colors=4,
                show_scalar_bar=False, show_edges=True, edge_color="#1a1a1a", line_width=lw)


def legend(fig):
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER],
               loc="lower center", ncol=4, frameon=False, fontsize=12)


def fig_meshes():
    s, targets = levels()
    fine = load()
    kinds = [str(k) for k in fine["part_kind"]]
    tips = fine["X"][np.isin(fine["part"], [list(fine["part_names"]).index(n)
                                            for n in ("index_pad", "middle_pad", "ring_pad")])]
    c = tips.mean(0) + [0.012, 0, -0.012]
    box = pv.Box(bounds=(c[0] - 0.035, c[0] + 0.035, c[1] - 0.035, c[1] + 0.035,
                         c[2] - 0.035, c[2] + 0.035))
    cols = [("fine", fine, len(fine["X"]), len(fine["T"]))]
    for t, r in zip(targets, s["levels"]):
        cols.append((f"target {t}", load(t), r["vertices"], r["tets"]))
    fig, ax = plt.subplots(2, len(cols), figsize=(4.3 * len(cols), 10.2))
    for j, (name, d, nv, nt) in enumerate(cols):
        surf = kind_surface(d["X"], d["T"], d["part"], kinds)
        lw = 0.35 if nv > 6000 else 0.7
        ax[0, j].imshow(shot(lambda pl: pl.add_mesh(surf, **kind_opts(lw)), "three-quarter (palm side)",
                             size=(760, 900)))
        ax[1, j].imshow(shot(lambda pl: pl.add_mesh(surf, **kind_opts(lw)),
                             ((0.55, -1.0, 0.45), (0, 0, 1)), size=(760, 760), zoom=1.0, focus=c,
                             bounds_mesh=box))
        kc = {k: int((np.array(kinds)[d["part"]] == k).sum()) for k in ORDER}
        ax[0, j].set_title(f"{nv:,} vertices, {nt:,} tets\n" + ("fine mesh" if name == "fine" else "mesh4PDE"),
                           fontsize=13)
        ax[1, j].set_title(f"fingertips  (pad tets {kc['pad']:,}, flexure {kc['flexure']:,})", fontsize=10.5)
        for a in ax[:, j]:
            a.axis("off")
    legend(fig)
    fig.suptitle(f"mesh4PDE coarsening of the compliant hand, collapses scored through the lowest "
                 f"{s['modes']} modes (up to {s['top_mode_hz']:.0f} Hz: flexion + fingertip/palm pad "
                 f"deformation)", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = os.path.join(R, "14_coarse_meshes.png")
    fig.savefig(path, dpi=100)
    print("wrote", os.path.relpath(path))


INK, INK2 = "#0b0b0b", "#52514e"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]     # coarse -> fine (ordinal blue)


def sims():
    """[(label, vertices, scene, sim, report, P or None)] fine first, then coarse levels."""
    s, targets = levels()
    fine = load()
    out = [("fine", len(fine["X"]), fine, dict(np.load(os.path.join(OUT, "sdm_sim.npz"))),
            json.load(open(os.path.join(OUT, "sim_report.json"))), None)]
    for t, r in zip(targets, s["levels"]):
        f = os.path.join(OUT, f"sdm_sim_{t}.npz")
        if os.path.exists(f):
            d = load(t)
            out.append((f"{r['vertices']:,} v", r["vertices"], d, dict(np.load(f)),
                        json.load(open(os.path.join(OUT, f"sim_report_{t}.json"))), d["P"]))
    return out


def tip_ids(fine):
    from sdm_sim import Hand
    h = Hand(fine)
    return [h.tip_vertices(f) for f in h.fingers], h.fingers


def compare(runs):
    """Prolong every coarse trajectory to the fine vertices (x_f ~ P x_c) and compare."""
    fine = runs[0]
    X = fine[2]["X"]
    tips, fingers = tip_ids(fine[2])
    res = []
    for lab, nv, sc, sim, rep, P in runs:
        pro = (lambda x: x.reshape(-1, 3)) if P is None else (lambda x, P=P: P @ x.reshape(-1, 3))
        dyn = np.array([pro(x.astype(float)) for x in sim["dyn_x"]])
        stat = pro(sim["static_x"][-1].astype(float))
        tipd = np.array([[np.linalg.norm(f[t].mean(0) - X[t].mean(0)) for t in tips] for f in dyn])
        res.append(dict(label=lab, nv=nv, t=sim["dyn_t"], dyn=dyn, stat=stat, tipd=tipd,
                        time=rep["t_static_s"] + rep["t_dynamic_s"], rep=rep))
    uf = res[0]["dyn"] - X
    for r in res:
        du = r["dyn"] - res[0]["dyn"]
        r["err_t"] = np.linalg.norm(du, axis=(1, 2)) / np.maximum(np.linalg.norm(uf, axis=(1, 2)), 1e-12)
        r["err_stat"] = np.linalg.norm(r["stat"] - res[0]["stat"]) / np.linalg.norm(res[0]["stat"] - X)
        r["tip_err_mm"] = np.abs(r["tipd"][-1] - res[0]["tipd"][-1]).max() * 1e3
    return res, fingers


def fig_sims():
    runs = sims()
    res, fingers = compare(runs)
    kinds = [str(k) for k in runs[0][2]["part_kind"]]
    n = len(runs)
    # ---- closing poses (statics, a = 1)
    fig, ax = plt.subplots(2, n, figsize=(4.3 * n, 10.4))
    bm = None
    for j, ((lab, nv, sc, sim, rep, P), r) in enumerate(zip(runs, res)):
        x1 = sim["static_x"][-1].reshape(-1, 3).astype(float)
        surf = kind_surface(x1, sc["T"], sc["part"], kinds)
        bm = bm or kind_surface(runs[0][3]["static_x"][-1].reshape(-1, 3).astype(float), runs[0][2]["T"],
                                runs[0][2]["part"], kinds)
        lw = 0.3 if nv > 6000 else 0.6
        for i, view in enumerate(["three-quarter (palm side)", "three-quarter (thumb side)"]):
            ax[i, j].imshow(shot(lambda pl: pl.add_mesh(surf, **kind_opts(lw)), view, size=(760, 860),
                                 zoom=1.25, bounds_mesh=bm))
            ax[i, j].axis("off")
        head = "fine mesh" if P is None else "mesh4PDE"
        ax[0, j].set_title(f"{head}: {nv:,} vertices\nsim {r['time']:.0f} s"
                           + ("" if P is None else f",  error {100 * r['err_stat']:.1f} %"), fontsize=13)
    legend(fig)
    fig.suptitle("Closed pose (statics, a = 1) on every mesh. Error = |P x_coarse - x_fine| / "
                 "|x_fine - X| over the fine vertices", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(R, "15_coarse_closing.png"), dpi=100)
    print("wrote renders/15_coarse_closing.png")

    # ---- curves: small multiples, one y-axis each
    col = [INK] + RAMP[len(RAMP) - (n - 1):][::-1] if n > 1 else [INK]
    col = [INK] + list(reversed(RAMP[:n - 1][::-1]))
    col = [INK] + RAMP[:n - 1]                    # coarse levels light -> dark as they get finer
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    for r, c in zip(res, col):
        lw = 2.4 if r["label"] == "fine" else 2.0
        ax[0].plot(r["t"], 1e3 * r["tipd"].mean(1), color=c, lw=lw,
                   ls="--" if r["label"] == "fine" else "-", label=r["label"], zorder=3 if r["label"] == "fine" else 2)
        if r["label"] != "fine":
            ax[1].plot(r["t"][1:], 100 * r["err_t"][1:], color=c, lw=2.0, label=r["label"])
            ax[1].annotate(r["label"], (r["t"][-1], 100 * r["err_t"][-1]), textcoords="offset points",
                           xytext=(6, 0), va="center", fontsize=10, color=INK2)
    co = res[1:]
    ax[2].plot([r["nv"] for r in co], [r["time"] for r in co], "-o", color=RAMP[2], lw=2, ms=8,
               mec="#fcfcfb", mew=2, label="mesh4PDE levels")
    ax[2].plot([res[0]["nv"]], [res[0]["time"]], "s", color=INK, ms=9, mec="#fcfcfb", mew=2, label="fine")
    for r in res:
        ax[2].annotate(f"{r['time']:.0f} s", (r["nv"], r["time"]), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=10, color=INK2)
    ax[0].set(xlabel="time [s]", ylabel="mean fingertip displacement [mm]",
              title="Closing (backward Euler, fingertips tracked on the fine vertices)")
    ax[1].set(xlabel="time [s]", ylabel="relative displacement error [%]",
              title="Error vs the fine simulation, |P x_c - x_f| / |x_f - X|")
    ax[1].set_xlim(0, res[0]["t"][-1] * 1.18)
    ax[2].set(xscale="log", yscale="log", xlabel="vertices", ylabel="wall time, statics + dynamics [s]",
              title="Cost")
    ax[0].legend(frameon=False, fontsize=10)
    ax[2].legend(frameon=False, fontsize=10, loc="upper left")
    for a in ax:
        a.grid(True, color="#e5e4df", lw=0.8)
        a.set_facecolor("#fcfcfb")
        a.title.set_fontsize(11.5)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "16_coarse_error.png"), dpi=120, facecolor="#fcfcfb")
    print("wrote renders/16_coarse_error.png")
    summary = [dict(label=r["label"], vertices=int(r["nv"]), sim_seconds=round(r["time"], 1),
                    static_rel_error=round(float(r["err_stat"]), 4),
                    dyn_rel_error_final=round(float(r["err_t"][-1]), 4),
                    dyn_rel_error_max=round(float(r["err_t"][1:].max()), 4),
                    tip_error_mm_final=round(float(r["tip_err_mm"]), 2)) for r in res]
    json.dump(summary, open(os.path.join(CO, "coarse_sim_summary.json"), "w"), indent=1)
    for s_ in summary:
        print(s_)
    video(runs)


def video(runs, fps=30):
    """Every level closing side by side (three-quarter, palm side)."""
    import imageio.v2 as imageio
    kinds = [str(k) for k in runs[0][2]["part_kind"]]
    size = (520, 600)
    bm = kind_surface(runs[0][3]["dyn_x"][0].reshape(-1, 3).astype(float), runs[0][2]["T"],
                      runs[0][2]["part"], kinds)
    pls = []
    for lab, nv, sc, sim, rep, P in runs:
        surf = kind_surface(sc["X"], sc["T"], sc["part"], kinds)
        pid = surf.point_data["vtkOriginalPointIds"]
        pl = pv.Plotter(window_size=size, off_screen=True)
        pl.set_background("white")
        pl.add_mesh(surf, **kind_opts(0.3 if nv > 6000 else 0.6))
        d = np.array([0.75, -1.0, 0.35]); d /= np.linalg.norm(d)
        c = np.array(bm.bounds).reshape(3, 2).mean(1)
        pl.camera_position = [tuple(c + d), tuple(c), (0, 0, 1)]
        pl.reset_camera(bounds=bm.bounds)
        pl.camera.zoom(1.25)
        pl.add_text(("fine" if P is None else "mesh4PDE") + f"  {nv:,} v", position="upper_left",
                    font_size=11, color="black")
        pls.append((pl, surf, pid, sim["dyn_x"]))
    path = os.path.join(R, "coarse_closing.mp4")
    w = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, quality=8)
    for k in range(len(runs[0][3]["dyn_x"])):
        imgs = []
        for pl, surf, pid, frames in pls:
            surf.points = frames[k].reshape(-1, 3).astype(float)[pid]
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for p in pls:
        p[0].close()
    print("wrote renders/coarse_closing.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", action="store_true")
    ap.add_argument("--sims", action="store_true")
    a = ap.parse_args()
    if a.meshes or not a.sims:
        fig_meshes()
    if a.sims:
        fig_sims()
