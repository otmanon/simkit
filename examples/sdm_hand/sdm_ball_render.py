"""Reduced hand grasping a rigid ball (contact on the fine surface only), rendered.

    python sdm_ball_render.py TAG FILE_TAG     # e.g. 1200_mixw3 1200_mixw3_kpin100000_ball
Writes renders/31_ball_<FILE_TAG>.png and ball_<FILE_TAG>.mp4 (adds the full space if
reduced_sim_fine_ball.npz exists).
"""
import os
import sys

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sdm_reduced as R
from sdm_render import shot, KIND_COL, KIND_LAB
from sdm_coarse_render import kind_surface, kind_opts, ORDER
from sdm_reduced_render import Fine, VIEW
from sdm_pca_render import interp

pv.OFF_SCREEN = True
REN = os.path.join(R.OUT, "renders")
BALL = ((-0.010, -0.036, 0.128), 0.030)
BALL_COL = "#c9b99a"


def ball_mesh():
    return pv.Sphere(radius=BALL[1], center=BALL[0], theta_resolution=64, phi_resolution=64)


def main(tag, ftag):
    F = Fine()
    sc, P = R.load_level(tag)
    d = dict(np.load(os.path.join(R.OUT, f"reduced_sim_{ftag}.npz")))
    ref_f = os.path.join(R.OUT, "reduced_sim_fine_ball.npz")
    ref = dict(np.load(ref_f)) if os.path.exists(ref_f) else None
    bm = ball_mesh()
    bounds = F.surf.merge(bm)
    rows = [("coarse", d), ("fine", d)] + ([("full", ref)] if ref is not None else [])
    avals = [0.0, 0.5, 0.75, 1.0]
    views = [VIEW, ((-1.0, -0.35, 0.15), (0, 0, 1))]
    fig, ax = plt.subplots(len(rows), len(avals) + 1, figsize=(3.9 * (len(avals) + 1), 4.2 * len(rows)))
    for i, (what, dd) in enumerate(rows):
        for j in range(len(avals) + 1):
            a = avals[j] if j < len(avals) else 1.0
            view = VIEW if j < len(avals) else views[1]
            x = interp(dd, a)
            if what == "coarse":
                s, o = kind_surface(x, sc["T"], sc["part"], F.kinds), kind_opts(0.5)
            else:
                s, o = F.at(x if what == "full" else P @ x), kind_opts(0.0) | dict(show_edges=False)

            def add(pl, s=s, o=o):
                pl.add_mesh(s, **o)
                pl.add_mesh(bm, color=BALL_COL, opacity=0.55, smooth_shading=True)
            ax[i, j].imshow(shot(add, view, size=(600, 650), bounds_mesh=bounds))
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"a = {a:.2f}" + ("" if j < len(avals) else "  (thumb side)"), fontsize=13)
        k = -1
        if what == "coarse":
            lab = f"coarse mesh\n{len(sc['X']):,} v, {len(sc['T']):,} tets\n(elastic integration)"
        elif what == "fine":
            lab = (f"reduced: fine hand\nx_f = B x ({3 * P.shape[1]:,} DOFs)\nstatics {float(dd['t_static']):.0f} s\n"
                   f"contact {dd['contact_force'][k]:.0f} N on\n{int(dd['contact_n'][k])} fine surface v")
        else:
            lab = (f"full space\n34,776 DOFs\nstatics {float(dd['t_static']):.0f} s\n"
                   f"contact {dd['contact_force'][k]:.0f} N on\n{int(dd['contact_n'][k])} surface v")
        ax[i, 0].text(-0.05, 0.5, lab, transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=11.5)
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER] +
               [Patch(color=BALL_COL, label="rigid ball, R = 30 mm (analytic SDF)")],
               loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle("Hyper-reduced hand grasping a rigid ball: elastic energy on the coarse mesh, contact (cubic "
                 "penalty) on the fine surface vertices x_s = B_s x, statics", fontsize=13)
    fig.tight_layout(rect=(0.08, 0.03, 1, 0.965))
    fig.savefig(os.path.join(REN, f"31_ball_{ftag}.png"), dpi=80)
    print(f"wrote renders/31_ball_{ftag}.png")

    import imageio.v2 as imageio
    panels = []
    for what, dd in rows:
        pl = pv.Plotter(window_size=(500, 560), off_screen=True)
        pl.set_background("white")
        if what == "coarse":
            s = kind_surface(sc["X"], sc["T"], sc["part"], F.kinds)
            pl.add_mesh(s, **kind_opts(0.5))
            pid = s.point_data["vtkOriginalPointIds"]
        else:
            s = F.at(F.X)
            pl.add_mesh(s, **kind_opts(0.0) | dict(show_edges=False))
            pid = F.pid
        pl.add_mesh(bm, color=BALL_COL, opacity=0.55, smooth_shading=True)
        dv = np.array([0.75, -1.0, 0.35]); dv /= np.linalg.norm(dv)
        c = np.array(bounds.bounds).reshape(3, 2).mean(1)
        pl.camera_position = [tuple(c + dv), tuple(c), (0, 0, 1)]
        pl.reset_camera(bounds=bounds.bounds)
        pl.camera.zoom(1.2)
        txt = pl.add_text("", position="upper_left", font_size=10, color="black")
        panels.append((pl, s, pid, dd, what, txt))
    w = imageio.get_writer(os.path.join(REN, f"ball_{ftag}.mp4"), fps=24, codec="libx264", macro_block_size=1,
                           quality=8)
    for a in np.concatenate([np.linspace(0, 1, 73), np.ones(18)]):
        imgs = []
        for pl, s, pid, dd, what, txt in panels:
            x = interp(dd, a)
            s.points = (x if what in ("coarse", "full") else P @ x)[pid]
            A = dd["static_a"]
            f = np.interp(a, A, dd["contact_force"])
            name = {"coarse": "coarse mesh", "fine": "reduced (fine hand)", "full": "full space"}[what]
            txt.SetText(2, f"{name}   a = {a:.2f}   contact {f:.0f} N")
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for p in panels:
        p[0].close()
    print(f"wrote renders/ball_{ftag}.mp4")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
