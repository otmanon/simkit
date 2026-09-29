"""Coarse tets that bridge a flexure joint (vertices on both sides of the flexure
mid-plane), coloured by the material they integrate: a 1.5 GPa tet spanning a
joint locks it. Writes renders/22_joint_bridges.png and prints counts.

    python sdm_joint_bridges.py
"""
import os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sdm_geometry import joints
from sdm_render import tet_grid, shot
from sdm_sim import OUT

pv.OFF_SCREEN = True
LEVELS = [300, 600, 1200, 2500, 5000]
STIFF, SOFT, PAD = "#d62728", "#f28e2b", "#76b7e0"


def bridging(X, T, finger_only=None):
    out = np.zeros(len(T), bool)
    for jt in joints():
        if finger_only and not jt["name"].startswith(finger_only):
            continue
        A = jt["A"]
        loc = (X - A[:3, 3]) @ A[:3, :3]
        s = loc[:, 2] - 0.5 * (jt["z0"] + jt["z1"])
        near = (np.abs(loc[:, 0]) < 0.014) & (loc[:, 1] > -0.02) & (loc[:, 1] < 0.03) & (np.abs(s) < 0.03)
        sv = s[T]
        out |= near[T].all(1) & (sv.min(1) < -0.002) & (sv.max(1) > 0.002)
    return out


def main():
    fig, ax = plt.subplots(2, len(LEVELS), figsize=(4.2 * len(LEVELS), 10))
    for j, lv in enumerate(LEVELS):
        d = dict(np.load(os.path.join(OUT, "coarse", f"sdm_tets_{lv}.npz")))
        X, T, E = d["X"], d["T"], d["E"]
        kinds = np.array([str(k) for k in d["part_kind"]])[d["part"]]
        allb = bridging(X, T)
        for i, finger in enumerate(["middle", "index"]):
            br = bridging(X, T, finger)
            col = np.where(E[br] >= 1e9, 0, np.where(kinds[br] == "pad", 2, 1)).astype(float)
            g = tet_grid(X, T[br], c=col)
            ctx = tet_grid(X, T[~br]).extract_surface(algorithm="dataset_surface")
            jt = [q for q in joints() if q["name"] == f"{finger}_flex1"][0]
            A = jt["A"]
            c = A[:3, :3] @ np.array([0, 0.011, 0.5 * (jt["z0"] + jt["z1"])]) + A[:3, 3]
            box = pv.Box(bounds=(c[0] - 0.02, c[0] + 0.02, c[1] - 0.045, c[1] + 0.045, c[2] - 0.055, c[2] + 0.06))

            def add(pl, g=g, ctx=ctx):
                pl.add_mesh(ctx, color="#bdbab2", opacity=0.18)
                pl.add_mesh(g, scalars="c", cmap=[STIFF, SOFT, PAD], clim=(-0.5, 2.5), n_colors=3,
                            show_scalar_bar=False, show_edges=True, edge_color="#111111", line_width=0.8)
            ax[i, j].imshow(shot(add, ((1.0, -0.15, 0.05), (0, 0, 1)), size=(640, 900), zoom=1.0,
                                 focus=c, bounds_mesh=box))
            ax[i, j].axis("off")
            ns = int((E[br] >= 1e9).sum())
            ax[i, j].set_title(f"{lv} v: {finger} finger\n{br.sum()} bridging tets, {ns} stiff", fontsize=11.5)
        print(f"{lv}: {allb.sum()} tets bridge a joint, {(E[allb] >= 1e9).sum()} of them 1.5 GPa, "
              f"{(kinds[allb] == 'flexure').sum()} flexure (6 MPa), {(kinds[allb] == 'pad').sum()} pad")
    fig.legend(handles=[Patch(color=STIFF, label="bridging tet integrates 1.5 GPa block material (locks the joint)"),
                        Patch(color=SOFT, label="bridging tet integrates 6 MPa flexure material"),
                        Patch(color=PAD, label="bridging tet integrates 0.2 MPa pad material")],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Coarse tets spanning a flexure joint (side view, fingers point up, palm to the left), "
                 "coloured by the material their centroid label gives them", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, "renders", "22_joint_bridges.png"), dpi=90)
    print("wrote renders/22_joint_bridges.png")


if __name__ == "__main__":
    main()
