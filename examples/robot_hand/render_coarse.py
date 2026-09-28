"""Render fine vs coarsened hands (mesh4PDE vs shortest-edge) side by side.

    python render_coarse.py [--target 200]

Each mesh is posed in the grasp by forward kinematics (a coarse tet's link
comes from its winding-number part label) and drawn with its edges, coloured
by Young's modulus. Writes ``output/renders/hand_coarsened_<n>.png``.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from allegro_kinematics import AllegroHand
from hand_geometry import XML
from hand_grasp import pose_vertices
from render_hand import WORLD, tet_grid, shot, views
from materials import MATERIALS, part_material

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

LEVELS = sorted({m["E"] for k, m in MATERIALS.items() if k != "polystyrene_GPPS"})
PALETTE = ["#3b4cc0", "#17becf", "#8cc665", "#d7301f", "#6a3d9a"]


def posed(X, T, part, names, meta, body_names, hand, q):
    parts_body = meta["parts"]
    vb = np.zeros(len(X), int)
    tb = np.array([body_names.index(parts_body[names[p]]) for p in part])
    vb[T.ravel()] = np.repeat(tb, 4)
    return pose_vertices(X, vb, hand, body_names, q) @ WORLD.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200)
    args = ap.parse_args()
    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    meta = json.load(open(os.path.join(OUT, "hand_meta.json")))
    names = [str(n) for n in scene["part_names"]]
    body_names = [str(b) for b in scene["body_names"]]
    hand, q = AllegroHand(XML), meta["grasp_q"]
    cup = names.index("cup")

    meshes = []
    keep = scene["part"] != cup
    meshes.append(("fine", scene["X"], scene["T"][keep], scene["part"][keep]))
    for method, label in [("mesh4pde", "mesh4PDE (PDE-aware)"),
                          ("shortest_edge", "shortest-edge (geometry only)")]:
        d = np.load(os.path.join(OUT, f"coarse_{method}_{args.target}.npz"))
        meshes.append((label, d["X"], d["T"], d["part"]))

    # the cup, rigid, as context
    cup_T = scene["T"][scene["part"] == cup]
    cup_grid = tet_grid(scene["X"] @ WORLD.T, cup_T).extract_surface(algorithm="dataset_surface")

    cols = []
    for label, X, T, part in meshes:
        Xw = posed(X, T, part, names, meta, body_names, hand, q)
        E = np.array([MATERIALS[part_material(names[p])]["E"] for p in part])
        g = tet_grid(Xw, T, E=np.searchsorted(LEVELS, E).astype(float))
        surf = g.extract_surface(algorithm="dataset_surface")
        center = Xw[np.unique(T)].mean(0)
        nv = len(np.unique(T))
        cols.append((f"{label}\n{nv} vertices, {len(T)} tets", surf, center))

    opts = dict(scalars="E", cmap=PALETTE, clim=(-0.5, len(LEVELS) - 0.5),
                n_colors=len(LEVELS), show_scalar_bar=False, show_edges=True,
                edge_color="#202020", line_width=0.6)
    rows = ["palm side, three-quarter", "back of hand"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6 * len(cols), 5.4 * len(rows)))
    for j, (title, surf, center) in enumerate(cols):
        for i, r in enumerate(rows):
            cam = views(cols[0][2])[r]

            def add(pl):
                pl.add_mesh(surf, **opts)
                pl.add_mesh(cup_grid, color="#e9e4d8", opacity=0.35)
            axes[i, j].imshow(shot(add, cam, size=(1000, 900)))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title, fontsize=13)
    mats = {part_material(n): MATERIALS[part_material(n)]["E"] for n in names if n != "cup"}
    handles = [Patch(color=PALETTE[LEVELS.index(E)],
                     label=f"{m.replace('_', ' ')}  E = "
                           + (f"{E/1e9:.1f} GPa" if E >= 1e8 else f"{E/1e6:.2f} MPa"))
               for m, E in sorted(mats.items(), key=lambda kv: kv[1])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11.5, frameon=False)
    fig.suptitle(f"Allegro hand coarsened to ~{args.target} vertices "
                 f"(cup shown translucent for context)", fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    path = os.path.join(OUT, "renders", f"hand_coarsened_{args.target}.png")
    fig.savefig(path, dpi=120)
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
