"""Render the Allegro hand + cup coloured by Young's modulus, from several angles.

    python render_hand.py            # fitted grasp pose (hand_meta.json)
    python render_hand.py --open     # flat meshing pose

The hand is posed by forward kinematics (each link's tets move rigidly with
the link), i.e. exactly the actuation the simulation prescribes.
Writes ``output/renders/hand_youngs_modulus.png`` and
``output/renders/hand_poisson_density.png``.
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
from hand_grasp import pose_vertices, vertex_bodies

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

# world frame: palm y (index finger side) points up, the cup stands upright.
# Map palm (x, y, z) -> world (z, y, x) mirrored so the hand is right-handed.
WORLD = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

FOCAL = (0.0, 0.0, 0.0)


def views(center):
    c = np.asarray(center)
    return {
        "back of hand": (c + [0.0, 0.05, 0.40], c),
        "thumb side": (c + [0.33, 0.12, 0.20], c),
        "palm side": (c + [-0.30, 0.18, -0.25], c),
        "from above": (c + [0.10, 0.40, 0.08], c),
        "fingertips": (c + [0.38, 0.02, -0.12], c),
    }


def tet_grid(X, T, **cell_data):
    cells = np.hstack([np.full((len(T), 1), 4), T]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(T), pv.CellType.TETRA), X)
    for k, v in cell_data.items():
        g.cell_data[k] = v
    return g


def shot(add, cam, size=(1100, 950)):
    pl = pv.Plotter(window_size=size)
    pl.set_background("white")
    add(pl)
    pl.camera_position = [tuple(cam[0]), tuple(cam[1]), (0, 1, 0)]
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def render(scene, X, out_dir, title, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    T = scene["T"]
    X = X @ WORLD.T
    E_levels = np.unique(scene["E"])
    E_rank = np.searchsorted(E_levels, scene["E"]).astype(float)
    palette = ["#3b4cc0", "#f2b134", "#8cc665", "#d7301f", "#6a3d9a"][:len(E_levels)]
    grid = tet_grid(X, T, E=E_rank, nu=scene["nu"], rho=scene["rho"])
    surf = grid.extract_surface(algorithm="dataset_surface")
    center = X.mean(0)
    cup_c = X[np.unique(T[scene["part"] == len(scene["part_names"]) - 1])].mean(0)
    # cut-away through the middle finger's plane (palm y = 0) shows the steel
    # joint shafts and the aluminium cores inside the rubber tips
    cut = grid.clip(normal=(0, 1, 0), origin=(0, 0.0, 0), crinkle=True, invert=True)
    E_opts = dict(scalars="E", cmap=palette, clim=(-0.5, len(E_levels) - 0.5),
                  n_colors=len(E_levels), show_scalar_bar=False)

    imgs = []
    for name, cam in views(center).items():
        imgs.append((name, shot(lambda pl: pl.add_mesh(surf, **E_opts), cam)))

    def add_cut(pl):
        pl.add_mesh(cut, **E_opts, show_edges=True, edge_color="#303030", line_width=0.3)
    cut_cam = (cup_c * [1, 0, 1] + [0.0, 0.30, 0.02], cup_c * [1, 0, 1] + [-0.03, 0, 0.0])
    imgs.append(("cut-away: steel shafts; rubber over Al tip cores",
                 shot(add_cut, cut_cam)))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.8))
    for ax, (name, img) in zip(axes.ravel(), imgs):
        ax.imshow(img)
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    handles = []
    for i, e in enumerate(E_levels):
        ids = [k for k in range(len(scene["part_names"]))
               if (scene["part"] == k).any() and scene["E"][scene["part"] == k][0] == e]
        mat = str(scene["material_names"][ids[0]])
        nu = scene["nu"][scene["part"] == ids[0]][0]
        rho = scene["rho"][scene["part"] == ids[0]][0]
        kinds = sorted({_kind(str(scene["part_names"][k])) for k in ids})
        es = f"{e/1e9:.1f} GPa" if e >= 1e8 else f"{e/1e6:.2f} MPa"
        handles.append(Patch(color=palette[i], label=(
            f"E = {es:>9s}   {mat.replace('_', ' ')}  ({', '.join(kinds)});  "
            f"ν = {nu:.2f},  ρ = {rho:.0f} kg/m³")))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11.5, frameon=False,
               title="Young's modulus (per tet, from winding-number labels)", title_fontsize=13)
    fig.suptitle(title + f"  ({len(X)} vertices, {len(T)} tets)", fontsize=15)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    path_E = os.path.join(out_dir, f"hand_youngs_modulus{suffix}.png")
    fig.savefig(path_E, dpi=130)
    plt.close(fig)

    # Poisson ratio + density
    cams = views(center)
    panels = [("Poisson ratio ν", "nu", "plasma", (0.28, 0.48), surf, cams["thumb side"]),
              ("density ρ [kg/m³] (cut-away)", "rho", "viridis", (1000, 8000), cut,
               (cup_c * [1, 0, 1] + [0.0, 0.30, 0.02], cup_c * [1, 0, 1] + [-0.03, 0, 0.0]))]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    for ax, (ttl, key, cmap, clim, mesh, cam) in zip(axes, panels):
        img = shot(lambda pl: pl.add_mesh(mesh, scalars=key, cmap=cmap, clim=clim,
                                          show_scalar_bar=False), cam)
        ax.imshow(img)
        ax.set_title(ttl, fontsize=13)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(*clim))
        fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
    fig.tight_layout()
    path_nr = os.path.join(out_dir, f"hand_poisson_density{suffix}.png")
    fig.savefig(path_nr, dpi=120)
    plt.close(fig)
    return path_E, path_nr


def _kind(part):
    if part == "cup":
        return "cup"
    for key, label in [("_shaft", "joint shafts"), ("_tip_rubber", "fingertips"),
                       ("_tip_core", "tip cores"), ("palm", "palm")]:
        if part.endswith(key) or part == key:
            return label
    return "phalanges"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="render the flat meshing pose")
    args = ap.parse_args()
    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    meta = json.load(open(os.path.join(OUT, "hand_meta.json")))
    q = {} if args.open else meta["grasp_q"]
    X = pose_vertices(scene["X"], vertex_bodies(scene), AllegroHand(XML),
                      list(scene["body_names"]), q)
    title = ("Allegro Hand v3 (open)" if args.open else
             "Allegro Hand v3 grasping a polystyrene cup")
    for p in render(scene, X, os.path.join(OUT, "renders"), title,
                    suffix="_open" if args.open else ""):
        print("wrote", os.path.relpath(p))
