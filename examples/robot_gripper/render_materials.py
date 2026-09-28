"""Render the labelled tet mesh coloured by material parameters, from several angles.

    python render_materials.py            # rest pose, from scene_tets.npz
    python render_materials.py --frame -1 # last frame of grasp_frames.npz

Writes ``gripper_youngs_modulus.png`` (Young's modulus, 5 views + a crinkle
cut-away through the mid-plane showing the per-tet winding-number labels) and
``gripper_poisson_density.png`` into ``examples/robot_gripper/output/renders``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output")

VIEWS = {
    "front":           dict(position=(0.0, 0.09, 0.42), focal=(0.0, 0.075, 0.0), up=(0, 1, 0)),
    "three-quarter":   dict(position=(0.27, 0.21, 0.30), focal=(0.0, 0.07, 0.0), up=(0, 1, 0)),
    "side":            dict(position=(0.42, 0.12, 0.0), focal=(0.0, 0.075, 0.0), up=(0, 1, 0)),
    "rear-high":       dict(position=(-0.22, 0.34, -0.26), focal=(0.0, 0.07, 0.0), up=(0, 1, 0)),
    "from below":      dict(position=(0.16, -0.20, 0.24), focal=(0.0, 0.07, 0.0), up=(0, 1, 0)),
}


def tet_grid(X, T, **cell_data):
    cells = np.hstack([np.full((len(T), 1), 4), T]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(T), pv.CellType.TETRA), X)
    for k, v in cell_data.items():
        g.cell_data[k] = v
    return g


def material_legend(scene):
    rows = []
    for i, (part, mat) in enumerate(zip(scene["part_names"], scene["material_names"])):
        sel = scene["part"] == i
        if not sel.any() or part.endswith("right"):
            continue
        E, nu, rho = scene["E"][sel][0], scene["nu"][sel][0], scene["rho"][sel][0]
        Es = f"{E/1e9:6.1f} GPa" if E >= 1e8 else f"{E/1e6:6.2f} MPa"
        rows.append(f"{part.replace('_left', ''):9s} {mat:19s} E={Es}  nu={nu:.2f}  rho={rho:5.0f}")
    return "\n".join(rows)


def render(scene, X, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = scene["T"]
    # Discrete E levels: a log colour scale squashes 69 / 72 / 200 GPa together,
    # so colour by the rank of each distinct modulus and label every level.
    E_levels = np.unique(scene["E"])
    E_rank = np.searchsorted(E_levels, scene["E"]).astype(float)
    palette = ["#3b4cc0", "#f2b134", "#8cc665", "#d7301f", "#6a3d9a"][:len(E_levels)]
    grid = tet_grid(X, T, E=E_rank, nu=scene["nu"], rho=scene["rho"])
    surf = grid.extract_surface(algorithm="dataset_surface")
    cut = grid.clip(normal=(0, 0, 1), origin=(0, 0, 0), crinkle=True)
    floor = pv.Plane(center=(0, -0.0005, 0), direction=(0, 1, 0), i_size=0.35, j_size=0.25)
    E_opts = dict(scalars="E", cmap=palette, clim=(-0.5, len(E_levels) - 0.5),
                  n_colors=len(E_levels), show_scalar_bar=False)

    def shot(add, cam, size=(1100, 950)):
        pl = pv.Plotter(window_size=size)
        pl.set_background("white")
        add(pl)
        pl.camera_position = cam
        pl.enable_anti_aliasing("ssaa")
        img = pl.screenshot(return_img=True)
        pl.close()
        return img

    imgs = []
    for name, cam in VIEWS.items():
        def add(pl):
            pl.add_mesh(surf, **E_opts)
            pl.add_mesh(floor, color="#d9d9d9", opacity=0.5)
        imgs.append((name, shot(add, [cam["position"], cam["focal"], cam["up"]])))

    def add_cut(pl):
        pl.add_mesh(cut, **E_opts, show_edges=True, edge_color="#303030", line_width=0.4)
    imgs.append(("cut-away at z = 0: per-tet winding-number labels",
                 shot(add_cut, [(0.06, 0.11, 0.30), (0.0, 0.07, 0.0), (0, 1, 0)])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.5))
    for ax, (name, img) in zip(axes.ravel(), imgs):
        ax.imshow(img)
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    handles = []
    for i, e in enumerate(E_levels):
        parts = [str(p) for k, p in enumerate(scene["part_names"])
                 if (scene["part"] == k).any() and scene["E"][scene["part"] == k][0] == e]
        mats = sorted({str(scene["material_names"][k]) for k, p in enumerate(scene["part_names"])
                       if str(p) in parts})
        k0 = list(scene["part_names"]).index(parts[0])
        nu = scene["nu"][scene["part"] == k0][0]
        rho = scene["rho"][scene["part"] == k0][0]
        es = f"{e/1e9:.1f} GPa" if e >= 1e8 else f"{e/1e6:.2f} MPa"
        who = sorted({p.replace("_left", "").replace("_right", "") for p in parts})
        handles.append(Patch(color=palette[i], label=(
            f"E = {es:>9s}   {', '.join(m.replace('_', ' ') for m in mats)}  "
            f"({', '.join(who)});  \u03bd = {nu:.2f},  \u03c1 = {rho:.0f} kg/m\u00b3")))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11.5, frameon=False,
               title="Young's modulus (per tet, from winding-number labels)",
               title_fontsize=13)
    fig.suptitle("Parallel-jaw gripper with soft silicone pads + polystyrene cup  "
                 f"({len(X)} vertices, {len(T)} tets)", fontsize=15)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    path_E = os.path.join(out_dir, "gripper_youngs_modulus.png")
    fig.savefig(path_E, dpi=130)
    plt.close(fig)

    # ---------------- Poisson ratio + density ---------------------------------
    pl = pv.Plotter(shape=(1, 2), window_size=(2400, 1000), border=False)
    cam = VIEWS["three-quarter"]
    for j, (key, title, clim, log) in enumerate([
            ("nu", "Poisson ratio nu", (0.27, 0.48), False),
            ("rho", "density rho [kg/m^3]", (1000, 8000), True)]):
        pl.subplot(0, j)
        pl.add_mesh(cut if j else surf, scalars=key, cmap="viridis" if j else "plasma",
                    clim=clim, log_scale=log,
                    scalar_bar_args=dict(title=title, n_labels=5, fmt="%.2f" if not j else "%.0f",
                                         title_font_size=18, label_font_size=15,
                                         width=0.6, position_x=0.2, position_y=0.04))
        pl.add_mesh(floor, color="#d9d9d9", opacity=0.5)
        pl.camera_position = [cam["position"], cam["focal"], cam["up"]]
        pl.add_text(title + (" (cut-away)" if j else ""), font_size=13, position="upper_left")
    path_nr = os.path.join(out_dir, "gripper_poisson_density.png")
    pl.screenshot(path_nr)
    pl.close()
    return path_E, path_nr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None,
                    help="render this frame of grasp_frames.npz instead of the rest pose")
    args = ap.parse_args()
    scene = dict(np.load(os.path.join(DATA, "scene_tets.npz")))
    X = scene["X"]
    if args.frame is not None:
        X = np.load(os.path.join(DATA, "grasp_frames.npz"))["frames"][args.frame].astype(float)
    for p in render(scene, X, os.path.join(DATA, "renders")):
        print("wrote", os.path.relpath(p))
