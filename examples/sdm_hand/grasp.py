"""Close the hyper-reduced hand in air, on a rigid ball and on a rigid cup.

The reduced model runs on the mesh4PDE coarse mesh (1,200 vertices, 3,600 DOFs):
elastic energy on the coarse tets, contact and friction on the fine surface
vertices only (see ``hand.py``). ``--full`` repeats every case in the full space
(34,008 DOFs) for comparison -- slow: about 8 min in air and 35 min on the cup.

Writes to ``examples/sdm_hand/results/``: ``grasp.json`` (wall time, Newton
iterations, fingertip travel, contact force per run), ``grasp_<case>.png`` (the
coarse mesh and the fine hand it drives at a = 0, 0.5, 0.75, 1) and
``grasp_cup.mp4``::

    python examples/sdm_hand/grasp.py [--full]
"""
import argparse
import json
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simkit
from hand import load_fine, load_coarse, build_hand_system, close_hand, fingertip_travel

pv.OFF_SCREEN = True
RESULTS = os.path.join(os.path.dirname(__file__), "results")
K_PIN_REDUCED = 1e5          # hinge pins in the subspace (1e7 locks joints it cannot bend exactly)
FRICTION = 1e10              # N/m^3
yaw = np.radians(-15)
OBJECTS = {
    "ball": dict(sdf=lambda P: simkit.sphere_sdf(P, (-0.010, -0.036, 0.128), 0.030),
                 mesh=lambda: pv.Sphere(radius=0.030, center=(-0.010, -0.036, 0.128), theta_resolution=64,
                                        phi_resolution=64)),
    # an open cup across the palm, turned 15 deg so the index finger reaches it first
    "cup": dict(sdf=lambda P: simkit.cup_sdf(P, (-0.015, -0.044, 0.120), 0.035, 0.003, 0.004, 0.100,
                                             R=np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                                                         [0, 0, 1]])),
                mesh=lambda: pv.Cylinder(center=(-0.015, -0.044, 0.120), direction=(np.cos(yaw), np.sin(yaw), 0),
                                         radius=0.035, height=0.100, resolution=96)),
}
KIND_COLOR = {"palm": "#8a8d91", "link": "#3b3f45", "flexure": "#f28e2b", "pad": "#76b7e0"}


def surface(X, T, part, kinds):
    """Boundary of a tet mesh, coloured by part kind."""
    order = list(KIND_COLOR)
    g = pv.UnstructuredGrid(np.hstack([np.full((len(T), 1), 4), T]).ravel(),
                            np.full(len(T), pv.CellType.TETRA), np.asarray(X, float))
    g.cell_data["k"] = np.array([order.index(kinds[p]) for p in part], float)
    return g.extract_surface(algorithm="dataset_surface")


def draw(pl, s, edges, obj):
    pl.add_mesh(s, scalars="k", cmap=list(KIND_COLOR.values()), clim=(-0.5, 3.5), n_colors=4,
                show_scalar_bar=False, show_edges=edges, edge_color="#1a1a1a", line_width=0.5)
    if obj is not None:
        pl.add_mesh(OBJECTS[obj]["mesh"](), color="#c9b99a", opacity=0.5, smooth_shading=True)


def camera(pl, bounds, zoom=1.2):
    d = np.array([0.75, -1.0, 0.35]) / np.linalg.norm([0.75, -1.0, 0.35])
    c = np.array(bounds).reshape(3, 2).mean(1)
    pl.camera_position = [tuple(c + d), tuple(c), (0, 0, 1)]
    pl.reset_camera(bounds=bounds)
    pl.camera.zoom(zoom)


def interp(run, a):
    i = int(np.clip(np.searchsorted(run["a"], a) - 1, 0, len(run["a"]) - 2))
    t = (a - run["a"][i]) / (run["a"][i + 1] - run["a"][i])
    return ((1 - t) * run["x"][i] + t * run["x"][i + 1]).reshape(-1, 3)


def render(case, fine, coarse, runs):
    kinds = list(fine["part_kind"])
    fs = surface(fine["X"], fine["T"], fine["part"], kinds)
    pid = fs.point_data["vtkOriginalPointIds"]
    bounds = fs.bounds if case == "air" else fs.merge(OBJECTS[case]["mesh"]()).bounds
    rows = [("coarse mesh (integration)", "reduced", True)] + [(f"{m}: fine hand", m, False) for m in runs]
    avals = [0.0, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(len(rows), len(avals), figsize=(3.8 * len(avals), 4.0 * len(rows)))
    for i, (label, model, is_coarse) in enumerate(rows):
        for j, a in enumerate(avals):
            x = interp(runs[model], a)
            if is_coarse:
                s = surface(x, coarse["T"], coarse["part"], kinds)
            else:
                s = fs.copy()
                s.points = (runs[model]["P"] @ x)[pid]
            pl = pv.Plotter(window_size=(560, 600), off_screen=True)
            pl.set_background("white")
            draw(pl, s, is_coarse, None if case == "air" else case)
            camera(pl, bounds)
            ax[i, j].imshow(pl.screenshot(return_img=True))
            pl.close()
            ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(f"a = {a:.2f}", fontsize=13)
        ax[i, 0].text(-0.05, 0.5, label, transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=11)
    fig.suptitle(f"{case}: reduced (3,600 DOFs) vs full space", fontsize=13)
    fig.tight_layout(rect=(0.08, 0, 1, 0.96))
    fig.savefig(os.path.join(RESULTS, f"grasp_{case}.png"), dpi=80)
    plt.close(fig)


def video(case, fine, run, fps=30):
    import imageio.v2 as imageio
    fs = surface(fine["X"], fine["T"], fine["part"], list(fine["part_kind"]))
    pid = fs.point_data["vtkOriginalPointIds"]
    pl = pv.Plotter(window_size=(900, 900), off_screen=True)
    pl.set_background("white")
    draw(pl, fs, False, case)
    camera(pl, fs.merge(OBJECTS[case]["mesh"]()).bounds, 1.15)
    txt = pl.add_text("", position="upper_left", font_size=13, color="black")
    w = imageio.get_writer(os.path.join(RESULTS, f"grasp_{case}.mp4"), fps=fps, codec="libx264", macro_block_size=1)
    for a in np.concatenate([np.linspace(0, 1, 121), np.ones(30)]):
        fs.points = (run["P"] @ interp(run, a))[pid]
        txt.SetText(2, f"hyper-reduced hand grasping a {case}   a = {a:.2f}")
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    w.close()
    pl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the full space (slow)")
    ap.add_argument("--cases", nargs="*", default=["air", "ball", "cup"])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    fine, rig = load_fine()
    coarse = load_coarse()
    report = []
    for case in args.cases:
        sdf = None if case == "air" else OBJECTS[case]["sdf"]
        fr = 0.0 if case == "air" else FRICTION
        models = {"reduced": dict(coarse=coarse, k_pin=K_PIN_REDUCED)}
        if args.full:
            models["full"] = dict(coarse=None)
        runs = {}
        for model, kw in models.items():
            sysd = build_hand_system(fine, rig, sdf=sdf, friction=fr, **kw)
            run = close_hand(sysd)
            run["P"] = sysd["P"]
            runs[model] = run
            row = dict(case=case, model=model, dofs=sysd["n_dof"], seconds=round(run["seconds"], 1),
                       newton_its=int(run["its"].sum()),
                       tip_travel_mm=(fingertip_travel(sysd, fine, run["x"][-1]) * 1e3).round(1).tolist())
            if sdf is not None:
                row.update(sysd["contact_report"](run["x"][-1]))
            report.append(row)
            print(row, flush=True)
            json.dump(report, open(os.path.join(RESULTS, "grasp.json"), "w"), indent=1)
        render(case, fine, coarse, runs)
        if case == "cup":
            video(case, fine, runs["reduced"])


if __name__ == "__main__":
    main()
