"""Hyper-reduced hand (1,200-vertex coarse mesh, 3,600 DOFs) closing in air, on a ball and on a cup
(friction on); ``--full`` also runs the full space (slow). -> results/grasp.json, grasp_<case>.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import simkit
from hand import load, build_hand_system, close_hand, fingertip_travel

pv.OFF_SCREEN = True
OUT = os.path.join(os.path.dirname(__file__), "results")
yaw = np.radians(-15)
R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
OBJECTS = {"air": (None, None),
           "ball": (lambda P: simkit.sphere_sdf(P, (-0.010, -0.036, 0.128), 0.030),
                    lambda: pv.Sphere(radius=0.030, center=(-0.010, -0.036, 0.128))),
           "cup": (lambda P: simkit.cup_sdf(P, (-0.015, -0.044, 0.120), 0.035, 0.003, 0.004, 0.100, R),
                   lambda: pv.Cylinder(center=(-0.015, -0.044, 0.120), direction=R[:, 0], radius=0.035, height=0.100))}
COLORS = ["#8a8d91", "#3b3f45", "#f28e2b", "#76b7e0"]


def surface(X, T, part, kinds):
    g = pv.UnstructuredGrid(np.hstack([np.full((len(T), 1), 4), T]).ravel(), np.full(len(T), 10), X)
    g.cell_data["k"] = [["palm", "link", "flexure", "pad"].index(kinds[p]) for p in part]
    return g.extract_surface(algorithm="dataset_surface")


os.makedirs(OUT, exist_ok=True)
fine, rig, coarse = load("hand_tets.npz"), load("hand_rig.npz"), load("hand_coarse_1200.npz")
kinds = list(fine["part_kind"])
fs = surface(fine["X"], fine["T"], fine["part"], kinds)
report = []
for case, (sdf, obj) in OBJECTS.items():
    rows = []
    for model, kw in [("reduced", dict(coarse=coarse, k_pin=1e5))] + [("full", {})] * ("--full" in sys.argv):
        sysd = build_hand_system(fine, rig, sdf=sdf, friction=1e10 * (sdf is not None), **kw)
        run = close_hand(sysd)
        report.append(dict(case=case, model=model, seconds=round(run["seconds"], 1), newton=int(run["its"].sum()),
                           tips_mm=(1e3 * fingertip_travel(sysd, fine, run["x"][-1])).round(1).tolist(),
                           **(sysd["contact"](run["x"][-1]) if sdf else {})))
        print(report[-1], flush=True)
        if model == "reduced":
            rows.append(("coarse mesh", [surface(x.reshape(-1, 3), coarse["T"], coarse["part"], kinds) for x in run["x"]]))
        rows.append((f"{model}: fine hand", [fs.copy() for _ in run["x"]]))
        for s, x in zip(rows[-1][1], run["x"]):
            s.points = (sysd["P"] @ x.reshape(-1, 3))[fs["vtkOriginalPointIds"]]
    fig, ax = plt.subplots(len(rows), 3, figsize=(11, 3.8 * len(rows)), squeeze=False)
    for i, (label, surfs) in enumerate(rows):
        for j, k in enumerate([0, 12, 24]):
            pl = pv.Plotter(window_size=(520, 560), off_screen=True)
            pl.add_mesh(surfs[k], scalars="k", cmap=COLORS, clim=(-0.5, 3.5), show_scalar_bar=False,
                        show_edges=label == "coarse mesh")
            if obj:
                pl.add_mesh(obj(), color="#c9b99a", opacity=0.5)
            pl.camera_position = [(0.75, -1.0, 0.35), (0, 0, 0), (0, 0, 1)]
            pl.reset_camera(bounds=fs.bounds)
            ax[i, j].imshow(pl.screenshot(return_img=True))
            ax[i, j].set_axis_off()
            ax[i, j].set_title(f"{label}, a = {k / 24:.1f}")
    fig.savefig(os.path.join(OUT, f"grasp_{case}.png"), dpi=80)
json.dump(report, open(os.path.join(OUT, "grasp.json"), "w"), indent=1)
