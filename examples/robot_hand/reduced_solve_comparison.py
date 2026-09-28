"""Reduced-solve accuracy of the coarse spaces: mesh4PDE vs shortest-edge.

Mirrors mesh4PDE's beam experiments. For each load ``f`` the fine static
solve ``A u = f`` (the spring-actuated elastic hand, palm pinned) is compared
with the Galerkin solve in each coarse space ``B = prolongation_to_subspace(P)``:

    u_r = B (B^T A B)^{-1} B^T f        (mesh4PDE's galerkin_projection / reduced_solve)
    err = ||u - u_r||_A / ||u||_A        (mesh4PDE's relative_energy_error)

Galerkin minimises exactly this norm over the subspace, so ``err`` is a property
of the coarse space, not of a solver. A second, more tangible number is the
relative displacement error on the loaded vertices.

Loads (palm frame, the hand's meshing pose):
* gravity                 -- the hand's own weight
* fingertip press         -- 1 N per fingertip, pressure on the palmar half of each rubber tip
* distal-pad press        -- 1 N per distal-phalanx silicone pad, on its outer face
* grasp                   -- 1 N on every pad and every fingertip at once

    MESH4PDE=~/mesh4pde python reduced_solve_comparison.py
    python reduced_solve_comparison.py --plot-only    # redraw from the saved JSON

Writes ``output/reduced_solve_errors.json`` and ``output/renders/reduced_solve_errors.png``.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy as sp
import igl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import galerkin_projection, prolongation_to_subspace, reduced_solve, relative_energy_error  # noqa: E402

from allegro_kinematics import AllegroHand  # noqa: E402
from hand_geometry import XML, BASE  # noqa: E402
from hand_springs import HandSprings, hand_operator  # noqa: E402

sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from simulate_grasp import solve_spd  # noqa: E402

OUT = os.path.join(HERE, "output")
TARGETS = [200, 1000, 2000]
METHODS = {"mesh4pde": ("mesh4PDE (PDE-aware)", "#2a78d6"),
           "shortest_edge": ("shortest-edge (geometry only)", "#eb6834")}


def surface_loads(X, T, part, names, meta, M):
    """Build the four load vectors (interleaved, length 3n)."""
    F, J, _ = igl.boundary_facets(T)
    N = igl.per_face_normals(X, F, np.array([0, 0, 1.0]))
    out_dir = X[F].mean(1) - X[T[J]].mean(1)          # make normals point outward
    N *= np.sign((N * out_dir).sum(1))[:, None]
    area = 0.5 * igl.doublearea(X, F)
    T0 = AllegroHand(XML).fk({}, BASE)
    fname = [names[p] for p in part[J]]

    def pressure(select):
        """{group: (vertices, forces)} for faces picked by select(name) -> (group, body)."""
        f = np.zeros_like(X)
        groups = {}
        for i, nm in enumerate(fname):
            g = select(nm)
            if g is None:
                continue
            group, body = g
            palmar = T0[body][:3, 0]
            if N[i] @ palmar < 0.3:
                continue
            groups.setdefault(group, []).append(i)
        for group, faces in groups.items():
            fg = np.zeros_like(X)
            for i in faces:
                for v in F[i]:
                    fg[v] -= N[i] * area[i] / 3.0
            fg /= np.linalg.norm(fg.sum(0))              # 1 N net per group
            f += fg
        return f

    tips = lambda nm: (nm, nm.split("_")[0] + "_distal") if nm.endswith("_tip_rubber") else None
    dpads = lambda nm: (nm, nm[:-4]) if nm.endswith("_distal_pad") else None
    allpad = lambda nm: (nm, "palm" if nm == "palm_pad" else nm[:-4]) if nm.endswith("_pad") else None
    f_tip, f_dpad, f_pads = pressure(tips), pressure(dpads), pressure(allpad)
    g = np.zeros_like(X)
    g[:, 1] = -9.81
    f_grav = (M @ g.reshape(-1)).reshape(-1, 3)
    return {"gravity": f_grav, "fingertip press": f_tip,
            "distal-pad press": f_dpad, "grasp": f_tip + f_pads}


def main():
    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    meta = json.load(open(os.path.join(OUT, "hand_meta.json")))
    names = [str(n) for n in scene["part_names"]]
    springs = HandSprings(scene, meta)
    A, M, X, T, part, _ = hand_operator(scene, meta, springs)
    loads = surface_loads(X, T, part, names, meta, M)

    t0 = time.time()
    u_full = {k: solve_spd(A, f.reshape(-1)) for k, f in loads.items()}
    print(f"fine solves ({A.shape[0]} DOFs) [{time.time()-t0:.0f}s]")
    for k, u in u_full.items():
        print(f"  {k:18s} max |u| = {np.abs(u).max()*1e6:8.2f} um")

    results = []
    for method in METHODS:
        for target in TARGETS:
            d = np.load(os.path.join(OUT, f"coarse_{method}_{target}.npz"))
            P = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]),
                                     shape=tuple(d["P_shape"]))
            if P.shape[0] != len(X):
                P = P.T.tocsc()
            B = prolongation_to_subspace(P=P, dof=3)
            Ar = galerkin_projection(operator=A, basis=B)
            row = dict(method=method, target=target, vertices=int(P.shape[1]))
            for k, f in loads.items():
                u_r = reduced_solve(reduced_operator=Ar, basis=B, f=f.reshape(-1, 1)).ravel()
                u = u_full[k]
                loaded = np.linalg.norm(f, axis=1) > 0
                sel = np.repeat(loaded, 3)
                row[k] = dict(
                    energy=float(relative_energy_error(u_full=u, u_reduced=u_r, A=A)),
                    displacement=float(np.linalg.norm(u[sel] - u_r[sel]) / np.linalg.norm(u[sel])))
            results.append(row)
            print(f"{method:14s} {row['vertices']:5d} v: " + "  ".join(
                f"{k} {row[k]['energy']:.3f}" for k in loads), flush=True)

    json.dump(results, open(os.path.join(OUT, "reduced_solve_errors.json"), "w"), indent=1)
    plot(results, list(loads))


def plot(results, loads):
    """Small multiples: one panel per load, energy-norm error vs coarse vertex count."""
    fig, axes = plt.subplots(1, len(loads), figsize=(4.0 * len(loads), 4.1), sharey=True)
    for ax, k in zip(axes, loads):
        for method, (label, colour) in METHODS.items():
            rows = [r for r in results if r["method"] == method]
            x = [r["vertices"] for r in rows]
            y = [r[k]["energy"] for r in rows]
            ax.plot(x, y, "-o", color=colour, lw=2, ms=8, mec="#fcfcfb", mew=2, label=label)
            ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=10, color="#3d3d3a")
        ax.set_title(k, fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([200, 1000, 2000])
        ax.set_xticklabels(["200", "1000", "2000"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([0.1, 0.2, 0.4, 0.6])
        ax.set_yticklabels(["0.1", "0.2", "0.4", "0.6"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(0.09, 0.75)
        ax.set_xlim(160, 2900)
        ax.set_xlabel("coarse vertices")
        ax.grid(True, which="major", color="#e5e4df", lw=0.8)
        ax.set_facecolor("#fcfcfb")
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0].set_ylabel("relative energy-norm error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.92))
    fig.suptitle("Galerkin reduced-solve error of the coarse hand vs the fine hand "
                 "(61,603 vertices); lower is better", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path = os.path.join(OUT, "renders", "reduced_solve_errors.png")
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        res = json.load(open(os.path.join(OUT, "reduced_solve_errors.json")))
        plot(res, [k for k in res[0] if k not in ("method", "target", "vertices")])
    else:
        main()
