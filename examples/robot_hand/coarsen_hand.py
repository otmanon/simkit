"""Coarsen the hand's tet mesh with mesh4PDE (PDE-aware) vs a geometric baseline.

mesh4PDE (github.com/otmanon/mesh4PDE) collapses edges in the order that costs
the *solution* the least: it scores each collapse through the lowest modes of
the Hessian of the energy you will actually solve. Here that operator is the
hand as it is actually simulated: heterogeneous volumetric elasticity --
silicone pads (0.6 MPa) and rubber tips (1.5 MPa) on aluminium (69-72 GPa) and
steel (205 GPa) -- plus the SimKit mass springs that join and actuate the links
(``hand_springs.py``), with only the palm and wrist pinned.

    A = H_elastic + H_springs + Q_pin                (hand_springs.hand_operator)
    eigenvalues, B = simkit.eigs(A, k, M)            (eigen_modes.py, saved)
    Xc, Tc, P = coarsen(X, T, B, eigenvalues, target_vertices)

``k`` defaults to "up to the 4th fingertip mode", so the fingertips are in the
basis the collapses are scored through.

The geometry-only control is mesh4PDE's ``coarsen_shortest_edge`` at the same
size. Coarse tets get their materials the same way the fine ones do: generalized
winding numbers against the part OBJs.

    MESH4PDE=/path/to/mesh4PDE python coarsen_hand.py [--targets 200 1000 2000]

Writes ``output/coarse_<method>_<n>.npz`` / ``.obj`` and ``coarse_summary.json``
(including a boundary-manifoldness check of every result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import igl


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import coarsen, coarsen_shortest_edge  # noqa: E402  (mesh4PDE)

sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from gripper_geometry import read_obj, write_obj  # noqa: E402
from materials import MATERIALS, part_material  # noqa: E402
from hand_springs import HandSprings, hand_operator  # noqa: E402
from mesh_checks import nonmanifold_boundary  # noqa: E402

OUT = os.path.join(HERE, "output")


def label_parts(X, T, names):
    C = X[T].mean(1)
    part = np.full(len(T), -1)
    for i, n in enumerate(names):
        if n == "cup":
            continue
        V, F = read_obj(os.path.join(OUT, "parts", f"{n}.obj"))
        todo = np.where(part < 0)[0]
        part[todo[igl.winding_number(V, F, C[todo]) > 0.5]] = i
    # a coarse tet can straddle a thin part: fall back to nearest part by centroid
    miss = np.where(part < 0)[0]
    if len(miss):
        best = np.full(len(miss), np.inf)
        for i, n in enumerate(names):
            if n == "cup":
                continue
            V, F = read_obj(os.path.join(OUT, "parts", f"{n}.obj"))
            d = np.abs(igl.signed_distance(C[miss], V, F)[0])
            better = d < best
            best[better], part[miss[better]] = d[better], i
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[200, 1000, 2000])
    ap.add_argument("--modes", type=int, default=None,
                    help="modes to score collapses with (default: up to and including "
                         "the 4th fingertip mode found by eigen_modes.py)")
    args = ap.parse_args()

    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    meta = json.load(open(os.path.join(OUT, "hand_meta.json")))
    names = [str(n) for n in scene["part_names"]]
    springs = HandSprings(scene, meta)
    t0 = time.time()
    A, M, X, T, part, aux = hand_operator(scene, meta, springs)
    print(f"fine hand: {len(X)} vertices, {len(T)} tets; operator = elastic + "
          f"{len(springs.E)} SimKit mass springs + palm pin [{time.time()-t0:.0f}s]")
    modes = np.load(os.path.join(OUT, "hand_modes.npz"))
    assert len(modes["keep"]) == len(X), "hand_modes.npz is stale: rerun eigen_modes.py"
    tips = list(modes["tips"])
    k = args.modes or (tips[min(3, len(tips) - 1)] + 1 if tips else modes["B"].shape[1])
    B, eigenvalues = modes["B"][:, :k], modes["eigenvalues"][:k]
    print(f"scoring collapses with the lowest {k} modes (fingertip modes: {[i + 1 for i in tips]})")

    summary = {"modes": int(k), "fine_vertices": int(len(X)), "runs": []}
    mats = [part_material(n) for n in names]
    for target in args.targets:
        for method in ["mesh4pde", "shortest_edge"]:
            t0 = time.time()
            if method == "mesh4pde":
                Xc, Tc, P = coarsen(X=X, T=T, B=B, eigenvalues=eigenvalues,
                                    target_vertices=target)
            else:
                Xc, Tc, P = coarsen_shortest_edge(X=X, T=T, target_vertices=target)
            part_c = label_parts(Xc, Tc, names)
            E_c = np.array([MATERIALS[mats[p]]["E"] for p in part_c])
            bv, be = nonmanifold_boundary(Tc)
            soft = {m: int(np.isin(part_c, [i for i, mm in enumerate(mats) if mm == m]).sum())
                    for m in ("silicone_shore_20A", "polyurethane_40A")}
            tip_parts = sorted({names[p] for p in part_c if names[p].endswith("_tip_rubber")})
            print(f"target {target:5d} {method:14s}: {len(Xc)} vertices, {len(Tc)} tets, "
                  f"non-manifold v/e {len(bv)}/{len(be)}, pad tets {soft['silicone_shore_20A']}, "
                  f"tip-rubber tets {soft['polyurethane_40A']} in {len(tip_parts)}/4 tips "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            tag = f"{method}_{target}"
            np.savez_compressed(os.path.join(OUT, f"coarse_{tag}.npz"),
                                X=Xc, T=Tc, part=part_c, E=E_c, part_names=np.array(names),
                                P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                                P_shape=np.array(P.shape))
            write_obj(os.path.join(OUT, f"coarse_{tag}.obj"), Xc, igl.boundary_facets(Tc)[0])
            summary["runs"].append(dict(target=target, method=method, vertices=int(len(Xc)),
                                        tets=int(len(Tc)), nonmanifold_vertices=len(bv),
                                        nonmanifold_edges=len(be), tips_kept=len(tip_parts), **soft))
    json.dump(summary, open(os.path.join(OUT, "coarse_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
