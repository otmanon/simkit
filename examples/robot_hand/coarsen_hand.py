"""Coarsen the hand's tet mesh with mesh4PDE (PDE-aware) vs a geometric baseline.

mesh4PDE (github.com/otmanon/mesh4PDE) collapses edges in the order that costs
the *solution* the least: it scores each collapse through the lowest modes of
the Hessian of the energy you will actually solve. Here that operator is the
hand's heterogeneous elasticity -- silicone pads (0.6 MPa) and rubber tips
(1.5 MPa) on aluminium (69-72 GPa) and steel (205 GPa) -- pinned where the
simulation prescribes motion (palm, wrist, steel joint shafts). Every link
contains a shaft, so all 17 shells are pinned and the operator is SPD.

    H = elastic_energy_hessian(X, T, mu, lam) + Q_pin       (mesh4PDE + simkit)
    eigenvalues, B = simkit.eigs(H, k, M)
    Xc, Tc, P = coarsen(X, T, B, eigenvalues, target_vertices)

The geometry-only control is mesh4PDE's ``coarsen_shortest_edge`` at the same
size. Coarse tets get their materials the same way the fine ones do: generalized
winding numbers against the part OBJs.

    MESH4PDE=/path/to/mesh4PDE python coarsen_hand.py [--target 200] [--modes 48]

Writes ``output/coarse_<method>_<n>.npz`` / ``.obj`` and
``output/renders/hand_coarsened_<n>.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy as sp
import igl

import simkit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import coarsen, coarsen_shortest_edge, elastic_energy_hessian  # noqa: E402  (mesh4PDE)

sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from gripper_geometry import read_obj, write_obj  # noqa: E402
from materials import MATERIALS, lame, part_material  # noqa: E402

OUT = os.path.join(HERE, "output")


def hand_submesh(scene):
    cup = list(scene["part_names"]).index("cup")
    keep = scene["part"] != cup
    T = scene["T"][keep]
    used = np.unique(T)
    remap = -np.ones(len(scene["X"]), np.int64)
    remap[used] = np.arange(len(used))
    return scene["X"][used], remap[T], {k: scene[k][keep] for k in ("part", "E", "nu", "rho")}


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
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--modes", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=1e11, help="pinning stiffness per vertex")
    args = ap.parse_args()

    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    names = [str(n) for n in scene["part_names"]]
    X, T, attr = hand_submesh(scene)
    print(f"fine hand: {len(X)} vertices, {len(T)} tets")
    mu, lam = lame(attr["E"], attr["nu"])

    driven_parts = [i for i, n in enumerate(names) if n in ("palm", "wrist") or n.endswith("_shaft")]
    bI = np.unique(T[np.isin(attr["part"], driven_parts)])
    t0 = time.time()
    H = elastic_energy_hessian(X=X, T=T, mu=mu.reshape(-1, 1), lam=lam.reshape(-1, 1))
    Q, _ = simkit.dirichlet_penalty(bI, X[bI], X.shape[0], args.gamma)
    A = (H + Q).tocsc()
    M = sp.sparse.kron(simkit.massmatrix(X, T, rho=attr["rho"].reshape(-1, 1)),
                       sp.sparse.identity(3)).tocsc()
    eigenvalues, B = simkit.eigs(A, k=args.modes, M=M)
    print(f"operator + {args.modes} modes in {time.time()-t0:.0f}s; "
          f"lowest eigenvalues {np.array2string(eigenvalues[:4], precision=3)}")
    # where do the modes live?  (share of modal kinetic energy per material)
    m_v = simkit.massmatrix(X, T, rho=attr["rho"].reshape(-1, 1)).diagonal()
    e_v = (m_v[:, None] * (B.reshape(len(X), 3, -1) ** 2).sum(1)).sum(1)
    mat_of_v = np.full(len(X), "", dtype=object)
    for t, p in zip(T, attr["part"]):
        mat_of_v[t] = part_material(names[p])
    for mname in sorted(set(mat_of_v)):
        print(f"  mode energy in {mname:20s}: {e_v[mat_of_v == mname].sum() / e_v.sum():6.1%}")

    results = {}
    for method in ["mesh4pde", "shortest_edge"]:
        t0 = time.time()
        if method == "mesh4pde":
            Xc, Tc, P = coarsen(X=X, T=T, B=B, eigenvalues=eigenvalues,
                                target_vertices=args.target)
        else:
            Xc, Tc, P = coarsen_shortest_edge(X=X, T=T, target_vertices=args.target)
        part_c = label_parts(Xc, Tc, names)
        mats = [part_material(n) for n in names]
        E_c = np.array([MATERIALS[mats[p]]["E"] for p in part_c])
        print(f"{method:14s}: {len(Xc)} vertices, {len(Tc)} tets  [{time.time()-t0:.0f}s]")
        for mname in sorted(set(mats) - {"polystyrene_GPPS"}):
            k = np.isin(part_c, [i for i, m in enumerate(mats) if m == mname])
            print(f"    {mname:20s} {k.sum():5d} tets")
        np.savez_compressed(os.path.join(OUT, f"coarse_{method}_{args.target}.npz"),
                            X=Xc, T=Tc, part=part_c, E=E_c, part_names=np.array(names),
                            P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                            P_shape=np.array(P.shape))
        F = igl.boundary_facets(Tc)[0]
        write_obj(os.path.join(OUT, f"coarse_{method}_{args.target}.obj"), Xc, F)
        results[method] = (Xc, Tc, part_c)
    json.dump({"target": args.target, "modes": args.modes,
               "fine_vertices": int(len(X)),
               "coarse_vertices": {m: int(len(r[0])) for m, r in results.items()}},
              open(os.path.join(OUT, f"coarse_summary_{args.target}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
