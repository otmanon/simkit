"""Coarsen the compliant hand with mesh4PDE at several vertex counts.

mesh4PDE (github.com/otmanon/mesh4PDE) collapses edges in the order that costs
the *solution* the least, scoring each collapse through the lowest modes of the
operator that is actually simulated:

    A = H_elastic + H_tendons + H_pins  at the open pose (a = 0), wrist pinned
    eigenvalues, B = simkit.eigs(A, k, M)
    Xc, Tc, P = coarsen(X, T, B, eigenvalues, target_vertices)

The basis has to reach the soft tissue: with the hinge pins the first 5 modes
are rigid flexion of the fingers and from mode ~8 on the spectrum is pad
deformation (``sdm_modes.py --until-soft``). ``k`` is therefore chosen as the
smallest count that holds at least ``--pad-modes`` pad modes (>= 50 % of the
mode's energy in one pad) for every fingertip and for the palm pad, and never
fewer than ``--min-modes``.

Coarse tets get their part (and so their material) from generalized winding
numbers against the part OBJs, exactly like the fine mesh. Each level is saved
as a scene ``output/coarse/sdm_tets_<n>.npz`` that ``sdm_sim.Hand`` runs as is,
plus the prolongation ``P`` (fine vertex = P @ coarse vertices), which is how the
coarse simulations are compared with the fine one.

    python sdm_coarsen.py [--targets 300 600 1200 2500 5000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy as sp
import igl

import simkit
import simkit.energies as energies

sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
if not os.path.isdir(sys.path[0]):
    sys.path[0] = "/home/user/mesh4pde"
from lib import coarsen  # noqa: E402  (mesh4PDE)

from sdm_geometry import read_obj, write_obj  # noqa: E402
from sdm_sim import Hand, OUT  # noqa: E402
from sdm_tets import label, tet_boundary_topology, signed_volumes  # noqa: E402

CO = os.path.join(OUT, "coarse")
PADS = ["index_pad", "middle_pad", "ring_pad", "little_pad", "thumb_pad", "palm_pad"]


def pad_modes(hand, B, lam):
    """Per pad: indices of the modes with >= 50 % of their energy in that pad."""
    out = {}
    for name in PADS:
        sel = hand.part == hand.names.index(name)
        J = simkit.deformation_jacobian(hand.X, hand.T[sel]).tocsc()
        J.resize((J.shape[0], 3 * hand.n))
        H = energies.stable_neo_hookean_hessian_x(hand.X, J, hand.mu[sel], hand.lam[sel],
                                                  hand.vol[sel], psd=True)
        share = np.einsum("ij,ij->j", B, H @ B) / lam
        out[name] = np.where(share >= 0.5)[0]
    return out


def fine_basis(hand, k_max, pad_modes_wanted, min_modes):
    fr = hand.free
    A = hand.hessian(hand.X.reshape(-1), 0.0)[fr][:, fr].tocsc()
    M = sp.sparse.diags(hand.m[fr])
    t0 = time.time()
    lam, Bf = simkit.eigs(A, k=k_max, M=M)
    o = np.argsort(lam)
    lam, Bf = lam[o], Bf[:, o]
    B = np.zeros((3 * hand.n, k_max))
    B[fr] = Bf
    pm = pad_modes(hand, B, lam)
    need = [pm[n][pad_modes_wanted - 1] + 1 if len(pm[n]) >= pad_modes_wanted else k_max
            for n in PADS]
    k = max(min_modes, max(need))
    print(f"fine eigs: {k_max} modes [{time.time() - t0:.0f}s]; pad modes per pad: "
          + ", ".join(f"{n.split('_')[0]} {[int(i) + 1 for i in pm[n][:pad_modes_wanted]]}"
                      for n in PADS), flush=True)
    print(f"-> scoring collapses with the lowest {k} modes "
          f"({np.sqrt(lam[k - 1]) / (2 * np.pi):.0f} Hz)", flush=True)
    return B[:, :k], lam[:k], {n: [int(i) + 1 for i in pm[n] if i < k] for n in PADS}


def locate(X, T, Q, k=48):
    """Index of the tet containing each query point (or the one it is least outside of),
    among the k tets with the nearest centroids."""
    cand = sp.spatial.cKDTree(X[T].mean(1)).query(Q, k=min(k, len(T)))[1]
    a, b, c, d = (X[T[cand][..., i]] for i in range(4))
    M = np.stack([b - a, c - a, d - a], -1)                      # (nq, k, 3, 3)
    lam = np.linalg.solve(M, (Q[:, None, :] - a)[..., None])[..., 0]
    bary = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], -1).min(-1)
    return cand[np.arange(len(Q)), bary.argmax(1)]


def homogenize(sc, fine):
    """Per coarse tet, the Reuss (series) average of the fine tets whose centroids lie in
    it: E = V / sum(V_i / E_i); nu and rho volume-averaged. Centroid labels alone weld
    a joint whose thin flexure no coarse centroid falls in; in series, the soft flexure
    governs the joint's compliance, which is what the Reuss bound keeps."""
    Cf = fine["X"][fine["T"]].mean(1)
    Vf = np.abs(signed_volumes(fine["X"], fine["T"]))
    Xc, Tc = np.asarray(sc["X"], float), np.asarray(sc["T"], np.int64)
    owner = locate(Xc, Tc, Cf)
    n = len(Tc)
    V = np.bincount(owner, Vf, n)
    has = V > 0
    E = np.asarray(sc["E"], float).copy()
    nu = np.asarray(sc["nu"], float).copy()
    rho = np.asarray(sc["rho"], float).copy()
    E[has] = V[has] / np.bincount(owner, Vf / fine["E"], n)[has]
    nu[has] = np.bincount(owner, Vf * fine["nu"], n)[has] / V[has]
    rho[has] = np.bincount(owner, Vf * fine["rho"], n)[has] / V[has]
    return dict(sc, E=E, nu=nu, rho=rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[300, 600, 1200, 2500, 5000])
    ap.add_argument("--k-max", type=int, default=120)
    ap.add_argument("--pad-modes", type=int, default=3)
    ap.add_argument("--min-modes", type=int, default=60)
    ap.add_argument("--max-extrapolation", type=float, default=0.1,
                    help="(first-stage) bound: refuse collapses that leave a fine vertex further than this (in "
                         "barycentric units) outside its coarse tet; without it (inf) the stiff "
                         "blocks, which move rigidly in every mode and so cost nothing to "
                         "collapse, shrink to slivers (22-28%% of the volume kept)")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="mesh4PDE min_quality: refuse collapses leaving a one-ring tet below this "
                         "mean-ratio quality (0 = off)")
    ap.add_argument("--suffix", default="", help="appended to the level's file names, e.g. _q30")
    ap.add_argument("--homogenize", action="store_true",
                    help="Reuss-average the fine moduli into each coarse tet (default: the coarse "
                         "tet's winding-number material, as for the fine mesh)")
    args = ap.parse_args()
    os.makedirs(CO, exist_ok=True)

    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = Hand(scene)
    X, T = hand.X, hand.T
    names, kinds = hand.names, np.array(hand.kinds)
    cache = os.path.join(CO, "fine_basis.npz")
    if os.path.exists(cache) and np.load(cache)["B"].shape[0] == 3 * len(X):
        d = np.load(cache, allow_pickle=True)
        B, lam, pad_idx = d["B"], d["eigenvalues"], d["pad_idx"].item()
        print(f"fine basis from cache: {B.shape[1]} modes", flush=True)
    else:
        B, lam, pad_idx = fine_basis(hand, args.k_max, args.pad_modes, args.min_modes)
        np.savez_compressed(cache, B=B, eigenvalues=lam, pad_idx=np.array(pad_idx, dtype=object))
    meshes = [read_obj(os.path.join(OUT, "parts", f"{n}.obj")) for n in names]
    mat_of_kind = {k: scene["mat"][scene["part"] == i][0] for i, k in enumerate(kinds)
                   if (scene["part"] == i).any()}
    E_of = {m: e for m, e in zip(scene["mat"], scene["E"])}
    nu_of = {m: v for m, v in zip(scene["mat"], scene["nu"])}
    rho_of = {m: r for m, r in zip(scene["mat"], scene["rho"])}

    fine_vol = signed_volumes(X, T).sum()
    summary = dict(max_extrapolation=args.max_extrapolation, fine_vertices=int(len(X)), fine_tets=int(len(T)), modes=int(B.shape[1]),
                   top_mode_hz=float(np.sqrt(lam[-1]) / (2 * np.pi)), pad_modes_in_basis=pad_idx,
                   levels=[])
    def make_scene(Xc, Tc):
        part = label(Xc, Tc, meshes, kinds)
        mat = np.array([mat_of_kind[k] for k in kinds])[part]
        return dict(X=Xc, T=Tc, part=part, part_names=np.array(names), part_kind=kinds, mat=mat,
                    material_names=scene["material_names"],
                    E=np.array([E_of[m] for m in mat]), nu=np.array([nu_of[m] for m in mat]),
                    rho=np.array([rho_of[m] for m in mat]))

    def oriented(Xc, Tc):
        Tc = np.asarray(Tc, np.int64)
        if (signed_volumes(Xc, Tc) < 0).mean() > 0.5:
            Tc = Tc[:, [0, 2, 1, 3]]
        return Tc

    for target in sorted(args.targets):
        t0 = time.time()
        # Stage 1 from the fine mesh with the fine basis. A refused collapse leaves
        # mesh4PDE's queue for good, so under a tight bound the loop can stop well
        # above the target; then restart from the coarse mesh (fresh queue, its own
        # modes) and, only if that stalls too, relax the bound one notch.
        schedule = [args.max_extrapolation] + [b for b in (0.3, 0.6, 1.0, 2.0) if b > args.max_extrapolation]
        bi, stages = 0, []
        Xs, Ts, Bs, lams, Ptot = X, T, B, lam, None
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)    # the shortfall is handled below
                Xc, Tc, P = coarsen(X=Xs, T=Ts, B=Bs, eigenvalues=lams, target_vertices=int(target),
                                    max_extrapolation=schedule[bi],
                                    **({"min_quality": args.min_quality} if args.min_quality else {}))
            Tc = oriented(Xc, Tc)
            Ptot = P if Ptot is None else (Ptot @ P).tocsc()
            stages.append(dict(bound=schedule[bi], vertices=int(len(Xc)), modes=int(Bs.shape[1])))
            print(f"  target {target}: stage {len(stages)} bound {schedule[bi]}: {len(Xs)} -> {len(Xc)} v",
                  flush=True)
            if len(Xc) <= target or (len(Xs) - len(Xc) < 0.03 * len(Xs) and bi == len(schedule) - 1):
                break
            if len(Xs) - len(Xc) < 0.03 * len(Xs):
                bi += 1
            h = Hand(make_scene(Xc, Tc))
            fr = h.free
            A = h.hessian(Xc.reshape(-1), 0.0)[fr][:, fr].tocsc()
            k = min(B.shape[1], len(fr) // 4)
            lams, Bf = simkit.eigs(A, k=k, M=sp.sparse.diags(h.m[fr]))
            o = np.argsort(lams)
            lams, Bs = lams[o], np.zeros((3 * h.n, k))
            Bs[fr] = Bf[:, o]
            Xs, Ts = Xc, Tc
        P = Ptot
        sc = homogenize(make_scene(Xc, Tc), scene) if args.homogenize else make_scene(Xc, Tc)
        vol = signed_volumes(Xc, Tc)
        top = tet_boundary_topology(Tc)
        kind_t = kinds[sc["part"]]
        counts = {k: int((kind_t == k).sum()) for k in ("palm", "link", "flexure", "pad")}
        missing = [n for i, n in enumerate(names) if not (sc["part"] == i).any()]
        np.savez_compressed(os.path.join(CO, f"sdm_tets_{target}{args.suffix}.npz"), **sc, pinned=coarse_pinned(P, X),
                            P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                            P_shape=np.array(P.shape))
        write_obj(os.path.join(CO, f"sdm_hand_{target}{args.suffix}.obj"), Xc, igl.boundary_facets(Tc)[0])
        row = dict(target=int(target), min_quality=args.min_quality, suffix=args.suffix, vertices=int(len(Xc)), tets=int(len(Tc)), stages=stages,
                   genus=top["genus"], closed_manifold=bool(top["closed_manifold"]),
                   components=int(top["components"]), min_tet_volume_mm3=float(vol.min() * 1e9),
                   inverted_tets=int((vol <= 0).sum()), tets_by_kind=counts,
                   volume_ratio=float(vol.sum() / fine_vol),
                   parts_without_tets=missing, P_min=float(P.data.min()),
                   seconds=float(time.time() - t0))
        summary["levels"].append(row)
        print(f"target {target:5d}: {len(Xc)} v, {len(Tc)} tets, genus {top['genus']:.0f}, "
              f"manifold {top['closed_manifold']}, volume {row['volume_ratio']:.3f}, tets {counts}, "
              f"parts lost {missing}, min P {P.data.min():.2f} [{row['seconds']:.0f}s]", flush=True)
    with open(os.path.join(CO, f"coarse_summary{args.suffix}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


def coarse_pinned(P, X_fine):
    """Coarse vertices that the fine pinned vertices (wrist base plane) interpolate from."""
    pin = np.where(X_fine[:, 2] < X_fine[:, 2].min() + 1e-7)[0]
    sub = P.tocsr()[pin]
    return np.unique(sub.indices[np.abs(sub.data) > 1e-12])


def add_pinned():
    """Store ``pinned`` (from P) in every saved level."""
    Xf = np.load(os.path.join(OUT, "sdm_tets.npz"))["X"]
    for f in sorted(os.listdir(CO)):
        if f.startswith("sdm_tets_") and f.endswith(".npz"):
            d = dict(np.load(os.path.join(CO, f)))
            P = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]), shape=tuple(d["P_shape"]))
            d["pinned"] = coarse_pinned(P, Xf)
            np.savez_compressed(os.path.join(CO, f), **d)
            print(f, "pinned", len(d["pinned"]))


def rehomogenize():
    """Re-apply ``homogenize`` to the saved levels (keeps the meshes and P)."""
    fine = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    for f in sorted(os.listdir(CO)):
        if f.startswith("sdm_tets_") and f.endswith(".npz"):
            d = dict(np.load(os.path.join(CO, f)))
            np.savez_compressed(os.path.join(CO, f), **homogenize(d, fine))
            print("homogenized", f)


if __name__ == "__main__":
    if "--add-pinned" in sys.argv:
        add_pinned()
    elif "--rehomogenize" in sys.argv:
        rehomogenize()
    else:
        main()
