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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[300, 600, 1200, 2500, 5000])
    ap.add_argument("--k-max", type=int, default=120)
    ap.add_argument("--pad-modes", type=int, default=3)
    ap.add_argument("--min-modes", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(CO, exist_ok=True)

    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = Hand(scene)
    X, T = hand.X, hand.T
    names, kinds = hand.names, np.array(hand.kinds)
    B, lam, pad_idx = fine_basis(hand, args.k_max, args.pad_modes, args.min_modes)
    np.savez_compressed(os.path.join(CO, "fine_basis.npz"), B=B, eigenvalues=lam)
    meshes = [read_obj(os.path.join(OUT, "parts", f"{n}.obj")) for n in names]
    mat_of_kind = {k: scene["mat"][scene["part"] == i][0] for i, k in enumerate(kinds)
                   if (scene["part"] == i).any()}
    E_of = {m: e for m, e in zip(scene["mat"], scene["E"])}
    nu_of = {m: v for m, v in zip(scene["mat"], scene["nu"])}
    rho_of = {m: r for m, r in zip(scene["mat"], scene["rho"])}

    summary = dict(fine_vertices=int(len(X)), fine_tets=int(len(T)), modes=int(B.shape[1]),
                   top_mode_hz=float(np.sqrt(lam[-1]) / (2 * np.pi)), pad_modes_in_basis=pad_idx,
                   levels=[])
    for target in sorted(args.targets):
        t0 = time.time()
        Xc, Tc, P = coarsen(X=X, T=T, B=B, eigenvalues=lam, target_vertices=int(target))
        Tc = np.asarray(Tc, np.int64)
        vol = signed_volumes(Xc, Tc)
        if (vol < 0).mean() > 0.5:
            Tc = Tc[:, [0, 2, 1, 3]]
            vol = -vol
        part = label(Xc, Tc, meshes, kinds)
        mat = np.array([mat_of_kind[k] for k in kinds])[part]
        top = tet_boundary_topology(Tc)
        kind_t = kinds[part]
        counts = {k: int((kind_t == k).sum()) for k in ("palm", "link", "flexure", "pad")}
        missing = [n for i, n in enumerate(names) if not (part == i).any()]
        np.savez_compressed(os.path.join(CO, f"sdm_tets_{target}.npz"), X=Xc, T=Tc, part=part,
                            part_names=np.array(names), part_kind=kinds, mat=mat,
                            material_names=scene["material_names"],
                            E=np.array([E_of[m] for m in mat]), nu=np.array([nu_of[m] for m in mat]),
                            rho=np.array([rho_of[m] for m in mat]),
                            P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                            P_shape=np.array(P.shape))
        write_obj(os.path.join(CO, f"sdm_hand_{target}.obj"), Xc, igl.boundary_facets(Tc)[0])
        row = dict(target=int(target), vertices=int(len(Xc)), tets=int(len(Tc)),
                   genus=top["genus"], closed_manifold=bool(top["closed_manifold"]),
                   components=int(top["components"]), min_tet_volume_mm3=float(vol.min() * 1e9),
                   inverted_tets=int((vol <= 0).sum()), tets_by_kind=counts,
                   parts_without_tets=missing, P_min=float(P.data.min()),
                   seconds=float(time.time() - t0))
        summary["levels"].append(row)
        print(f"target {target:5d}: {len(Xc)} v, {len(Tc)} tets, genus {top['genus']:.0f}, "
              f"manifold {top['closed_manifold']}, tets {counts}, parts lost {missing}, "
              f"min P {P.data.min():.2f} [{row['seconds']:.0f}s]", flush=True)
    with open(os.path.join(CO, "coarse_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
