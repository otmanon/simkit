"""Tetrahedralize the unified hand surface and label per-tet materials.

1. TetGen (``-pq1.5/10 -m``) meshes the unified genus-0 surface with a
   background sizing field: ~2.2 mm in and near the flexures (two to three
   element layers across the 4-5 mm slabs), ~3.5 mm in the pads, ~2 mm at
   the tendon attachment points (which are vertices of the input surface), growing to 12 mm in the palm and the middle
   of the stiff blocks.
2. Every part OBJ (``output/parts``) is tested with libigl's generalized
   winding number at the tet centroids; priority pads > flexures > links >
   palm.
3. The boundary of the tet mesh is checked to be one closed 2-manifold of genus
   0 -- for the hand with pads (the simulated mesh) and, for comparison, the
   same pipeline run on the hand without pads.

    python sdm_tets.py      # -> output/sdm_tets.npz, output/tet_report.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import igl
import pyvista as pv
import tetgen

from sdm_geometry import (OUT, HandParams, build_parts, read_obj, surface_topology,
                          fmt_topology, manifold_to_VF, tendon_anchors)

# name -> (E [Pa], nu, rho [kg/m^3])
MATERIALS = {
    "stiff polyurethane (links, palm)": dict(E=1.5e9, nu=0.35, rho=1150.0),
    "soft elastomer (flexure joints)": dict(E=0.6e6, nu=0.45, rho=1050.0),
    "softer elastomer (pads)": dict(E=0.2e6, nu=0.45, rho=1030.0),
}
KIND_MATERIAL = {"palm": "stiff polyurethane (links, palm)",
                 "link": "stiff polyurethane (links, palm)",
                 "flexure": "soft elastomer (flexure joints)",
                 "pad": "softer elastomer (pads)"}
PRIORITY = ("pad", "flexure", "link", "palm")      # first match wins


def sizing_field(V, parts, h_flex=0.0022, h_pad=0.0035, h_tip=0.0013, h_coarse=0.012, grad=0.6,
                 spacing=0.0025, h_anchor=0.002):
    """Background tet grid carrying TetGen's ``target_size`` point field:
    h = min(h_coarse, h_kind + grad * distance to the nearest flexure / pad box),
    with h_kind = h_flex for flexures, h_tip for fingertip pads, h_pad for the palm pad."""
    lo, hi = V.min(0) - 0.005, V.max(0) + 0.005
    g = pv.ImageData(dimensions=tuple(np.ceil((hi - lo) / spacing).astype(int) + 1),
                     spacing=(spacing,) * 3, origin=lo).triangulate()
    P = np.asarray(g.points)
    h = np.full(len(P), h_coarse)
    for n, (M, kind) in parts.items():
        if kind not in ("flexure", "pad"):
            continue
        Vb, _ = manifold_to_VF(M)             # 8 corners of a (rotated) box
        c = Vb.mean(0)
        _, _, Rt = np.linalg.svd(Vb - c)      # its local axes
        half = np.abs((Vb - c) @ Rt.T).max(0)
        d = np.linalg.norm(np.maximum(np.abs((P - c) @ Rt.T) - half, 0), axis=1)
        # fingertip pads are meshed finely (several tets through their thickness);
        # the large palm pad keeps the coarser pad size
        hk = h_flex if kind == "flexure" else (h_pad if n == "palm_pad" else h_tip)
        h = np.minimum(h, hk + grad * d)
    # fine spots at the tendon anchors so that a surface vertex lies close to
    # every intended attachment point
    for _, _, pa, _, pb in tendon_anchors():
        for q in (pa, pb):
            h = np.minimum(h, h_anchor + 1.2 * np.linalg.norm(P - q, axis=1))
    g.point_data["target_size"] = h
    return g


def tetrahedralize(V, F, parts):
    tg = tetgen.TetGen(V, F.astype(np.int32))
    X, T = tg.tetrahedralize(order=1, quality=True, minratio=1.5, mindihedral=10,
                             metric=True, bgmesh=sizing_field(V, parts))[:2]
    return np.asarray(X, float), np.asarray(T, np.int64)


def label(X, T, part_meshes, kinds):
    """Per-tet part index by generalized winding number, in PRIORITY order."""
    C = X[T].mean(1)
    part = np.full(len(T), -1)
    for i in sorted(range(len(kinds)), key=lambda i: PRIORITY.index(kinds[i])):
        V, F = part_meshes[i]
        w = igl.winding_number(V, F, C)
        part[(part < 0) & (w > 0.5)] = i
    if (part < 0).any():                      # (none expected for box CSG)
        TT = igl.tet_tet_adjacency(T)[0]
        for _ in range(10):
            for t in np.where(part < 0)[0]:
                nb = part[TT[t][TT[t] >= 0]]
                nb = nb[nb >= 0]
                if len(nb):
                    part[t] = np.bincount(nb).argmax()
    return part


def tet_boundary_topology(T):
    Fb = igl.boundary_facets(np.asarray(T))[0]
    return surface_topology(None, Fb, merge_tol=0)


def signed_volumes(X, T):
    a, b, c, d = (X[T[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def build():
    p = HandParams()
    report = {}
    # the pad-free hand: tet-boundary genus only (for the before/after check)
    V0, F0 = read_obj(os.path.join(OUT, "sdm_hand_nopads.obj"))
    X0, T0 = tetrahedralize(V0, F0, build_parts(p, pads=False))
    top0 = tet_boundary_topology(T0)
    report["tet_boundary_nopads"] = top0 | dict(n_vertices=len(X0), n_tets=len(T0))
    print(fmt_topology(f"tet boundary, no pads ({len(X0)} v, {len(T0)} tets)", top0))

    parts = build_parts(p, pads=True)
    names = list(parts)
    V, F = read_obj(os.path.join(OUT, "sdm_hand.obj"))
    X, T = tetrahedralize(V, F, parts)
    vol = signed_volumes(X, T)
    if (vol < 0).mean() > 0.5:
        T = T[:, [0, 2, 1, 3]]
        vol = -vol
    top = tet_boundary_topology(T)
    report["tet_boundary_pads"] = top | dict(n_vertices=len(X), n_tets=len(T))
    print(fmt_topology(f"tet boundary, with pads ({len(X)} v, {len(T)} tets)", top))
    assert top["closed_manifold"] and top["components"] == 1 and top["genus"] == 0
    assert top0["closed_manifold"] and top0["components"] == 1 and top0["genus"] == 0

    meshes = [read_obj(os.path.join(OUT, "parts", f"{n}.obj")) for n in names]
    kinds = np.array([parts[n][1] for n in names])
    part = label(X, T, meshes, kinds)
    mats = list(MATERIALS)
    mat = np.array([mats.index(KIND_MATERIAL[k]) for k in kinds])[part]
    E = np.array([MATERIALS[mats[m]]["E"] for m in mat])
    nu = np.array([MATERIALS[mats[m]]["nu"] for m in mat])
    rho = np.array([MATERIALS[mats[m]]["rho"] for m in mat])
    counts = {}
    for k in PRIORITY[::-1]:
        sel = kinds[part] == k
        counts[k] = dict(tets=int(sel.sum()), volume_cm3=float(vol[sel].sum() * 1e6))
        print(f"  {k:8s} {counts[k]['tets']:6d} tets  {counts[k]['volume_cm3']:7.2f} cm^3")
    report["label_counts"] = counts
    report["min_tet_volume_mm3"] = float(vol.min() * 1e9)
    report["materials"] = MATERIALS
    np.savez_compressed(os.path.join(OUT, "sdm_tets.npz"), X=X, T=T, part=part,
                        part_names=np.array(names), part_kind=kinds, mat=mat,
                        material_names=np.array(mats), E=E, nu=nu, rho=rho)
    with open(os.path.join(OUT, "tet_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    return report


if __name__ == "__main__":
    build()
