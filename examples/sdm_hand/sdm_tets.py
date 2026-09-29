"""Tetrahedralize the unified SDM-hand surface and label per-tet materials.

1. TetGen (``-pq1.5/10 -m``) meshes the unified genus-0 surface with a
   background sizing field: ~2.4 mm near the flexures and pads (at least two
   element layers across the 5 mm proximal flexure), growing to 12 mm in the
   palm and the middle of the stiff links.
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
                          fmt_topology)

# name -> (E [Pa], nu, rho [kg/m^3])
MATERIALS = {
    "stiff polyurethane (links, palm)": dict(E=1.5e9, nu=0.35, rho=1150.0),
    "soft elastomer (flexure joints)": dict(E=0.6e6, nu=0.45, rho=1050.0),
    "softer elastomer (fingertip pads)": dict(E=0.2e6, nu=0.45, rho=1030.0),
}
KIND_MATERIAL = {"palm": "stiff polyurethane (links, palm)",
                 "link": "stiff polyurethane (links, palm)",
                 "flexure": "soft elastomer (flexure joints)",
                 "pad": "softer elastomer (fingertip pads)"}
PRIORITY = ("pad", "flex", "prox", "dist", "palm")   # first match wins


def part_kind(name):
    if name.startswith("pad"):
        return "pad"
    if name.startswith("flex"):
        return "flexure"
    if name.startswith(("prox", "dist")):
        return "link"
    return "palm"


def sizing_field(V, parts, h_fine=0.0024, h_coarse=0.012, grad=0.7, spacing=0.0025):
    """Background tet grid with ``target_size`` = h_fine + grad * dist(flexures, pads)."""
    boxes = []
    for n, M in parts.items():
        if n.startswith(("flex", "pad")):
            b = M.bounding_box()
            boxes.append((np.array(b[:3]), np.array(b[3:])))
    lo, hi = V.min(0) - 0.005, V.max(0) + 0.005
    g = pv.ImageData(dimensions=tuple(np.ceil((hi - lo) / spacing).astype(int) + 1),
                     spacing=(spacing,) * 3, origin=lo).triangulate()
    P = np.asarray(g.points)
    d = np.full(len(P), np.inf)
    for a, b in boxes:
        d = np.minimum(d, np.linalg.norm(np.maximum(np.maximum(a - P, P - b), 0), axis=1))
    g.point_data["target_size"] = np.minimum(h_coarse, h_fine + grad * d)
    return g


def tetrahedralize(V, F, parts):
    tg = tetgen.TetGen(V, F.astype(np.int32))
    X, T = tg.tetrahedralize(order=1, quality=True, minratio=1.5, mindihedral=10,
                             metric=True, bgmesh=sizing_field(V, parts))[:2]
    return np.asarray(X, float), np.asarray(T, np.int64)


def label(X, T, part_meshes, names):
    """Per-tet part index by generalized winding number, in PRIORITY order."""
    C = X[T].mean(1)
    part = np.full(len(T), -1)
    order = sorted(range(len(names)),
                   key=lambda i: [names[i].startswith(p) for p in PRIORITY].index(True))
    for i in order:
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
    return surface_topology(np.zeros((T.max() + 1, 3)), Fb, merge_tol=0)


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
    part = label(X, T, meshes, names)
    kinds = np.array([part_kind(n) for n in names])
    mats = list(MATERIALS)
    mat_of_part = np.array([mats.index(KIND_MATERIAL[k]) for k in kinds])
    mat = mat_of_part[part]
    E = np.array([MATERIALS[mats[m]]["E"] for m in mat])
    nu = np.array([MATERIALS[mats[m]]["nu"] for m in mat])
    rho = np.array([MATERIALS[mats[m]]["rho"] for m in mat])
    counts = {}
    for k in ("palm", "link", "flexure", "pad"):
        sel = kinds[part] == k
        counts[k] = dict(tets=int(sel.sum()), volume_cm3=float(vol[sel].sum() * 1e6))
    report["label_counts"] = counts
    report["min_tet_volume_mm3"] = float(vol.min() * 1e9)
    report["materials"] = MATERIALS
    for k, c in counts.items():
        print(f"  {k:8s} {c['tets']:6d} tets  {c['volume_cm3']:7.2f} cm^3")
    np.savez_compressed(os.path.join(OUT, "sdm_tets.npz"), X=X, T=T, part=part,
                        part_names=np.array(names), part_kind=kinds, mat=mat,
                        material_names=np.array(mats), E=E, nu=nu, rho=rho)
    with open(os.path.join(OUT, "tet_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    return report


if __name__ == "__main__":
    build()
