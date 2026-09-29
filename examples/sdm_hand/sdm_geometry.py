"""Simplified box-CSG geometry of the SDM Hand (Dollar & Howe, IJRR 2010).

The SDM Hand is a four-finger, eight-joint, single-actuator underactuated
hand made by shape deposition manufacturing: stiff polyurethane links joined
by soft elastomer flexure joints, soft fingertip pads, and one tendon that
closes all fingers. Two fingers oppose the other two across the palm.

Here every part is a rectangular prism (``manifold3d.Manifold.cube``):

* palm                         120 x 100 x 20 mm
* 4 proximal links             20 x 20 x 62 mm   (joint-to-joint 70 mm)
* 4 distal links               20 x 20 x 45 mm
* 4 proximal flexures          5 (thick) x 16 (wide) x 8 (long) mm  (softer)
* 4 distal flexures            7 (thick) x 16 (wide) x 8 (long) mm  (stiffer)
* 4 fingertip pads (optional)  4.5 x 18 x 35 mm on the palmar face of the distal link

Each flexure overlaps the parts it joins by 0.5 mm so the CSG union is a single
solid. The union with and without pads is checked to be a single closed,
oriented 2-manifold of genus 0 (Euler characteristic 2).

Frame: metres; palm on ``0 <= z <= 0.02``, fingers along ``+z``; fingers at
``x = +-0.05`` close towards ``x = 0`` (the palmar side of each finger faces
the palm centre). In the simulation gravity acts along ``+z`` -- the hand is
mounted facing down, as on the arm it was designed for -- and renders show it
that way (palm at the top, fingers hanging).

    python sdm_geometry.py        # -> output/parts/*.obj, output/sdm_hand*.obj
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import manifold3d as m3d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


@dataclass
class HandParams:
    palm: tuple = (0.120, 0.100, 0.020)      # x, y, z extents
    finger_x: float = 0.050                   # |x| of the finger centre lines
    finger_y: tuple = (-0.025, 0.025)         # y of the two fingers per side
    link_w: float = 0.020                     # square link cross-section
    prox_len: float = 0.062
    dist_len: float = 0.045
    flex_len: float = 0.008                   # visible flexure length
    flex_w: float = 0.016                     # flexure width (y)
    flex_t_prox: float = 0.005                # flexure thickness (x): proximal
    flex_t_dist: float = 0.007                #   distal (thicker -> stiffer)
    overlap: float = 0.0005                   # flexure embedding into links
    pad_t: float = 0.0045                     # pad thickness (proud of the face)
    pad_w: float = 0.018
    pad_z: tuple = (0.007, 0.042)             # pad span along the distal link
                                              # (from the distal link's base)


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def finger_names(p=HandParams()):
    return [f"{side}{j}" for side in ("L", "R") for j in range(len(p.finger_y))]


def z_levels(p=HandParams()):
    """z of: palm top, prox link base/top, distal link base/top."""
    z0 = p.palm[2]
    zp0 = z0 + p.flex_len
    zp1 = zp0 + p.prox_len
    zd0 = zp1 + p.flex_len
    zd1 = zd0 + p.dist_len
    return z0, zp0, zp1, zd0, zd1


def build_parts(p=HandParams(), pads=True):
    """Return an ordered dict ``{name: Manifold}``."""
    parts = {}
    px, py, pz = p.palm
    parts["palm"] = box((-px / 2, -py / 2, 0), (px / 2, py / 2, pz))
    z0, zp0, zp1, zd0, zd1 = z_levels(p)
    w, o = p.link_w / 2, p.overlap
    for side, s in (("L", -1.0), ("R", 1.0)):
        for j, yc in enumerate(p.finger_y):
            n = f"{side}{j}"
            xc = s * p.finger_x
            parts[f"prox_{n}"] = box((xc - w, yc - w, zp0), (xc + w, yc + w, zp1))
            parts[f"dist_{n}"] = box((xc - w, yc - w, zd0), (xc + w, yc + w, zd1))
            for key, t, za, zb in (("flexprox", p.flex_t_prox, z0, zp0),
                                   ("flexdist", p.flex_t_dist, zp1, zd0)):
                parts[f"{key}_{n}"] = box((xc - t / 2, yc - p.flex_w / 2, za - o),
                                          (xc + t / 2, yc + p.flex_w / 2, zb + o))
            if pads:
                xin = xc - s * w                    # palmar face of the finger
                xa, xb = xin - s * p.pad_t, xin + s * o
                parts[f"pad_{n}"] = box((min(xa, xb), yc - p.pad_w / 2, zd0 + p.pad_z[0]),
                                        (max(xa, xb), yc + p.pad_w / 2, zd0 + p.pad_z[1]))
    return parts


def union(parts):
    return m3d.Manifold.batch_boolean(list(parts.values()), m3d.OpType.Add)


def manifold_to_VF(M):
    mesh = M.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    return V, F


def write_obj(path, V, F):
    with open(path, "w") as f:
        f.write(f"# {os.path.basename(path)}  ({len(V)} v, {len(F)} f)  units: metres\n")
        np.savetxt(f, V, fmt="v %.9f %.9f %.9f")
        np.savetxt(f, F + 1, fmt="f %d %d %d")


def read_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F, dtype=np.int64)


# ------------------------------------------------------------------ topology
def surface_topology(V, F, merge_tol=1e-10):
    """Euler characteristic / genus of a triangle mesh.

    Coincident vertices are merged first (so a mesh that is only "closed up to
    duplicated vertices" is still judged on its true connectivity). Returns a
    dict with V, E, F, chi, components, boundary loops, manifold flags and the
    genus  g = (2 C - chi - B) / 2  (= (2 - chi)/2 for one closed component).
    """
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    key = np.round(V / merge_tol).astype(np.int64) if merge_tol else None
    if key is not None:
        _, inv = np.unique(key, axis=0, return_inverse=True)
        F = inv.ravel()[F]
    used = np.unique(F)
    remap = -np.ones(F.max() + 1, np.int64)
    remap[used] = np.arange(len(used))
    F = remap[F]
    nV = len(used)
    E_dir = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E_und = np.sort(E_dir, 1)
    cnt = Counter(map(tuple, E_und))
    nE = len(cnt)
    boundary = [e for e, c in cnt.items() if c == 1]
    nonmanifold_edges = sum(1 for c in cnt.values() if c > 2)
    # consistent orientation: every directed edge appears at most once
    dcnt = Counter(map(tuple, E_dir))
    oriented = all(c == 1 for c in dcnt.values())
    # connected components (union-find over edges)
    parent = np.arange(nV)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cnt:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    C = len({find(v) for v in range(nV)})
    # boundary loops
    B = 0
    if boundary:
        bpar = {}

        def bfind(a):
            while bpar.setdefault(a, a) != a:
                a = bpar[a]
            return a
        for a, b in boundary:
            ra, rb = bfind(a), bfind(b)
            if ra != rb:
                bpar[ra] = rb
        B = len({bfind(v) for e in boundary for v in e})
    chi = nV - nE + len(F)
    genus = (2 * C - chi - B) / 2
    # vertex manifoldness: faces around each vertex form one fan
    from collections import defaultdict
    link = defaultdict(list)
    for f in F:
        for k in range(3):
            link[f[k]].append((f[(k + 1) % 3], f[(k + 2) % 3]))
    bad_v = 0
    for v, es in link.items():
        adj = defaultdict(set)
        for a, b in es:
            adj[a].add(b)
            adj[b].add(a)
        start = next(iter(adj))
        seen, stack = {start}, [start]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != len(adj):
            bad_v += 1
    return dict(V=nV, E=nE, F=len(F), chi=int(chi), components=C, boundary_loops=B,
                nonmanifold_edges=nonmanifold_edges, nonmanifold_vertices=bad_v,
                oriented=oriented, genus=genus,
                closed_manifold=(not boundary and nonmanifold_edges == 0 and bad_v == 0))


def fmt_topology(name, t):
    return (f"{name}: V={t['V']} E={t['E']} F={t['F']}  chi=V-E+F={t['chi']}  "
            f"components={t['components']}  boundary loops={t['boundary_loops']}  "
            f"closed 2-manifold={t['closed_manifold']}  oriented={t['oriented']}  "
            f"genus={t['genus']:g}")


def export(out_dir=OUT, p=HandParams()):
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    report = {"params": asdict(p)}
    for tag, pads in (("nopads", False), ("pads", True)):
        parts = build_parts(p, pads=pads)
        U = union(parts)
        n_shells = len(U.decompose())
        V, F = manifold_to_VF(U)
        top = surface_topology(V, F)
        top["manifold3d_genus"] = int(U.genus())
        top["shells"] = n_shells
        name = "sdm_hand.obj" if pads else "sdm_hand_nopads.obj"
        write_obj(os.path.join(out_dir, name), V, F)
        report[f"surface_{tag}"] = top
        print(fmt_topology(f"unified surface ({tag})", top),
              f" [manifold3d genus()={top['manifold3d_genus']}, shells={n_shells}]")
        assert top["closed_manifold"] and top["components"] == 1 and top["genus"] == 0
        if pads:
            for name, M in parts.items():
                V, F = manifold_to_VF(M)
                write_obj(os.path.join(out_dir, "parts", f"{name}.obj"), V, F)
            report["parts"] = list(parts)
    with open(os.path.join(out_dir, "geometry_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    r = export()
    print(f"wrote {len(r['parts'])} part OBJs + unified OBJs to {OUT}")
