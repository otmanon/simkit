"""Box-CSG geometry of an anthropomorphic SDM-style compliant hand.

Construction follows the SDM Hand (Dollar & Howe, "The Highly Adaptive SDM
Hand", IJRR 2010): stiff polyurethane links joined by thin, soft elastomer
flexure joints, soft fingertip pads, tendon actuation from a single actuator.
The layout is anthropomorphic (the hand shape the user approved):

* palm plate 90 (x) x 95 (z) x 22 (y) mm, wrist block 50 x 18 x 30 mm below it
* four fingers on the palm's top edge, three phalanges each (lengths in mm)
    index 40/25/20 (w 19), middle 45/28/22 (w 20), ring 42/26/21 (w 19),
    little 32/20/18 (w 16); phalanx thickness 17 mm; ~2 mm gaps
* an opposable thumb off the radial (-x) side near the wrist: metacarpal
  38 mm + proximal 30 mm + distal 24 mm (widths 24/20/19, thickness 16 mm)
* between every pair of consecutive blocks a soft flexure slab 6 mm long,
  5 mm thick (4 mm for the thumb), 4 mm narrower than the phalanx, placed at
  the DORSAL side so that palmar tendons curl the finger towards the palm
* soft pads on the palmar face of every distal phalanx, plus a palm pad

Every block is a ``manifold3d.Manifold.cube``; flexures embed 0.5 mm into the
blocks they join, so the union is ONE solid. It is checked to be a single
closed, oriented 2-manifold with Euler characteristic 2, i.e. genus 0, with
and without the pads.

Frame (metres): ``x`` across the hand (thumb at -x), ``z`` up along the
fingers, ``y`` the thickness with the PALMAR face at ``y = 0`` and the dorsal
face at ``y = 0.022``. The bottom face of the wrist (``z = -0.030``) is pinned
in the simulation.

Changes w.r.t. the quick preview script (``hand_quick.py``): the thumb is
(1) moved 4 mm further out (3.4 mm clearance) and its first flexure extended
16 mm into the palm, so that only that flexure joins it to the palm --
in the preview the metacarpal block overlapped the palm by 481 mm^3, which
would have welded the thumb's first joint solid; (2) tilted +15 deg (towards
the palmar side) instead of -15 deg (towards the back of the hand), and
pronated 50 deg about its own axis so that its pad faces the fingers and it
flexes across the palm (opposition) rather than straight forward.

    python sdm_geometry.py        # -> output/parts/*.obj, output/sdm_hand*.obj
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field

import numpy as np
import manifold3d as m3d

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
mm = 1e-3


@dataclass
class HandParams:
    palm_t: float = 22 * mm                   # palm thickness (y)
    palm_x: tuple = (-45 * mm, 45 * mm)
    palm_z: tuple = (0.0, 95 * mm)
    wrist: tuple = (-25 * mm, 25 * mm, 2 * mm, 20 * mm, -30 * mm, 0.5 * mm)
    # finger: (x centre, width, (proximal, middle, distal) lengths)
    fingers: dict = field(default_factory=lambda: {
        "index": (-32 * mm, 19 * mm, (40 * mm, 25 * mm, 20 * mm)),
        "middle": (-10.5 * mm, 20 * mm, (45 * mm, 28 * mm, 22 * mm)),
        "ring": (11 * mm, 19 * mm, (42 * mm, 26 * mm, 21 * mm)),
        "little": (31 * mm, 16 * mm, (32 * mm, 20 * mm, 18 * mm))})
    phal_t: float = 17 * mm                   # phalanx thickness (y)
    flex_len: float = 6 * mm
    flex_t: float = 5 * mm
    flex_inset: float = 2 * mm                # flexure narrower by 2 mm per side
    flex_dorsal_gap: float = 1 * mm           # flexure's dorsal face at y = palm_t - 1 mm
    overlap: float = 0.5 * mm
    pad_t: float = 8 * mm                     # fingertip / thumb pad thickness
    # thumb, built along local +z with its palmar side at local -y
    thumb_seg: tuple = ((38 * mm, 24 * mm), (30 * mm, 20 * mm), (24 * mm, 19 * mm))
    thumb_half_t: float = 8 * mm
    thumb_flex_t: float = 4 * mm
    thumb_pronation: float = 50.0             # deg about the thumb's own axis
    thumb_tilt: float = 15.0                  # deg about x (towards the palmar side)
    thumb_abduction: float = -50.0            # deg about y (outwards)
    thumb_base: tuple = (-52 * mm, 11 * mm, 18 * mm)
    thumb_root_embed: float = 16 * mm          # first thumb flexure reaches into the palm
    palm_pad: tuple = (-38 * mm, 38 * mm, -3 * mm, 0.5 * mm, 20 * mm, 85 * mm)


def box(x0, x1, y0, y1, z0, z1):
    lo = np.array([min(x0, x1), min(y0, y1), min(z0, z1)])
    hi = np.array([max(x0, x1), max(y0, y1), max(z0, z1)])
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def thumb_transform(p=HandParams()):
    """4x4 matrix local thumb frame -> hand frame (same order as the CSG ops)."""
    def rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.eye(3)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R
    R = rot(1, p.thumb_abduction) @ rot(0, p.thumb_tilt) @ rot(2, p.thumb_pronation)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = R, p.thumb_base
    return A


def build_parts(p=HandParams(), pads=True):
    """Ordered ``{name: (Manifold, kind)}``; kind in palm / link / flexure / pad."""
    parts = {}
    T = p.palm_t
    parts["wrist"] = (box(*p.wrist), "palm")
    parts["palm"] = (box(*p.palm_x, 0, T, *p.palm_z), "palm")
    ov, fl = p.overlap, p.flex_len
    y_f1 = T - p.flex_dorsal_gap
    y_p0 = (T - p.phal_t) / 2                  # palmar face of the phalanges
    for f, (xc, w, L) in p.fingers.items():
        z = p.palm_z[1]
        for i, l in enumerate(L):
            parts[f"{f}_flex{i}"] = (box(xc - w / 2 + p.flex_inset, xc + w / 2 - p.flex_inset,
                                         y_f1 - p.flex_t, y_f1, z - ov, z + fl + ov), "flexure")
            z += fl
            parts[f"{f}_ph{i}"] = (box(xc - w / 2, xc + w / 2, y_p0, y_p0 + p.phal_t, z, z + l), "link")
            z += l
        if pads:
            parts[f"{f}_pad"] = (box(xc - w / 2 + 1.5 * mm, xc + w / 2 - 1.5 * mm,
                                     y_p0 - p.pad_t, y_p0 + ov, z - l + 3 * mm, z - 2 * mm), "pad")
    # thumb in its local frame
    A = thumb_transform(p)
    ht = p.thumb_half_t
    zc = 0.0
    local = []
    for i, (l, w) in enumerate(p.thumb_seg):
        z_lo = zc - (p.thumb_root_embed if i == 0 else ov)
        local.append((f"thumb_flex{i}", box(-w / 2 + p.flex_inset, w / 2 - p.flex_inset,
                                            ht - p.thumb_flex_t, ht, z_lo, zc + fl + ov), "flexure"))
        local.append((f"thumb_ph{i}", box(-w / 2, w / 2, -ht, ht, zc + fl, zc + fl + l), "link"))
        zc += fl + l
    if pads:
        local.append(("thumb_pad", box(-8 * mm, 8 * mm, -ht - p.pad_t, -ht + ov,
                                       zc - l + 3 * mm, zc - 2 * mm), "pad"))
    for name, M, kind in local:
        parts[name] = (M.transform(A[:3, :]), kind)
    if pads:
        parts["palm_pad"] = (box(*p.palm_pad), "pad")
    return parts


def joints(p=HandParams()):
    """Every flexure joint: dict(name, below, above, frame A (4x4), z0, z1 local,
    y_palmar local, x centre local). Tendons are attached from these."""
    out = []
    y_p0 = (p.palm_t - p.phal_t) / 2
    for f, (xc, w, L) in p.fingers.items():
        z = p.palm_z[1]
        prev = "palm"
        for i, l in enumerate(L):
            A = np.eye(4)
            A[0, 3] = xc
            out.append(dict(name=f"{f}_flex{i}", below=prev, above=f"{f}_ph{i}", A=A,
                            z0=z, z1=z + p.flex_len,
                            y_below=0.0 if prev == "palm" else y_p0, y_above=y_p0))
            prev = f"{f}_ph{i}"
            z += p.flex_len + l
    A = thumb_transform(p)
    zc, prev = 0.0, "palm"
    for i, (l, w) in enumerate(p.thumb_seg):
        out.append(dict(name=f"thumb_flex{i}", below=prev, above=f"thumb_ph{i}", A=A,
                        z0=zc, z1=zc + p.flex_len, y_below=-p.thumb_half_t,
                        y_above=-p.thumb_half_t))
        prev = f"thumb_ph{i}"
        zc += p.flex_len + l
    return out


ANCHOR_OFFSET = 4 * mm      # tendon anchors sit 4 mm from the joint on each side
ANCHOR_OFFSET_PADDED = 1.5 * mm   # ... except on distal blocks (pad starts at 3 mm)


def tendon_anchors(p=HandParams()):
    """[(joint name, block below, point below, block above, point above)] -- the
    palmar-side attachment points (hand frame) of the tendon across each joint."""
    out = []
    parts = build_parts(p, pads=False)
    for jt in joints(p):
        A = jt["A"]
        w = lambda q: A[:3, :3] @ np.asarray(q) + A[:3, 3]
        off = ANCHOR_OFFSET_PADDED if jt["above"].endswith("ph2") else ANCHOR_OFFSET
        pa = w((0.0, jt["y_below"], jt["z0"] - ANCHOR_OFFSET))
        pb = w((0.0, jt["y_above"], jt["z1"] + off))
        # snap onto the surface of the block it is attached to (the thumb's
        # palm anchor would otherwise lie inside the palm)
        pa, pb = (closest_on(parts[n][0], q) for n, q in ((jt["below"], pa), (jt["above"], pb)))
        out.append((jt["name"], jt["below"], pa, jt["above"], pb))
    return out


def closest_on(M, q):
    import igl
    V, F = manifold_to_VF(M)
    _, _, c = igl.point_mesh_squared_distance(np.atleast_2d(q), V, F)
    return c[0]


def insert_points(V, F, P, snap=2e-4):
    """Insert surface points P into the triangle mesh (V, F) by 1-to-3 face or
    1-to-2 edge splits (orientation preserving; topology unchanged) so that the
    tet mesh has a vertex exactly at every tendon attachment point."""
    import igl
    V, F = np.asarray(V, float).copy(), np.asarray(F).copy()
    for p in P:
        _, fi, c = igl.point_mesh_squared_distance(np.atleast_2d(p), V, F)
        fi, c = int(fi[0]), c[0]
        tri = F[fi]
        dv = np.linalg.norm(V[tri] - c, axis=1)
        if dv.min() < snap:
            continue
        A_, B_, C_ = V[tri]
        n = np.cross(B_ - A_, C_ - A_)
        b = np.array([np.dot(np.cross(C_ - B_, c - B_), n), np.dot(np.cross(A_ - C_, c - C_), n),
                      np.dot(np.cross(B_ - A_, c - A_), n)]) / np.dot(n, n)
        m = len(V)
        V = np.vstack([V, c])
        k = int(np.argmin(b))
        if b[k] < 0.02:                              # on edge (i, j) opposite corner k
            i, j = tri[(k + 1) % 3], tri[(k + 2) % 3]
            new, drop = [], []
            for f_id, f in enumerate(F):
                for r in range(3):
                    if (f[r], f[(r + 1) % 3]) in ((i, j), (j, i)):
                        a_, b_, o = f[r], f[(r + 1) % 3], f[(r + 2) % 3]
                        new += [(a_, m, o), (m, b_, o)]
                        drop.append(f_id)
            F = np.vstack([np.delete(F, drop, 0), np.array(new)])
        else:
            a_, b_, c_ = tri
            F = np.vstack([np.delete(F, fi, 0), [(a_, b_, m), (b_, c_, m), (c_, a_, m)]])
    return V, F


def union(parts):
    return m3d.Manifold.batch_boolean([M for M, _ in parts.values()], m3d.OpType.Add)


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
    """Euler characteristic and genus of a triangle mesh.

    Coincident vertices are merged first (``merge_tol``; 0 = use indices as
    given). Returns V, E, F, chi, #components C, #boundary loops B, manifold /
    orientation flags and  genus g = (2C - chi - B)/2  (= (2 - chi)/2 for one
    closed component).
    """
    F = np.asarray(F, np.int64)
    if merge_tol:
        key = np.round(np.asarray(V, float) / merge_tol).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        F = inv.ravel()[F]
    used = np.unique(F)
    remap = -np.ones(F.max() + 1, np.int64)
    remap[used] = np.arange(len(used))
    F = remap[F]
    nV = len(used)
    E_dir = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    cnt = Counter(map(tuple, np.sort(E_dir, 1)))
    boundary = [e for e, c in cnt.items() if c == 1]
    nonmanifold_edges = sum(1 for c in cnt.values() if c > 2)
    oriented = all(c == 1 for c in Counter(map(tuple, E_dir)).values())
    parent = list(range(nV))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cnt:
        parent[find(a)] = find(b)
    C = len({find(v) for v in range(nV)})
    B = 0
    if boundary:
        bp = {}

        def bfind(a):
            while bp.setdefault(a, a) != a:
                a = bp[a]
            return a
        for a, b in boundary:
            bp[bfind(a)] = bfind(b)
        B = len({bfind(v) for e in boundary for v in e})
    chi = nV - len(cnt) + len(F)
    # vertex manifoldness: the faces around a vertex form a single fan
    link = defaultdict(lambda: defaultdict(set))
    for f in F:
        for k in range(3):
            a, b = f[(k + 1) % 3], f[(k + 2) % 3]
            link[f[k]][a].add(b)
            link[f[k]][b].add(a)
    bad_v = 0
    for v, adj in link.items():
        start = next(iter(adj))
        seen, stack = {start}, [start]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        bad_v += len(seen) != len(adj)
    return dict(V=nV, E=len(cnt), F=len(F), chi=int(chi), components=C, boundary_loops=B,
                nonmanifold_edges=nonmanifold_edges, nonmanifold_vertices=bad_v,
                oriented=oriented, genus=(2 * C - chi - B) / 2,
                closed_manifold=(not boundary and nonmanifold_edges == 0 and bad_v == 0))


def fmt_topology(name, t):
    return (f"{name}: V={t['V']} E={t['E']} F={t['F']}  chi=V-E+F={t['chi']}  "
            f"components={t['components']}  boundary loops={t['boundary_loops']}  "
            f"closed 2-manifold={t['closed_manifold']}  oriented={t['oriented']}  "
            f"genus={t['genus']:g}")


def stiff_overlaps(parts):
    """Volumes (mm^3) where two stiff blocks overlap directly (would weld a joint)."""
    stiff = [n for n, (_, k) in parts.items() if k in ("palm", "link")]
    res = {}
    for i, a in enumerate(stiff):
        for b in stiff[i + 1:]:
            v = (parts[a][0] ^ parts[b][0]).volume()
            if v > 1e-12 and {a, b} != {"palm", "wrist"}:
                res[f"{a}&{b}"] = v * 1e9
    return res


def export(out_dir=OUT, p=HandParams()):
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    for f in os.listdir(os.path.join(out_dir, "parts")):
        os.remove(os.path.join(out_dir, "parts", f))
    report = {"params": {k: v for k, v in asdict(p).items()}}
    for tag, pads in (("nopads", False), ("pads", True)):
        parts = build_parts(p, pads=pads)
        U = union(parts)
        V, F = manifold_to_VF(U)
        if pads:   # vertices at the tendon attachment points
            V, F = insert_points(V, F, [q for t in tendon_anchors(p) for q in (t[2], t[4])])
        top = surface_topology(V, F)
        top["manifold3d_genus"] = int(U.genus())
        top["shells"] = len(U.decompose())
        write_obj(os.path.join(out_dir, "sdm_hand.obj" if pads else "sdm_hand_nopads.obj"), V, F)
        report[f"surface_{tag}"] = top
        print(fmt_topology(f"unified surface ({tag})", top),
              f" [manifold3d genus()={top['manifold3d_genus']}, shells={top['shells']}]")
        assert top["closed_manifold"] and top["components"] == 1 and top["genus"] == 0
        if pads:
            ov = stiff_overlaps(parts)
            assert not ov, f"stiff blocks overlap (joint welded): {ov}"
            for name, (M, kind) in parts.items():
                write_obj(os.path.join(out_dir, "parts", f"{name}.obj"), *manifold_to_VF(M))
            report["parts"] = {n: k for n, (_, k) in parts.items()}
    with open(os.path.join(out_dir, "geometry_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    return report


if __name__ == "__main__":
    r = export()
    print(f"wrote {len(r['parts'])} part OBJs + unified OBJs to {OUT}")
