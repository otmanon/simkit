"""Build the compliant hand: box-CSG geometry, tet mesh, materials and actuation rig.

A simplified SDM-style hand (Dollar & Howe, IJRR 2010): stiff polyurethane blocks
joined by thin elastomer flexures, soft fingertip and palm pads, four fingers and
an opposable thumb carried by a tapered thenar mound. The flexures sit on the
palmar side (hinges at the palmar corners, so closing never drives the blocks
into each other); one actuator spring per joint on the dorsal side lengthens to
flex it. The union is one genus-0 solid.

Writes to ``data/sdm_hand/``:

* ``hand.obj``        boundary surface of the tet mesh (metres)
* ``hand_tets.npz``   ``X, T``, per-tet ``part`` (index into ``part_names``),
                      ``part_kind``, and materials ``E, nu, rho``
* ``hand_rig.npz``    actuator springs (anchor points on the blocks either side of
                      each joint, rest length, contraction ratio), hinge axes (two
                      points per joint), wrist-base vertices, fingertip vertices

Needs ``manifold3d``, ``tetgen``, ``pyvista`` and ``libigl``::

    python examples/sdm_hand/build_hand.py
"""
import os
from dataclasses import dataclass, field

import igl
import manifold3d as m3d
import numpy as np
import pyvista as pv
import tetgen

from simkit.filesystem import get_data_directory

OUT = os.path.join(get_data_directory(), "sdm_hand")
mm = 1e-3
# kind -> (E [Pa], nu, rho [kg/m^3])
MATERIALS = {"palm": (1.5e9, 0.35, 1150.0), "link": (1.5e9, 0.35, 1150.0),
             "flexure": (6.0e6, 0.45, 1050.0), "pad": (0.2e6, 0.45, 1030.0)}
PRIORITY = ("pad", "flexure", "link", "palm")               # label order: first match wins
TARGET_DEG = {"finger": (32.0, 34.0, 26.0), "thumb": (30.0, 26.0, 26.0)}   # flexion per joint at a = 1
ANCHOR_OFFSET = 4 * mm                                    # actuator anchors 4 mm from each joint


@dataclass
class HandParams:
    palm_t: float = 22 * mm                               # palm thickness (y); palmar face at y = 0
    palm_x: tuple = (-45 * mm, 45 * mm)
    palm_z: tuple = (0.0, 95 * mm)
    wrist: tuple = (-25 * mm, 25 * mm, 2 * mm, 20 * mm, -30 * mm, 0.5 * mm)
    # finger: (x centre, width, (proximal, middle, distal) lengths)
    fingers: dict = field(default_factory=lambda: {
        "index": (-32 * mm, 19 * mm, (40 * mm, 25 * mm, 20 * mm)),
        "middle": (-10.5 * mm, 20 * mm, (45 * mm, 28 * mm, 22 * mm)),
        "ring": (11 * mm, 19 * mm, (42 * mm, 26 * mm, 21 * mm)),
        "little": (31 * mm, 16 * mm, (32 * mm, 20 * mm, 18 * mm))})
    phal_t: float = 17 * mm
    flex_len: float = 6 * mm
    flex_t: float = 5 * mm
    flex_inset: float = 2 * mm
    overlap: float = 0.5 * mm
    pad_t: float = 8 * mm
    # thumb, built along local +z with its palmar side at local -y
    thumb_seg: tuple = ((26 * mm, 24 * mm), (30 * mm, 20 * mm), (24 * mm, 19 * mm))
    thumb_half_t: float = 8 * mm
    thumb_flex_t: float = 4 * mm
    thumb_pronation: float = 50.0
    thumb_tilt: float = 15.0
    thumb_abduction: float = -50.0
    thumb_base: tuple = (-61 * mm, 8 * mm, 25.5 * mm)
    thenar_z: tuple = (0.0, 60 * mm)
    thenar_depth: float = 12 * mm
    palm_pad: tuple = (-38 * mm, 38 * mm, -3 * mm, 0.5 * mm, 20 * mm, 85 * mm)


def box(x0, x1, y0, y1, z0, z1):
    lo = np.minimum([x0, y0, z0], [x1, y1, z1])
    hi = np.maximum([x0, y0, z0], [x1, y1, z1])
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def to_VF(M):
    mesh = M.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(float), np.asarray(mesh.tri_verts).astype(np.int64)


def thumb_frame(p):
    """4x4 transform from the thumb's local frame to the hand frame."""
    def rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.eye(3)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
        return R
    A = np.eye(4)
    A[:3, :3] = rot(1, p.thumb_abduction) @ rot(0, p.thumb_tilt) @ rot(2, p.thumb_pronation)
    A[:3, 3] = p.thumb_base
    return A


def build_parts(p):
    """Ordered ``{name: (Manifold, kind)}``, kind in palm / link / flexure / pad."""
    T, ov, fl = p.palm_t, p.overlap, p.flex_len
    A = thumb_frame(p)
    # thenar mound: hull of a slab on the palm's radial side and the thumb's base section
    plate = box(-p.thumb_seg[0][1] / 2, p.thumb_seg[0][1] / 2, -p.thumb_half_t, p.thumb_half_t, -mm, 0.0)
    slab = box(p.palm_x[0], p.palm_x[0] + p.thenar_depth, 0.0, T, *p.thenar_z)
    thenar = m3d.Manifold.batch_hull([plate.transform(A[:3, :]), slab]) ^ \
        box(-0.2, 0.2, -0.2, 0.2, -0.4, 0.0).transform(A[:3, :])
    parts = {"wrist": (box(*p.wrist), "palm"), "palm": (box(*p.palm_x, 0, T, *p.palm_z) + thenar, "palm")}
    y0 = (T - p.phal_t) / 2                                # palmar face of the phalanges
    for f, (xc, w, L) in p.fingers.items():
        z = p.palm_z[1]
        for i, l in enumerate(L):
            parts[f"{f}_flex{i}"] = (box(xc - w / 2 + p.flex_inset, xc + w / 2 - p.flex_inset,
                                         y0, y0 + p.flex_t, z - ov, z + fl + ov), "flexure")
            parts[f"{f}_ph{i}"] = (box(xc - w / 2, xc + w / 2, y0, y0 + p.phal_t, z + fl, z + fl + l), "link")
            z += fl + l
        parts[f"{f}_pad"] = (box(xc - w / 2 + 1.5 * mm, xc + w / 2 - 1.5 * mm, y0 - p.pad_t, y0 + ov,
                                 z - l + 3 * mm, z - 2 * mm), "pad")
    ht, zc = p.thumb_half_t, 0.0
    for i, (l, w) in enumerate(p.thumb_seg):
        parts[f"thumb_flex{i}"] = (box(-w / 2 + p.flex_inset, w / 2 - p.flex_inset, -ht, -ht + p.thumb_flex_t,
                                       zc - ov, zc + fl + ov).transform(A[:3, :]), "flexure")
        parts[f"thumb_ph{i}"] = (box(-w / 2, w / 2, -ht, ht, zc + fl, zc + fl + l).transform(A[:3, :]), "link")
        zc += fl + l
    parts["thumb_pad"] = (box(-8 * mm, 8 * mm, -ht - p.pad_t, -ht + ov, zc - l + 3 * mm,
                              zc - 2 * mm).transform(A[:3, :]), "pad")
    parts["palm_pad"] = (box(*p.palm_pad), "pad")
    return parts


def joints(p):
    """Per flexure joint: blocks below/above, frame A (local z along the digit, local
    x the hinge axis), local z-span of the flexure, hinge height y_mid and the local
    y of the dorsal faces the actuator is anchored to."""
    out = []
    y0 = (p.palm_t - p.phal_t) / 2
    for f, (xc, w, L) in p.fingers.items():
        A = np.eye(4)
        A[0, 3] = xc
        z, prev = p.palm_z[1], "palm"
        for i, l in enumerate(L):
            out.append(dict(name=f"{f}_flex{i}", finger=f, idx=i, below=prev, above=f"{f}_ph{i}", A=A,
                            z0=z, z1=z + p.flex_len, y_mid=y0 + p.flex_t / 2,
                            y_below=p.palm_t if prev == "palm" else y0 + p.phal_t, y_above=y0 + p.phal_t))
            prev, z = f"{f}_ph{i}", z + p.flex_len + l
    A, zc, prev = thumb_frame(p), 0.0, "palm"
    for i, (l, w) in enumerate(p.thumb_seg):
        out.append(dict(name=f"thumb_flex{i}", finger="thumb", idx=i, below=prev, above=f"thumb_ph{i}", A=A,
                        z0=zc, z1=zc + p.flex_len, y_mid=-p.thumb_half_t + p.thumb_flex_t / 2,
                        y_below=p.thumb_half_t, y_above=p.thumb_half_t))
        prev, zc = f"thumb_ph{i}", zc + p.flex_len + l
    return out


def sizing_field(V, parts, h_flex=0.0022, h_pad=0.0035, h_tip=0.0013, h_coarse=0.012, grad=0.6, spacing=0.0025):
    """Background grid with TetGen's ``target_size``: fine in the flexures and pads."""
    lo, hi = V.min(0) - 0.005, V.max(0) + 0.005
    g = pv.ImageData(dimensions=tuple(np.ceil((hi - lo) / spacing).astype(int) + 1),
                     spacing=(spacing,) * 3, origin=lo).triangulate()
    P = np.asarray(g.points)
    h = np.full(len(P), h_coarse)
    for n, (M, kind) in parts.items():
        if kind in ("flexure", "pad"):
            Vb = to_VF(M)[0]
            c = Vb.mean(0)
            Rt = np.linalg.svd(Vb - c)[2]
            half = np.abs((Vb - c) @ Rt.T).max(0)
            d = np.linalg.norm(np.maximum(np.abs((P - c) @ Rt.T) - half, 0), axis=1)
            hk = h_flex if kind == "flexure" else (h_pad if n == "palm_pad" else h_tip)
            h = np.minimum(h, hk + grad * d)
    g.point_data["target_size"] = h
    return g


def part_vertices(T, part, pid):
    """Vertices whose incident tets all belong to part ``pid`` (all its vertices if
    fewer than 4)."""
    inside = np.zeros(T.max() + 1, bool)
    inside[np.unique(T[part == pid])] = True
    touch = inside.copy()
    inside[np.unique(T[part != pid])] = False
    return np.nonzero(inside if inside.sum() >= 4 else touch)[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    p = HandParams()
    parts = build_parts(p)
    names = list(parts)
    kinds = np.array([parts[n][1] for n in names])
    V, F = to_VF(m3d.Manifold.batch_boolean([M for M, _ in parts.values()], m3d.OpType.Add))
    X, T = tetgen.TetGen(V, F.astype(np.int32)).tetrahedralize(
        order=1, quality=True, minratio=1.5, mindihedral=10, metric=True, bgmesh=sizing_field(V, parts))[:2]
    X, T = np.asarray(X, float), np.asarray(T, np.int64)
    a, b, c, d = (X[T[:, i]] for i in range(4))
    if (np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) < 0).mean() > 0.5:
        T = T[:, [0, 2, 1, 3]]
    Fb = igl.boundary_facets(T)[0]
    chi = len(np.unique(Fb)) - len(np.unique(np.sort(np.vstack([Fb[:, [0, 1]], Fb[:, [1, 2]], Fb[:, [2, 0]]]), 1),
                                             axis=0)) + len(Fb)
    assert chi == 2, "the tet boundary is not one closed genus-0 surface"

    # per-tet part by generalized winding number, pads > flexures > links > palm
    C = X[T].mean(1)
    part = np.full(len(T), -1)
    for i in sorted(range(len(names)), key=lambda i: PRIORITY.index(kinds[i])):
        Vp, Fp = to_VF(parts[names[i]][0])
        part[(part < 0) & (igl.winding_number(Vp, Fp, C) > 0.5)] = i
    assert (part >= 0).all()
    E, nu, rho = (np.array([MATERIALS[k][j] for k in kinds[part]]) for j in range(3))
    np.savez_compressed(os.path.join(OUT, "hand_tets.npz"), X=X, T=T, part=part, part_names=np.array(names),
                        part_kind=kinds, E=E, nu=nu, rho=rho)
    with open(os.path.join(OUT, "hand.obj"), "w") as f:
        used = np.unique(Fb)
        r = -np.ones(len(X), int)
        r[used] = np.arange(len(used))
        np.savetxt(f, X[used], fmt="v %.9f %.9f %.9f")
        np.savetxt(f, r[Fb] + 1, fmt="f %d %d %d")

    # rig: one dorsal actuator per joint (lengthens by r * theta at a = 1), hinge axes
    rig = {k: [] for k in ("spring_below", "spring_pa", "spring_above", "spring_pb", "l_rest", "c",
                           "hinge_below", "hinge_above", "hinge_q")}
    for jt in joints(p):
        A = jt["A"]
        world = lambda q: A[:3, :3] @ np.asarray(q) + A[:3, 3]
        pa = world((0.0, jt["y_below"], jt["z0"] - ANCHOR_OFFSET))
        pb = world((0.0, jt["y_above"], jt["z1"] + ANCHOR_OFFSET))
        pa, pb = (igl.point_mesh_squared_distance(q[None], *to_VF(parts[n][0]))[2][0]
                  for n, q in ((jt["below"], pa), (jt["above"], pb)))
        hinge = world((0.0, jt["y_mid"], 0.5 * (jt["z0"] + jt["z1"])))
        axis = A[:3, 0]
        u = (pb - pa) / np.linalg.norm(pb - pa)
        arm = abs(np.cross(u, axis) @ (hinge - pa)) / np.linalg.norm(np.cross(u, axis))
        l_rest = np.linalg.norm(pb - pa)
        theta = np.radians(TARGET_DEG["thumb" if jt["finger"] == "thumb" else "finger"][jt["idx"]])
        xs = (X[part_vertices(T, part, names.index(jt["name"]))] - hinge) @ axis
        for k, v in zip(rig, (jt["below"], pa, jt["above"], pb, l_rest, -arm * theta / l_rest,
                              jt["below"], jt["above"], [hinge + xs.min() * axis, hinge + xs.max() * axis])):
            rig[k].append(v)
    fingers = list(p.fingers) + ["thumb"]
    tips = []
    for f in fingers:                                      # far face of the distal block
        v = part_vertices(T, part, names.index(f"{f}_ph2"))
        A = [j for j in joints(p) if j["name"] == f"{f}_flex2"][0]["A"]
        s = (X[v] - A[:3, 3]) @ A[:3, 2]
        tips.append(v[s > s.max() - 1e-6])
    np.savez_compressed(os.path.join(OUT, "hand_rig.npz"), **{k: np.array(v) for k, v in rig.items()},
                        base=np.nonzero(X[:, 2] < X[:, 2].min() + 1e-7)[0], fingers=np.array(fingers),
                        tip_ids=np.concatenate(tips), tip_ptr=np.cumsum([0] + [len(t) for t in tips]))
    print(f"wrote {OUT}: {len(X)} vertices, {len(T)} tets, {len(Fb)} surface triangles")


if __name__ == "__main__":
    main()
