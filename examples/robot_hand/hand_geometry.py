"""Allegro Hand (real link meshes) + steel joint shafts + rubber fingertips + a cup.

Starts from the *actual* Wonik Allegro Hand v3 link meshes and kinematics from
MuJoCo Menagerie (see ``allegro_kinematics.py``) and turns them into
simulation-ready solids:

* every link is posed in the flat reference pose ``q = 0``;
* adjacent links interleave at the knuckles in the CAD, so each parent link has
  a Minkowski-dilated copy of its child subtracted from it -- a real 0.3 mm
  running clearance -- which keeps every link a separate closed shell that the
  joints can rotate independently;
* each rubber fingertip cap is fused onto its distal link, and split into a
  rubber shell over an aluminium core (the real tips are soft covers);
* every driven link gets a hardened-steel joint shaft: a cylinder on the
  joint axis intersected with the link itself, so it is always inside it.

Outputs (``output/``)::

    parts/<name>.obj   closed per-part meshes (winding-number material labels)
    hand.obj           all parts unioned into one surface (one shell per link)
    cup.obj            the fragile object (in the palm frame, grasp pose)
    hand_meta.json     part -> rigid body, joint axes/origins/limits
                       (``hand_grasp.py`` adds the fitted grasp pose)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import manifold3d as m3d

from allegro_kinematics import AllegroHand, quat_to_R

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "robot_gripper"))
from gripper_geometry import CupParams, build_cup, write_obj, read_obj  # noqa: E402

# vendored copy of mujoco_menagerie/wonik_allegro (BSD-2-Clause, see its LICENSE)
XML = os.path.join(HERE, "wonik_allegro", "right_hand.xml")

CLEARANCE = 3e-4          # knuckle running clearance [m]
SHAFT_RADIUS = 2.5e-3     # steel joint shaft radius [m]
TIP_CORE_SCALE = 0.62     # aluminium core of the rubber tip, relative size

# palm frame = world frame: fingers along +z, they curl towards +x (the palm
# faces +x), finger spread along y (index at +y).
BASE = np.eye(4)
BASE[:3, :3] = quat_to_R([0, 1, 0, 1]).T

FINGERS = ["ff", "mf", "rf", "th"]


def load_stl(path):
    import trimesh
    tm = trimesh.load(path)
    return np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64)


def to_manifold(V, F):
    return m3d.Manifold(m3d.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                                 tri_verts=np.ascontiguousarray(F, np.uint32)))


def manifold_VF(M):
    mesh = M.to_mesh()
    return (np.asarray(mesh.vert_properties)[:, :3].astype(float),
            np.asarray(mesh.tri_verts).astype(np.int64))


def transform(V, T):
    return V @ T[:3, :3].T + T[:3, 3]


def cylinder_along(axis, center, r, length, segments=24):
    """Manifold cylinder of radius r centred on ``center`` along ``axis``."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c = m3d.Manifold.cylinder(length, r, r, segments).translate((0, 0, -length / 2))
    z = np.array([0, 0, 1.0])
    v = np.cross(z, a)
    s, cth = np.linalg.norm(v), z @ a
    if s < 1e-9:
        R = np.eye(3) if cth > 0 else np.diag([1, -1, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
    M = np.zeros((3, 4))
    M[:, :3], M[:, 3] = R, center
    return c.transform(M)


def build_hand_parts(hand: AllegroHand):
    """Return ``(parts, meta)``; ``parts`` maps part name -> Manifold (palm frame, q=0)."""
    T = hand.fk({}, BASE)
    link = {}
    for b in hand.bodies.values():
        for mesh, pos, mat in b.meshes:
            V, F = load_stl(hand.mesh_path(mesh))
            M = to_manifold(transform(V + pos, T[b.name]), F)
            key = b.name
            link[key] = link[key] + M if key in link else M

    # knuckle clearance: parent -= dilate(child)
    ball = m3d.Manifold.sphere(CLEARANCE, 8)
    for b in hand.bodies.values():
        if b.parent is None or b.joint is None or b.parent.name not in link:
            continue   # tips have no joint: they stay bonded to the distal link
        link[b.parent.name] = link[b.parent.name] - link[b.name].minkowski_sum(ball)
    # the thumb base also cuts into the palm
    link["palm"] = link["palm"] - link["th_base"].minkowski_sum(ball)

    parts = {"palm": link["palm"]}
    meta = {"bodies": {}, "joints": {}}
    for f in FINGERS:
        for seg in ["base", "proximal", "medial", "distal"]:
            name = f"{f}_{seg}"
            parts[name] = link[name]
        # rubber tip over an aluminium core, fused onto the distal link
        tip = link[f"{f}_tip"]
        lo, hi = np.array(tip.bounding_box()[:3]), np.array(tip.bounding_box()[3:])
        c = 0.5 * (lo + hi)
        s = TIP_CORE_SCALE
        core = tip.translate(tuple(-c)).scale((s, s, s)).translate(tuple(c))
        parts[f"{f}_tip_core"] = core ^ tip
        parts[f"{f}_tip_rubber"] = tip - core

    # hardened steel joint shafts, one per driven link
    for b in hand.bodies.values():
        if b.joint is None:
            continue
        Tb = T[b.name]
        axis = Tb[:3, :3] @ b.axis
        shaft = cylinder_along(axis, Tb[:3, 3], SHAFT_RADIUS, 0.03) ^ link[b.name]
        if shaft.volume() > 0:
            parts[f"{b.name}_shaft"] = shaft
        meta["joints"][b.joint] = dict(body=b.name, origin=Tb[:3, 3].tolist(),
                                       axis=axis.tolist(), range=list(b.range))
    for b in hand.bodies.values():
        meta["bodies"][b.name] = dict(parent=b.parent.name if b.parent else None,
                                      joint=b.joint)
    return parts, meta


def rigid_body_of_part(name):
    """Kinematic body that carries a part (tips ride on the distal link)."""
    if name == "palm":
        return "palm"
    if name.endswith("_shaft"):
        return name[: -len("_shaft")]
    if "_tip" in name:
        return name.split("_")[0] + "_distal"
    return name


# --------------------------------------------------------------------------- #
# Grasp: an upright cup held in a thumb-up side grasp.                         #
# --------------------------------------------------------------------------- #
@dataclass
class CupParamsTall(CupParams):
    """A tall, thin-walled polystyrene cup sized for the (large) Allegro hand."""
    radius_bottom: float = 0.034
    radius_top: float = 0.042
    height: float = 0.130


CUP_AXIS_XZ = (0.058, 0.068)   # cup axis (along palm y) in the palm's xz-plane
CUP_Y_BOTTOM = -0.075


def cup_in_palm_frame(cp: CupParams, y_bottom=CUP_Y_BOTTOM, axis_xz=CUP_AXIS_XZ):
    """Upright cup whose axis is the palm's y axis (index finger on top)."""
    cup = build_cup(cp)                      # base on y = 0, axis +y
    return cup.translate((axis_xz[0], y_bottom, axis_xz[1]))


def export(out_dir, cp=CupParamsTall()):
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    hand = AllegroHand(XML)
    parts, meta = build_hand_parts(hand)
    for name, M in parts.items():
        write_obj(os.path.join(out_dir, "parts", f"{name}.obj"), *manifold_VF(M))
    unified = m3d.Manifold.batch_boolean(list(parts.values()), m3d.OpType.Add)
    # coincident tip/distal faces can leave zero-volume sliver shells: drop them
    unified = m3d.Manifold.compose([s for s in unified.decompose() if s.volume() > 1e-9])
    write_obj(os.path.join(out_dir, "hand.obj"), *manifold_VF(unified))
    write_obj(os.path.join(out_dir, "cup.obj"), *manifold_VF(cup_in_palm_frame(cp)))
    meta["parts"] = {n: rigid_body_of_part(n) for n in parts}
    meta["n_shells"] = len(unified.decompose())
    with open(os.path.join(out_dir, "hand_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    meta = export(os.path.join(HERE, "output"))
    print(f"{len(meta['parts'])} parts, unified hand has {meta['n_shells']} shells")
