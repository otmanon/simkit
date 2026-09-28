"""Parametric parallel-jaw robot gripper with soft fingertip pads, plus a cup.

The gripper is modelled on commercial electric parallel grippers (Robotiq
Hand-E, Schunk EGP, OnRobot RG2-FT class): a machined aluminium housing with an
ISO 9409 tool flange, a hardened-steel linear guide rail, two aluminium jaw
carriages riding on that rail, aluminium fingers bolted to the carriages, and a
moulded silicone pad on the inside face of each finger.

Every part is built analytically with ``manifold3d`` constructive solid
geometry, so every mesh is closed, manifold and consistently oriented -- which
is what both TetGen and libigl's generalized winding number need.

Outputs (in ``examples/robot_gripper/output`` by default)::

    parts/<part>.obj     one closed mesh per part (used for winding numbers)
    gripper.obj          all gripper parts unioned into one surface
                         (three shells: housing+rail, left jaw, right jaw --
                         the jaws slide on the rail so they are not fused to it)
    cup.obj              the fragile object

Units are metres. ``+y`` is up, ``x`` is the closing axis, ``z`` is depth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
import manifold3d as m3d


@dataclass
class GripperParams:
    # housing (aluminium 6061-T6)
    housing_half_width: float = 0.075
    housing_y: tuple = (0.105, 0.145)
    housing_half_depth: float = 0.025
    flange_radius: float = 0.0315          # ISO 9409-1-50-4-M6
    flange_height: float = 0.010
    # guide rail (hardened stainless steel)
    rail_half_width: float = 0.070
    rail_y: tuple = (0.098, 0.1055)
    rail_half_depth: float = 0.008
    # jaw carriage (aluminium, wraps the rail with 0.5 mm clearance)
    carriage_half_width: float = 0.012
    carriage_y: tuple = (0.086, 0.0975)
    carriage_half_depth: float = 0.018
    clearance: float = 0.0005
    # finger (aluminium 7075-T6)
    finger_thickness: float = 0.010
    finger_y: tuple = (0.018, 0.0865)
    finger_half_depth: float = 0.012
    # silicone fingertip pad
    pad_thickness: float = 0.008
    pad_y: tuple = (0.026, 0.080)
    pad_half_depth: float = 0.011
    pad_round: float = 0.0015
    # jaw opening: |x| of the pad contact face when fully open
    pad_face_open: float = 0.040


@dataclass
class CupParams:
    # thin-walled disposable polystyrene cup (brittle, fragile)
    radius_bottom: float = 0.031
    radius_top: float = 0.034
    height: float = 0.085
    wall: float = 0.0025
    base: float = 0.003
    rim_radius: float = 0.0022
    segments: int = 48


def _box(x0, x1, y0, y1, z0, z1):
    lo = np.array([min(x0, x1), min(y0, y1), min(z0, z1)])
    hi = np.array([max(x0, x1), max(y0, y1), max(z0, z1)])
    return m3d.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def _rounded_box(x0, x1, y0, y1, z0, z1, r, segments=12):
    lo = np.array([min(x0, x1), min(y0, y1), min(z0, z1)]) + r
    hi = np.array([max(x0, x1), max(y0, y1), max(z0, z1)]) - r
    s = m3d.Manifold.sphere(r, segments)
    corners = [s.translate((x, y, z)) for x in (lo[0], hi[0])
               for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return m3d.Manifold.batch_hull(corners)


def _cyl_y(r, y0, y1, segments=48):
    # manifold cylinders are along +z; rotate so the axis is +y
    return (m3d.Manifold.cylinder(y1 - y0, r, r, segments)
            .rotate((-90.0, 0.0, 0.0)).translate((0.0, y0, 0.0)))


def build_gripper_parts(p: GripperParams = GripperParams()):
    """Return ``{part_name: manifold3d.Manifold}`` for the gripper in its open pose."""
    parts = {}
    hw, hd = p.housing_half_width, p.housing_half_depth
    housing = _rounded_box(-hw, hw, *p.housing_y, -hd, hd, 0.003)
    housing = housing + _cyl_y(p.flange_radius, p.housing_y[1] - 0.001,
                               p.housing_y[1] + p.flange_height)
    # pilot recess on the flange face and a cable boss on the side, for realism
    housing = housing - _cyl_y(0.0125, p.housing_y[1] + p.flange_height - 0.003,
                               p.housing_y[1] + p.flange_height + 0.001)
    parts["housing"] = housing

    rw, rd = p.rail_half_width, p.rail_half_depth
    parts["rail"] = _box(-rw, rw, p.rail_y[0], p.rail_y[1], -rd, rd)

    for side, s in (("left", -1.0), ("right", 1.0)):
        face = p.pad_face_open
        pad_x0, pad_x1 = face, face + p.pad_thickness
        fin_x0, fin_x1 = pad_x1, pad_x1 + p.finger_thickness
        cx = 0.5 * (fin_x0 + fin_x1) + 0.004
        cw, cd, c = p.carriage_half_width, p.carriage_half_depth, p.clearance

        # carriage: a block under the rail with two cheeks hugging its sides
        carriage = _box(cx - cw, cx + cw, p.carriage_y[0], p.rail_y[0] - c, -cd, cd)
        for zs in (-1.0, 1.0):
            carriage = carriage + _box(cx - cw, cx + cw, p.rail_y[0] - c - 0.0005,
                                       p.rail_y[1] - 0.002,
                                       zs * (rd + c), zs * cd)

        # finger: plate with a tapered tip and an outboard stiffening rib
        fd = p.finger_half_depth
        finger = _box(fin_x0, fin_x1, p.finger_y[0] + 0.012, p.finger_y[1], -fd, fd)
        tip = m3d.Manifold.batch_hull([
            _box(fin_x0, fin_x1, p.finger_y[0] + 0.012, p.finger_y[0] + 0.0125, -fd, fd),
            _box(fin_x0, fin_x0 + 0.6 * p.finger_thickness,
                 p.finger_y[0], p.finger_y[0] + 0.0005, -0.8 * fd, 0.8 * fd),
        ])
        rib = m3d.Manifold.batch_hull([
            _box(fin_x1 - 0.0005, fin_x1 + 0.006, p.finger_y[1] - 0.001,
                 p.finger_y[1], -0.003, 0.003),
            _box(fin_x1 - 0.0005, fin_x1, p.finger_y[1] - 0.030,
                 p.finger_y[1] - 0.0295, -0.003, 0.003),
        ])
        finger = finger + tip + rib

        pd = p.pad_half_depth
        pad = _rounded_box(pad_x0, pad_x1 + 0.0005, *p.pad_y, -pd, pd, p.pad_round, 8)

        mirror = (lambda M: M.mirror((1.0, 0.0, 0.0))) if s < 0 else (lambda M: M)
        parts[f"carriage_{side}"] = mirror(carriage)
        parts[f"finger_{side}"] = mirror(finger)
        parts[f"pad_{side}"] = mirror(pad)
    return parts


def build_cup(c: CupParams = CupParams()):
    """Tapered thin-walled cup with a rolled rim bead, base on ``y = 0``."""
    rb, rt, H, t = c.radius_bottom, c.radius_top, c.height, c.wall
    n = c.segments
    outer = m3d.Manifold.cylinder(H, rb, rt, n)
    slope = (rt - rb) / H
    inner = m3d.Manifold.cylinder(H, rb - t + slope * c.base, rt - t + 1e-4, n) \
        .translate((0, 0, c.base))
    cup = outer - inner.translate((0, 0, 1e-4))
    # rolled rim bead: revolve a small circle
    circ = m3d.CrossSection.circle(c.rim_radius, 16).translate((rt - 0.5 * t, 0.0))
    bead = m3d.Manifold.revolve(circ, n).translate((0, 0, H - c.rim_radius * 0.6))
    cup = cup + bead
    return cup.rotate((-90.0, 0.0, 0.0))


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


def export(out_dir, gp=GripperParams(), cp=CupParams()):
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    parts = build_gripper_parts(gp)
    for name, M in parts.items():
        V, F = manifold_to_VF(M)
        write_obj(os.path.join(out_dir, "parts", f"{name}.obj"), V, F)
    unified = m3d.Manifold.batch_boolean(list(parts.values()), m3d.OpType.Add)
    unified = unified.simplify(2e-4)   # collapse sliver edges left by the union
    V, F = manifold_to_VF(unified)
    write_obj(os.path.join(out_dir, "gripper.obj"), V, F)
    n_shells = len(unified.decompose())
    V, F = manifold_to_VF(build_cup(cp))
    write_obj(os.path.join(out_dir, "cup.obj"), V, F)
    return dict(parts=list(parts), shells=n_shells,
                gripper_params=asdict(gp), cup_params=asdict(cp))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    info = export(os.path.abspath(ap.parse_args().out))
    print(f"wrote {len(info['parts'])} parts, unified gripper has {info['shells']} shells")
