"""Minimal forward kinematics for the Wonik Allegro Hand MJCF (MuJoCo Menagerie).

Parses ``right_hand.xml`` (body tree, ``pos``/``quat`` offsets, hinge-joint
axes and ranges, visual mesh geoms) with the standard library only, so the
real Allegro link meshes can be posed at any joint configuration without
MuJoCo installed.

Menagerie: https://github.com/google-deepmind/mujoco_menagerie/tree/main/wonik_allegro
(BSD-2-Clause, derived from SimLab's allegro_hand_ros URDF).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_R(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _vec(s, default):
    return np.array([float(v) for v in s.split()]) if s is not None else np.array(default, float)


@dataclass
class Body:
    name: str
    parent: "Body | None"
    pos: np.ndarray
    R: np.ndarray
    joint: str | None = None
    axis: np.ndarray | None = None
    range: tuple | None = None
    meshes: list = field(default_factory=list)      # [(mesh_name, pos (3,), material)]
    children: list = field(default_factory=list)


class AllegroHand:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.asset_dir = os.path.join(os.path.dirname(xml_path), "assets")
        root = ET.parse(xml_path).getroot()
        self.defaults = {}
        self._parse_defaults(root.find("default"), {})
        self.bodies: dict[str, Body] = {}
        self.joints: list[str] = []
        wb = root.find("worldbody")
        for b in wb.findall("body"):
            self._parse_body(b, None, "allegro_right")

    # MJCF default classes are nested; each class inherits its parent's attrs.
    def _parse_defaults(self, node, inherited):
        for d in node.findall("default"):
            cls = d.get("class")
            attrs = {k: dict(v) for k, v in inherited.items()}
            for child in d:
                if child.tag != "default":
                    attrs.setdefault(child.tag, {}).update(child.attrib)
            self.defaults[cls] = attrs
            self._parse_defaults(d, attrs)

    def _attrs(self, el, cls):
        cls = el.get("class", cls)
        a = dict(self.defaults.get(cls, {}).get(el.tag, {}))
        a.update(el.attrib)
        return a

    def _parse_body(self, el, parent, cls):
        cls = el.get("childclass", cls)
        pos = _vec(el.get("pos"), [0, 0, 0])
        R = quat_to_R(_vec(el.get("quat"), [1, 0, 0, 0]))
        b = Body(el.get("name"), parent, pos, R)
        j = el.find("joint")
        if j is not None:
            ja = self._attrs(j, cls)
            b.joint = ja["name"]
            b.axis = _vec(ja.get("axis"), [0, 0, 1])
            b.range = tuple(_vec(ja.get("range"), [-np.pi, np.pi]))
            self.joints.append(b.joint)
        for g in el.findall("geom"):
            ga = self._attrs(g, cls)
            if ga.get("type") == "mesh" and "mesh" in ga:
                b.meshes.append((ga["mesh"], _vec(ga.get("pos"), [0, 0, 0]),
                                 ga.get("material", "black")))
        self.bodies[b.name] = b
        if parent is not None:
            parent.children.append(b)
        for c in el.findall("body"):
            self._parse_body(c, b, cls)

    def fk(self, q: dict | None = None, base=np.eye(4)):
        """World 4x4 transform of every body for joint angles ``q`` (by name)."""
        q = q or {}
        out = {}

        def rec(b, Tp):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = b.R, b.pos
            if b.joint is not None:
                J = np.eye(4)
                J[:3, :3] = axis_angle_R(b.axis, q.get(b.joint, 0.0))
                T = T @ J
            Tw = Tp @ T
            out[b.name] = Tw
            for c in b.children:
                rec(c, Tw)

        for b in self.bodies.values():
            if b.parent is None:
                rec(b, base)
        return out

    def mesh_path(self, name):
        return os.path.join(self.asset_dir, f"{name}.stl")
