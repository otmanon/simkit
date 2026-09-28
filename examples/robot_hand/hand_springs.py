"""Mass-spring joints and actuators for the Allegro hand (SimKit mass springs).

The 17 link shells are separate tet meshes. They are tied into one mechanism
with SimKit's mass-spring energy (``0.5 k (|x_i - x_j| - l0)^2`` per spring):

* **Hinge springs** -- at every joint, ``n_stations`` points are spread along
  the joint axis (over +-``half_span``); at each, the child vertex nearest the
  point is tied to the ``k_nn`` parent vertices nearest it. The anchors are
  (nearly) collinear with the axis, so rotation about it is (nearly) free while
  every other relative motion of the two links stretches springs: a compliant
  line hinge. (Anchoring at a few clustered near-axis vertices instead leaves
  tiny links free to pivot about a point -- spurious near-mechanism modes.)
* **Actuator springs** -- an agonist/antagonist pair per joint, crossing the
  joint on opposite sides of the axis at a moment arm ``arm`` (a motor as a
  pair of antagonistic linear actuators). Actuation changes their rest lengths:
  ``l0(q)`` is the distance between the two attachment vertices when every link
  is posed by forward kinematics at joint angles ``q``, so the springs' zero-
  energy configuration is the hand at ``q``.

Only the palm and the wrist flange are pinned (the robot flange). Everything
else -- phalanges, steel shafts, silicone pads, rubber tips -- is free.
"""
from __future__ import annotations

import json
import os

import numpy as np
import scipy as sp
from scipy.spatial import cKDTree

import simkit
import simkit.energies as energies

from allegro_kinematics import AllegroHand
from hand_geometry import XML
from hand_grasp import body_transforms, vertex_bodies
from materials import lame

HERE = os.path.dirname(os.path.abspath(__file__))


class HandSprings:
    def __init__(self, scene, meta, n_stations=5, half_span=6e-3, k_nn=4, arm=9e-3,
                 span=8e-3, k_hinge=2e4, k_act=5e4, k_off_axis=50.0):
        self.X = scene["X"].astype(float)
        self.body_names = [str(b) for b in scene["body_names"]]
        self.vb = vertex_bodies(scene)
        self.hand = AllegroHand(XML)
        X, vb, bn = self.X, self.vb, self.body_names

        hinge, hinge_k, act, act_joint = [], [], [], []
        for jname, J in meta["joints"].items():
            o, a = np.array(J["origin"]), np.array(J["axis"], float)
            a /= np.linalg.norm(a)
            child, parent = J["body"], meta["bodies"][J["body"]]["parent"]
            ci = np.where(vb == bn.index(child))[0]
            pi = np.where(vb == bn.index(parent))[0]

            c_tree, p_tree = cKDTree(X[ci]), cKDTree(X[pi])
            pairs = []
            for t in np.linspace(-half_span, half_span, n_stations):
                station = o + t * a
                c = ci[c_tree.query(station)[1]]
                for j in np.atleast_1d(p_tree.query(station, k=k_nn)[1]):
                    pairs.append((c, pi[j]))
            pairs = list(dict.fromkeys(pairs))
            # Off-axis (wobble) stiffness of a line hinge ~ sum_k k * t_k^2 over
            # the anchors' axial offsets t_k: a short child link (anchors close
            # together) needs stiffer springs, or it pivots about a point.
            tk = (X[[c for c, _ in pairs]] - o) @ a
            spread2 = max(((tk - tk.mean()) ** 2).sum(), 1e-12)
            hinge += pairs
            hinge_k += [max(k_hinge, k_off_axis / spread2)] * len(pairs)

            # actuator pair: w points from the joint into the child, d is the
            # lever direction (perpendicular to the axis and to w)
            w = X[ci].mean(0) - o
            w -= (w @ a) * a
            if np.linalg.norm(w) < 2e-3:
                w = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
            w /= np.linalg.norm(w)
            d = np.cross(a, w)
            for sgn in (1.0, -1.0):
                pc = o + span * w + sgn * arm * d
                pp = o - span * w + sgn * arm * d
                vc = ci[np.argmin(np.linalg.norm(X[ci] - pc, axis=1))]
                vp = pi[np.argmin(np.linalg.norm(X[pi] - pp, axis=1))]
                act.append((vc, vp))
                act_joint.append(jname)

        self.E_hinge = np.array(hinge, np.int64)
        self.E_act = np.array(act, np.int64)
        self.act_joint = np.array(act_joint)
        self.E = np.vstack([self.E_hinge, self.E_act])
        self.k = np.r_[np.array(hinge_k), np.full(len(self.E_act), k_act)]
        self.ym = self.k.reshape(-1, 1)
        self.vol = np.ones((len(self.E), 1))
        self.l0_rest = np.linalg.norm(X[self.E[:, 0]] - X[self.E[:, 1]], axis=1)

    def rest_lengths(self, q):
        """Hinge springs keep their rest length; actuator springs take the length
        their attachment points have when the links are posed at ``q``."""
        A = body_transforms(self.hand, self.body_names, q)
        P = np.einsum("nij,nj->ni", A[self.vb, :3, :3], self.X) + A[self.vb, :3, 3]
        l0 = self.l0_rest.copy()
        na = len(self.E_hinge)
        l0[na:] = np.linalg.norm(P[self.E_act[:, 0]] - P[self.E_act[:, 1]], axis=1)
        return l0

    # SimKit mass-spring energy on the full vertex set
    def energy(self, x, l0):
        return energies.mass_springs_energy_x(x.reshape(-1, 3), self.E, self.ym, self.vol,
                                              l0.reshape(-1, 1))

    def gradient(self, x, l0):
        return energies.mass_springs_gradient_x(x.reshape(-1, 3), self.E, self.ym, self.vol,
                                                l0.reshape(-1, 1)).ravel()

    def hessian(self, x, l0, psd=True):
        return energies.mass_springs_hessian_x(x.reshape(-1, 3), self.E, self.ym, self.vol,
                                               l0.reshape(-1, 1), psd=psd)


def hand_operator(scene, meta, springs, gamma=1e11, keep=None):
    """Rest-state operator of the actuated hand, restricted to vertex set ``keep``.

    ``A = H_elastic + H_springs + Q_pin`` (interleaved DOFs), with the palm and
    wrist pinned by a Dirichlet penalty, and the lumped mass matrix ``M``.
    Returns ``A, M, X, T, part`` on the kept vertices.
    """
    names = [str(n) for n in scene["part_names"]]
    cup = names.index("cup")
    if keep is None:
        keep = np.unique(scene["T"][scene["part"] != cup])
    remap = -np.ones(len(scene["X"]), np.int64)
    remap[keep] = np.arange(len(keep))
    tmask = scene["part"] != cup
    T = remap[scene["T"][tmask]]
    X = scene["X"][keep]
    part = scene["part"][tmask]
    mu, lam = lame(scene["E"][tmask], scene["nu"][tmask])
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    H_el = energies.macklin_mueller_neo_hookean_hessian_x(X, J, mu.reshape(-1, 1),
                                                          lam.reshape(-1, 1), vol, psd=False)
    E = remap[springs.E]
    ok = (E >= 0).all(1)
    H_sp = energies.mass_springs_hessian_x(X, E[ok], springs.ym[ok], springs.vol[ok],
                                           springs.l0_rest[ok].reshape(-1, 1), psd=True)
    pinned_parts = [i for i, n in enumerate(names) if n in ("palm", "wrist")]
    bI = np.unique(T[np.isin(part, pinned_parts)])
    Q, _ = simkit.dirichlet_penalty(bI, X[bI], X.shape[0], gamma)
    A = (H_el + H_sp + Q).tocsc()
    M = sp.sparse.kron(simkit.massmatrix(X, T, rho=scene["rho"][tmask].reshape(-1, 1)),
                       sp.sparse.identity(3)).tocsc()
    return A, M, X, T, part, dict(H_el=H_el, H_sp=H_sp, Q=Q, mu=mu, lam=lam, J=J, vol=vol)


if __name__ == "__main__":
    out = os.path.join(HERE, "output")
    scene = dict(np.load(os.path.join(out, "scene_tets.npz")))
    meta = json.load(open(os.path.join(out, "hand_meta.json")))
    s = HandSprings(scene, meta)
    print(f"{len(s.E_hinge)} hinge springs, {len(s.E_act)} actuator springs")
    l0g = s.rest_lengths(meta["grasp_q"])
    na = len(s.E_hinge)
    for j in dict.fromkeys(s.act_joint):
        m = np.where(s.act_joint == j)[0] + na
        print(f"  {j}: actuator rest lengths {np.round(s.l0_rest[m]*1e3,1)} mm -> "
              f"{np.round(l0g[m]*1e3,1)} mm at the grasp")
