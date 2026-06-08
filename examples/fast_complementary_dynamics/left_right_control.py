"""Fast Complementary Dynamics: left/right rig-driven beam demo.

Reproduces a minimal slice of "Fast Complementary Dynamics via Skinning
Eigenmodes" (Benchekroun et al., SIGGRAPH 2023). A 2D beam is driven by an
animator rig (a single global affine handle) and we sweep that handle
left/right.

Two worlds are exercised side-by-side:

- ``CoDyLeftRightWorld`` uses Fast Complementary Dynamics: the simulation
  subspace ``B`` is built ORTHOGONAL to the rig ``J`` (momentum-leaking
  constraint), so the secondary motion ``B @ z`` is purely complementary to
  the rig motion ``J @ p``.
- ``PinLeftRightWorld`` uses Dirichlet pins as a baseline (a small center
  region is hard-pinned and swept left/right).

Everything is built from flat simkit functions -- there is no library class.
The Fast-CoDy local/global solve is a couple of small closures driven by
``simkit.solvers.block_coord``; the precomputed operators live in plain dicts.

Run from the repository root with the ``viz`` and ``video`` extras installed::

    python examples/fast_complementary_dynamics/left_right_control.py

Resulting ``.mp4`` and ``.gif`` files are written to
``examples/fast_complementary_dynamics/results/`` (gitignored).
"""

import os
import shutil
from pathlib import Path

import igl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sps

from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.common_selections import center_indices
from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.diffuse_field import diffuse_field
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.lbs_jacobian import lbs_jacobian
from simkit.lbs_weight_space_constraint import lbs_weight_space_constraint
from simkit.massmatrix import massmatrix
from simkit.matplotlib.Frame import Frame
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.TriangleMesh import TriangleMesh, light_red
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
from simkit.polar_svd import polar_svd
from simkit.project_into_subspace import project_into_subspace
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.solvers import block_coord
from simkit.spectral_cubature import spectral_cubature
from simkit.volume import volume
from simkit.ympr_to_lame import ympr_to_lame


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# =============================================================================
# Fast Complementary Dynamics core (flat functions)
#
# All of the math that used to live in ``FastCoDySim`` is here as a builder
# (``cody_precompute``) that returns a dict of precomputed operators, plus a
# single timestep (``cody_step``) made of two closures handed to
# ``block_coord``:
#
#   local step  : per-cluster best-fit rotation via polar SVD
#   global step : one Cholesky back-solve in the reduced subspace ``B``
#
# The reduced unknown is ``z`` (subspace coords); the full deformation is
# ``x = J @ p + B @ z`` where ``J @ p`` is the rig and ``B @ z`` is the
# complementary secondary motion.
# =============================================================================

class CodySim:
    """A small *local* Fast-CoDy simulator (state + precomputed operators).

    Defining a tiny class inside an example is fine -- the no-class rule is
    about the simkit *library*, not example scripts. The numerics are entirely
    flat functions; this just bundles the operators and per-step state.
    """

    def __init__(self, X, T, J, B, labels, ym=1e5, rho=1e3, h=1e-2,
                 max_iter=10, Q0=None):
        self.X, self.T = X, T
        self.dim = X.shape[1]
        self.J, self.B = J, B
        self.h = float(h)
        self.max_iter = int(max_iter)
        self.pre = cody_precompute(X, T, J, B, labels, ym=ym, rho=rho, h=h,
                                   Q0=Q0)

    def rest_state(self):
        z = np.zeros((self.B.shape[1], 1))
        z_dot = np.zeros_like(z)
        p = self.pre["p0"].copy()
        p_dot = np.zeros_like(p)
        p_next = p.copy()
        return z, p, z_dot, p_dot, p_next

    def step(self, z, p, z_dot, p_dot, p_next, b_ext=None):
        return cody_step(self.pre, z, z_dot, p, p_dot, p_next,
                         h=self.h, max_iter=self.max_iter, b_ext=b_ext)


def cody_precompute(X, T, J, B, labels, ym=1e5, rho=1e3, h=1e-2, Q0=None):
    """Precompute all Fast-CoDy operators for a fixed mesh / rig / subspace."""
    n, dim = X.shape
    Mv = sps.kron(massmatrix(X, T, rho=rho), sps.identity(dim)).tocsc()
    G = deformation_jacobian(X, T)
    mu, _lam = ympr_to_lame(ym, 0)
    vol = volume(X, T)
    AMu = sps.diags((mu * vol).flatten())
    AMue = sps.kron(AMu, sps.identity(dim * dim))
    L = (G.T @ AMue @ G).tocsc()

    # clustered cubature: one rotation per cluster instead of one per element
    P, _Pm = cluster_grouping_matrices(np.asarray(labels).flatten(), X, T)
    PAMue = sps.kron(P @ AMu, sps.identity(dim * dim))

    AMuPGB = (PAMue @ G) @ B
    AMuPGJ = (PAMue @ G) @ J
    BLB = B.T @ L @ B
    BMB = B.T @ Mv @ B
    BMJ = B.T @ Mv @ J
    BLJ = B.T @ L @ J

    Q = np.zeros((B.shape[1], B.shape[1])) if Q0 is None else Q0
    Hsys = BLB + BMB / h ** 2 + Q
    chol = sp.linalg.cho_factor(Hsys)

    JMJ = J.T @ Mv @ J
    JMy = J.T @ Mv @ X.reshape(-1, 1)
    p0 = project_into_subspace(X.reshape(-1, 1), J, M=Mv, BMB=JMJ, BMy=JMy)

    return dict(dim=dim, AMuPGB=AMuPGB, AMuPGJ=AMuPGJ, BMB=BMB, BMJ=BMJ,
                BLJ=BLJ, chol=chol, p0=p0)


def cody_step(pre, z, z_dot, p, p_dot, p_next, h=1e-2, max_iter=10, b_ext=None):
    """One Fast-CoDy timestep: local/global block-coordinate descent on ``z``."""
    dim = pre["dim"]
    AMuPGB, AMuPGJ = pre["AMuPGB"], pre["AMuPGJ"]
    BMB, BMJ, BLJ, chol = pre["BMB"], pre["BMJ"], pre["BLJ"], pre["chol"]

    y = z + h * z_dot                      # inertial target (subspace coords)
    q = p_next - (p + h * p_dot)           # rig acceleration term
    b = np.zeros((z.shape[0], 1)) if b_ext is None else b_ext

    def local(zv):
        # best-fit rotation per cluster from the current deformation gradient
        c = AMuPGB @ zv + AMuPGJ @ p_next
        return polar_svd(c.reshape(-1, dim, dim))[0].reshape(-1, 1)

    def glob(zv, r):
        rhs = (BMB @ y / h ** 2 - BMJ @ q / h ** 2
               + AMuPGB.T @ r - BLJ @ p_next - b)
        return sp.linalg.cho_solve(chol, rhs)

    return block_coord(z.copy(), glob, local, tolerance=0.0, max_iter=max_iter)


def cody_subspace(X, T, num_modes=10, num_clusters=20):
    """Rig + rig-orthogonal skinning-eigenmode subspace + cubature labels.

    The orthogonality is the heart of complementary dynamics: we constrain the
    skinning eigenmodes against a *momentum-leaking* operator so that the
    simulation subspace ``B`` carries no low-frequency rig motion.
    """
    n, dim = X.shape
    M = massmatrix(X, T)
    Mv = sps.kron(M, sps.identity(dim))

    # rig: single global affine handle (linear blend skinning Jacobian)
    J = lbs_jacobian(X, np.ones((n, 1)))

    # momentum-leaking weights: 1 in the interior, falling to 0 at the boundary
    bI = np.unique(igl.boundary_facets(T)[0])
    d = diffuse_field(X, T, bI, np.ones((bI.shape[0], 1)), dt=1, normalize=True)
    De = sps.kron(sps.diags((1 - d).flatten()), sps.identity(dim))

    # orthogonality (momentum-leaking) constraint -> weight-space constraint
    O = De @ Mv @ J
    Aeq = lbs_weight_space_constraint(X, T, O.T)

    # rig-orthogonal skinning eigenmodes + clustered cubature labels
    W, _E, B = skinning_eigenmodes(X, T, num_modes, Aeq=Aeq)
    B = orthonormalize(B, M=Mv)
    _cI, _cW, labels = spectral_cubature(X, T, W, num_clusters,
                                         return_labels=True)
    return J, B, labels


class PinLeftRightWorld():
    def __init__(self, X, T, bI=None):
        dim = X.shape[1]
        M = massmatrix(X, T)
        Mv = sps.kron(M, sps.identity(dim))

        # rig space: identity (positions themselves), pins drive the motion
        J = X.reshape(-1, 1)

        # simulation subspace: plain skinning eigenmodes (no rig orthogonality)
        W, _E, B = skinning_eigenmodes(X, T, 10)
        B = orthonormalize(B, M=Mv)
        _cI, _cW, l = spectral_cubature(X, T, W, 20, return_labels=True)

        if bI is None:
            bI = center_indices(X, 0.1)[1]
        self.bI = bI
        bc = np.zeros((self.bI.shape[0], 2))

        self.gamma = 1e12
        Q0, b0 = dirichlet_penalty(self.bI, bc, X.shape[0], self.gamma)
        BQB = B.T @ Q0 @ B

        self.sim = CodySim(X, T, J, B, l, ym=1e5, rho=1e3, max_iter=10, Q0=BQB)
        self.X = X
        self.T = T
        self.J = J
        self.B = B

    def simulate_periodic_x(self, num_timesteps=500, period=100, a=1.0):
        [z, p, z_dot, p_dot, p_next] = self.sim.rest_state()
        p_next = p.copy()

        Zs = np.zeros((z.shape[0], num_timesteps + 1))
        Ps = np.zeros((p.shape[0], num_timesteps + 1))
        Zs[:, 0] = z.flatten()
        Ps[:, 0] = p.flatten()

        for i in range(num_timesteps):
            bc = np.zeros((self.bI.shape[0], 2))
            bc[:, 0] = a * np.sin(2 * np.pi * i / period)
            b = dirichlet_penalty(self.bI, bc, self.X.shape[0], self.gamma,
                                  only_b=True)[0]

            z_next = self.sim.step(z, p, z_dot, p_dot, p_next,
                                   b_ext=self.B.T @ b)

            z_dot = (z_next - z) / self.sim.h
            p_dot = (p_next - p) / self.sim.h
            z = z_next.copy()

            Zs[:, i + 1] = z.flatten()
            Ps[:, i + 1] = p.flatten()

        return Zs, Ps

    def render(self, Zs, Ps, path=None, save_tmp=False):
        Xs = self.B @ Zs + self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        purple = np.array([201, 148, 199]) / 255
        mesh = TriangleMesh(X, T, linewidths=1, outlinewidth=1,
                            facecolors=purple, edgecolors=purple)
        pc = PointCloud(X[self.bI, :], color=[0.75, 0.75, 0.75, 1], size=50,
                        linewidth=0.5)

        if path is not None:
            dir = os.path.join(os.path.dirname(path), Path(path).stem)
            os.makedirs(dir, exist_ok=True)

        assert (Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)
            mesh.update_vertex_positions(X)
            pc.update_vertex_positions(X[self.bI, :])
            plt.pause(0.0001)
            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))

        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp and path is not None:
            shutil.rmtree(dir)


class CoDyLeftRightWorld():
    def __init__(self, X, T):
        J, B, l = cody_subspace(X, T, num_modes=10, num_clusters=20)
        self.sim = CodySim(X, T, J, B, l, ym=1e5, rho=1e3, max_iter=10)
        self.X = X
        self.T = T
        self.J = J
        self.B = B

    def simulate_periodic_x(self, num_timesteps=500, period=100, a=1.0):
        [z, p, z_dot, p_dot, p_next] = self.sim.rest_state()
        p_next = p.copy()

        Zs = np.zeros((z.shape[0], num_timesteps + 1))
        Ps = np.zeros((p.shape[0], num_timesteps + 1))
        Zs[:, 0] = z.flatten()
        Ps[:, 0] = p.flatten()

        for i in range(num_timesteps):
            p_next[-2] = a * np.sin(2 * np.pi * i / period)
            z_next = self.sim.step(z, p, z_dot, p_dot, p_next)

            z_dot = (z_next - z) / self.sim.h
            p_dot = (p_next - p) / self.sim.h
            p = p_next.copy()
            z = z_next.copy()

            Zs[:, i + 1] = z.flatten()
            Ps[:, i + 1] = p.flatten()

        return Zs, Ps

    def render(self, Zs, Ps, path=None, save_tmp=False):
        Xs = self.B @ Zs + self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        P = Ps[:, 0].reshape(3, 2).T
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        mesh = TriangleMesh(X, T, linewidths=1, outlinewidth=1)
        frame = Frame(P)

        if path is not None:
            dir = os.path.join(os.path.dirname(path), Path(path).stem)
            os.makedirs(dir, exist_ok=True)

        assert (Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)
            P = Ps[:, i].reshape(3, 2).T
            mesh.update_vertex_positions(X)
            frame.update_frame(P)
            plt.pause(0.0001)
            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))

        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp and path is not None:
            shutil.rmtree(dir)

    def render_rig(self, Zs, Ps, path=None, save_tmp=False):
        Xs = self.J @ Ps
        T = self.T
        X = Xs[:, 0].reshape(-1, 2)
        P = Ps[:, 0].reshape(3, 2).T
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.ion()
        plt.clf()
        plt.axis('off')
        plt.axis('equal')
        plt.axis('tight')
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.margins(x=0, y=0)
        mesh = TriangleMesh(X, T, facecolors=light_red, edgecolors=light_red,
                            linewidths=1, outlinewidth=1)
        frame = Frame(P)
        if path is not None:
            dir = os.path.join(os.path.dirname(path), Path(path).stem)
            os.makedirs(dir, exist_ok=True)

        assert (Zs.shape[1] == Ps.shape[1])

        for i in range(Zs.shape[1]):
            X = Xs[:, i].reshape(-1, 2)
            P = Ps[:, i].reshape(3, 2).T
            mesh.update_vertex_positions(X)
            frame.update_frame(P)
            plt.pause(0.0001)
            if path is not None:
                plt.savefig(os.path.join(dir, f"{i:04d}.png"))

        plt.clf()
        if path is not None:
            video_from_image_dir(dir, path, fps=30)
            path_gif = path.replace(".mp4", ".gif")
            mp4_to_gif(path, path_gif)

        if not save_tmp and path is not None:
            shutil.rmtree(dir)


def _make_beam(width=40, height=5, thickness=0.1):
    """Construct the rotated, thin 2D beam used in the demo."""
    [X, T] = igl.triangulated_grid(width, height)
    X[:, 1] *= thickness
    Y = X.copy()
    X[:, 0] = -Y[:, 1]
    X[:, 1] = Y[:, 0]
    X = normalize_and_center(X)
    return X, T


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, T = _make_beam()

    # Complementary Dynamics: rig drives the global handle, simulation
    # subspace is orthogonal to the rig.
    world_cd = CoDyLeftRightWorld(X, T)
    Zs, Ps = world_cd.simulate_periodic_x(num_timesteps=300, period=150, a=3)
    world_cd.render(Zs, Ps, path=os.path.join(RESULTS_DIR, "beam_cody_left_right.mp4"))

    # Pinned baseline: a small center region is hard-pinned and swept left/right.
    world_pin = PinLeftRightWorld(X, T, bI=center_indices(X, 0.1)[1])
    Zs, Ps = world_pin.simulate_periodic_x(num_timesteps=300, period=150, a=3)
    world_pin.render(Zs, Ps, path=os.path.join(RESULTS_DIR, "beam_pin_0.1_left_right.mp4"))


if __name__ == "__main__":
    main()
