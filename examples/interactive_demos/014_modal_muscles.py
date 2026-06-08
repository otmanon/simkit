"""Tutorial 014 - Modal Muscles.

A 2D creature whose deformation lives in a reduced skinning-eigenmode subspace
``B`` (state ``z``, so ``x = B @ z``). A small set of displacement modes ``D``
act as **muscles**: dialing each muscle's amplitude ``a`` drives a *clustered
plastic stretch tensor* that the creature's flesh tries to match, while passive
ARAP elasticity + inertia resist. Drag the per-muscle sliders to contract and
relax the muscles and watch the body deform and move (*Modal Muscles* /
*Actuators a la Mode*).

Like tutorials 007 / 013, the simulator lives in a small *local* class (the
no-class rule is about the simkit library, not example scripts). Its ``step()``
hands two closures to ``simkit.solvers.block_coord``:

    local step  : clustered best-fit rotations (polar SVD) for the passive
                  ARAP term AND the active plastic-stretch target
    global step : one Cholesky back-solve in the reduced subspace ``B``

Run from ``examples/interactive_demos`` (so ``utils`` is importable)::

    python 014_modal_muscles.py
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps

import simkit as sk
from simkit.clustered_plastic_stretch_tensor import clustered_plastic_stretch_tensor
from simkit.fast_sandwich_transform_clustered import fast_sandwich_transform_clustered
from simkit.solvers import block_coord


# ============================================================================
# Local modal-muscle simulator: state + precomputed operators.
#
# Subspace:  x = B @ z       (skinning eigenmodes, reduced coords z)
# Muscles:   a  (one amplitude per muscle mode in D); a drives the active
#            plastic stretch tensor K(z, a) that the flesh tries to match.
# ============================================================================

class ModalMuscleSim:
    """Reduced modal-muscle creature.

    Exposes ``U`` (current full vertex positions, n x dim) for the viewer, and
    ``n`` / ``dim`` for the scaffolding. The interactive UI writes muscle
    amplitudes onto ``self.amp`` (one entry per muscle mode).
    """

    def __init__(self, X, T, num_modes=8, num_active_modes=4, num_clusters=20,
                 mu=1e5, gamma=1e5, rho=1e3, h=0.02, max_iter=10,
                 gravity=0.0, contact=False, ground_y=None, num_contact=30,
                 alpha=1.0):
        n, dim = X.shape
        self.X, self.T = X, T
        self.n, self.dim = n, dim
        self.h = float(h)
        self.max_iter = int(max_iter)
        self.contact = contact

        Mv = sps.kron(sk.massmatrix(X, T, rho=rho), sps.identity(dim)).tocsc()
        J = sk.deformation_jacobian(X, T)
        vol = sk.volume(X, T)
        nT = T.shape[0]

        # ---- subspace + muscle modes + cubature --------------------------
        W, _E, B = sk.skinning_eigenmodes(X, T, num_modes)
        B = sk.orthonormalize(B, M=Mv)
        _E2, Dfull = sk.linear_modal_analysis(X, T, num_active_modes + 1)
        # skip mode 0 (rigid); use the next `num_active_modes` as muscles
        modeset = list(range(1, num_active_modes + 1))
        # append the rest pose so the activation "a" can encode the rest state
        D = np.hstack([Dfull[:, modeset], X.reshape(-1, 1)])
        self.modeset = modeset
        self.limit_a = sk.limit_actuation_dirichlet_energy(X, T, Dfull, max_s=2.0)[modeset]

        _cI, _cW, labels = sk.spectral_cubature(X, T, W, num_clusters, return_labels=True)
        l = np.asarray(labels).flatten().astype(int)
        d = np.zeros(nT, dtype=int)   # single active cluster

        # ---- passive ARAP (clustered) ------------------------------------
        mu_v = np.ones((nT, 1)) * mu
        gamma_v = np.ones((nT, 1)) * gamma
        P, _ = sk.cluster_grouping_matrices(l, X, T)
        A = sps.diags(vol.flatten())
        Mu = sps.diags(mu_v.flatten())
        AMue = sps.kron(A @ Mu, sps.identity(dim * dim))
        PAMue = sps.kron(P @ (A @ Mu), sps.identity(dim * dim))
        AMuPJB = (PAMue @ J) @ B
        L_passive = J.T @ AMue @ J

        # ---- active muscle force -----------------------------------------
        Gamma = sps.diags(gamma_v.flatten())
        AGammae = sps.kron(A @ Gamma, sps.identity(dim * dim))
        L_active = J.T @ (AGammae @ J)
        JD = J @ D
        BJAgamma = B.T @ (J.T @ AGammae)

        self.B, self.D = B, D
        self.Mv = Mv
        self.AMuPJB = AMuPJB
        self.BMB = B.T @ Mv @ B
        self.BMy = B.T @ Mv @ X.reshape(-1, 1)
        self.DMD = D.T @ Mv @ D
        self.DMy = D.T @ Mv @ X.reshape(-1, 1)
        self.num_passive_clusters = int(l.max()) + 1
        self.num_active_clusters = int(d.max()) + 1
        self.K = clustered_plastic_stretch_tensor(X, T, d, B, D, w=(vol * gamma_v).reshape(-1, 1))
        self.fst = fast_sandwich_transform_clustered(BJAgamma, JD, d, dim=dim)

        # constant reduced force: gravity (optional)
        if gravity != 0.0:
            g = B.T @ sk.gravity_force(X, T, a=gravity, rho=rho).reshape(-1, 1)
            self.b = -g
        else:
            self.b = np.zeros((B.shape[1], 1))

        H = self.B.T @ L_passive @ self.B + self.B.T @ L_active @ self.B \
            + self.BMB / self.h ** 2
        self.chol = sp.linalg.cho_factor(H)

        # ---- optional ground contact -------------------------------------
        if contact:
            if ground_y is None:
                ground_y = float(X[:, 1].min())
            self.alpha = float(alpha)
            self.plane_pos = np.array([[0.0], [ground_y]])
            try:
                import igl
                fI = np.unique(igl.boundary_facets(T)[0])
            except Exception:
                fI = np.arange(n)
            cI = fI[sk.farthest_point_sampling(X[fI, :], min(num_contact, fI.shape[0]))]
            cI = np.unique(cI)
            S = sk.selection_matrix(cI, n)
            Se = sps.kron(S, sps.identity(dim))
            self.Je = Se @ B
            self.JeQi = sp.linalg.cho_solve(self.chol, self.Je.T).T

        # ---- state -------------------------------------------------------
        self.z = sk.project_into_subspace(X.reshape(-1, 1), B, M=Mv,
                                          BMB=self.BMB, BMy=self.BMy)
        self.z_dot = np.zeros_like(self.z)
        self.a0 = sk.project_into_subspace(X.reshape(-1, 1), D, M=Mv,
                                           BMB=self.DMD, BMy=self.DMy)
        # user-controlled muscle amplitudes (one per muscle mode)
        self.amp = np.zeros(len(modeset))
        self.U = self._positions()

    # -- the two modal-muscle block-coordinate sub-steps ------------------
    def _build_a(self):
        a = self.a0.copy()
        a[:-1, 0] = self.amp
        return a

    def _contact_projection(self, z, f, z_curr):
        dim, h = self.dim, self.h
        z_dot_tent = (sp.linalg.cho_solve(self.chol, f) - z_curr) / h
        Pp = (self.Je @ z).reshape(-1, dim)
        under = (Pp[:, 1] < self.plane_pos[1]).flatten()
        local_vel = (self.Je @ z_dot_tent).reshape(-1, dim)
        closer = local_vel[:, 1] < 0
        ci = np.where(under * closer)[0]
        if ci.shape[0] == 0:
            return np.zeros_like(f)
        idx = (np.repeat(ci[:, None], dim, axis=1) * dim + np.arange(dim)).flatten()
        JeI = self.Je[idx, :]
        L = self.JeQi[idx, :]
        vt = local_vel[ci, 0]
        v = np.zeros((ci.shape[0], dim)); v[:, 0] = (1.0 - self.alpha) * vt
        p = (JeI @ z_curr).reshape(-1, dim)
        local_f = (L @ f).reshape(-1, dim)
        bb = (v * h + p - local_f).reshape(-1, 1)
        if L.shape[0] >= L.shape[1]:
            c = np.linalg.solve(L.T @ L, L.T @ bb)
        else:
            c = L.T @ np.linalg.solve(L @ L.T, bb)
        return c

    def step(self):
        dim, h = self.dim, self.h
        a = self._build_a()
        z = self.z
        y = z + h * self.z_dot
        k = self.BMB @ y / h ** 2
        z_curr = z.copy()

        def local_step(zc):
            c = self.AMuPJB @ zc
            R_p = sk.polar_svd(c.reshape(-1, dim, dim))[0]
            C = self.K(zc, a)
            R_a = sk.polar_svd(C)[0]
            return np.vstack([R_p.reshape(-1, 1), R_a.reshape(-1, 1)])

        def global_step(zc, r):
            pd = self.num_passive_clusters * dim ** 2
            e_p = self.AMuPJB.T @ r[:pd]
            ad = self.num_active_clusters * dim ** 2
            e_a = self.fst(r[pd:pd + ad].reshape(-1, dim, dim)) @ a
            f = k + e_p + e_a - self.b
            if self.contact:
                f = f + self._contact_projection(zc, f, z_curr)
            return sp.linalg.cho_solve(self.chol, f)

        z_next = block_coord(z.copy(), global_step, local_step,
                             tolerance=1e-6, max_iter=self.max_iter)
        self.z_dot = (z_next - z) / h
        self.z = z_next
        self.U = self._positions()

    def reset(self):
        self.z = sk.project_into_subspace(self.X.reshape(-1, 1), self.B, M=self.Mv,
                                          BMB=self.BMB, BMy=self.BMy)
        self.z_dot = np.zeros_like(self.z)
        self.amp[:] = 0.0
        self.U = self._positions()

    def _positions(self):
        return (self.B @ self.z).reshape(self.n, self.dim)


# ============================================================================
# Main viewer wiring (guarded so importing the file never opens a window).
# ============================================================================

def main():
    import polyscope as ps
    import polyscope.imgui as psim
    from utils import RollingPlot, Viewer2D, triangulated_grid

    # a simple 2D "worm" creature; muscle modes will bend / squash it
    X, T = triangulated_grid(nx=60, ny=8, width=4.0, height=0.6)
    X = sk.normalize_and_center(X)
    sim = ModalMuscleSim(X, T, num_modes=8, num_active_modes=4,
                         num_clusters=20, mu=1e5, gamma=1e5, rho=1e3,
                         h=0.02, max_iter=10, contact=False)

    viewer = Viewer2D(X, T, camera_distance=6.0)

    # one slider per muscle mode; range scaled by the dirichlet-energy limit
    slider_max = np.maximum(sim.limit_a, 0.5)
    auto = {"on": False, "t": 0.0}

    def callback():
        if psim.Button("Reset"):
            sim.reset()
        psim.SameLine()
        _, auto["on"] = psim.Checkbox("auto wiggle", auto["on"])

        psim.Text("Muscle amplitudes")
        for i in range(len(sim.amp)):
            changed, val = psim.SliderFloat(
                f"muscle {i}", float(sim.amp[i]),
                v_min=-float(slider_max[i]), v_max=float(slider_max[i]))
            if changed:
                sim.amp[i] = val

        if auto["on"]:
            auto["t"] += sim.h
            for i in range(len(sim.amp)):
                sim.amp[i] = slider_max[i] * np.sin(
                    2 * np.pi * (auto["t"] / 1.0 + 0.15 * i))

        psim.Text(f"vertices: {sim.n}   triangles: {sim.T.shape[0]}   "
                  f"subspace dim: {sim.B.shape[1]}   muscles: {len(sim.amp)}")

        sim.step()
        viewer.refresh(sim.U)

    viewer.show(callback)


if __name__ == "__main__":
    main()
