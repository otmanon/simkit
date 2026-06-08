"""Shared scaffolding for the subspace_mfem examples - flat-function edition.

The ``simkit.sims`` module has been removed. The elastic FEM and subspace Mixed
FEM (MFEM) simulators are now composed *inline* from flat simkit functions. To
keep the example driver scripts (drop / slingshot / interactive) readable, we
wrap that composition in two tiny *local* sim classes here:

* ``FEMSim``  - full/subspace elastic FEM, stepped with ``newton_solver``.
* ``MFEMSim`` - subspace Mixed FEM (auxiliary stretch ``a`` + constraint),
  stepped with the flat ``sqp_mfem`` solver (which eliminates the Lagrange
  multiplier and updates only ``[du, da]``).

Defining small local classes in an *example* file is fine - the no-class rule
is about the SimKit *library*, not example glue. Both classes expose a minimal,
matching interface used by the drivers: ``X``, ``T``, ``B``, ``q``, ``dim``,
``h``, ``rest_state()`` and ``step(...)``.

MFEM math reference: https://www.dgp.toronto.edu/projects/subspace-mfem/ , Sec 4.
"""

import numpy as np
import igl
import os
from pathlib import Path

import simkit as sk
from simkit.solvers import sqp_mfem, newton_solver
from simkit.energies import (elastic_energy_S, elastic_gradient_S, elastic_hessian_S,
                             elastic_energy_z, elastic_gradient_z, elastic_hessian_z,
                             ElasticEnergyZPrecomp,
                             quadratic_energy, quadratic_gradient, quadratic_hessian,
                             kinetic_energy_be, kinetic_gradient_be, kinetic_hessian_be)


ag = -9.81
k_pin = 1e8


# =============================================================================
# mesh IO
# =============================================================================

def load_mesh(geometry_path):
    file_extension = geometry_path.split(".")[-1]
    if file_extension == "obj":
        [X, _, _, T, _, _] = igl.readOBJ(geometry_path)
        X = X[:, 0:2]
    elif file_extension == "mesh":
        [X, T, _] = igl.readMESH(geometry_path)
    else:
        raise ValueError("Geometry path must be a .obj or .mesh file")
    return X, T


def normalize_mesh(X):
    X = X - X.mean(axis=0)
    X = X / max(X.max(axis=0) - X.min(axis=0))
    return X


def compute_subspace(X, T, m, k, mu=None, bI=None):
    """Skinning-eigenmode subspace B + spectral cubature (cI, cW)."""
    if mu is not None:
        if isinstance(mu, str):
            mu = np.load(mu).reshape(-1, 1)
        elif isinstance(mu, (float, int)):
            mu = mu * np.ones((T.shape[0], 1))
        elif isinstance(mu, np.ndarray):
            assert mu.shape[0] == T.shape[0] and mu.shape[1] == 1
    assert mu.shape[1] == 1 and mu.shape[0] == T.shape[0]

    if isinstance(bI, str):
        bI = np.load(bI).astype(int)

    print("Computing skinning eigenmodes subspace ....")
    if m is None:
        W = E = B = None
    else:
        [W, E, B] = sk.skinning_eigenmodes(X, T, m, mu=mu, bI=bI)
    print("Done!")

    print("Computing spectral cubature points ....")
    if k is None:
        cI = cW = labels = None
    else:
        [cI, cW, labels] = sk.spectral_cubature(X, T, W, k, return_labels=True)
    print("Done!")

    return W, E, B, cI, cW, labels


# =============================================================================
# tiny per-step external-force container (replaces the old sim_params object)
# =============================================================================

class SimParams:
    """Lightweight bag so the drivers can read ``sim.sim_params.h``."""
    def __init__(self, h, max_iter, do_line_search):
        self.h = h
        self.max_iter = max_iter
        self.do_line_search = do_line_search


def _as_tet_array(ym, T):
    if isinstance(ym, str):
        ym = np.load(ym).reshape(-1, 1)
    elif isinstance(ym, (float, int)):
        ym = ym * np.ones((T.shape[0], 1))
    assert ym.shape[0] == T.shape[0] and ym.shape[1] == 1
    return ym


# =============================================================================
# FEM sim (subspace), stepped with newton_solver
# =============================================================================

class FEMSim:
    def __init__(self, X, T, ym, rho, h, max_iter, do_line_search,
                 B=None, cI=None, cW=None, q=None, material="fcr"):
        import scipy as sp
        self.X, self.T = X, T
        self.dim = X.shape[1]
        dim = self.dim
        self.material = material
        self.sim_params = SimParams(h, max_iter, do_line_search)

        if B is None:
            B = sp.sparse.identity(X.shape[0] * dim)
        self.B = B
        self.q = (X.reshape(-1, 1) if q is None else q)

        if cI is None:
            cI = np.arange(0, T.shape[0])
        self.cI = cI

        ym = _as_tet_array(ym, T)
        mu, lam = sk.ympr_to_lame(ym, 0.0 * ym)
        self.mu = mu[cI]
        self.lam = lam[cI]
        self.vol = (sk.volume(X, T).reshape(-1, 1) if cW is None else cW.reshape(-1, 1))

        Mn = sk.massmatrix(X, T, rho=rho)
        Mv = sp.sparse.kron(Mn, sp.sparse.identity(dim))
        self.M = Mv
        self.kin_pre = B.T @ Mv @ B

        G = sk.selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim * dim))
        J = sk.deformation_jacobian(X, T)
        self.el_pre = ElasticEnergyZPrecomp(B, self.q, Ge, J, dim)

        self._Q = None
        self._b = None
        self._z_curr = None
        self._z_prev = None

    def rest_state(self):
        import scipy as sp
        M = sp.sparse.kron(sk.massmatrix(self.X, self.T), sp.sparse.identity(self.dim))
        z = sk.project_into_subspace(self.X.reshape(-1, 1) - self.q, self.B, M=M)
        return z, np.zeros_like(z)

    def energy(self, z):
        return (kinetic_energy_be(z, self._z_curr, self._z_prev, self.kin_pre, self.sim_params.h)
                + elastic_energy_z(z, self.mu, self.lam, self.vol, self.material, self.el_pre)
                + quadratic_energy(z, self._Q, self._b))

    def gradient(self, z):
        return (kinetic_gradient_be(z, self._z_curr, self._z_prev, self.kin_pre, self.sim_params.h)
                + elastic_gradient_z(z, self.mu, self.lam, self.vol, self.material, self.el_pre)
                + quadratic_gradient(z, self._Q, self._b))

    def hessian(self, z):
        return (kinetic_hessian_be(self.kin_pre, self.sim_params.h)
                + elastic_hessian_z(z, self.mu, self.lam, self.vol, self.material, self.el_pre)
                + quadratic_hessian(self._Q))

    def step(self, z_curr, z_prev, z_dot=None, Q_ext=None, b_ext=None, return_info=False):
        import scipy as sp
        h = self.sim_params.h
        self._z_curr, self._z_prev = z_curr, z_prev
        self._Q = (Q_ext if Q_ext is not None
                   else sp.sparse.csc_matrix((z_curr.shape[0], z_curr.shape[0])))
        self._b = (b_ext if b_ext is not None else np.zeros((z_curr.shape[0], 1)))

        v = (z_curr - z_prev) / h
        z0 = z_curr + v * h
        out = newton_solver(z0, self.energy, self.gradient, self.hessian,
                            tolerance=1e-6, max_iter=self.sim_params.max_iter,
                            do_line_search=self.sim_params.do_line_search,
                            return_info=return_info)
        return out


# =============================================================================
# MFEM sim (subspace), stepped with sqp_mfem (lambda eliminated)
# =============================================================================

class MFEMSim:
    def __init__(self, X, T, ym, rho, h, max_iter, do_line_search,
                 B=None, cI=None, cW=None, q=None, material="arap"):
        import scipy as sp
        self.X, self.T = X, T
        self.dim = X.shape[1]
        dim = self.dim
        self.material = material
        self.sim_params = SimParams(h, max_iter, do_line_search)

        if B is None:
            B = sp.sparse.identity(X.shape[0] * dim)
        self.B = B
        self.q = (X.reshape(-1, 1) if q is None else q)

        if cI is None:
            cI = np.arange(0, T.shape[0])
        self.cI = cI

        ym = _as_tet_array(ym, T)
        mu, lam = sk.ympr_to_lame(ym, 0.0 * ym)
        self.mu = mu[cI]
        self.lam = lam[cI]
        self.vol = (sk.volume(X, T).reshape(-1, 1) if cW is None else cW.reshape(-1, 1))

        Mn = sk.massmatrix(X, T, rho=rho)
        Mv = sp.sparse.kron(Mn, sp.sparse.identity(dim))
        self.M = Mv
        self.kin_pre = B.T @ Mv @ B

        G = sk.selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim * dim))
        J = sk.deformation_jacobian(X, T)
        self.GJB = Ge @ J @ B
        self.GJq = Ge @ J @ self.q

        self.C, self.Ci = sk.symmetric_stretch_map(cI.shape[0], dim)
        if dim == 2:
            self.w = np.kron(self.vol, np.array([[1., 1., 2.]]).T)
        else:
            self.w = np.kron(self.vol, np.array([[1., 1., 1., 2., 2., 2.]]).T)
        self.W = sp.sparse.diags(self.w.flatten())

        self.nz = B.shape[1]
        self.na = self.Ci.shape[0]

        self._Q = None
        self._b = None
        self._z_curr = None
        self._z_prev = None

    def rest_state(self):
        """Returns z (positions), s (stretch=identity), la (zeros), z_dot (zeros)."""
        import scipy as sp
        dim = self.dim
        z = sk.project_into_subspace(
            self.X.reshape(-1, 1) - self.q, self.B,
            M=sp.sparse.kron(sk.massmatrix(self.X, self.T), sp.sparse.identity(dim)))
        cc = dim * (dim + 1) // 2
        s = np.ones((self.cI.shape[0], cc)); s[:, dim:] = 0.0
        s = s.reshape(-1, 1)
        la = np.zeros_like(s)
        return z, s, la, np.zeros_like(z)

    def _split(self, p):
        return p[:self.nz], p[self.nz:self.nz + self.na]

    def energy(self, p):
        u, a = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        F = (self.GJB @ u + self.GJq).reshape(-1, self.dim, self.dim)
        el = elastic_energy_S(A, self.mu, self.lam, self.vol, self.material)
        kin = kinetic_energy_be(u, self._z_curr, self._z_prev, self.kin_pre, self.sim_params.h)
        quad = quadratic_energy(u, self._Q, self._b)
        c = (self.Ci @ sk.stretch(F) - a)
        constraint = (c.T @ (self.w * c))
        return el + kin + quad + constraint

    def grad_blocks(self, p):
        u, a = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        F = (self.GJB @ u + self.GJq).reshape(-1, self.dim, self.dim)
        f_u = (kinetic_gradient_be(u, self._z_curr, self._z_prev, self.kin_pre, self.sim_params.h)
               + quadratic_gradient(u, self._Q, self._b))
        f_z = elastic_gradient_S(A, self.mu, self.lam, self.vol, self.material).reshape(-1, 1)
        f_mu = (self.w * (self.Ci @ sk.stretch(F) - a))
        return [f_u, f_z, f_mu]

    def hess_blocks(self, p):
        import scipy as sp
        u, a = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        H_u = kinetic_hessian_be(self.kin_pre, self.sim_params.h) + quadratic_hessian(self._Q)
        dsdz = sk.stretch_gradient_dz(u, self.GJB, Ci=self.Ci, dim=self.dim, GJq=self.GJq)
        G_u = dsdz @ self.W
        H_z = elastic_hessian_S(A, self.mu, self.lam, self.vol, self.material)
        H_z = sp.sparse.block_diag([h for h in H_z])
        G_z = -self.W
        G_zi = sp.sparse.diags(1.0 / G_z.diagonal())
        return [H_u, H_z, G_u, G_z, G_zi]

    def step(self, z_curr, z_prev, a, la=None, Q_ext=None, b_ext=None, return_info=False):
        """Step subspace MFEM. ``la`` is accepted for API compatibility but the
        flat sqp_mfem eliminates the multiplier, so it is recomputed as zeros."""
        import scipy as sp
        self._z_curr, self._z_prev = z_curr, z_prev
        self._Q = (Q_ext if Q_ext is not None
                   else sp.sparse.csc_matrix((z_curr.shape[0], z_curr.shape[0])))
        self._b = (b_ext if b_ext is not None else np.zeros((z_curr.shape[0], 1)))

        p = np.vstack([z_curr, a])
        p = sqp_mfem(p, self.energy, self.hess_blocks, self.grad_blocks,
                     tolerance=1e-6, max_iter=self.sim_params.max_iter,
                     do_line_search=self.sim_params.do_line_search)
        z_next, a_next = self._split(p)
        la_next = np.zeros_like(a_next)  # multiplier eliminated by the flat solver
        if return_info:
            return z_next, a_next, la_next, {}
        return z_next, a_next, la_next


# =============================================================================
# builders (kept for backward-compatible call sites in the drivers)
# =============================================================================

def create_mfem_sim(X, T, ym, rho, h, max_iter, do_line_search, B=None, cI=None, cW=None):
    print("Initializing MFEM simulation ....")
    q = X.reshape(-1, 1)
    sim = MFEMSim(X, T, ym, rho, h, max_iter, do_line_search, B=B, cI=cI, cW=cW, q=q)
    print("Done!")
    return sim


def create_fem_sim(X, T, ym, rho, h, max_iter, do_line_search, B=None, cI=None, cW=None):
    print("Initializing FEM simulation ....")
    q = X.reshape(-1, 1)
    sim = FEMSim(X, T, ym, rho, h, max_iter, do_line_search, B=B, cI=cI, cW=cW, q=q)
    print("Done!")
    return sim


# =============================================================================
# animation viewer (unchanged)
# =============================================================================

def view_animation(X, T, U, path=None, fps=60, eye_pos=None, look_at=None):
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    if eye_pos is not None and look_at is not None:
        ps.look_at(eye_pos, look_at)
    else:
        ps.look_at(np.array([0, 0, 3]), np.array([0, 0, 0]))
    dim = X.shape[1]

    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    if T.shape[1] == 3:
        mesh = ps.register_surface_mesh("mesh", X, T, edge_width=0.01)
    elif T.shape[1] == 4:
        mesh = ps.register_volume_mesh("mesh", X, T, edge_width=0.01)
    else:
        raise ValueError("T must be 3 or 4")

    for i in range(U.shape[1]):
        x = X.reshape(-1, 1) + U[:, [i]]
        mesh.update_vertex_positions(x.reshape(-1, dim))
        ps.frame_tick()
        if path is not None:
            ps.screenshot(dirstem + "/" + str(i + 1).zfill(4) + ".png", transparent_bg=True)

    if path is not None:
        sk.filesystem.video_from_image_dir(dirstem, path, fps=fps)
        sk.filesystem.mp4_to_gif(path, path.replace(".mp4", ".gif"))

    ps.remove_all_structures()
    return
