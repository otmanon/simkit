"""Tutorial 015 - Interactive subspace Mixed FEM (MFEM) in 3D.

The 3D sibling of ``012_interactive_mixed_fem.py``. Instead of a flat beam this
loads the tetrahedral **crab** mesh (the same asset and pin set used by
``examples/subspace_mfem/interactive.py``) and simulates it in a *reduced space*:
we solve for a handful of skinning-eigenmode weights ``B`` plus a small set of
per-cubature-point stretch auxiliary variables ``a`` rather than every vertex.

This is the subspace Mixed FEM method from
https://www.dgp.toronto.edu/projects/subspace-mfem/ . An auxiliary stretch
variable ``a`` (one symmetric stretch per cubature tet) is tied to the
deformation by the constraint ``S(F(u)) = a``. The elastic energy is evaluated
on the cheap, element-local ``a`` instead of on the dense subspace deformation,
so per-iteration cost stays tiny even though ``B`` is dense. The constrained
system is solved with the flat ``sqp_mfem`` solver, which eliminates the
Lagrange multiplier and updates only ``[du, da]``.

The ``MixedFEMSim`` class below is identical in spirit to 012 - the no-class rule
is about the library, not example scripts - and is fully dimension-generic, so
the only real difference here is loading a real 3D mesh, pinning the crab's feet,
and wiring the 3D viewer / mouse handle from ``utils``.

Run with ``python 015_interactive_mixed_fem_3D.py`` for the GUI. The sim class is
importable and steppable headlessly (see the ``if __name__ == "__main__"`` guard
at the bottom).
"""
import os

import igl
import numpy as np
import scipy as sp

import simkit as sk
from simkit.solvers import sqp_mfem
from simkit.energies import (elastic_energy_S, elastic_gradient_S, elastic_hessian_S,
                             quadratic_energy, quadratic_gradient, quadratic_hessian,
                             kinetic_energy_be, kinetic_gradient_be, kinetic_hessian_be)


# ============================================================================
# Crab mesh loading (same asset + pin set as examples/subspace_mfem/interactive.py)
# ============================================================================

def load_crab():
    """Load + normalize the crab tet mesh, its precomputed *heterogeneous*
    per-tet Young's modulus (soft 1e6 joints / stiff 1e12 shell), and its
    foot-pin indices - the same three assets ``subspace_mfem/interactive.py``
    consumes via ``crabConfig``.
    """
    data = sk.filesystem.get_data_directory()
    X, T, _ = igl.readMESH(os.path.join(data, "3d", "crab", "crab.mesh"))
    ym = np.load(os.path.join(data, "3d", "crab", "mu.npy")).reshape(-1, 1)  # (t, 1)
    bI = np.load(os.path.join(data, "3d", "crab", "bI.npy")).reshape(-1)

    # center + unit-box scale (matches normalize_mesh in the subspace_mfem demo).
    X = X - X.mean(axis=0)
    X = X / max(X.max(axis=0) - X.min(axis=0))
    return X, T, ym, bI


# ============================================================================
# Local subspace-MFEM sim class (dimension-generic; identical math to 012).
#
# State vector solved by sqp_mfem is  p = [u; a; ll]  where
#   u  : (nz, 1) subspace (skinning-eigenmode) weights  -> positions x = B u + q
#   a  : (na, 1) per-cubature symmetric-stretch auxiliary variables
#   ll : (na, 1) Lagrange multipliers of the constraint S(F) = a
#
# The handle writes full-space Q_h / b_h (n*dim square / n*dim vector); we keep
# the subspace projections up to date inside step().
# ============================================================================

class MixedFEMSim:
    def __init__(self, X, T, m=12, k=300, ym=1e10, pr=0.45, rho=1e3, h=0.01,
                 material="macklin-mueller-neo-hookean", pin_idx=None, K_pin=1e8,
                 max_iter=4):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        dim = self.dim
        self.h = float(h)
        self.material = material
        self._newton_iters = int(max_iter)

        # rest geometry offset q so that x = B @ u + q
        self.q = X.reshape(-1, 1)

        # ``ym`` may be a scalar or a per-tet (t, 1) array of Young's moduli
        # (heterogeneous material - e.g. the crab's soft joints / stiff shell).
        ym_tet = np.asarray(ym, dtype=float).reshape(-1, 1)
        if ym_tet.shape[0] == 1:
            ym_tet = np.full((T.shape[0], 1), float(ym_tet))

        # ---- subspace + cubature -------------------------------------------
        # Weight the skinning eigenmodes by the per-tet stiffness so stiff
        # regions move rigidly and soft regions carry the deformation modes.
        W, E, B = sk.skinning_eigenmodes(X, T, m, mu=ym_tet)
        cI, cW, _ = sk.spectral_cubature(X, T, W, k, return_labels=True)
        self.B = B
        self.cI = cI
        self.vol = cW.reshape(-1, 1)

        # ---- material ------------------------------------------------------
        # Per-tet Lame params, then gathered down to the cubature tets.
        mu, lam = sk.ympr_to_lame(ym_tet, pr)        # (t, 1) each
        self.mu  = mu[cI]
        self.lam = lam[cI]

        # ---- kinetic / elastic precompute ----------------------------------
        Mn = sk.massmatrix(X, T, rho=rho)
        Mv = sp.sparse.kron(Mn, sp.sparse.identity(dim))
        self.M = Mv
        self.kin_pre = B.T @ Mv @ B

        G  = sk.selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim * dim))
        J  = sk.deformation_jacobian(X, T)
        self.GJB = Ge @ J @ B
        self.GJq = Ge @ J @ self.q

        # symmetric-stretch maps; weights w for the constraint metric
        self.C, self.Ci = sk.symmetric_stretch_map(cI.shape[0], dim)
        if dim == 2:
            self.w = np.kron(self.vol, np.array([[1., 1., 2.]]).T)
        else:
            self.w = np.kron(self.vol, np.array([[1., 1., 1., 2., 2., 2.]]).T)
        self.W = sp.sparse.diags(self.w.flatten())
        self.Wi = sp.sparse.diags(1.0 / self.w.flatten())

        # Augmented-Lagrangian penalty weight (NOT the mass density `rho`).
        # Keep this SMALL (see 012 for the full note); 0 is fine here.
        self.rho_aug = 0

        self.nz = B.shape[1]
        self.na = self.Ci.shape[0]

        # ---- pin (full space, projected into subspace) ---------------------
        D = self.n * dim
        if pin_idx is not None and len(pin_idx) > 0:
            bI = np.asarray(pin_idx).reshape(-1)
            bc0 = X[bI, :]
            Qp, bp = sk.dirichlet_penalty(bI, bc0, self.n, K_pin)
        else:
            Qp = sp.sparse.csc_matrix((D, D)); bp = np.zeros((D, 1))
        self.Q_pin, self.b_pin = Qp, bp

        # gravity as a constant full-space linear term
        self.b_grav = -sk.gravity_force(X, T, a=-9.81, rho=rho).reshape(-1, 1)

        # ---- mouse handle full-space soft pin (written by MouseHandle3D) ----
        self.Q_h = sp.sparse.csc_matrix((D, D))
        self.b_h = np.zeros((D, 1))

        # ---- dynamic state -------------------------------------------------
        self.u, self.a, self.ll = self._rest_state()
        self.u_prev = self.u.copy()
        # full-space vertex positions exposed to the viewer / handle
        self.U = (B @ self.u).reshape(-1, dim) + self.q.reshape(-1, dim)

        # per-step external blocks (set in step())
        self._Q = self.kin_pre * 0.0
        self._b = np.zeros((self.nz, 1))

    # ------------------------------------------------------------------------
    def _rest_state(self):
        dim = self.dim
        u = sk.project_into_subspace(self.X.reshape(-1, 1) - self.q, self.B, M=self.M)
        cc = dim * (dim + 1) // 2
        a = np.ones((self.cI.shape[0], cc)); a[:, dim:] = 0.0
        ll = np.zeros_like(a).real
        return u, a.reshape(-1, 1), ll.reshape(-1, 1)

    def _split(self, p):
        return p[:self.nz], p[self.nz:self.nz + self.na], p[self.nz + self.na:]

    # ------- energy / gradient blocks / hessian blocks (augmented Lagrangian) --
    def energy(self, p):
        u, a, ll = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        F = (self.GJB @ u + self.GJq).reshape(-1, self.dim, self.dim)
        el  = elastic_energy_S(A, self.mu, self.lam, self.vol, self.material)
        kin = kinetic_energy_be(u, self.u, self.u_prev, self.kin_pre, self.h)
        quad = quadratic_energy(u, self._Q, self._b)
        c  = (self.Ci @ sk.stretch(F) - a)
        wc = self.w * c
        lag = float(ll.T @ wc)                       # Lagrangian term
        aug = 0.5 * self.rho_aug * float(c.T @ wc)   # 0.5 rho c^T W c
        return el + kin + quad + lag + aug

    def grad_blocks(self, p):
        u, a, ll = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        F = (self.GJB @ u + self.GJq).reshape(-1, self.dim, self.dim)

        wc = self.w * (self.Ci @ sk.stretch(F) - a)   # = W c

        # penalty gradient w.r.t. u needs G_u; recompute the same dsdz as the Hessian
        dsdz = sk.stretch_gradient_dz(u, self.GJB, Ci=self.Ci, dim=self.dim, GJq=self.GJq)
        G_u = dsdz @ self.W

        f_u = (kinetic_gradient_be(u, self.u, self.u_prev, self.kin_pre, self.h)
               + quadratic_gradient(u, self._Q, self._b)
               + self.rho_aug * (G_u @ wc))
        f_z = (elastic_gradient_S(A, self.mu, self.lam, self.vol, self.material).reshape(-1, 1)
               - self.rho_aug * wc)     # dc/da = -I
        f_ll = wc
        return [f_u, f_z, f_ll]

    def hess_blocks(self, p):
        u, a, lam = self._split(p)
        A = a.reshape(-1, self.dim * (self.dim + 1) // 2)
        dsdz = sk.stretch_gradient_dz(u, self.GJB, Ci=self.Ci, dim=self.dim, GJq=self.GJq)
        G_u = dsdz @ self.W

        H_u = kinetic_hessian_be(self.kin_pre, self.h) + quadratic_hessian(self._Q)
        H_u = H_u + self.rho_aug * (G_u @ self.Wi @ G_u.T)   # GN penalty term in u

        H_z = elastic_hessian_S(A, self.mu, self.lam, self.vol, self.material)
        H_z = sp.sparse.block_diag([h for h in H_z])
        H_z = H_z + self.rho_aug * self.W                    # GN penalty term in a

        G_z = -self.W
        G_zi = sp.sparse.diags(1.0 / G_z.diagonal())
        return [H_u, H_z, G_u, G_z, G_zi]

    # ------------------------------------------------------------------------
    def step(self):
        # Assemble full-space external quadratic / linear penalty acting on
        # absolute positions x = B u + q, then project into the subspace u.
        Q_full = self.Q_pin + self.Q_h
        b_full = self.b_pin + self.b_h + self.b_grav
        self._Q = self.B.T @ Q_full @ self.B
        self._b = self.B.T @ (Q_full @ self.q + b_full)

        p = np.vstack([self.u, self.a, self.ll])
        p = sqp_mfem(p, self.energy, self.hess_blocks, self.grad_blocks,
                     tolerance=1e-3, max_iter=self._newton_iters, do_line_search=True)
        u_next, self.a, self.ll = self._split(p)

        self.u_prev = self.u.copy()
        self.u = u_next.copy()
        self.U = (self.B @ self.u).reshape(-1, self.dim) + self.q.reshape(-1, self.dim)


# ============================================================================
# GUI wiring (only runs when executed directly; importing is side-effect free).
# ============================================================================

def _run_gui():
    import polyscope as ps
    import polyscope.imgui as psim
    from utils import MouseHandle3D, RollingPlot, TutorialUI, Viewer3D

    X, T, ym, pin_idx = load_crab()
    sim = MixedFEMSim(X, T, m=12, k=300, ym=ym, h=0.01,
                      material="macklin-mueller-neo-hookean",
                      pin_idx=pin_idx, max_iter=4)

    sims = {"Subspace MFEM": sim}
    viewer = Viewer3D(X, T, camera_eye=(1.2, 0.8, 2.0), camera_target=(0.0, 0.0, 0.0))
    ps.register_point_cloud("pinned", X[pin_idx], radius=0.006, material="flat",
                            color=(0.4, 0.4, 0.4), enabled=True)
    sel_pc, target_pc = viewer.add_handle_markers()
    handle = MouseHandle3D(sims, sel_pc, target_pc, K_handle=5e8)

    ui = TutorialUI(sims, handle=handle, show_material=False, show_integrator=False,
                    show_handle_mode=True)

    elastic_plot = RollingPlot("elastic energy", height=120.0)
    ui.on_reset(elastic_plot.clear)

    def reset_mfem():
        sim.u, sim.a, sim.ll = sim._rest_state()
        sim.u_prev = sim.u.copy()
        sim.U = (sim.B @ sim.u).reshape(-1, sim.dim) + sim.q.reshape(-1, sim.dim)
    ui.on_reset(reset_mfem)

    def callback():
        ui.draw()
        if ui.handle_enabled:
            handle.update()
        sim.step()
        viewer.refresh(sim.U)
        handle.refresh_markers()

        A = sim.a.reshape(-1, sim.dim * (sim.dim + 1) // 2)
        elastic_plot.push(float(elastic_energy_S(A, sim.mu, sim.lam, sim.vol, sim.material)))
        psim.Text(f"subspace dim: {sim.nz}    cubature pts: {sim.cI.shape[0]}    "
                  f"vertices: {sim.n}    pinned: {len(pin_idx)}")
        elastic_plot.draw()

    viewer.show(callback)


if __name__ == "__main__":
    _run_gui()
