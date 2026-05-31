"""Tutorial 007 - Interactive dynamics.

Same beam + handle UI as 006, plus mass and a selectable time integrator.
Grabbing a vertex while the beam is mid-swing throws inertia around -- you can
see overshoot, ringing, and (with Backward Euler) numerical damping.

Three sim classes, one per integrator. Each owns its own ``Q_h`` / ``b_h``
matrices (the mouse handle writes onto them directly), its own history, and
its own ``energy`` / ``gradient`` / ``hessian`` -- with the kinetic term layered
on top of the shared elastic + pin potential.
"""
import numpy as np
import polyscope.imgui as psim
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.massmatrix import massmatrix
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
from simkit.volume import volume
import simkit.energies as energies

from utils import MouseHandle2D, RollingPlot, TutorialUI, Viewer2D, triangulated_grid


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=15, ny=5, width=2.0, height=0.6)
n, dim = X.shape

RHO = 1.0
J   = deformation_jacobian(X, T)
vol = volume(X, T)
M_n = massmatrix(X, T, rho=RHO)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()

pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], n, 1e4)


# ============================================================================
# Three sim classes (no gravity, no contact in this tutorial).
# E_pot(x) = E_elastic + 1/2 x^T Q_pin x + b_pin^T x
#                      + 1/2 x^T Q_h x + b_h^T x      (mouse handle)
# ============================================================================

class ElasticSimBE:
    """Backward Euler."""

    def __init__(self, X, T, J, vol, M, mu, lam, Q_pin, b_pin, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M = J, vol, M
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)

        D = self.n * self.dim
        self.U      = X.copy()
        self.U_prev = X.copy()
        self.Q_h    = sp.sparse.csc_matrix((D, D))
        self.b_h    = np.zeros((D, 1))

        self._solver = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=5, do_line_search=True),
        )

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el  = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                 + float((self.b_pin.T @ xc)[0, 0]))
        E_h   = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                 + float((self.b_h.T @ xc)[0, 0]))
        E_kin = float(energies.kinetic_energy_be(
            xc, self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1), self.M, self.h))
        return E_el + E_pin + E_h + E_kin

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el  = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin = self.Q_pin @ xc + self.b_pin
        g_h   = self.Q_h   @ xc + self.b_h
        g_kin = energies.kinetic_gradient_be(
            xc, self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1), self.M, self.h)
        return g_el + g_pin + g_h + g_kin

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el  = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_kin = energies.kinetic_hessian_be(self.M, self.h)
        return H_el + self.Q_pin + self.Q_h + H_kin

    def step(self):
        x_next = self._solver.solve(self.U.flatten().reshape(-1, 1))
        self.U_prev[:] = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimBDF2:
    """BDF2."""

    def __init__(self, X, T, J, vol, M, mu, lam, Q_pin, b_pin, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M = J, vol, M
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)

        D = self.n * self.dim
        self.U       = X.copy()
        self.U_prev  = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()
        self.Q_h     = sp.sparse.csc_matrix((D, D))
        self.b_h     = np.zeros((D, 1))

        self._solver = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=5, do_line_search=True),
        )

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el  = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                 + float((self.b_pin.T @ xc)[0, 0]))
        E_h   = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                 + float((self.b_h.T @ xc)[0, 0]))
        E_kin = float(energies.kinetic_energy_bdf2(
            xc,
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.M, self.h,
        ))
        return E_el + E_pin + E_h + E_kin

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el  = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin = self.Q_pin @ xc + self.b_pin
        g_h   = self.Q_h   @ xc + self.b_h
        g_kin = energies.kinetic_gradient_bdf2(
            xc,
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.M, self.h,
        )
        return g_el + g_pin + g_h + g_kin

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el  = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_kin = energies.kinetic_hessian_bdf2(self.M, self.h)
        return H_el + self.Q_pin + self.Q_h + H_kin

    def step(self):
        x_next = self._solver.solve(self.U.flatten().reshape(-1, 1))
        self.U_prev3[:] = self.U_prev2
        self.U_prev2[:] = self.U_prev
        self.U_prev[:]  = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)



# ---------- build sims + viewer + UI --------------------------------------
common = dict(X=X, T=T, J=J, vol=vol, M=M, mu=1.0, lam=1.0,
              Q_pin=Q_pin, b_pin=b_pin, h=0.02)
sims = {
    "Backward Euler": ElasticSimBE  (**common),
    "BDF2":           ElasticSimBDF2(**common),
}

viewer = Viewer2D(X, T)
viewer.add_pin_markers(X[pin_idx])
sel_pc, target_pc = viewer.add_handle_markers()
handle = MouseHandle2D(sims, sel_pc, target_pc, K_handle=1e4)

ui = TutorialUI(sims, handle=handle, show_material=False)

elastic_plot = RollingPlot("elastic energy", height=120.0)
kinetic_plot = RollingPlot("kinetic energy", height=120.0)
ui.on_reset(elastic_plot.clear)
ui.on_reset(kinetic_plot.clear)
ui.on_switch(elastic_plot.clear)
ui.on_switch(kinetic_plot.clear)


def callback():
    ui.draw()
    handle.update()
    ui.sim.step()
    viewer.refresh(ui.sim.U)
    handle.refresh_markers()

    # explicit per-component energies (sim.energy lumps them)
    elastic_plot.push(float(energies.macklin_mueller_neo_hookean_energy_x(
        ui.sim.U, J, ui.sim.mu, ui.sim.lam, vol)))
    v_flat = (ui.sim.V.flatten() if hasattr(ui.sim, "V")
              else (ui.sim.U - ui.sim.U_prev).flatten() / ui.sim.h)
    kinetic_plot.push(0.5 * float(v_flat @ (M @ v_flat)))

    psim.Text(f"vertices: {ui.sim.n}    triangles: {ui.sim.T.shape[0]}    pinned: {len(pin_idx)}")
    elastic_plot.draw()
    kinetic_plot.draw()


viewer.show(callback)
