"""Tutorial 009 - Contact against a static floor (2D).

A patch sits above a horizontal floor. Gravity pulls it down; the floor pushes
back via plane-contact penalty springs. Left-click + drag any vertex to throw
the patch around, then release to let it fall back.

Three sim classes, one per integrator. Each minimizes (or for FE: evaluates)
the same potential -- elastic + plane contact + mouse handle - gravity --
with a kinetic term layered on top.
"""
import numpy as np
import polyscope.imgui as psim
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.gravity_force import gravity_force
from simkit.integrators import backward_euler, bdf2
from simkit.massmatrix import massmatrix
from simkit.volume import volume
import simkit.energies as energies

from utils import (
    MouseHandle2D, TutorialUI, Viewer2D, lame_from_E_nu, triangulated_grid,
)


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=12, ny=8, width=0.8, height=0.5)
X[:, 1] += 1.0
n, dim = X.shape

RHO = 1e3
J   = deformation_jacobian(X, T)
vol = volume(X, T)
M_n = massmatrix(X, T, rho=RHO)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
f_g = gravity_force(X, T, a=-9.8, rho=RHO).reshape(-1, 1)

floor_y = -0.6
floor_p = np.array([0.0, floor_y])
floor_n = np.array([0.0, 1.0])
mu0, lam0 = lame_from_E_nu(E=1e5, nu=0.4)


# ============================================================================
# Three self-contained sim classes.
# E_pot(x) = E_elastic + E_floor + 1/2 x^T Q_h x + b_h^T x - f_g^T x
# ============================================================================

class ElasticSimBE:
    """Backward Euler."""

    def __init__(self, X, T, J, vol, M, M_n, f_g, mu, lam,
                 K_contact, p_floor, n_floor, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.K_contact = float(K_contact)
        self.p_floor, self.n_floor = p_floor, n_floor
        self.h = float(h)

        D = self.n * self.dim
        self.U      = X.copy()
        self.U_prev = X.copy()
        self.Q_h    = sp.sparse.csc_matrix((D, D))
        self.b_h    = np.zeros((D, 1))

        self._newton_iters = 5

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el    = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_floor = float(energies.contact_springs_plane_energy(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n))
        E_h     = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                   + float((self.b_h.T @ xc)[0, 0]))
        E_grav  = -float((self.f_g.T @ xc)[0, 0])
        return E_el + E_floor + E_h + E_grav

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el    = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_floor = energies.contact_springs_plane_gradient(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        g_h     = self.Q_h @ xc + self.b_h
        g_grav  = -self.f_g
        return g_el + g_floor + g_h + g_grav

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el    = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_floor = energies.contact_springs_plane_hessian(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        return H_el + H_floor + self.Q_h

    def step(self):
        x_next = backward_euler(
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.energy, self.gradient, self.hessian, self.M, self.h,
            max_iter=self._newton_iters, do_line_search=True)
        self.U_prev[:] = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimBDF2:
    """BDF2 (three history slots)."""

    def __init__(self, X, T, J, vol, M, M_n, f_g, mu, lam,
                 K_contact, p_floor, n_floor, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.K_contact = float(K_contact)
        self.p_floor, self.n_floor = p_floor, n_floor
        self.h = float(h)

        D = self.n * self.dim
        self.U       = X.copy()
        self.U_prev  = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()
        self.Q_h     = sp.sparse.csc_matrix((D, D))
        self.b_h     = np.zeros((D, 1))

        self._newton_iters = 5

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el    = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_floor = float(energies.contact_springs_plane_energy(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n))
        E_h     = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                   + float((self.b_h.T @ xc)[0, 0]))
        E_grav  = -float((self.f_g.T @ xc)[0, 0])
        return E_el + E_floor + E_h + E_grav

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el    = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_floor = energies.contact_springs_plane_gradient(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        g_h     = self.Q_h @ xc + self.b_h
        g_grav  = -self.f_g
        return g_el + g_floor + g_h + g_grav

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el    = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_floor = energies.contact_springs_plane_hessian(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        return H_el + H_floor + self.Q_h

    def step(self):
        x_next = bdf2(
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.energy, self.gradient, self.hessian, self.M, self.h,
            max_iter=self._newton_iters, do_line_search=True)
        self.U_prev3[:] = self.U_prev2
        self.U_prev2[:] = self.U_prev
        self.U_prev[:]  = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)




# ---------- build sims + viewer + UI --------------------------------------
common = dict(
    X=X, T=T, J=J, vol=vol, M=M, M_n=M_n, f_g=f_g,
    mu=mu0, lam=lam0,
    K_contact=1e5, p_floor=floor_p, n_floor=floor_n, h=0.02,
)
sims = {
    "Backward Euler": ElasticSimBE  (**common),
    "BDF2":           ElasticSimBDF2(**common),
}

viewer = Viewer2D(X, T)
viewer.add_floor_line(y=floor_y)
sel_pc, target_pc = viewer.add_handle_markers(radius=0.001)
handle = MouseHandle2D(sims, sel_pc, target_pc, K_handle=5e9)

ui = TutorialUI(sims, handle=handle, show_contact_K=True)


def callback():
    ui.draw()
    handle.update()
    ui.sim.step()
    viewer.refresh(ui.sim.U)
    handle.refresh_markers()

    psim.Text(f"vertices: {ui.sim.n}    triangles: {ui.sim.T.shape[0]}")


viewer.show(callback)
