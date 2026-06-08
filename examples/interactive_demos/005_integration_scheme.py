"""Tutorial 005 - Integration schemes: Backward Euler vs. BDF2 vs. Forward Euler.

A cantilever beam under gravity, sagging. The dropdown swaps between three
integrators that share the same potential (elastic + pin + gravity). Each sim
class exposes only the *potential* energy / gradient / hessian; the time
stepping is delegated to ``simkit.integrators``: the implicit schemes
(``backward_euler``, ``bdf2``) layer on the inertial term and run a Newton
solve internally, while ``forward_euler`` skips the implicit minimize and takes
an explicit acceleration step instead.

Each integrator is written out as its own class. Read one top to bottom and the
rest follow by analogy.
"""
import numpy as np
import polyscope.imgui as psim
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.integrators import backward_euler, bdf2, forward_euler
from simkit.massmatrix import massmatrix
from simkit.volume import volume
import simkit.energies as energies

from utils import (
    RollingPlot, Viewer2D, lame_from_E_nu, triangulated_grid,
)


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=12, ny=4, width=2.0, height=0.4)
n, dim = X.shape

RHO = 1.0
J   = deformation_jacobian(X, T)
vol = volume(X, T)
M_n = massmatrix(X, T, rho=RHO)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
f_g = gravity_force(X, T, a=-2.0, rho=RHO).reshape(-1, 1)

pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], n, 1e4)

mu0, lam0 = lame_from_E_nu(E=1.0, nu=0.45)


# ============================================================================
# E_pot(x) = E_elastic + 1/2 x^T Q_pin x + b_pin^T x - f_g^T x
# Three classes differ only in the kinetic term layered on top.
# ============================================================================

class ElasticSimBE:
    """Backward Euler."""

    def __init__(self, X, T, J, vol, M, f_g, mu, lam, Q_pin, b_pin, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.f_g = J, vol, M, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)

        self.U      = X.copy()
        self.U_prev = X.copy()

        self._newton_iters = 5

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el   = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin  = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                  + float((self.b_pin.T @ xc)[0, 0]))
        E_grav = -float((self.f_g.T @ xc)[0, 0])
        return E_el + E_pin + E_grav

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el   = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin  = self.Q_pin @ xc + self.b_pin
        g_grav = -self.f_g
        return g_el + g_pin + g_grav

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el  = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H_el + self.Q_pin

    def step(self):
        # backward_euler adds the inertial term to this potential and Newton-solves.
        x_next = backward_euler(
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.energy, self.gradient, self.hessian, self.M, self.h,
            max_iter=self._newton_iters, do_line_search=True)
        self.U_prev[:] = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimBDF2:
    """BDF2 (3 history slots, second-order accurate, less damping than BE)."""

    def __init__(self, X, T, J, vol, M, f_g, mu, lam, Q_pin, b_pin, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.f_g = J, vol, M, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)

        self.U       = X.copy()
        self.U_prev  = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()

        self._newton_iters = 5

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el   = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin  = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                  + float((self.b_pin.T @ xc)[0, 0]))
        E_grav = -float((self.f_g.T @ xc)[0, 0])
        return E_el + E_pin + E_grav

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el   = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin  = self.Q_pin @ xc + self.b_pin
        g_grav = -self.f_g
        return g_el + g_pin + g_grav

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el  = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H_el + self.Q_pin

    def step(self):
        # bdf2 reconstructs both velocities from the 4-level position history.
        x_next = bdf2(
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.energy, self.gradient, self.hessian, self.M, self.h,
            max_iter=self._newton_iters, do_line_search=True)
        self.U_prev3[:] = self.U_prev2
        self.U_prev2[:] = self.U_prev
        self.U_prev[:]  = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimFE:
    """Forward Euler (explicit). step does M a = -grad E_pot, then U += h V, V += h a.
    Pinned dofs get a hard clamp because the soft-pin spring is not enough for FE
    to stay stable at reasonable timesteps."""

    def __init__(self, X, T, J, vol, M, f_g, mu, lam, Q_pin, b_pin, h, pin_idx):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.f_g = J, vol, M, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)
        self.pin_idx = pin_idx

        self.U = X.copy()
        self.V = np.zeros_like(X)

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el   = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin  = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                  + float((self.b_pin.T @ xc)[0, 0]))
        E_grav = -float((self.f_g.T @ xc)[0, 0])
        return E_el + E_pin + E_grav

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el   = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin  = self.Q_pin @ xc + self.b_pin
        g_grav = -self.f_g
        return g_el + g_pin + g_grav

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H_el + self.Q_pin

    def step(self):
        # forward_euler reads the force off self.gradient and lumps the mass.
        x_next, v_next = forward_euler(
            self.U.reshape(-1, 1), self.V.reshape(-1, 1),
            self.gradient, self.M, self.h)
        self.U[:] = x_next.reshape(self.n, self.dim)
        self.V[:] = v_next.reshape(self.n, self.dim)
        if len(self.pin_idx):
            self.U[self.pin_idx] = self.X[self.pin_idx]
            self.V[self.pin_idx] = 0.0


# ---------- build sims -----------------------------------------------------
h0 = 0.005
common = dict(X=X, T=T, J=J, vol=vol, M=M, f_g=f_g, mu=mu0, lam=lam0,
              Q_pin=Q_pin, b_pin=b_pin, h=h0)
sims = {
    "Backward Euler": ElasticSimBE  (**common),
    "BDF2":           ElasticSimBDF2(**common),
    "Forward Euler":  ElasticSimFE  (**common, pin_idx=pin_idx),
}
names = list(sims.keys())
active_idx = 0
ym = 1.0
POISSON = 0.45
h = h0


# ---------- viewer ---------------------------------------------------------
viewer = Viewer2D(X, T)
viewer.add_pin_markers(X[pin_idx])

elastic_plot = RollingPlot("elastic energy", height=100.0)
kinetic_plot = RollingPlot("kinetic energy", height=100.0)
total_plot   = RollingPlot("total energy",   height=100.0)


def clear_history(new_idx, old_idx):
    """When the integrator changes, copy state from the old sim into the new
    and reset every history slot to that shared starting pose."""
    old = sims[names[old_idx]]
    new = sims[names[new_idx]]
    new.U[:] = old.U
    for attr in ("U_prev", "U_prev2", "U_prev3"):
        if hasattr(new, attr):
            getattr(new, attr)[:] = old.U
    if hasattr(new, "V"):
        new.V[:] = 0.0


def apply_material():
    mu, lam = lame_from_E_nu(E=ym, nu=POISSON)
    for s in sims.values():
        s.mu[:] = mu
        s.lam[:] = lam


def apply_h():
    for s in sims.values():
        s.h = float(h)


def callback():
    global active_idx, h, ym

    if psim.Button("Reset"):
        for s in sims.values():
            s.U[:] = s.X
            for attr in ("U_prev", "U_prev2", "U_prev3"):
                if hasattr(s, attr):
                    getattr(s, attr)[:] = s.X
            if hasattr(s, "V"):
                s.V[:] = 0.0
        elastic_plot.clear(); kinetic_plot.clear(); total_plot.clear()

    old_idx = active_idx
    changed_int, active_idx = psim.Combo("Integrator", active_idx, names)
    if changed_int:
        clear_history(active_idx, old_idx)
        elastic_plot.clear(); kinetic_plot.clear(); total_plot.clear()

    changed_h, h = psim.SliderFloat("dt (h)", h, v_min=0.0005, v_max=0.05)
    if changed_h:
        apply_h()

    changed_ym, ym = psim.SliderFloat("Young's modulus", ym, v_min=0.05, v_max=5.0)
    if changed_ym:
        apply_material()

    sim = sims[names[active_idx]]
    sim.step()
    viewer.refresh(sim.U)

    # Pull the per-component energies out explicitly for plotting; sim.energy
    # would lump them together.
    x_flat = sim.U.flatten()
    e_el   = float(energies.macklin_mueller_neo_hookean_energy_x(sim.U, J, sim.mu, sim.lam, vol))
    v_flat = sim.V.flatten() if hasattr(sim, "V") else (sim.U - sim.U_prev).flatten() / sim.h
    e_kin  = 0.5 * float(v_flat @ (M @ v_flat))
    e_grav = -float((f_g.T @ x_flat.reshape(-1, 1))[0, 0])
    elastic_plot.push(e_el)
    kinetic_plot.push(e_kin)
    total_plot.push(e_el + e_kin + e_grav)

    psim.Text(f"vertices: {sim.n}    triangles: {sim.T.shape[0]}    pinned: {len(pin_idx)}")
    psim.Text(f"integrator: {names[active_idx]}")
    elastic_plot.draw()
    kinetic_plot.draw()
    total_plot.draw()


viewer.show(callback)
