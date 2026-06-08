"""Tutorial 008 - Contact against a movable ball.

A small deformable patch hangs from its top edge while a ball follows the
cursor and shoves it around through penalty-based contact springs. Sliders
expose Young's modulus ``E``, Poisson ratio ``nu`` (via ``log10(0.5 - nu)``
so near-incompressible is reachable), and the contact stiffness ``K``. Press
``Space`` to delete the ball + reset; left-click to bring it back.

Four sim classes, one per integrator (Static / BE / BDF2 / FE). Each class is
fully self-contained: pick one, read it top to bottom, no base class to chase.
"""
import numpy as np
import polyscope.imgui as psim
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.integrators import backward_euler, bdf2
from simkit.massmatrix import massmatrix
from simkit.solvers import newton_solver
from simkit.volume import volume
import simkit.energies as energies

from utils import (
    RollingPlot, Viewer2D, ball_mesh_2d, lame_from_E_nu,
    screen_to_world_2d, triangulated_grid,
)


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=12, ny=8, width=0.8, height=0.5)
X[:, 1] += 0.3
n, dim = X.shape

RHO = 1e3
J   = deformation_jacobian(X, T)
vol = volume(X, T)
M_n = massmatrix(X, T, rho=RHO)
M   = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()
f_g = gravity_force(X, T, a=-9.8, rho=RHO).reshape(-1, 1)

pin_idx = np.where(X[:, 1] >= X[:, 1].max() - 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], n, 5e6)

BALL_R = 0.12


# ============================================================================
# Shared potential
#   E_pot(x) = E_elastic + E_pin + E_sphere_contact(K, center, BALL_R) - f_g^T x
# Static minimizes E_pot. The others minimize E_pot + E_kin_<scheme>.
# ============================================================================

class ElasticSimStatic:
    """No time, no kinetic. Each frame, minimize the potential from rest."""

    def __init__(self, X, T, J, vol, M, M_n, f_g, mu, lam,
                 Q_pin, b_pin, K_contact, ball_center, ball_radius, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.K_contact = float(K_contact)
        self.ball_center = np.asarray(ball_center, dtype=float)
        self.ball_radius = float(ball_radius)
        self.h = float(h)
        self.U = X.copy()

        self._newton_iters = 8

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el      = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin     = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                     + float((self.b_pin.T @ xc)[0, 0]))
        E_grav    = -float((self.f_g.T @ xc)[0, 0])
        E_contact = float(energies.contact_springs_sphere_energy(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n))
        return E_el + E_pin + E_grav + E_contact

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el      = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin     = self.Q_pin @ xc + self.b_pin
        g_grav    = -self.f_g
        g_contact = energies.contact_springs_sphere_gradient(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return g_el + g_pin + g_grav + g_contact

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el      = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_contact = energies.contact_springs_sphere_hessian(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return H_el + self.Q_pin + H_contact

    def step(self):
        x_next = newton_solver(self.U.flatten().reshape(-1, 1), self.energy, self.gradient, self.hessian, max_iter=self._newton_iters, do_line_search=True)
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimBE:
    """Backward Euler: E(x) = E_pot(x) + E_kin_BE(x; U, U_prev, M, h)."""

    def __init__(self, X, T, J, vol, M, M_n, f_g, mu, lam,
                 Q_pin, b_pin, K_contact, ball_center, ball_radius, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.K_contact = float(K_contact)
        self.ball_center = np.asarray(ball_center, dtype=float)
        self.ball_radius = float(ball_radius)
        self.h = float(h)

        self.U      = X.copy()
        self.U_prev = X.copy()

        self._newton_iters = 8

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el      = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin     = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                     + float((self.b_pin.T @ xc)[0, 0]))
        E_grav    = -float((self.f_g.T @ xc)[0, 0])
        E_contact = float(energies.contact_springs_sphere_energy(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n))
        return E_el + E_pin + E_grav + E_contact

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el      = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin     = self.Q_pin @ xc + self.b_pin
        g_grav    = -self.f_g
        g_contact = energies.contact_springs_sphere_gradient(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return g_el + g_pin + g_grav + g_contact

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el      = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_contact = energies.contact_springs_sphere_hessian(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return H_el + self.Q_pin + H_contact

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
                 Q_pin, b_pin, K_contact, ball_center, ball_radius, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.K_contact = float(K_contact)
        self.ball_center = np.asarray(ball_center, dtype=float)
        self.ball_radius = float(ball_radius)
        self.h = float(h)

        self.U       = X.copy()
        self.U_prev  = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()

        self._newton_iters = 8

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el      = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin     = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                     + float((self.b_pin.T @ xc)[0, 0]))
        E_grav    = -float((self.f_g.T @ xc)[0, 0])
        E_contact = float(energies.contact_springs_sphere_energy(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n))
        return E_el + E_pin + E_grav + E_contact

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el      = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin     = self.Q_pin @ xc + self.b_pin
        g_grav    = -self.f_g
        g_contact = energies.contact_springs_sphere_gradient(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return g_el + g_pin + g_grav + g_contact

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el      = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_contact = energies.contact_springs_sphere_hessian(
            xn, self.K_contact, self.ball_center, self.ball_radius, M=self.M_n)
        return H_el + self.Q_pin + H_contact

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



# ---------- build sims, viewer, plots --------------------------------------
log_E = 5.0
log_nu_compl = -1.0
log_K_contact = 6.0
mu0, lam0 = lame_from_E_nu(E=10.0 ** log_E, nu=0.5 - 10.0 ** log_nu_compl)

ball_p = np.array([0.6, 0.3])
ball_active = True

common = dict(
    X=X, T=T, J=J, vol=vol, M=M, M_n=M_n, f_g=f_g,
    mu=mu0, lam=lam0, Q_pin=Q_pin, b_pin=b_pin,
    K_contact=10.0 ** log_K_contact, ball_center=ball_p, ball_radius=BALL_R,
    h=0.02,
)
sims = {
    "Static":         ElasticSimStatic(**common),
    "Backward Euler": ElasticSimBE    (**common),
    "BDF2":           ElasticSimBDF2  (**common)
}
names = list(sims.keys())
active_idx = 0
h = common["h"]

viewer = Viewer2D(X, T)
viewer.add_pin_markers(X[pin_idx], radius=0.012)
ball_X, ball_T = ball_mesh_2d(radius=BALL_R, n_segments=48)
ball_mesh = viewer.add_ball(ball_X, ball_T, ball_p)

elastic_plot = RollingPlot("elastic energy", height=120.0)
kinetic_plot = RollingPlot("kinetic energy", height=120.0)
contact_plot = RollingPlot("contact energy", height=120.0)


def reset_all():
    for s in sims.values():
        s.U[:] = s.X
        for attr in ("U_prev", "U_prev2", "U_prev3"):
            if hasattr(s, attr):
                getattr(s, attr)[:] = s.X
        if hasattr(s, "V"):
            s.V[:] = 0.0
    elastic_plot.clear(); kinetic_plot.clear(); contact_plot.clear()


def callback():
    global active_idx, h, log_E, log_nu_compl, log_K_contact, ball_p, ball_active

    if psim.Button("Reset"):
        reset_all()

    old_idx = active_idx
    changed_int, active_idx = psim.Combo("Integrator", active_idx, names)
    if changed_int:
        old, new = sims[names[old_idx]], sims[names[active_idx]]
        new.U[:] = old.U
        for attr in ("U_prev", "U_prev2", "U_prev3"):
            if hasattr(new, attr):
                getattr(new, attr)[:] = old.U
        if hasattr(new, "V"):
            new.V[:] = 0.0
        kinetic_plot.clear()

    changed_h, h = psim.SliderFloat("dt (h)", h, v_min=0.001, v_max=0.05)
    if changed_h:
        for s in sims.values():
            s.h = float(h)

    E_val = 10.0 ** log_E
    nu_val = 0.5 - 10.0 ** log_nu_compl
    changed_E, log_E = psim.SliderFloat(
        f"log10 E  (E = {E_val:.2e})", log_E, v_min=2.0, v_max=8.0)
    changed_nu, log_nu_compl = psim.SliderFloat(
        f"log10 (0.5 - nu)  (nu = {nu_val:.4f})",
        log_nu_compl, v_min=-4.0, v_max=-0.31)
    if changed_E or changed_nu:
        mu, lam = lame_from_E_nu(E=10.0 ** log_E, nu=0.5 - 10.0 ** log_nu_compl)
        for s in sims.values():
            s.mu[:] = mu; s.lam[:] = lam

    K_val = 10.0 ** log_K_contact
    changed_K, log_K_contact = psim.SliderFloat(
        f"log10 contact penalty  (K = {K_val:.2e})",
        log_K_contact, v_min=2.0, v_max=9.0)
    if changed_K and ball_active:
        for s in sims.values():
            s.K_contact = 10.0 ** log_K_contact

    # input: ball follows cursor; Space deletes; click respawns
    io = psim.GetIO()
    if psim.IsKeyPressed(psim.ImGuiKey_Space):
        ball_active = False
        reset_all()
        for s in sims.values():
            s.K_contact = 0.0
        ball_mesh.set_enabled(False)

    if not ball_active and psim.IsMouseClicked(0) and not io.WantCaptureMouse:
        ball_active = True
        for s in sims.values():
            s.K_contact = 10.0 ** log_K_contact
        ball_mesh.set_enabled(True)

    if ball_active and not io.WantCaptureMouse:
        ball_p = screen_to_world_2d(psim.GetMousePos()).astype(float).copy()
        for s in sims.values():
            s.ball_center[:] = ball_p

    # ---- step the active sim ----
    sim = sims[names[active_idx]]
    sim.step()

    viewer.refresh(sim.U)
    if ball_active:
        ball_mesh.update_vertex_positions(ball_X + ball_p[None, :])

    # explicit per-component energies for plotting
    elastic_plot.push(float(energies.macklin_mueller_neo_hookean_energy_x(
        sim.U, J, sim.mu, sim.lam, vol)))
    if hasattr(sim, "V"):
        v_flat = sim.V.flatten()
    elif hasattr(sim, "U_prev"):
        v_flat = (sim.U - sim.U_prev).flatten() / sim.h
    else:
        v_flat = np.zeros(sim.n * sim.dim)
    kinetic_plot.push(0.5 * float(v_flat @ (M @ v_flat)))
    contact_plot.push(float(energies.contact_springs_sphere_energy(
        sim.U, sim.K_contact, sim.ball_center, sim.ball_radius, M=sim.M_n)))

    psim.Text(f"vertices: {sim.n}    triangles: {sim.T.shape[0]}    pinned: {len(pin_idx)}")
    if ball_active:
        psim.Text(f"ball: ({ball_p[0]:.2f}, {ball_p[1]:.2f})  r = {BALL_R:.2f}")
        psim.Text("move mouse to push the ball; SPACE to delete + reset")
    else:
        psim.Text("ball deleted - click anywhere in scene to bring it back")
    elastic_plot.draw()
    kinetic_plot.draw()
    contact_plot.draw()


viewer.show(callback)
