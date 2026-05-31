"""Tutorial 003 - Resolution scaling: how mesh size affects solver cost.

Three cantilever beams (coarse / mid / fine) share physics and dt; the radio
button picks which one steps each frame, the others freeze. Each Newton
iteration is wall-clock timed and plotted so the per-iteration cost gap
between resolutions is visible.

The BDF2 sim class has the standard five methods, but the timing loop
(``newton_step_timed``) is unrolled inline below so you can see exactly where
the time goes: assemble g, assemble H, spsolve, line-search.
"""
import time

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import scipy as sp

from simkit.backtracking_line_search import backtracking_line_search
from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.massmatrix import massmatrix
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
from simkit.volume import volume
import simkit.energies as energies

from utils import BLUE, RollingPlot, init_2d_scene, triangulated_grid


# ---------- shared params --------------------------------------------------
MU = 1.0e3
LAM = 1.0e3
RHO = 1.0
G_ACC = -9.8
DT = 0.02
NEWTON_ITERS = 5
HISTORY_LEN = 200


class ElasticSimBDF2:
    """Macklin-Mueller Neo-Hookean beam stepped with BDF2; pin matrices supplied as inputs.

    No handle, no contact -- this tutorial measures the cost of resolution
    against a fixed potential, not interactivity.
    """

    def __init__(self, X, T, J, vol, M, M_n, f_g, mu, lam, Q_pin, b_pin, h):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin
        self.h = float(h)

        self.U       = X.copy()
        self.U_prev  = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()

    def energy(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        E_el   = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin  = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                  + float((self.b_pin.T @ xc)[0, 0]))
        E_grav = -float((self.f_g.T @ xc)[0, 0])
        E_kin  = float(energies.kinetic_energy_bdf2(
            xc,
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.M, self.h,
        ))
        return E_el + E_pin + E_grav + E_kin

    def gradient(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        g_el   = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin  = self.Q_pin @ xc + self.b_pin
        g_grav = -self.f_g
        g_kin  = energies.kinetic_gradient_bdf2(
            xc,
            self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1),
            self.U_prev2.reshape(-1, 1), self.U_prev3.reshape(-1, 1),
            self.M, self.h,
        )
        return g_el + g_pin + g_grav + g_kin

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el  = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_kin = energies.kinetic_hessian_bdf2(self.M, self.h)
        return H_el + self.Q_pin + H_kin

    def step(self):
        x_next = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=NEWTON_ITERS, do_line_search=True),
        ).solve(self.U.flatten().reshape(-1, 1))
        self.U_prev3[:] = self.U_prev2
        self.U_prev2[:] = self.U_prev
        self.U_prev[:]  = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


def make_beam(nx, ny, y_offset):
    """Build one cantilever beam: mesh, precomputed operators, left-edge pin."""
    X, T = triangulated_grid(nx=nx, ny=ny, width=2.0, height=0.3)
    X[:, 1] += y_offset
    J   = deformation_jacobian(X, T)
    vol = volume(X, T)
    M_n = massmatrix(X, T, rho=RHO)
    M   = sp.sparse.kron(M_n, sp.sparse.eye(2)).tocsc()
    f_g = gravity_force(X, T, a=G_ACC, rho=RHO).reshape(-1, 1)
    pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
    Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], X.shape[0], 1e6)
    sim = ElasticSimBDF2(X, T, J, vol, M, M_n, f_g, MU, LAM, Q_pin, b_pin, DT)
    return sim, pin_idx


def newton_step_timed(sim, dt):
    """One BDF2 step with per-iteration wall-clock timing.

    The pin and kinetic Hessian pieces don't depend on ``x``, so we hoist them
    out of the inner loop. The remaining per-iter work -- ``macklin_mueller_neo_hookean_hessian``,
    sparse solve, line search -- is what the rolling plot measures.
    """
    x_curr  = sim.U.flatten().reshape(-1, 1)
    H_const = sim.Q_pin + energies.kinetic_hessian_bdf2(sim.M, dt)

    def Etot(x):
        return sim.energy(x.flatten())

    x = x_curr.copy()
    iter_times_ms = []
    for _ in range(NEWTON_ITERS):
        t0 = time.perf_counter()
        g = sim.gradient(x.flatten())
        H_el = energies.macklin_mueller_neo_hookean_hessian_x(
            x.reshape(-1, sim.dim), sim.J, sim.mu, sim.lam, sim.vol, psd=True)
        H = H_el + H_const
        dx = sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
        alpha, _, _ = backtracking_line_search(Etot, x, g, dx)
        x = x + alpha * dx
        iter_times_ms.append((time.perf_counter() - t0) * 1000.0)
        if np.linalg.norm(alpha * dx) < 1e-6:
            break

    sim.U_prev3[:] = sim.U_prev2
    sim.U_prev2[:] = sim.U_prev
    sim.U_prev[:]  = sim.U
    sim.U[:] = x.reshape(sim.n, sim.dim)
    return iter_times_ms


# ---------- build three beams + scene --------------------------------------
init_2d_scene(camera_distance=6.0)

beams = []
for name, nx, ny, y_off, color in [
    ("coarse", 8,  3,  +0.9, np.array([153, 216, 201]) / 255),
    ("mid",    20, 5,   0.0, np.array([158, 188, 218]) / 255),
    ("fine",   60, 10, -0.9, np.array([251, 180, 174]) / 255),
]:
    sim, pin_idx = make_beam(nx, ny, y_off)
    beams.append({
        "name": name,
        "sim": sim,
        "pin_idx": pin_idx,
        "mesh": ps.register_surface_mesh(
            f"{name} beam", sim.U, sim.T, material="flat", color=color, edge_width=1.5),
        "pin_pc": ps.register_point_cloud(
            f"{name} pinned", sim.U[pin_idx], radius=0.012, material="flat", color=BLUE),
        "plot": RollingPlot(
            f"{name} ms/Newton-iter", length=HISTORY_LEN, height=90.0, fmt="{:.2f} ms"),
    })

active_idx = 1
paused = False


def callback():
    global active_idx, paused

    if psim.Button("Reset all sims"):
        for b in beams:
            sim = b["sim"]
            sim.U[:] = sim.X
            sim.U_prev[:]  = sim.X
            sim.U_prev2[:] = sim.X
            sim.U_prev3[:] = sim.X
            b["plot"].clear()
            b["mesh"].update_vertex_positions(sim.U)
            b["pin_pc"].update_point_positions(sim.U[b["pin_idx"]])
    psim.SameLine()
    if psim.Button("Resume" if paused else "Pause"):
        paused = not paused

    psim.Text(f"BDF2  dt={DT}  Newton iters/step={NEWTON_ITERS}    {'PAUSED' if paused else 'running'}")
    psim.Separator()
    psim.Text("active sim (only one runs at a time):")
    for i, b in enumerate(beams):
        if psim.RadioButton(
            f"{b['name']}  ({b['sim'].n} verts, {b['sim'].T.shape[0]} tris)", active_idx == i
        ):
            active_idx = i
    psim.Separator()

    active = beams[active_idx]
    if not paused:
        for ms in newton_step_timed(active["sim"], DT):
            active["plot"].push(ms)
        active["mesh"].update_vertex_positions(active["sim"].U)
        active["pin_pc"].update_point_positions(active["sim"].U[active["pin_idx"]])

    for i, b in enumerate(beams):
        vals = b["plot"].values
        last = vals[-1] if vals else 0.0
        avg = (sum(vals) / len(vals)) if vals else 0.0
        tag = "*" if i == active_idx else " "
        psim.Text(
            f"{tag} {b['name']:>6}: verts={b['sim'].n:>5}  tris={b['sim'].T.shape[0]:>5}"
            f"   last={last:7.2f} ms   avg={avg:7.2f} ms"
        )
        b["plot"].draw()


ps.set_user_callback(callback)
ps.show()
