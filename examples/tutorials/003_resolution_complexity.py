"""Cantilever beam under gravity at three mesh resolutions (BDF2).

Three independent cantilever beams (coarse / mid / fine) all run in lock-step
in the same polyscope window. Each step uses BDF2 implicit time integration
and a hand-rolled Newton loop so we can time each Newton iteration in
isolation. The wall-clock cost of one Newton iteration is plotted on a rolling
basis per beam so the resolution scaling is visible live.
"""

import time

import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.backtracking_line_search import backtracking_line_search
import simkit.energies as energies


# ---------- shared sim params -----------------------------------------------
MU = 1.0e3
LAM = 1.0e3
RHO = 1.0
G_ACC = -9.8
K_PIN = 1e6
DT = 0.02
NEWTON_ITERS = 5
HISTORY_LEN = 200

BEAM_WIDTH = 2.0
BEAM_HEIGHT = 0.3
Y_GAP = 0.45  # vertical spacing between stacked rest poses


# ---------- one beam --------------------------------------------------------
class BeamSim:
    """Single cantilever beam: rest mesh, dynamics state, and a timed BDF2 step."""

    def __init__(self, name, nx, ny, y_offset, color):
        self.name = name
        self.color = color
        X, T = triangulated_grid(nx=nx, ny=ny, width=BEAM_WIDTH, height=BEAM_HEIGHT)
        X[:, 1] += y_offset
        self.X = X
        self.T = T
        self.n, self.dim = X.shape

        self.mu = np.full((T.shape[0], 1), MU)
        self.lam = np.full((T.shape[0], 1), LAM)
        self.J = deformation_jacobian(X, T)
        self.vol = volume(X, T)
        M_n = massmatrix(X, T, rho=RHO)
        self.M = sp.sparse.kron(M_n, sp.sparse.eye(self.dim)).tocsc()

        self.f_g = gravity_force(X, T, a=G_ACC, rho=RHO).reshape(-1, 1)

        # cantilever: pin the left edge of the rest pose
        self.pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
        self.Q_pin, self.b_pin = dirichlet_penalty(self.pin_idx, X[self.pin_idx, :], self.n, K_PIN)

        # BDF2 history (positions only; velocities reconstructed inside the
        # kinetic energy from these levels).
        self.U = X.copy()
        self.U_prev = X.copy()
        self.U_prev2 = X.copy()
        self.U_prev3 = X.copy()

        # constant pieces of the Newton system
        self.kin_H = energies.kinetic_hessian_bdf2(self.M, DT)
        self.H_const = self.Q_pin + self.kin_H  # elastic Hessian gets added per-iter

        # rolling per-Newton-iteration wall-clock (milliseconds)
        self.iter_times_ms = []

        self.mesh = None
        self.pin_pc = None

    def reset(self):
        self.U[:] = self.X
        self.U_prev[:] = self.X
        self.U_prev2[:] = self.X
        self.U_prev3[:] = self.X
        self.iter_times_ms.clear()

    def step(self):
        """One BDF2 step. Each Newton iteration's wall time is appended to the rolling buffer."""
        x_curr = self.U.flatten().reshape(-1, 1)
        x_prev = self.U_prev.flatten().reshape(-1, 1)
        x_prev2 = self.U_prev2.flatten().reshape(-1, 1)
        x_prev3 = self.U_prev3.flatten().reshape(-1, 1)

        def Etot(x):
            xc = x.reshape(-1, 1)
            E_el = float(energies.neo_hookean_energy_x(xc.reshape(-1, self.dim), self.J, self.mu, self.lam, self.vol))
            E_pin = 0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0]) + float((self.b_pin.T @ xc)[0, 0])
            E_g = -float((self.f_g.T @ xc)[0, 0])
            E_k = energies.kinetic_energy_bdf2(xc, x_curr, x_prev, x_prev2, x_prev3, self.M, DT)
            return E_el + E_pin + E_g + E_k

        x = x_curr.copy()
        for _ in range(NEWTON_ITERS):
            t0 = time.perf_counter()

            x_pos = x.reshape(-1, self.dim)
            g = (
                energies.neo_hookean_gradient_x(x_pos, self.J, self.mu, self.lam, self.vol)
                + self.Q_pin @ x + self.b_pin
                - self.f_g
                + energies.kinetic_gradient_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, self.M, DT)
            )
            H = energies.neo_hookean_hessian_x(x_pos, self.J, self.mu, self.lam, self.vol, psd=True) + self.H_const
            dx = sp.sparse.linalg.spsolve(H.tocsc(), -g).reshape(-1, 1)
            alpha, _, _ = backtracking_line_search(Etot, x, g, dx)
            x = x + alpha * dx

            self.iter_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if np.linalg.norm(alpha * dx) < 1e-6:
                break

        del self.iter_times_ms[:-HISTORY_LEN]

        self.U_prev3[:] = self.U_prev2
        self.U_prev2[:] = self.U_prev
        self.U_prev[:] = self.U
        self.U[:] = x.reshape(self.n, self.dim)


# ---------- build the three beams ------------------------------------------
beams = [
    BeamSim("coarse", nx=8,  ny=3,  y_offset=+0.9, color=np.array([153, 216, 201]) / 255),
    BeamSim("mid",    nx=20, ny=5,  y_offset=+0.0, color=np.array([158, 188, 218]) / 255),
    BeamSim("fine",   nx=60, ny=10, y_offset=-0.9, color=np.array([251, 180, 174]) / 255),
]

black = np.array([0.0, 0.0, 0.0])
blue = np.array([0.2, 0.4, 0.85])

# index of the beam currently being simulated (mutually exclusive)
active_idx = 1
paused = False


# ---------- callback --------------------------------------------------------
def callback():
    global active_idx, paused

    if psim.Button("Reset all sims"):
        for b in beams:
            b.reset()
            b.mesh.update_vertex_positions(b.U)
            b.pin_pc.update_point_positions(b.U[b.pin_idx])
    psim.SameLine()
    if psim.Button("Resume" if paused else "Pause"):
        paused = not paused

    psim.Text(f"BDF2  dt={DT}  newton iters/step={NEWTON_ITERS}  history={HISTORY_LEN}    {'PAUSED' if paused else 'running'}")
    psim.Separator()
    psim.Text("active sim (only one runs at a time):")
    for i, b in enumerate(beams):
        if psim.RadioButton(f"{b.name}  ({b.n} verts, {b.T.shape[0]} tris)", active_idx == i):
            active_idx = i
    psim.Separator()

    # only step the selected beam; the others freeze in place
    active = beams[active_idx]
    if not paused:
        active.step()
        active.mesh.update_vertex_positions(active.U)
        active.pin_pc.update_point_positions(active.U[active.pin_idx])

    for i, b in enumerate(beams):
        last_ms = b.iter_times_ms[-1] if b.iter_times_ms else 0.0
        mean_ms = (sum(b.iter_times_ms) / len(b.iter_times_ms)) if b.iter_times_ms else 0.0
        tag = "*" if i == active_idx else " "
        psim.Text(f"{tag} {b.name:>6}: verts={b.n:>5}  tris={b.T.shape[0]:>5}   last={last_ms:7.2f} ms   avg={mean_ms:7.2f} ms")
        psim.PlotLines(
            f"{b.name} ms/Newton-iter",
            b.iter_times_ms,
            overlay_text=f"{last_ms:.2f} ms",
            graph_size=(0.0, 90.0),
        )


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.look_at(np.array([0, 0, 6]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")

for b in beams:
    b.mesh = ps.register_surface_mesh(f"{b.name} beam", b.U, b.T, material="flat", color=b.color, edge_width=1.5)
    b.pin_pc = ps.register_point_cloud(f"{b.name} pinned", b.U[b.pin_idx], radius=0.012, material="flat", color=blue)

ps.set_user_callback(callback)
ps.show()
