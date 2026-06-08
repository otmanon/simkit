"""Tutorial 004 - Newton vs. Gradient Descent: solver choice matters.

A static (no time, no inertia) beam pinned on the left, with a fixed handle
vertex on the right edge soft-pinned to wherever you click. Each frame the
solver minimizes elastic + pin + handle energy from the current pose. The
dropdown swaps between Newton and Gradient Descent; the iteration count
needed to track the cursor tells the story.

The sim class is just a static potential -- energy = elastic + pin + handle --
exposed as ``energy`` / ``gradient`` / ``hessian`` for the solvers below.
"""
import numpy as np
import polyscope.imgui as psim
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers import gradient_descent, newton_solver
from simkit.volume import volume
import simkit.energies as energies

from utils import RollingPlot, Viewer2D, screen_to_world_2d, triangulated_grid


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=15, ny=5, width=2.0, height=0.6)
n, dim = X.shape

J   = deformation_jacobian(X, T)
vol = volume(X, T)

pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], n, 1e4)

# fixed handle vertex: middle of the right edge
right_edge = np.where(X[:, 0] >= X[:, 0].max() - 1e-6)[0]
mid_y = 0.5 * (X[:, 1].min() + X[:, 1].max())
HANDLE_VERTEX = int(right_edge[np.argmin(np.abs(X[right_edge, 1] - mid_y))])
K_HANDLE = 1e4


class ElasticSimStatic:
    """Static Macklin-Mueller Neo-Hookean with fixed pin + soft handle.

    No time, no kinetic. Each Newton/GD outer iteration minimizes
        E(x) = E_elastic(x) + 1/2 x^T Q_pin x + b_pin^T x
                              + 1/2 x^T Q_h   x + b_h^T x
    """

    def __init__(self, X, T, J, vol, mu, lam, Q_pin, b_pin):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol = J, vol
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.Q_pin, self.b_pin = Q_pin, b_pin

        D = self.n * self.dim
        self.U = X.copy()
        self.Q_h = sp.sparse.csc_matrix((D, D))
        self.b_h = np.zeros((D, 1))

    def energy(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        E_el  = float(energies.macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                 + float((self.b_pin.T @ xc)[0, 0]))
        E_h   = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                 + float((self.b_h.T @ xc)[0, 0]))
        return E_el + E_pin + E_h

    def gradient(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        g_el  = energies.macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin = self.Q_pin @ xc + self.b_pin
        g_h   = self.Q_h   @ xc + self.b_h
        return g_el + g_pin + g_h

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el = energies.macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H_el + self.Q_pin + self.Q_h

    def step(self):
        x_next = newton_solver(
            self.U.flatten().reshape(-1, 1),
            self.energy, self.gradient, self.hessian,
            max_iter=5, do_line_search=True,
        )
        self.U[:] = x_next.reshape(self.n, self.dim)


sim = ElasticSimStatic(X, T, J, vol, mu=1.0, lam=1.0, Q_pin=Q_pin, b_pin=b_pin)


def set_handle_target(target):
    """Rebuild the soft-pin matrices that constrain HANDLE_VERTEX to ``target``."""
    bI = np.array([HANDLE_VERTEX])
    y = np.asarray(target, dtype=float).reshape(1, sim.dim)
    sim.Q_h, sim.b_h = dirichlet_penalty(bI, y, sim.n, K_HANDLE)


handle_target = X[HANDLE_VERTEX].copy()
set_handle_target(handle_target)


# ---------- solver config (the point of this tutorial) --------------------
SOLVER_NAMES = ["Newton", "Gradient Descent"]
solver_choice = 0
MAX_ITERS = 5
NEWTON_LS = True
GD_LS = True
GD_STEP = 1.0


def build_solver():
    """Return a ``solve(x0) -> (x, info)`` closure for the current UI config."""
    if solver_choice == 0:
        return lambda x0: newton_solver(
            x0, sim.energy, sim.gradient, sim.hessian,
            max_iter=MAX_ITERS, do_line_search=NEWTON_LS, return_info=True,
        )
    return lambda x0: gradient_descent(
        x0, sim.energy, sim.gradient,
        max_iter=MAX_ITERS, do_line_search=GD_LS, step_size=GD_STEP, return_info=True,
    )


solver = build_solver()


# ---------- viewer ---------------------------------------------------------
viewer = Viewer2D(X, T)
viewer.add_pin_markers(X[pin_idx])
sel_pc, handle_pc = viewer.add_handle_markers()
sel_pc.set_enabled(True)
handle_pc.set_enabled(True)

energy_plot = RollingPlot("elastic energy", length=200, height=150.0)
iter_plot   = RollingPlot("iters this step", length=200, height=150.0, fmt="{:.1f}")


def callback():
    global solver, solver_choice, MAX_ITERS, NEWTON_LS, GD_LS, GD_STEP, handle_target

    changed_solver, solver_choice = psim.Combo("solver", solver_choice, SOLVER_NAMES)
    changed_iters,  MAX_ITERS     = psim.SliderInt("max iterations", MAX_ITERS, 1, 100)
    if solver_choice == 0:
        changed_ls, NEWTON_LS = psim.Checkbox("Newton line search", NEWTON_LS)
        changed_step = False
    else:
        changed_ls, GD_LS = psim.Checkbox("GD line search", GD_LS)
        changed_step, GD_STEP = psim.SliderFloat("GD step size", GD_STEP, 1e-4, 1e4)
    if changed_solver or changed_iters or changed_ls or changed_step:
        solver = build_solver()
        iter_plot.clear()

    if psim.Button("rerun solver from rest"):
        sim.U[:] = sim.X
        energy_plot.clear()
        iter_plot.clear()

    # click/drag (away from imgui) moves the fixed handle's target
    io = psim.GetIO()
    if not io.WantCaptureMouse and psim.IsMouseDown(0):
        handle_target = screen_to_world_2d(psim.GetMousePos())[: sim.dim]
        set_handle_target(handle_target)

    # minimize energy once
    x_col, info = solver(sim.U.flatten().reshape(-1, 1))
    sim.U[:] = x_col.reshape(sim.n, sim.dim)
    iters_run = info["iters"] + 1
    iter_plot.push(iters_run)

    viewer.refresh(sim.U)
    sel_pc.update_point_positions(sim.U[HANDLE_VERTEX].reshape(1, sim.dim))
    handle_pc.update_point_positions(handle_target.reshape(1, sim.dim))

    e_elastic = float(energies.macklin_mueller_neo_hookean_energy_x(sim.U, sim.J, sim.mu, sim.lam, sim.vol))
    energy_plot.push(e_elastic)

    psim.Text(f"vertices: {sim.n}    triangles: {sim.T.shape[0]}")
    psim.Text(f"solver: {SOLVER_NAMES[solver_choice]}    iters this step: {iters_run}/{MAX_ITERS}")
    psim.Text(f"handle vertex {HANDLE_VERTEX} -> target ({handle_target[0]:.2f}, {handle_target[1]:.2f})")
    psim.Text("click/drag in the viewport to move the target")
    energy_plot.draw()
    iter_plot.draw()


viewer.show(callback)
