"""Tutorial 006 - Interactive static deformation.

Beam pinned on the left. Left-click any other vertex and drag -- that vertex
becomes a soft-pinned handle that follows the cursor; release to let go. Each
frame the solver minimizes elastic + pin + handle energy from the current
pose, so the beam continuously catches up to wherever you're holding it.

No mass, no gravity, no time: this is just static equilibrium under a moving
boundary condition.
"""
import numpy as np
import polyscope.imgui as psim

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
from simkit.volume import volume
import scipy as sp
import simkit.energies as energies

from utils import MouseHandle2D, RollingPlot, Viewer2D, triangulated_grid


# ---------- mesh + precomputed operators -----------------------------------
X, T = triangulated_grid(nx=15, ny=5, width=2.0, height=0.6)
n, dim = X.shape

J   = deformation_jacobian(X, T)
vol = volume(X, T)

pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx], n, 1e4)


class ElasticSimStatic:
    """Static neo-Hookean with fixed pin + soft mouse handle.

    No kinetic, no time. Each frame: minimize
        E(x) = E_elastic + 1/2 x^T Q_pin x + b_pin^T x
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

        self._solver = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=5, do_line_search=True),
        )

    def energy(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        E_el  = float(energies.neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_pin = (0.5 * float((xc.T @ (self.Q_pin @ xc))[0, 0])
                 + float((self.b_pin.T @ xc)[0, 0]))
        E_h   = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                 + float((self.b_h.T @ xc)[0, 0]))
        return E_el + E_pin + E_h

    def gradient(self, x):
        xn = x.reshape(-1, self.dim); xc = x.reshape(-1, 1)
        g_el  = energies.neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_pin = self.Q_pin @ xc + self.b_pin
        g_h   = self.Q_h   @ xc + self.b_h
        return g_el + g_pin + g_h

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el = energies.neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H_el + self.Q_pin + self.Q_h

    def step(self):
        x_next = self._solver.solve(self.U.flatten().reshape(-1, 1))
        self.U[:] = x_next.reshape(self.n, self.dim)


sim = ElasticSimStatic(X, T, J, vol, mu=1.0, lam=1.0, Q_pin=Q_pin, b_pin=b_pin)

viewer = Viewer2D(X, T)
viewer.add_pin_markers(X[pin_idx])
sel_pc, target_pc = viewer.add_handle_markers()
handle = MouseHandle2D(sim, sel_pc, target_pc, K_handle=1e4)

energy_plot = RollingPlot("elastic energy", height=150.0)


def callback():
    if psim.Button("Reset"):
        sim.U[:] = sim.X
        handle._clear()
        energy_plot.clear()

    handle.update()
    sim.step()
    viewer.refresh(sim.U)
    handle.refresh_markers()

    e_el = float(energies.neo_hookean_energy_x(sim.U, J, sim.mu, sim.lam, vol))
    energy_plot.push(e_el)

    psim.Text(f"vertices: {sim.n}    triangles: {sim.T.shape[0]}    pinned: {len(pin_idx)}")
    psim.Text(handle.status)
    energy_plot.draw()


viewer.show(callback)
