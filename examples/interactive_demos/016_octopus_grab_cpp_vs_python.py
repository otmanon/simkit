"""016 -- Octopus grab: neo-Hookean elastodynamics, C++ vs Python backend.

Loads the octopus tetrahedral mesh from the ``data/`` submodule, pins its
belly, and lets you poke it:

- **left-click** anywhere on the octopus to create a soft penalty handle on
  the patch of vertices around the hit point,
- **drag** to pull the patch (the target moves in the camera-facing plane),
- **release** to destroy the handle.

The ``Backend`` radio in the UI flips simkit between the C++ core
(``simkit._cpp``) and the pure NumPy implementation, so you can feel the
difference in the step-time plot. Every step is one backward-Euler
timestep with one (or more, see the slider) Newton iteration on the true neo-Hookean energy
(``simkit.energies.neo_hookean``), a Dirichlet penalty on the pinned belly,
the mouse handle, and optional gravity.

Run from anywhere::

    python examples/interactive_demos/016_octopus_grab_cpp_vs_python.py
    python examples/interactive_demos/016_octopus_grab_cpp_vs_python.py --headless 4

``--headless N`` skips polyscope, grabs the tip of one arm with a scripted
pull, steps ``N`` times under each backend and prints the timings. Useful to
check the simulation (and the backend switch) without a display.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy as sp

import simkit
import simkit.energies as energies
from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.energies.kinetic import be_target, kinetic_energy_be, kinetic_gradient_be, kinetic_hessian_be
from simkit.massmatrix import massmatrix
from simkit.normalize_and_center import normalize_and_center
from simkit.read_mesh import read_mesh
from simkit.backtracking_line_search import backtracking_line_search
from simkit.sparse_solve import sparse_solve
from simkit.volume import volume

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import lame_from_E_nu  # noqa: E402

RHO = 1000.0
PIN_RADIUS = 0.12     # belly: vertices this close to the centroid are pinned
K_PIN = 1.0e7
BACKENDS = ["cpp", "python"]


# --------------------------------------------------------------------------- #
# Mesh                                                                        #
# --------------------------------------------------------------------------- #
def load_octopus():
    """Octopus tet mesh, centred and scaled to a bounding box of width 2."""
    path = os.path.join(simkit.filesystem.get_data_directory(), "3d", "octopus", "octopus.mesh")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Initialise the data submodule first: git submodule update --init data"
        )
    X, T, _ = read_mesh(path)
    X = normalize_and_center(X)
    return X, T


def belly_pins(X, radius=PIN_RADIUS):
    """Indices of the vertices within ``radius`` of the centroid (the belly)."""
    return np.where(np.linalg.norm(X, axis=1) < radius)[0]


# --------------------------------------------------------------------------- #
# Simulation                                                                  #
# --------------------------------------------------------------------------- #
class OctopusSim:
    """Backward-Euler neo-Hookean elastodynamics with pins, handle, gravity.

    The energy is ``E_el + E_pin + E_handle + E_grav``; ``backward_euler`` adds
    the kinetic term and runs Newton with line search. ``Q_h`` / ``b_h`` are the
    mouse-handle penalty, written by ``GrabHandle`` and zero when nothing is
    grabbed.
    """

    def __init__(self, X, T, E=1.0e5, nu=0.4, h=0.02, newton_iters=1, gravity=False):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        D = self.n * self.dim
        self.J = deformation_jacobian(X, T)
        self.vol = volume(X, T)
        self.M_n = massmatrix(X, T, rho=RHO)
        self.M = sp.sparse.kron(self.M_n, sp.sparse.eye(self.dim)).tocsc()
        self.f_g = gravity_force(X, T, a=-9.8, rho=RHO).reshape(-1, 1)
        self.gravity = bool(gravity)
        mu, lam = lame_from_E_nu(E, nu)
        self.mu = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.h = float(h)
        self.newton_iters = int(newton_iters)

        self.pin_idx = belly_pins(X)
        self.Q_pin, self.b_pin = dirichlet_penalty(self.pin_idx, X[self.pin_idx], self.n, K_PIN)
        self.Q_h = sp.sparse.csc_matrix((D, D))
        self.b_h = np.zeros((D, 1))

        self.U = X.copy()
        self.U_prev = X.copy()
        self.last_step_time = 0.0
        self.last_timing = {}
        self.solve_method = "-"

    def set_material(self, E, nu):
        mu, lam = lame_from_E_nu(E, nu)
        self.mu[:] = mu
        self.lam[:] = lam

    def reset(self):
        self.U[:] = self.X
        self.U_prev[:] = self.X

    def energy(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        E_el = energies.neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol)
        E_pin = 0.5 * (xc.T @ (self.Q_pin @ xc)).item() + (self.b_pin.T @ xc).item()
        E_h = 0.5 * (xc.T @ (self.Q_h @ xc)).item() + (self.b_h.T @ xc).item()
        E_g = -(self.f_g.T @ xc).item() if self.gravity else 0.0
        E = E_el + E_pin + E_h + E_g
        # An inverted tet makes log(det F) NaN; report it as infinite energy so the
        # line search rejects that step instead of comparing against NaN.
        return E if np.isfinite(E) else np.inf

    def gradient(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        g = energies.neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g = g + self.Q_pin @ xc + self.b_pin + self.Q_h @ xc + self.b_h
        if self.gravity:
            g = g - self.f_g
        return g

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H = energies.neo_hookean_hessian_x(xn, self.J, self.mu, self.lam, self.vol, psd=True)
        return H + self.Q_pin + self.Q_h

    def step(self):
        """One backward-Euler timestep; returns the wall-clock seconds it took.

        This is ``simkit.integrators.backward_euler`` written out, so the demo
        can time each piece of a Newton iteration separately (assembled
        gradient, assembled Hessian, linear solve, line search) and report
        which direct solver handled the system. The only behavioural
        difference is the starting point: the inertial guess ``x + h v`` can
        invert tets when the handle yanks an arm, and neo-Hookean is undefined
        there, so in that case Newton starts from the current pose instead.
        """
        t_start = time.perf_counter()
        x_curr = self.U.reshape(-1, 1)
        x_prev = self.U_prev.reshape(-1, 1)
        M, h = self.M, self.h
        timing = dict(gradient=0.0, hessian=0.0, solve=0.0, line_search=0.0)

        def energy(x):
            return self.energy(x) + kinetic_energy_be(x, x_curr, x_prev, M, h)

        x = be_target(x_curr, x_prev, h)
        if not np.isfinite(self.energy(x)):
            x = x_curr.copy()
        for _ in range(self.newton_iters):
            t0 = time.perf_counter()
            g = self.gradient(x) + kinetic_gradient_be(x, x_curr, x_prev, M, h)
            t1 = time.perf_counter()
            H = self.hessian(x) + kinetic_hessian_be(M, h)
            t2 = time.perf_counter()
            dx, self.solve_method = sparse_solve(H, -g, return_method=True)
            t3 = time.perf_counter()
            alpha, x, _ = backtracking_line_search(energy, x, g, dx)
            t4 = time.perf_counter()
            timing["gradient"] += t1 - t0
            timing["hessian"] += t2 - t1
            timing["solve"] += t3 - t2
            timing["line_search"] += t4 - t3
            if np.linalg.norm(alpha * dx) < 1e-6:
                break
        self.U_prev[:] = self.U
        self.U[:] = x.reshape(self.n, self.dim)
        timing["total"] = time.perf_counter() - t_start
        self.last_timing = timing
        self.last_step_time = timing["total"]
        return self.last_step_time


# --------------------------------------------------------------------------- #
# Handle: a soft penalty on a patch of vertices that follows the mouse        #
# --------------------------------------------------------------------------- #
class GrabHandle:
    """Soft penalty handle created on click, moved on drag, destroyed on release.

    ``grab(hit)`` selects the vertices within ``selection_radius`` of the hit
    point and anchors them at their current positions; ``move(target)``
    translates the whole patch's target positions by the mouse displacement;
    ``release()`` removes the penalty. The mouse plumbing (picking, ray-plane
    intersection) lives in the polyscope loop below so this class is usable
    headless.
    """

    def __init__(self, sim, K_handle=2.0e4, selection_radius=0.08):
        self.sim = sim
        self.K_handle = float(K_handle)
        self.selection_radius = float(selection_radius)
        self.idx = None
        self.P0 = None
        self.target0 = None
        self.target = None
        self.status = "left-click the octopus to grab; drag to pull; release to let go"

    @property
    def active(self):
        return self.idx is not None

    def grab(self, hit):
        hit = np.asarray(hit, dtype=float).reshape(1, -1)
        d = np.linalg.norm(self.sim.U - hit, axis=1)
        idx = np.where(d < self.selection_radius)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(d))])
        self.idx = idx
        self.P0 = self.sim.U[idx].copy()
        self.target0 = hit.reshape(-1).copy()
        self.target = self.target0.copy()
        self._write()
        self.status = f"grabbed {idx.size} vertices"

    def move(self, target):
        if not self.active:
            return
        self.target = np.asarray(target, dtype=float).reshape(-1).copy()
        self._write()

    def release(self):
        if self.active:
            self.status = "released; click again to grab"
        self.idx = None
        self.P0 = None
        self.target0 = None
        self.target = None
        D = self.sim.n * self.sim.dim
        self.sim.Q_h = sp.sparse.csc_matrix((D, D))
        self.sim.b_h = np.zeros((D, 1))

    def targets(self):
        """Current target positions of the grabbed patch, (k, dim)."""
        return self.P0 + (self.target - self.target0)[None, :]

    def _write(self):
        Q, b = dirichlet_penalty(self.idx, self.targets(), self.sim.n, self.K_handle)
        self.sim.Q_h = Q
        self.sim.b_h = b


# --------------------------------------------------------------------------- #
# Interactive demo (polyscope)                                                #
# --------------------------------------------------------------------------- #
def main_interactive(args):
    import polyscope as ps
    import polyscope.imgui as psim

    from utils import BLUE, LIGHT_GREEN, RED, RollingPlot, _cursor_ray, _pick_world_point, _ray_plane

    X, T = load_octopus()
    sim = OctopusSim(X, T, h=args.h, newton_iters=args.newton_iters, gravity=args.gravity)
    handle = GrabHandle(sim)

    if not simkit.cpp_available():
        print("simkit._cpp is not built: only the Python backend is available (pip install -e .)")
    state = dict(
        backend=0 if simkit.backend() == "cpp" else 1,
        running=True,
        log_E=5.0,
        log_nu_compl=-1.0,
        own_mouse=True,
        plane_p=None,
        plane_n=None,
        ema={b: None for b in BACKENDS},
        parts={b: {} for b in BACKENDS},
    )
    simkit.set_backend(BACKENDS[state["backend"]])
    plot = RollingPlot("step time (ms)", length=120, fmt="{:.0f}")

    ps.init()
    ps.remove_all_structures()
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    ps.set_ground_plane_mode("tile_reflection")
    ps.look_at(np.array([0.0, 1.6, 3.2]), np.array([0.0, 0.0, 0.0]))
    ps.set_do_default_mouse_interaction(False)
    mesh = ps.register_volume_mesh("octopus", X, tets=T, color=LIGHT_GREEN, interior_color=LIGHT_GREEN,
                                   edge_width=0.0, material="flat")
    ps.register_point_cloud("pinned belly", X[sim.pin_idx], radius=0.006, material="flat", color=BLUE)
    sel_pc = ps.register_point_cloud("handle patch", np.zeros((1, 3)), radius=0.008, material="flat",
                                     color=RED, enabled=False)
    tgt_pc = ps.register_point_cloud("handle target", np.zeros((1, 3)), radius=0.014, material="flat",
                                     color=RED, enabled=False)

    def mouse():
        win_pos = psim.GetMousePos()
        if psim.IsMouseClicked(0) and not psim.GetIO().want_capture_mouse:
            hit = _pick_world_point(win_pos)
            if hit is None:
                handle.release()
                handle.status = "click landed on empty space"
            else:
                handle.grab(hit)
                state["plane_p"] = hit.copy()
                state["plane_n"] = np.asarray(ps.get_view_camera_parameters().get_look_dir(), dtype=float)
                sel_pc.set_enabled(True)
                tgt_pc.set_enabled(True)
        if handle.active and psim.IsMouseDown(0):
            ray_o, ray_d = _cursor_ray(win_pos)
            target = _ray_plane(ray_o, ray_d, state["plane_p"], state["plane_n"])
            if target is not None and np.all(np.isfinite(target)):
                handle.move(target)
        if psim.IsMouseReleased(0) and handle.active:
            handle.release()
            sel_pc.set_enabled(False)
            tgt_pc.set_enabled(False)

    def ui():
        psim.Text(f"octopus: {sim.n} vertices, {sim.T.shape[0]} tets, {sim.pin_idx.size} pinned")
        psim.Separator()
        psim.Text("Backend")
        psim.SameLine()
        for i, name in enumerate(["C++ (simkit._cpp)", "Python (NumPy)"]):
            if i == 0 and not simkit.cpp_available():
                psim.TextDisabled("C++ not built")
            elif psim.RadioButton(name, state["backend"] == i):
                state["backend"] = i
                simkit.set_backend(BACKENDS[i])
                plot.clear()
            if i == 0:
                psim.SameLine()
        psim.Text(f"{'running means (ms / step)':28s}" + "".join(f"{b:>10s}" for b in BACKENDS))
        for key, label in [("gradient", "assembled gradient"), ("hessian", "assembled Hessian (PSD)"),
                           ("solve", "linear solve"), ("line_search", "line search"), ("total", "whole step")]:
            row = f"  {label:26s}"
            for b in BACKENDS:
                v = state["parts"][b].get(key)
                row += f"{v * 1e3:10.0f}" if v is not None else f"{'-':>10s}"
            psim.Text(row)
        psim.Text(f"  linear solver: {sim.solve_method}  (cpp: Cholesky LLT, then LDLT; python: SuperLU)")
        plot.draw()
        psim.Separator()
        _, state["running"] = psim.Checkbox("Run", state["running"])
        psim.SameLine()
        if psim.Button("Step once"):
            do_step()
        psim.SameLine()
        if psim.Button("Reset"):
            sim.reset()
            handle.release()
        changed_mode, state["own_mouse"] = psim.Checkbox("Click-to-grab (uncheck to orbit the camera)", state["own_mouse"])
        if changed_mode:
            ps.set_do_default_mouse_interaction(not state["own_mouse"])
            handle.release()
        _, sim.newton_iters = psim.SliderInt("Newton iterations / step", sim.newton_iters, v_min=1, v_max=6)
        _, sim.h = psim.SliderFloat("dt (h)", sim.h, v_min=0.002, v_max=0.05)
        _, sim.gravity = psim.Checkbox("Gravity", sim.gravity)
        E_val = 10.0 ** state["log_E"]
        nu_val = 0.5 - 10.0 ** state["log_nu_compl"]
        cE, state["log_E"] = psim.SliderFloat(f"log10 Young's E (E = {E_val:.1e})", state["log_E"], v_min=3.0, v_max=7.0)
        cN, state["log_nu_compl"] = psim.SliderFloat(f"log10 (0.5 - nu) (nu = {nu_val:.3f})", state["log_nu_compl"], v_min=-3.0, v_max=-0.31)
        if cE or cN:
            sim.set_material(10.0 ** state["log_E"], 0.5 - 10.0 ** state["log_nu_compl"])
        _, handle.K_handle = psim.SliderFloat("handle stiffness K", handle.K_handle, v_min=1.0e3, v_max=1.0e9, power=10.0)
        _, handle.selection_radius = psim.SliderFloat("grab radius", handle.selection_radius, v_min=0.01, v_max=0.4)
        psim.Text(handle.status)

    def do_step():
        dt = sim.step()
        b = BACKENDS[state["backend"]]
        state["ema"][b] = dt if state["ema"][b] is None else 0.9 * state["ema"][b] + 0.1 * dt
        parts = state["parts"][b]
        for key, v in sim.last_timing.items():
            parts[key] = v if parts.get(key) is None else 0.9 * parts[key] + 0.1 * v
        plot.push(dt * 1e3)

    def callback():
        ui()
        if state["own_mouse"]:
            mouse()
        if state["running"]:
            do_step()
        mesh.update_vertex_positions(sim.U)
        if handle.active:
            sel_pc.update_point_positions(sim.U[handle.idx])
            tgt_pc.update_point_positions(handle.target.reshape(1, 3))

    ps.set_user_callback(callback)
    ps.show()


# --------------------------------------------------------------------------- #
# Headless check                                                              #
# --------------------------------------------------------------------------- #
def main_headless(n_steps, args):
    X, T = load_octopus()
    print(f"octopus: {X.shape[0]} vertices, {T.shape[0]} tets; C++ available: {simkit.cpp_available()}")
    tip = int(np.argmax(np.linalg.norm(X, axis=1)))  # tip of the longest arm
    pull = np.array([0.0, 0.15, 0.0])
    results = {}
    for b in BACKENDS:
        if b == "cpp" and not simkit.cpp_available():
            continue
        simkit.set_backend(b)
        sim = OctopusSim(X, T, h=args.h, newton_iters=args.newton_iters, gravity=args.gravity)
        handle = GrabHandle(sim)
        handle.grab(X[tip])
        times = []
        for k in range(n_steps):
            handle.move(X[tip] + pull * (k + 1) / n_steps)
            times.append(sim.step())
            disp = np.linalg.norm(sim.U - X, axis=1).max()
            tm = sim.last_timing
            print(f"  [{b:6s}] step {k + 1}/{n_steps}: {times[-1] * 1e3:6.0f} ms "
                  f"(gradient {tm['gradient'] * 1e3:4.0f}, hessian {tm['hessian'] * 1e3:5.0f}, "
                  f"solve {tm['solve'] * 1e3:5.0f} via {sim.solve_method}, line search {tm['line_search'] * 1e3:4.0f}), "
                  f"max displacement {disp:.3f}")
        handle.release()
        assert np.all(np.isfinite(sim.U)), "simulation produced non-finite positions"
        results[b] = (np.mean(times), sim.U.copy())
    for b, (mean_t, _) in results.items():
        print(f"{b:7s}: {mean_t * 1e3:7.0f} ms / step on average")
    if len(results) == 2:
        diff = np.abs(results["cpp"][1] - results["python"][1]).max()
        print(f"max |U_cpp - U_python| after {n_steps} steps: {diff:.2e}")
        print(f"speedup: {results['python'][0] / results['cpp'][0]:.1f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--headless", type=int, metavar="N", default=0, help="run N scripted steps without a window")
    ap.add_argument("--h", type=float, default=0.02, help="timestep")
    ap.add_argument("--newton-iters", type=int, default=1, help="Newton iterations per step")
    ap.add_argument("--gravity", action="store_true", help="start with gravity enabled")
    args = ap.parse_args()
    if args.headless > 0:
        main_headless(args.headless, args)
    else:
        main_interactive(args)
