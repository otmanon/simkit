import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import screen_to_world_2d, triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
from simkit.solvers.GradientDescentSolver import GradientDescentSolver, GradientDescentSolverParams
import simkit.energies as energies


# ---------- mesh & material -------------------------------------------------
X, T = triangulated_grid(nx=15, ny=5, width=2.0, height=0.6)
U = X.copy()
n, dim = X.shape

mu = np.full((T.shape[0], 1), 1.0)
lam = np.full((T.shape[0], 1), 1.0)

J = deformation_jacobian(X, T)
vol = volume(X, T)

# ---------- pinning ---------------------------------------------------------
K_PIN = 1e4       # soft-pin stiffness for the fixed left edge
K_HANDLE = 1e4    # soft-pin stiffness for the draggable handle

pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx, :], n, K_PIN)

# fixed handle at the middle of the right edge
right_edge = np.where(X[:, 0] >= X[:, 0].max() - 1e-6)[0]
mid_y = 0.5 * (X[:, 1].min() + X[:, 1].max())
selected_index = int(right_edge[np.argmin(np.abs(X[right_edge, 1] - mid_y))])
handle_target = X[selected_index].copy().astype(float)
Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
b_handle = np.zeros((n * dim, 1))

# ---------- optimization ---------------------------------------------------
MAX_ITERS = 5
NEWTON_LINE_SEARCH = True
GD_LINE_SEARCH = True
GD_STEP_SIZE = 1.0
SOLVER_NAMES = ["Newton", "Gradient Descent"]
solver_choice = 0  # 0 = Newton, 1 = Gradient Descent

def rebuild_handle():
    global Q_handle, b_handle
    bI = np.array([selected_index])
    y = handle_target.reshape(1, dim)
    Q_handle, b_handle = dirichlet_penalty(bI, y, n, K_HANDLE)


rebuild_handle()


def elastic_energy(x_flat):
    x = x_flat.reshape(-1, dim)
    return float(energies.neo_hookean_energy_x(x, J, mu, lam, vol))


def total_energy(x_flat):
    xc = x_flat.reshape(-1, 1)
    E_pin = 0.5 * float((xc.T @ (Q_pin @ xc))[0, 0]) + float((b_pin.T @ xc)[0, 0])
    E_han = 0.5 * float((xc.T @ (Q_handle @ xc))[0, 0]) + float((b_handle.T @ xc)[0, 0])
    return elastic_energy(x_flat) + E_pin + E_han


def total_gradient(x_flat):
    x = x_flat.reshape(-1, dim)
    g = energies.neo_hookean_gradient_x(x, J, mu, lam, vol)
    xc = x_flat.reshape(-1, 1)
    g = g + Q_pin @ xc + b_pin + Q_handle @ xc + b_handle
    return g


def total_hessian(x_flat):
    x = x_flat.reshape(-1, dim)
    H = energies.neo_hookean_hessian_x(x, J, mu, lam, vol, psd=True)
    return H + Q_pin + Q_handle


def build_solver():
    if solver_choice == 0:
        return NewtonSolver(
            energy_func=total_energy,
            gradient_func=total_gradient,
            hessian_func=total_hessian,
            p=NewtonSolverParams(max_iter=MAX_ITERS, do_line_search=NEWTON_LINE_SEARCH),
        )
    else:
        return GradientDescentSolver(
            energy_func=total_energy,
            gradient_func=total_gradient,
            p=GradientDescentSolverParams(
                max_iter=MAX_ITERS,
                do_line_search=GD_LINE_SEARCH,
                step_size=GD_STEP_SIZE,
            ),
        )


solver = build_solver()


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
red = np.array([0.85, 0.2, 0.2])
blue = np.array([0.2, 0.4, 0.85])

ENERGY_HISTORY_LEN = 200
energy_history = []
iter_history = []


# ---------- callback --------------------------------------------------------
def callback():
    global handle_target, U, solver, solver_choice, MAX_ITERS, NEWTON_LINE_SEARCH, GD_LINE_SEARCH, GD_STEP_SIZE

    # solver controls
    changed_solver, solver_choice = psim.Combo("solver", solver_choice, SOLVER_NAMES)
    changed_iters, MAX_ITERS = psim.SliderInt("max iterations", MAX_ITERS, 1, 100)
    if solver_choice == 0:
        changed_ls, NEWTON_LINE_SEARCH = psim.Checkbox("newton line search", NEWTON_LINE_SEARCH)
        changed_step = False
    else:
        changed_ls, GD_LINE_SEARCH = psim.Checkbox("gradient descent line search", GD_LINE_SEARCH)
        changed_step, GD_STEP_SIZE = psim.SliderFloat("gradient descent step size", GD_STEP_SIZE, 1e-4, 1e4)
    if changed_solver or changed_iters or changed_ls or changed_step:
        solver = build_solver()
        iter_history.clear()

    if psim.Button("rerun solver from rest"):
        U[:] = X
        energy_history.clear()
        iter_history.clear()

    # click/drag (away from UI) updates the target position the handle is pulled toward
    io = psim.GetIO()
    if not io.WantCaptureMouse and psim.IsMouseDown(0):
        win_pos = psim.GetMousePos()
        handle_target = screen_to_world_2d(win_pos)[:dim].astype(float)
        rebuild_handle()

    # minimize energy
    x_col, info = solver.solve(U.flatten().reshape(-1, 1), return_info=True)
    U[:] = x_col.reshape(n, dim)
    iters_run = info['iters'] + 1
    iter_history.append(iters_run)
    del iter_history[:-ENERGY_HISTORY_LEN]

    mesh.update_vertex_positions(U)
    pc.update_point_positions(U)
    pin_pc.update_point_positions(U[pin_idx])
    handle_pc.update_point_positions(handle_target.reshape(1, dim))
    sel_pc.update_point_positions(U[selected_index].reshape(1, dim))

    e_elastic = elastic_energy(U.flatten())
    energy_history.append(e_elastic)
    del energy_history[:-ENERGY_HISTORY_LEN]

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}")
    psim.Text(f"pinned (left edge): {len(pin_idx)}")
    psim.Text(f"solver: {SOLVER_NAMES[solver_choice]}    iters this step: {iters_run}/{MAX_ITERS}")
    psim.Text(f"handle vertex {selected_index} -> target ({handle_target[0]:.2f}, {handle_target[1]:.2f})")
    psim.Text("click/drag in the viewport to move the target; press the button to restart from rest")

    psim.PlotLines(
        f"elastic energy (last {ENERGY_HISTORY_LEN})",
        energy_history,
        overlay_text=f"E_elastic: {e_elastic:.3f}",
        graph_size=(0.0, 150.0),
    )

    psim.PlotLines(
        f"solver iterations (last {ENERGY_HISTORY_LEN})",
        [float(v) for v in iter_history],
        overlay_text=f"iters: {iters_run}    avg: {(sum(iter_history) / max(len(iter_history), 1)):.2f}",
        scale_min=0.0,
        scale_max=float(MAX_ITERS),
        graph_size=(0.0, 150.0),
    )


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.look_at(np.array([0, 0, 5]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")

mesh = ps.register_surface_mesh("mesh", U, T, material="flat", color=light_green, edge_width=2)
pc = ps.register_point_cloud("vertices", U, radius=0.012, material="flat", color=black)
pin_pc = ps.register_point_cloud("pinned", U[pin_idx], radius=0.022, material="flat", color=blue)
sel_pc = ps.register_point_cloud("selected", U[selected_index].reshape(1, dim), radius=0.028, material="flat", color=red)
handle_pc = ps.register_point_cloud("handle target", handle_target.reshape(1, dim), radius=0.028, material="flat", color=red)

ps.set_do_default_mouse_interaction(False)
ps.set_user_callback(callback)
ps.show()
