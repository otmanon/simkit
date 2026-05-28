import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import screen_to_world_2d, triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
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

selected_index = None
handle_target = None
Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
b_handle = np.zeros((n * dim, 1))

# ---------- optimization ---------------------------------------------------
NEWTON_ITERS = 5

def rebuild_handle():
    global Q_handle, b_handle
    if selected_index is None or handle_target is None:
        Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
        b_handle = np.zeros((n * dim, 1))
        return
    bI = np.array([selected_index])
    y = handle_target.reshape(1, dim)
    Q_handle, b_handle = dirichlet_penalty(bI, y, n, K_HANDLE)


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


solver = NewtonSolver(
    energy_func=total_energy,
    gradient_func=total_gradient,
    hessian_func=total_hessian,
    p=NewtonSolverParams(max_iter=NEWTON_ITERS, do_line_search=True),
)


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
red = np.array([0.85, 0.2, 0.2])
blue = np.array([0.2, 0.4, 0.85])

ENERGY_HISTORY_LEN = 200
energy_history = []


# ---------- callback --------------------------------------------------------
def callback():
    global selected_index, handle_target, U

    if psim.Button("Reset"):
        U[:] = X
        selected_index = None
        handle_target = None
        rebuild_handle()
        energy_history.clear()
        sel_pc.set_enabled(False)
        handle_pc.set_enabled(False)

    win_pos = psim.GetMousePos()

    # left-click near a vertex to grab it as the new handle
    if psim.IsMouseClicked(0):
        pos = screen_to_world_2d(win_pos)
        distance = np.linalg.norm(U - pos.reshape(-1, 2), axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] < 0.15:
            selected_index = nearest
            handle_target = U[selected_index].copy()
            rebuild_handle()
            sel_pc.set_enabled(True)
            handle_pc.set_enabled(True)

    # while held: drag the handle target
    if selected_index is not None and psim.IsMouseDown(0):
        handle_target = screen_to_world_2d(win_pos)[:dim].astype(float)
        rebuild_handle()

    # release to drop the handle
    if psim.IsMouseReleased(0):
        selected_index = None
        handle_target = None
        rebuild_handle()
        sel_pc.set_enabled(False)
        handle_pc.set_enabled(False)

    # minimize energy
    x_col = solver.solve(U.flatten().reshape(-1, 1))
    U[:] = x_col.reshape(n, dim)

    mesh.update_vertex_positions(U)
    pc.update_point_positions(U)
    pin_pc.update_point_positions(U[pin_idx])
    if selected_index is not None:
        handle_pc.update_point_positions(handle_target.reshape(1, dim))
        sel_pc.update_point_positions(U[selected_index].reshape(1, dim))

    e_elastic = elastic_energy(U.flatten())
    energy_history.append(e_elastic)
    del energy_history[:-ENERGY_HISTORY_LEN]

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}")
    psim.Text(f"pinned (left edge): {len(pin_idx)}")
    if selected_index is not None:
        psim.Text(f"dragging vertex {selected_index} -> ({handle_target[0]:.2f}, {handle_target[1]:.2f})")
    else:
        psim.Text("left-click a vertex and drag to move it; release to drop")

    psim.PlotLines(
        f"elastic energy (last {ENERGY_HISTORY_LEN})",
        energy_history,
        overlay_text=f"E_elastic: {e_elastic:.3f}",
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
sel_pc = ps.register_point_cloud("selected", np.zeros((1, dim)), radius=0.028, material="flat", color=red, enabled=False)
handle_pc = ps.register_point_cloud("handle target", np.zeros((1, dim)), radius=0.028, material="flat", color=red, enabled=False)

ps.set_do_default_mouse_interaction(False)
ps.set_user_callback(callback)
ps.show()
