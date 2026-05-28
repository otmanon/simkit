import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import screen_to_world_2d, triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import simkit.energies as energies


# ---------- mesh & material -------------------------------------------------
X, T = triangulated_grid(nx=15, ny=5, width=2.0, height=0.6)
n, dim = X.shape

mu = np.full((T.shape[0], 1), 1.0)
lam = np.full((T.shape[0], 1), 1.0)
rho = 1.0

J = deformation_jacobian(X, T)
vol = volume(X, T)
M = sp.sparse.kron(massmatrix(X, T, rho=rho), sp.sparse.eye(dim)).tocsc()

# ---------- pinning ---------------------------------------------------------
K_PIN = 1e4
K_HANDLE = 1e4
pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx, :], n, K_PIN)

selected_index = None
handle_target = None
Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
b_handle = np.zeros((n * dim, 1))

# ---------- dynamics state --------------------------------------------------
U = X.copy()
U_prev = X.copy()
U_prev2 = X.copy()
U_prev3 = X.copy()
V = np.zeros((n, dim))

INTEGRATORS = ["Backward Euler", "BDF2", "Forward Euler"]
integrator_idx = 0
h = 0.02
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


def reset_history():
    """Snap velocity to zero, keep current pose."""
    U_prev[:] = U
    U_prev2[:] = U
    U_prev3[:] = U
    V[:] = 0.0


# ---------- energies --------------------------------------------------------
def elastic_E(x_flat):
    return float(energies.neo_hookean_energy_x(x_flat.reshape(-1, dim), J, mu, lam, vol))


def elastic_g(x_flat):
    return energies.neo_hookean_gradient_x(x_flat.reshape(-1, dim), J, mu, lam, vol)


def elastic_H(x_flat):
    return energies.neo_hookean_hessian_x(x_flat.reshape(-1, dim), J, mu, lam, vol, psd=True)


def pin_E(x_flat):
    xc = x_flat.reshape(-1, 1)
    Ep = 0.5 * float((xc.T @ (Q_pin @ xc))[0, 0]) + float((b_pin.T @ xc)[0, 0])
    Eh = 0.5 * float((xc.T @ (Q_handle @ xc))[0, 0]) + float((b_handle.T @ xc)[0, 0])
    return Ep + Eh


def pin_g(x_flat):
    xc = x_flat.reshape(-1, 1)
    return Q_pin @ xc + b_pin + Q_handle @ xc + b_handle


def pin_H():
    return Q_pin + Q_handle


# ---------- integrators -----------------------------------------------------
def implicit_step(make_kin_E, make_kin_g, kin_H):
    """Drive simkit's NewtonSolver to minimize elastic + pin + kinetic."""
    def Etot(x):
        return elastic_E(x) + pin_E(x) + make_kin_E(x.reshape(-1, 1))

    def gtot(x):
        return elastic_g(x) + pin_g(x) + make_kin_g(x.reshape(-1, 1))

    def Htot(x):
        return elastic_H(x) + pin_H() + kin_H

    solver = NewtonSolver(
        Etot, gtot, Htot,
        NewtonSolverParams(max_iter=NEWTON_ITERS, do_line_search=True),
    )
    return solver.solve(U.flatten().reshape(-1, 1)).reshape(n, dim)


def step_be():
    x_curr = U.flatten().reshape(-1, 1)
    x_prev = U_prev.flatten().reshape(-1, 1)
    return implicit_step(
        lambda x: energies.kinetic_energy_be(x, x_curr, x_prev, M, h),
        lambda x: energies.kinetic_gradient_be(x, x_curr, x_prev, M, h),
        energies.kinetic_hessian_be(M, h),
    )


def step_bdf2():
    x_curr = U.flatten().reshape(-1, 1)
    x_prev = U_prev.flatten().reshape(-1, 1)
    x_prev2 = U_prev2.flatten().reshape(-1, 1)
    x_prev3 = U_prev3.flatten().reshape(-1, 1)
    return implicit_step(
        lambda x: energies.kinetic_energy_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h),
        lambda x: energies.kinetic_gradient_bdf2(x, x_curr, x_prev, x_prev2, x_prev3, M, h),
        energies.kinetic_hessian_bdf2(M, h),
    )


def step_fe():
    """Explicit forward Euler: x_{k+1} = x_k + h v_k; v_{k+1} = v_k + h M^{-1} f."""
    x = U.flatten()
    f = -(elastic_g(x) + pin_g(x))            # (n*d, 1) net force
    a = (f.flatten() / M.diagonal()).reshape(n, dim)
    U_new = U + h * V
    V_new = V + h * a
    return U_new, V_new


def advance():
    global U, U_prev, U_prev2, U_prev3, V
    if integrator_idx == 0:
        U_next = step_be()
        V[:] = (U_next - U) / h
    elif integrator_idx == 1:
        U_next = step_bdf2()
        V[:] = (U_next - U) / h
    else:
        U_next, V_new = step_fe()
        V[:] = V_new
    U_prev3[:] = U_prev2
    U_prev2[:] = U_prev
    U_prev[:] = U
    U[:] = U_next


# ---------- diagnostics -----------------------------------------------------
def kinetic_value():
    v = V.flatten()
    return 0.5 * float(v @ (M @ v))


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
red = np.array([0.85, 0.2, 0.2])
blue = np.array([0.2, 0.4, 0.85])

HISTORY_LEN = 200
elastic_history = []
kinetic_history = []


# ---------- callback --------------------------------------------------------
def callback():
    global selected_index, handle_target, integrator_idx, h

    # --- UI controls ---
    if psim.Button("Reset"):
        U[:] = X
        reset_history()
        selected_index = None
        handle_target = None
        rebuild_handle()
        elastic_history.clear()
        kinetic_history.clear()
        sel_pc.set_enabled(False)
        handle_pc.set_enabled(False)

    changed_int, integrator_idx = psim.Combo("Integrator", integrator_idx, INTEGRATORS)
    if changed_int:
        reset_history()
        elastic_history.clear()
        kinetic_history.clear()
    _, h = psim.SliderFloat("dt (h)", h, v_min=0.001, v_max=0.05)

    # --- mouse interaction: left-click + drag to grab a vertex ---
    win_pos = psim.GetMousePos()
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

    if selected_index is not None and psim.IsMouseDown(0):
        handle_target = screen_to_world_2d(win_pos)[:dim].astype(float)
        rebuild_handle()

    if psim.IsMouseReleased(0):
        selected_index = None
        handle_target = None
        rebuild_handle()
        sel_pc.set_enabled(False)
        handle_pc.set_enabled(False)

    # --- step the simulation ---
    advance()

    # --- update visualization ---
    mesh.update_vertex_positions(U)
    pc.update_point_positions(U)
    pin_pc.update_point_positions(U[pin_idx])
    if selected_index is not None:
        handle_pc.update_point_positions(handle_target.reshape(1, dim))
        sel_pc.update_point_positions(U[selected_index].reshape(1, dim))

    # --- diagnostics ---
    e_el = elastic_E(U.flatten())
    e_kin = kinetic_value()
    elastic_history.append(e_el)
    kinetic_history.append(e_kin)
    del elastic_history[:-HISTORY_LEN]
    del kinetic_history[:-HISTORY_LEN]

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}    pinned: {len(pin_idx)}")
    if selected_index is not None:
        psim.Text(f"dragging vertex {selected_index} -> ({handle_target[0]:.2f}, {handle_target[1]:.2f})")
    else:
        psim.Text("left-click a vertex and drag to move it; release to drop")

    psim.PlotLines(
        f"elastic energy (last {HISTORY_LEN})",
        elastic_history,
        overlay_text=f"E_elastic: {e_el:.3f}",
        graph_size=(0.0, 120.0),
    )
    psim.PlotLines(
        f"kinetic energy (last {HISTORY_LEN})",
        kinetic_history,
        overlay_text=f"E_kinetic: {e_kin:.3f}",
        graph_size=(0.0, 120.0),
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
