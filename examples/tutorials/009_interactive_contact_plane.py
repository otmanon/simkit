import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import screen_to_world_2d, triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import simkit.energies as energies


# ---------- mesh & material -------------------------------------------------
X, T = triangulated_grid(nx=12, ny=8, width=0.8, height=0.5)
X[:, 1] += 1.0  # lift the rest pose above the floor so it has somewhere to fall
n, dim = X.shape

# Material: Young's modulus E and Poisson ratio nu are the user-facing knobs.
# Stored Lame parameters mu, lam are derived (plane strain).
log_E = 5.0          # log10(E),         E_init = 1e5
log_nu_compl = -1.0  # log10(0.5 - nu),  nu_init = 0.4
rho = 1.0e3


def lame_from_E_nu(E: float, nu: float):
    mu_s = E / (2.0 * (1.0 + nu))
    lam_s = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu_s, lam_s


mu = np.full((T.shape[0], 1), 0.0)
lam = np.full((T.shape[0], 1), 0.0)


def update_material():
    E = 10.0 ** log_E
    nu = 0.5 - 10.0 ** log_nu_compl
    mu_s, lam_s = lame_from_E_nu(E, nu)
    mu[:] = mu_s
    lam[:] = lam_s


update_material()

J = deformation_jacobian(X, T)
vol = volume(X, T)
M_n = massmatrix(X, T, rho=rho)                      # (n, n) vertex masses
M = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()  # (n*d, n*d)

# ---------- gravity (constant body force -> linear potential) ---------------
g_acc = -9.8
f_g = gravity_force(X, T, a=g_acc, rho=rho).reshape(-1, 1)  # (n*d, 1)

# ---------- floor (penalty contact) ----------------------------------------
log_K_contact = 5.0   # log10(K_CONTACT)
K_CONTACT = 10.0 ** log_K_contact
floor_y = -0.6
floor_p = np.array([0.0, floor_y])
floor_n = np.array([0.0, 1.0])

# ---------- mouse handle ---------------------------------------------------
K_HANDLE = 5e9
selected_index = None
handle_target = None
Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
b_handle = np.zeros((n * dim, 1))


def rebuild_handle():
    global Q_handle, b_handle
    if selected_index is None or handle_target is None:
        Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
        b_handle = np.zeros((n * dim, 1))
        return
    bI = np.array([selected_index])
    y = handle_target.reshape(1, dim)
    Q_handle, b_handle = dirichlet_penalty(bI, y, n, K_HANDLE)


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


def reset_history():
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


def contact_E(x_flat):
    return float(energies.contact_springs_plane_energy(x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n))


def contact_g(x_flat):
    return energies.contact_springs_plane_gradient(x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n)


def contact_H(x_flat):
    return energies.contact_springs_plane_hessian(x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n)


def handle_E(x_flat):
    xc = x_flat.reshape(-1, 1)
    return 0.5 * float((xc.T @ (Q_handle @ xc))[0, 0]) + float((b_handle.T @ xc)[0, 0])


def handle_g(x_flat):
    xc = x_flat.reshape(-1, 1)
    return Q_handle @ xc + b_handle


def gravity_E(x_flat):
    return -float((f_g.T @ x_flat.reshape(-1, 1))[0, 0])


# ---------- integrators -----------------------------------------------------
def implicit_step(make_kin_E, make_kin_g, kin_H):
    def Etot(x):
        return elastic_E(x) + contact_E(x) + handle_E(x) + gravity_E(x) + make_kin_E(x.reshape(-1, 1))

    def gtot(x):
        return (
            elastic_g(x)
            + contact_g(x)
            + handle_g(x)
            - f_g
            + make_kin_g(x.reshape(-1, 1))
        )

    def Htot(x):
        return elastic_H(x) + contact_H(x) + Q_handle + kin_H

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
    f = -(elastic_g(x) + contact_g(x) + handle_g(x)) + f_g
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


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
red = np.array([0.85, 0.2, 0.2])
gray = np.array([0.4, 0.4, 0.4])


# ---------- callback --------------------------------------------------------
def callback():
    global selected_index, handle_target, integrator_idx, h
    global log_E, log_nu_compl, log_K_contact, K_CONTACT

    # --- UI controls ---
    if psim.Button("Reset"):
        U[:] = X
        reset_history()
        selected_index = None
        handle_target = None
        rebuild_handle()
        sel_pc.set_enabled(False)
        handle_pc.set_enabled(False)

    changed_int, integrator_idx = psim.Combo("Integrator", integrator_idx, INTEGRATORS)
    if changed_int:
        reset_history()
    _, h = psim.SliderFloat("dt (h)", h, v_min=0.001, v_max=0.05)

    # --- material / contact sliders (log scale) ---
    E_val = 10.0 ** log_E
    nu_val = 0.5 - 10.0 ** log_nu_compl
    changed_E, log_E = psim.SliderFloat(
        f"log10 Young's E  (E = {E_val:.2e})", log_E, v_min=2.0, v_max=8.0
    )
    changed_nu, log_nu_compl = psim.SliderFloat(
        f"log10 (0.5 - nu)  (nu = {nu_val:.4f})", log_nu_compl, v_min=-4.0, v_max=-0.31
    )
    if changed_E or changed_nu:
        update_material()
    changed_K, log_K_contact = psim.SliderFloat(
        f"log10 contact penalty  (K = {K_CONTACT:.2e})", log_K_contact, v_min=2.0, v_max=9.0
    )
    if changed_K:
        K_CONTACT = 10.0 ** log_K_contact

    # --- mouse interaction: left-click + drag to grab a vertex ---
    win_pos = psim.GetMousePos()
    if psim.IsMouseClicked(0):
        pos = screen_to_world_2d(win_pos)
        distance = np.linalg.norm(U - pos.reshape(-1, 2), axis=1)
        nearest = int(np.argmin(distance))
        # only pick if the click was close enough to a vertex
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
    if selected_index is not None:
        handle_pc.update_point_positions(handle_target.reshape(1, dim))
        sel_pc.update_point_positions(U[selected_index].reshape(1, dim))

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}")
    if selected_index is not None:
        psim.Text(f"dragging vertex {selected_index} -> ({handle_target[0]:.2f}, {handle_target[1]:.2f})")
    else:
        psim.Text("left-click a vertex and drag to move it; release to drop")


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.look_at(np.array([0, 0, 5]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")

mesh = ps.register_surface_mesh("mesh", U, T, material="flat", color=light_green, edge_width=2)
pc = ps.register_point_cloud("vertices", U, radius=0.001, material="flat", color=black)
sel_pc = ps.register_point_cloud("selected", np.zeros((1, dim)), radius=0.001, material="flat", color=red, enabled=False)
handle_pc = ps.register_point_cloud("handle target", np.zeros((1, dim)), radius=0.001, material="flat", color=red, enabled=False)

# floor: draw a wide line at y = floor_y as a curve network
floor_nodes = np.array([[-3.0, floor_y, -0.005], [3.0, floor_y, -0.005]])
floor_edges = np.array([[0, 1]])
floor_curve = ps.register_curve_network("floor", floor_nodes, floor_edges, material="flat", color=gray, radius=0.006)

ps.set_do_default_mouse_interaction(False)
ps.set_user_callback(callback)
ps.show()
