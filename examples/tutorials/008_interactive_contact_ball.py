import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import screen_to_world_2d, triangulated_grid, ball_mesh_2d

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import simkit.energies as energies


# ----------  deformable shape -----------------------------------------------
X, T = triangulated_grid(nx=12, ny=8, width=0.8, height=0.5)
X[:, 1] += 0.3        # lift slightly so the top-edge pin is above the origin
n, dim = X.shape

# Young's modulus / Poisson ratio -> Lame parameters (plane strain).
log_E = 5.0
log_nu_compl = -1.0
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
M_n = massmatrix(X, T, rho=rho)                       # (n, n)
M = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()   # (n*d, n*d)

# gravity (constant body force -> linear potential)
g_acc = -9.8
f_g = gravity_force(X, T, a=g_acc, rho=rho).reshape(-1, 1)

# soft pin: top edge holds the shape in place so it has something to hang from
K_PIN = 5e6
pin_idx = np.where(X[:, 1] >= X[:, 1].max() - 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx, :], n, K_PIN)


# ---------- ball (user-controlled obstacle) --------------------------------
BALL_R = 0.12
ball_p = np.array([0.6, 0.3])      # current center (follows cursor)
ball_active = True

ball_X, ball_T = ball_mesh_2d(radius=BALL_R, n_segments=48)

log_K_contact = 6.0
K_CONTACT = 10.0 ** log_K_contact


# ---------- dynamics state --------------------------------------------------
U = X.copy()
U_prev = X.copy()
U_prev2 = X.copy()
U_prev3 = X.copy()
V = np.zeros((n, dim))

INTEGRATORS = ["Static", "Backward Euler", "BDF2", "Forward Euler"]
integrator_idx = 0       # default: static (quasi-static equilibrium)
h = 0.02
NEWTON_ITERS = 8


def reset_history():
    U_prev[:] = U
    U_prev2[:] = U
    U_prev3[:] = U
    V[:] = 0.0


def reset_scene():
    U[:] = X
    reset_history()
    elastic_history.clear()
    kinetic_history.clear()
    contact_history.clear()


# ---------- energies --------------------------------------------------------
def elastic_E(x_flat):
    return float(energies.neo_hookean_energy_x(x_flat.reshape(-1, dim), J, mu, lam, vol))


def elastic_g(x_flat):
    return energies.neo_hookean_gradient_x(x_flat.reshape(-1, dim), J, mu, lam, vol)


def elastic_H(x_flat):
    return energies.neo_hookean_hessian_x(x_flat.reshape(-1, dim), J, mu, lam, vol, psd=True)


def pin_E(x_flat):
    xc = x_flat.reshape(-1, 1)
    return 0.5 * float((xc.T @ (Q_pin @ xc))[0, 0]) + float((b_pin.T @ xc)[0, 0])


def pin_g(x_flat):
    return Q_pin @ x_flat.reshape(-1, 1) + b_pin


def contact_E(x_flat):
    if not ball_active:
        return 0.0
    return float(energies.contact_springs_sphere_energy(
        x_flat.reshape(-1, dim), K_CONTACT, ball_p, BALL_R, M=M_n))


def contact_g(x_flat):
    if not ball_active:
        return np.zeros((n * dim, 1))
    return energies.contact_springs_sphere_gradient(
        x_flat.reshape(-1, dim), K_CONTACT, ball_p, BALL_R, M=M_n)


def contact_H(x_flat):
    if not ball_active:
        return sp.sparse.csc_matrix((n * dim, n * dim))
    return energies.contact_springs_sphere_hessian(
        x_flat.reshape(-1, dim), K_CONTACT, ball_p, BALL_R, M=M_n)


def gravity_E(x_flat):
    return -float((f_g.T @ x_flat.reshape(-1, 1))[0, 0])


# ---------- integrators -----------------------------------------------------
def implicit_step(make_kin_E, make_kin_g, kin_H):
    def Etot(x):
        return (
            elastic_E(x) + pin_E(x) + contact_E(x) + gravity_E(x)
            + make_kin_E(x.reshape(-1, 1))
        )

    def gtot(x):
        return (
            elastic_g(x) + pin_g(x) + contact_g(x) - f_g
            + make_kin_g(x.reshape(-1, 1))
        )

    def Htot(x):
        return elastic_H(x) + Q_pin + contact_H(x) + kin_H

    solver = NewtonSolver(
        Etot, gtot, Htot,
        NewtonSolverParams(max_iter=NEWTON_ITERS, do_line_search=True),
    )
    return solver.solve(U.flatten().reshape(-1, 1)).reshape(n, dim)


_ZERO_KIN_H = sp.sparse.csc_matrix((n * dim, n * dim))
_ZERO_KIN_g = np.zeros((n * dim, 1))


def step_static():
    return implicit_step(lambda _: 0.0, lambda _: _ZERO_KIN_g, _ZERO_KIN_H)


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
    x = U.flatten()
    f = -(elastic_g(x) + pin_g(x) + contact_g(x)) + f_g
    a = (f.flatten() / M.diagonal()).reshape(n, dim)
    return U + h * V, V + h * a


def advance():
    global U, U_prev, U_prev2, U_prev3, V
    name = INTEGRATORS[integrator_idx]
    if name == "Static":
        U_next = step_static()
        V[:] = 0.0
    elif name == "Backward Euler":
        U_next = step_be()
        V[:] = (U_next - U) / h
    elif name == "BDF2":
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
blue = np.array([0.2, 0.4, 0.85])
ball_color = np.array([0.95, 0.45, 0.35])

HISTORY_LEN = 200
elastic_history = []
kinetic_history = []
contact_history = []


# ---------- callback --------------------------------------------------------
def callback():
    global integrator_idx, h
    global log_E, log_nu_compl, log_K_contact, K_CONTACT
    global ball_p, ball_active

    # --- UI controls ---
    if psim.Button("Reset"):
        reset_scene()

    changed_int, integrator_idx = psim.Combo("Integrator", integrator_idx, INTEGRATORS)
    if changed_int:
        reset_history()
        kinetic_history.clear()
    _, h = psim.SliderFloat("dt (h)", h, v_min=0.001, v_max=0.05)

    E_val = 10.0 ** log_E
    nu_val = 0.5 - 10.0 ** log_nu_compl
    changed_E, log_E = psim.SliderFloat(
        f"log10 E  (E = {E_val:.2e})", log_E, v_min=2.0, v_max=8.0
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

    # --- input: ball follows cursor; SPACE deletes ball + resets; click reactivates ---
    io = psim.GetIO()
    win_pos = psim.GetMousePos()

    if psim.IsKeyPressed(psim.ImGuiKey_Space):
        ball_active = False
        reset_scene()
        ball_mesh.set_enabled(False)

    if not ball_active and psim.IsMouseClicked(0) and not io.WantCaptureMouse:
        ball_active = True
        ball_mesh.set_enabled(True)

    if ball_active and not io.WantCaptureMouse:
        ball_p = screen_to_world_2d(win_pos).astype(float).copy()

    # --- step the simulation ---
    advance()

    # --- update visualization ---
    mesh.update_vertex_positions(U)
    if ball_active:
        ball_mesh.update_vertex_positions(ball_X + ball_p[None, :])

    # --- diagnostics ---
    e_el = elastic_E(U.flatten())
    e_kin = 0.5 * float(V.flatten() @ (M @ V.flatten()))
    e_con = contact_E(U.flatten())
    elastic_history.append(e_el)
    kinetic_history.append(e_kin)
    contact_history.append(e_con)
    del elastic_history[:-HISTORY_LEN]
    del kinetic_history[:-HISTORY_LEN]
    del contact_history[:-HISTORY_LEN]

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}    pinned: {len(pin_idx)}")
    if ball_active:
        psim.Text(f"ball: ({ball_p[0]:.2f}, {ball_p[1]:.2f})  r = {BALL_R:.2f}")
        psim.Text("move mouse to push the ball; SPACE to delete + reset")
    else:
        psim.Text("ball deleted - click anywhere in scene to bring it back")

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
    psim.PlotLines(
        f"ball contact energy (last {HISTORY_LEN})",
        contact_history,
        overlay_text=f"E_contact: {e_con:.3f}",
        graph_size=(0.0, 120.0),
    )


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.look_at(np.array([0, 0, 5]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")

mesh = ps.register_surface_mesh("mesh", U, T, material="flat", color=light_green, edge_width=2)
ball_mesh = ps.register_surface_mesh(
    "ball", ball_X + ball_p[None, :], ball_T, material="flat", color=ball_color, edge_width=1
)

# soft pin markers
ps.register_point_cloud("pinned", X[pin_idx], radius=0.012, material="flat", color=blue)

ps.set_do_default_mouse_interaction(False)
ps.set_user_callback(callback)
ps.show()
