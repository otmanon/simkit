import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import tetrahedralized_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.gravity_force import gravity_force
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import simkit.energies as energies


# ---------- mesh & material -------------------------------------------------
X, T = tetrahedralized_grid(nx=6, ny=4, nz=4, width=0.8, height=0.4, depth=0.4)
X[:, 1] += 1.0    # drop height above the floor
n, dim = X.shape

# Material: Young's modulus E and Poisson ratio nu are the user-facing knobs.
log_E = 5.0           # log10(E),         E_init = 1e5
log_nu_compl = -1.0   # log10(0.5 - nu),  nu_init = 0.4
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
f_g = gravity_force(X, T, a=g_acc, rho=rho).reshape(-1, 1)  # (n*d, 1)

# ---------- floor (penalty contact) ----------------------------------------
log_K_contact = 5.0
K_CONTACT = 10.0 ** log_K_contact
floor_y = -0.6
floor_p = np.array([0.0, floor_y, 0.0])
floor_n = np.array([0.0, 1.0, 0.0])

# ---------- mouse handle ---------------------------------------------------
K_HANDLE = 5e9
selected_index = None
handle_target = None       # (3,) world point
drag_plane_origin = None   # 3D point used as drag-plane reference
drag_plane_normal = None   # 3D drag-plane normal (camera look_dir)
Q_handle = sp.sparse.csc_matrix((n * dim, n * dim))
b_handle = np.zeros((n * dim, 1))

# Interaction mode: True = our click/drag handle interface; False = polyscope camera
interaction_enabled = True

# Status line shown in the GUI
info_message = "left-click a mesh vertex to grab; drag to move; release to drop"


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


def clear_handle():
    global selected_index, handle_target, drag_plane_origin, drag_plane_normal
    selected_index = None
    handle_target = None
    drag_plane_origin = None
    drag_plane_normal = None
    rebuild_handle()
    sel_pc.set_enabled(False)
    handle_pc.set_enabled(False)


# ---------- energies --------------------------------------------------------
def elastic_E(x_flat):
    return float(energies.neo_hookean_energy_x(x_flat.reshape(-1, dim), J, mu, lam, vol))


def elastic_g(x_flat):
    return energies.neo_hookean_gradient_x(x_flat.reshape(-1, dim), J, mu, lam, vol)


def elastic_H(x_flat):
    return energies.neo_hookean_hessian_x(x_flat.reshape(-1, dim), J, mu, lam, vol, psd=True)


def contact_E(x_flat):
    return float(energies.contact_springs_plane_energy(
        x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n))


def contact_g(x_flat):
    return energies.contact_springs_plane_gradient(
        x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n)


def contact_H(x_flat):
    return energies.contact_springs_plane_hessian(
        x_flat.reshape(-1, dim), K_CONTACT, floor_p, floor_n, M=M_n)


def handle_E(x_flat):
    xc = x_flat.reshape(-1, 1)
    return 0.5 * float((xc.T @ (Q_handle @ xc))[0, 0]) + float((b_handle.T @ xc)[0, 0])


def handle_g(x_flat):
    return Q_handle @ x_flat.reshape(-1, 1) + b_handle


def gravity_E(x_flat):
    return -float((f_g.T @ x_flat.reshape(-1, 1))[0, 0])


# ---------- integrators -----------------------------------------------------
def implicit_step(make_kin_E, make_kin_g, kin_H):
    def Etot(x):
        return elastic_E(x) + contact_E(x) + handle_E(x) + gravity_E(x) + make_kin_E(x.reshape(-1, 1))

    def gtot(x):
        return (
            elastic_g(x) + contact_g(x) + handle_g(x) - f_g
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
    x = U.flatten()
    f = -(elastic_g(x) + contact_g(x) + handle_g(x)) + f_g
    a = (f.flatten() / M.diagonal()).reshape(n, dim)
    return U + h * V, V + h * a


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


# ---------- 3D picking helpers ----------------------------------------------
def _is_finite_point(p):
    return p is not None and np.all(np.isfinite(p))


def pick_world_point(win_pos):
    """World position under the cursor, or None if the pixel is empty space."""
    try:
        p = ps.screen_coords_to_world_position(np.asarray(win_pos, dtype=float))
    except Exception:
        return None
    p = np.asarray(p, dtype=float)
    if not _is_finite_point(p):
        return None
    return p


def cursor_ray(win_pos):
    """Camera position + normalized world-space ray direction for the cursor."""
    params = ps.get_view_camera_parameters()
    cam_pos = np.asarray(params.get_position(), dtype=float)
    ray_dir = np.asarray(
        ps.screen_coords_to_world_ray(np.asarray(win_pos, dtype=float)),
        dtype=float,
    )
    nrm = np.linalg.norm(ray_dir)
    if nrm > 0:
        ray_dir = ray_dir / nrm
    return cam_pos, ray_dir


def intersect_ray_plane(ray_o, ray_d, plane_p, plane_n):
    """Intersect a ray with a plane; returns the world-space hit or None if parallel."""
    denom = float(np.dot(ray_d, plane_n))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(plane_p - ray_o, plane_n) / denom)
    return ray_o + t * ray_d


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
red = np.array([0.85, 0.2, 0.2])


# ---------- callback --------------------------------------------------------
def callback():
    global integrator_idx, h
    global log_E, log_nu_compl, log_K_contact, K_CONTACT
    global selected_index, handle_target, drag_plane_origin, drag_plane_normal
    global interaction_enabled, info_message

    # --- UI controls ---
    if psim.Button("Reset"):
        U[:] = X
        reset_history()
        clear_handle()
        info_message = "scene reset"

    psim.SameLine()
    changed_mode, interaction_enabled = psim.Checkbox(
        "Click-to-drag handle (uncheck for camera)", interaction_enabled
    )
    if changed_mode:
        # When we own the mouse, disable polyscope's camera mouse handling.
        ps.set_do_default_mouse_interaction(not interaction_enabled)
        clear_handle()
        info_message = (
            "handle mode: click a mesh vertex to grab"
            if interaction_enabled
            else "camera mode: polyscope owns the mouse"
        )

    changed_int, integrator_idx = psim.Combo("Integrator", integrator_idx, INTEGRATORS)
    if changed_int:
        reset_history()
    _, h = psim.SliderFloat("dt (h)", h, v_min=0.001, v_max=0.05)

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

    # --- mouse interaction (only when our handle mode is on) -------------
    if interaction_enabled:
        win_pos = psim.GetMousePos()

        if psim.IsMouseClicked(0):
            hit = pick_world_point(win_pos)
            if hit is None:
                clear_handle()
                info_message = "you clicked on infinity — click on a mesh"
            else:
                # nearest vertex in 3D world space
                distances = np.linalg.norm(U - hit.reshape(1, dim), axis=1)
                selected_index = int(np.argmin(distances))
                handle_target = U[selected_index].copy()
                # drag plane: passes through the picked world point, camera-facing
                drag_plane_origin = hit.copy()
                drag_plane_normal = np.asarray(
                    ps.get_view_camera_parameters().get_look_dir(), dtype=float
                )
                rebuild_handle()
                sel_pc.set_enabled(True)
                handle_pc.set_enabled(True)
                info_message = f"grabbed vertex {selected_index}"

        if selected_index is not None and psim.IsMouseDown(0):
            ray_o, ray_d = cursor_ray(win_pos)
            new_target = intersect_ray_plane(
                ray_o, ray_d, drag_plane_origin, drag_plane_normal
            )
            if _is_finite_point(new_target):
                handle_target = new_target.astype(float)
                rebuild_handle()

        if psim.IsMouseReleased(0):
            if selected_index is not None:
                info_message = "released — left-click a mesh vertex to grab again"
            clear_handle()

    # --- step the simulation ---
    advance()

    # --- update visualization ---
    mesh.update_vertex_positions(U)
    if selected_index is not None:
        handle_pc.update_point_positions(handle_target.reshape(1, dim))
        sel_pc.update_point_positions(U[selected_index].reshape(1, dim))

    psim.Text(f"vertices: {n}    tets: {T.shape[0]}")
    psim.Text(info_message)


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.set_up_dir("y_up")
ps.set_front_dir("z_front")
ps.set_ground_plane_mode("tile_reflection")
ps.look_at(np.array([2.0, 1.2, 3.5]), np.array([0.0, 0.0, 0.0]))

mesh = ps.register_volume_mesh(
    "block", U, tets=T, color=light_green, interior_color=light_green,
    edge_width=1.0, material="flat",
)

sel_pc = ps.register_point_cloud(
    "selected", np.zeros((1, dim)), radius=0.012, material="flat",
    color=red, enabled=False,
)
handle_pc = ps.register_point_cloud(
    "handle target", np.zeros((1, dim)), radius=0.012, material="flat",
    color=red, enabled=False,
)

# Pin polyscope's ground plane to the contact plane: freeze the scene bbox so
# its bottom along the up-axis sits at floor_y (polyscope auto-anchors the
# ground at the bbox bottom).
bb_lo = np.array([
    float(X[:, 0].min()) - 1.0,
    floor_y,
    float(X[:, 2].min()) - 1.0,
])
bb_hi = np.array([
    float(X[:, 0].max()) + 1.0,
    float(X[:, 1].max()) + 1.0,
    float(X[:, 2].max()) + 1.0,
])
ps.set_automatically_compute_scene_extents(False)
ps.set_bounding_box(bb_lo, bb_hi)

# Handle interface owns the mouse by default; toggle the checkbox to switch.
ps.set_do_default_mouse_interaction(not interaction_enabled)
ps.set_user_callback(callback)
ps.show()
