import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import scipy as sp

from utils import triangulated_grid

from simkit.deformation_jacobian import deformation_jacobian
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.gravity_force import gravity_force
from simkit.ympr_to_lame import ympr_to_lame
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams
import simkit.energies as energies


# ---------- mesh & material -------------------------------------------------
# A slender cantilever beam pinned along its left edge and sagging under
# gravity. The mesh is intentionally coarse and the default Young's modulus
# / time step are tuned so that explicit Forward Euler stays stable.
X, T = triangulated_grid(nx=12, ny=4, width=2.0, height=0.4)
n, dim = X.shape

rho = 1.0
poisson = 0.45
ym = 1.0
mu_val, lam_val = ympr_to_lame(ym, poisson)
mu = np.full((T.shape[0], 1), mu_val)
lam = np.full((T.shape[0], 1), lam_val)

gravity_a = -2.0

J = deformation_jacobian(X, T)
vol = volume(X, T)
M = sp.sparse.kron(massmatrix(X, T, rho=rho), sp.sparse.eye(dim)).tocsc()
M_diag = M.diagonal()

# Gravity body force, lumped via the mass matrix.
f_grav_col = gravity_force(X, T, a=gravity_a, rho=rho).flatten().reshape(-1, 1)

# ---------- pinning ---------------------------------------------------------
# Implicit integrators see the pin as a stiff quadratic penalty in the
# total energy. Forward Euler bypasses that penalty (it would force a tiny
# dt) and instead hard-clamps the pinned DOFs each step.
K_PIN = 1e4
pin_idx = np.where(X[:, 0] <= X[:, 0].min() + 1e-6)[0]
Q_pin, b_pin = dirichlet_penalty(pin_idx, X[pin_idx, :], n, K_PIN)


# ---------- dynamics state --------------------------------------------------
U = X.copy()
U_prev = X.copy()
U_prev2 = X.copy()
U_prev3 = X.copy()
V = np.zeros((n, dim))

INTEGRATORS = ["Backward Euler", "BDF2", "Forward Euler"]
integrator_idx = 0
h = 0.005
NEWTON_ITERS = 5


def reset_history():
    U_prev[:] = U
    U_prev2[:] = U
    U_prev3[:] = U
    V[:] = 0.0


def update_material(new_ym):
    m, l = ympr_to_lame(new_ym, poisson)
    mu[:] = m
    lam[:] = l


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


def gravity_E(x_flat):
    # f = -dE/dx => E_grav(x) = -f_grav^T x
    return -float((f_grav_col.T @ x_flat.reshape(-1, 1))[0, 0])


def gravity_g(_x_flat):
    return -f_grav_col


# ---------- integrators -----------------------------------------------------
def implicit_step(make_kin_E, make_kin_g, kin_H):
    def Etot(x):
        return elastic_E(x) + pin_E(x) + gravity_E(x) + make_kin_E(x.reshape(-1, 1))

    def gtot(x):
        return elastic_g(x) + pin_g(x) + gravity_g(x) + make_kin_g(x.reshape(-1, 1))

    def Htot(x):
        return elastic_H(x) + Q_pin + kin_H

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
    """Explicit Forward Euler with mask-based pinning.

    x_{k+1} = x_k + h v_k
    v_{k+1} = v_k + h M^{-1} (f_elastic + f_grav)
    Pinned DOFs are clamped after the update so the explicit step does not
    have to resolve the stiff pin penalty.
    """
    x = U.flatten()
    f = -(elastic_g(x) + gravity_g(x)).flatten()
    a = (f / M_diag).reshape(n, dim)
    U_new = U + h * V
    V_new = V + h * a
    U_new[pin_idx] = X[pin_idx]
    V_new[pin_idx] = 0.0
    return U_new, V_new


def advance():
    global U
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


def kinetic_value():
    v = V.flatten()
    return 0.5 * float(v @ (M @ v))


# ---------- viz colors ------------------------------------------------------
light_green = np.array([153, 216, 201]) / 255
black = np.array([0.0, 0.0, 0.0])
blue = np.array([0.2, 0.4, 0.85])

HISTORY_LEN = 200
elastic_history = []
kinetic_history = []
total_history = []


# ---------- callback --------------------------------------------------------
def callback():
    global integrator_idx, h, ym

    if psim.Button("Reset"):
        U[:] = X
        reset_history()
        elastic_history.clear()
        kinetic_history.clear()
        total_history.clear()

    changed_int, integrator_idx = psim.Combo("Integrator", integrator_idx, INTEGRATORS)
    if changed_int:
        reset_history()
        elastic_history.clear()
        kinetic_history.clear()
        total_history.clear()

    _, h = psim.SliderFloat("dt (h)", h, v_min=0.0005, v_max=0.05)

    changed_ym, ym = psim.SliderFloat("Young's modulus", ym, v_min=0.05, v_max=5.0)
    if changed_ym:
        update_material(ym)

    advance()

    mesh.update_vertex_positions(U)
    pc.update_point_positions(U)
    pin_pc.update_point_positions(U[pin_idx])

    e_el = elastic_E(U.flatten())
    e_kin = kinetic_value()
    e_grav = gravity_E(U.flatten())
    e_tot = e_el + e_kin + e_grav
    elastic_history.append(e_el)
    kinetic_history.append(e_kin)
    total_history.append(e_tot)
    del elastic_history[:-HISTORY_LEN]
    del kinetic_history[:-HISTORY_LEN]
    del total_history[:-HISTORY_LEN]

    psim.Text(f"vertices: {n}    triangles: {T.shape[0]}    pinned: {len(pin_idx)}")
    psim.Text(f"integrator: {INTEGRATORS[integrator_idx]}")

    psim.PlotLines(
        f"elastic energy (last {HISTORY_LEN})",
        elastic_history,
        overlay_text=f"E_elastic: {e_el:.3f}",
        graph_size=(0.0, 100.0),
    )
    psim.PlotLines(
        f"kinetic energy (last {HISTORY_LEN})",
        kinetic_history,
        overlay_text=f"E_kinetic: {e_kin:.3f}",
        graph_size=(0.0, 100.0),
    )
    psim.PlotLines(
        f"total energy (last {HISTORY_LEN})",
        total_history,
        overlay_text=f"E_total: {e_tot:.3f}",
        graph_size=(0.0, 100.0),
    )


# ---------- polyscope setup -------------------------------------------------
ps.init()
ps.remove_all_structures()
ps.look_at(np.array([0, 0, 5]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")

mesh = ps.register_surface_mesh("mesh", U, T, material="flat", color=light_green, edge_width=2)
pc = ps.register_point_cloud("vertices", U, radius=0.012, material="flat", color=black)
pin_pc = ps.register_point_cloud("pinned", U[pin_idx], radius=0.022, material="flat", color=blue)

ps.set_user_callback(callback)
ps.show()
