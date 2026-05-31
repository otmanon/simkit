"""Tutorial 010 - LBS subspace contact in 3D.

Same scene as ``010_interactive_contact_plane_3D.py`` (a tet block falling onto
a soft floor with a drag-handle), but the Backward-Euler integrator runs in a
**reduced linear-blend-skinning (LBS) subspace** instead of the full mesh DOFs.
The full-space sim is kept here as a side-by-side reference: toggle via the
"Integrator" combo.

LBS subspace recipe (built at module scope so you can read it top-to-bottom):

* **Weights.** Take the ``m`` lowest generalized-eigenvectors of the Dirichlet
  Laplacian ``(L, M)``. These are the smoothest scalar modes on the mesh; we
  use them as per-bone skinning weights.
* **Basis.** ``B = lbs_jacobian(X, W)`` lifts those weights into a full LBS
  Jacobian. ``x = X + B z`` reconstructs deformed positions from a stack of
  per-bone affine-transform DOFs ``z`` (rest is ``z = 0``).
* **Elastic cubature.** Sample ``k_e`` random tets and only evaluate the
  Macklin-Mueller Neo-Hookean energy at those, weighted by their actual rest volumes scaled by
  ``t / k_e`` (Monte-Carlo).  Plug straight into the displacement ``_u`` tier
  with the precomputed ``JB_cub = (sel J) B`` as the "Jacobian" and
  ``Jx0_cub = (sel J) x_rest`` as the offset. The ``_u`` API expects
  ``u.shape[1] == dim`` so we reshape ``z`` from ``(r, 1)`` to ``(r/dim, dim)``;
  ``r`` is always divisible by ``dim`` for LBS (= num_bones * (dim+1) * dim).
* **Contact cubature.** Pick ``k_c`` farthest-point-sampled vertices, precompute
  ``SB = (sel B)`` so ``S_v @ B @ z`` gives the contact-vertex positions in
  one matmul.
* **Reduced mass / gravity / handle.** ``BMB = B^T M B`` (small dense (r, r))
  for the kinetic term. ``B^T f_g`` for gravity. The Dirichlet handle projects
  on-the-fly as ``B^T Q_h B`` / ``B^T (Q_h x + b_h)`` because the handle
  matrices change when the user picks a new vertex.

Two visualizations toggle in polyscope: the cubature tet centroids and the
contact-sample vertices, both updated each frame.
"""
import numpy as np
import polyscope as ps
import scipy as sp

from simkit.deformation_jacobian import deformation_jacobian
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.eigs import eigs
from simkit.energies.contact_springs_plane import (
    contact_springs_plane_energy,
    contact_springs_plane_gradient,
    contact_springs_plane_hessian,
)
from simkit.energies.kinetic import (
    kinetic_energy_be,
    kinetic_gradient_be,
    kinetic_hessian_be,
)
from simkit.energies.macklin_mueller_neo_hookean import (
    macklin_mueller_neo_hookean_energy_u,
    macklin_mueller_neo_hookean_gradient_u,
    macklin_mueller_neo_hookean_hessian_u,
    macklin_mueller_neo_hookean_energy_x,
    macklin_mueller_neo_hookean_gradient_x,
    macklin_mueller_neo_hookean_hessian_x,
)
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.gravity_force import gravity_force
from simkit.lbs_jacobian import lbs_jacobian
from simkit.massmatrix import massmatrix
from simkit.orthonormalize import orthonormalize
from simkit.selection_matrix import selection_matrix
from simkit.volume import volume
from simkit.solvers.NewtonSolver import NewtonSolver, NewtonSolverParams

from utils import (
    MouseHandle3D, TutorialUI, Viewer3D,
    lame_from_E_nu, tetrahedralized_grid,
)


# ============================================================================
# Mesh + precomputed operators (the sim classes consume these as inputs)
# ============================================================================
X, T = tetrahedralized_grid(nx=10, ny=10, nz=10, width=0.8, height=0.4, depth=0.4)
X[:, 1] += 1.0
n, dim = X.shape
t = T.shape[0]

RHO = 1e3
J     = deformation_jacobian(X, T)                          # per-tet sparse
vol   = volume(X, T)                                        # per-tet (Tx1)
M_n   = massmatrix(X, T, rho=RHO)                           # (n,n) per-vertex
M     = sp.sparse.kron(M_n, sp.sparse.eye(dim)).tocsc()     # (n*d, n*d)
f_g   = gravity_force(X, T, a=-9.8, rho=RHO).reshape(-1, 1)

floor_p = np.array([0.0, -0.6, 0.0])
floor_n = np.array([0.0,  1.0, 0.0])
mu0, lam0 = lame_from_E_nu(E=1e5, nu=0.4)


# ============================================================================
# LBS subspace: smooth eigenmodes -> skinning weights -> LBS Jacobian B
# ============================================================================
NUM_BONES  = 10                                   # m: number of LBS handles
NUM_CUB_T  = min(100, t)                          # k_e: elastic-cubature tets
NUM_CUB_V  = min(50, n)                           # k_c: contact-cubature verts

# Laplacian (n,n) and matching mass (n,n) for the generalized eigenproblem.
L = dirichlet_laplacian(X, T)
_, W_complex = eigs(L, k=NUM_BONES, M=M_n)
W = np.real(np.asarray(W_complex))                # (n, m) - real, may be signed

# LBS basis. The raw LBS Jacobian has redundant columns (e.g., the constant
# eigenmode contributes a global affine that overlaps with every other bone),
# so mass-orthonormalize and drop the null directions; without this the
# Hessian below becomes singular and the Newton solve goes off the rails.
B_raw = lbs_jacobian(X, W)                        # (n*d, m*(d+1)*d)
B = orthonormalize(B_raw, M=M)                    # (n*d, r), r <= m*(d+1)*d
r = B.shape[1]
# Round r down to a multiple of dim so the ``_u`` API's reshape works.
r = (r // dim) * dim
B = B[:, :r]

# Integrate the *displacement* z; rest is z = 0 and ``x = X.flatten() + B @ z``.
X_flat = X.reshape(-1, 1)


# ============================================================================
# Elastic cubature: pick k_e random tets; build C_F (selects per-element F).
# JB_cub = C_F J B is the only matrix the elastic energy needs per call.
# Jx0_cub = C_F J X.flatten() is the rest deformation gradient at those tets
# (i.e. vec(I) repeated k_e times).  ``vol_cub`` is rescaled by t/k_e so the
# Monte-Carlo estimator matches the full-mesh energy in expectation.
# ============================================================================
rng = np.random.default_rng(0)
elasticI = np.sort(rng.choice(t, size=NUM_CUB_T, replace=False))
C        = selection_matrix(elasticI, t)                          # (k_e, t)
C_F      = sp.sparse.kron(C, sp.sparse.eye(dim * dim)).tocsc()    # (k_e*d*d, t*d*d)
JB_cub   = np.asarray((C_F @ J @ B))                              # (k_e*d*d, r)
Jx0_cub  = np.asarray((C_F @ J @ X_flat))                         # (k_e*d*d, 1)
vol_cub  = vol[elasticI] * (t / NUM_CUB_T)                        # (k_e, 1)


# ============================================================================
# Contact cubature: pick k_c farthest-point-sampled vertices; build
# S_v (selects per-vertex). SB = S_v B; contact positions = X_bar + SB z.
# ============================================================================
contactI = farthest_point_sampling(X, NUM_CUB_V)                  # (k_c,)
S_v      = sp.sparse.kron(
    selection_matrix(contactI, n), sp.sparse.eye(dim)).tocsc()    # (k_c*d, n*d)
SB       = np.asarray(S_v @ B)                                    # (k_c*d, r)
X_bar_contact = X[contactI]                                       # (k_c, d)
M_contact = sp.sparse.diags(
    np.asarray(M_n.diagonal())[contactI])                         # (k_c, k_c)


# ============================================================================
# Reduced kinetic mass + reduced gravity (constants, precomputed once).
# ============================================================================
BMB = np.asarray(B.T @ M @ B)                                     # (r, r) dense
BTfg = B.T @ f_g                                                  # (r, 1)


# ============================================================================
# Sim classes
#
# Both minimize the same potential
#
#     E_pot(x) = E_elastic(x)                       Macklin-Mueller Neo-Hookean
#              + E_floor(x)                         penalty springs vs. plane
#              + 1/2 x^T Q_h x + b_h^T x            mouse handle (zeroed = none)
#              - f_g^T x                            gravity
#
# plus a Backward-Euler kinetic term.  The full-space variant is included for
# A/B testing; the subspace variant evaluates the elastic and contact terms
# only at their cubature samples and projects everything else through B.
# ============================================================================

class ElasticSimBE:
    """Full-space Backward Euler. Kept for side-by-side comparison."""

    def __init__(self, X, T, J, vol, M, M_n, f_g,
                 mu, lam, K_contact, p_floor, n_floor, h, newton_iters=5):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.J, self.vol, self.M, self.M_n, self.f_g = J, vol, M, M_n, f_g
        self.mu  = np.full((T.shape[0], 1), float(mu))
        self.lam = np.full((T.shape[0], 1), float(lam))
        self.K_contact = float(K_contact)
        self.p_floor, self.n_floor = p_floor, n_floor
        self.h = float(h)

        D = self.n * self.dim
        self.U      = X.copy()
        self.U_prev = X.copy()
        self.Q_h    = sp.sparse.csc_matrix((D, D))
        self.b_h    = np.zeros((D, 1))

        self._solver = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=newton_iters, do_line_search=True),
        )

    def energy(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        E_el    = float(macklin_mueller_neo_hookean_energy_x(xn, self.J, self.mu, self.lam, self.vol))
        E_floor = float(contact_springs_plane_energy(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n))
        E_h     = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
                   + float((self.b_h.T @ xc)[0, 0]))
        E_grav  = -float((self.f_g.T @ xc)[0, 0])
        E_kin   = float(kinetic_energy_be(
            xc, self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1), self.M, self.h))
        return E_el + E_floor + E_h + E_grav + E_kin

    def gradient(self, x):
        xn = x.reshape(-1, self.dim)
        xc = x.reshape(-1, 1)
        g_el    = macklin_mueller_neo_hookean_gradient_x(xn, self.J, self.mu, self.lam, self.vol)
        g_floor = contact_springs_plane_gradient(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        g_h     = self.Q_h @ xc + self.b_h
        g_grav  = -self.f_g
        g_kin   = kinetic_gradient_be(
            xc, self.U.reshape(-1, 1), self.U_prev.reshape(-1, 1), self.M, self.h)
        return g_el + g_floor + g_h + g_grav + g_kin

    def hessian(self, x):
        xn = x.reshape(-1, self.dim)
        H_el    = macklin_mueller_neo_hookean_hessian_x(
            xn, self.J, self.mu, self.lam, self.vol, psd=True)
        H_floor = contact_springs_plane_hessian(
            xn, self.K_contact, self.p_floor, self.n_floor, M=self.M_n)
        H_kin   = kinetic_hessian_be(self.M, self.h)
        return H_el + H_floor + self.Q_h + H_kin

    def step(self):
        x_next = self._solver.solve(self.U.flatten().reshape(-1, 1))
        self.U_prev[:] = self.U
        self.U[:] = x_next.reshape(self.n, self.dim)


class ElasticSimBESubspace:
    """LBS-subspace Backward Euler. Variable: z (per-bone affine displacement)."""

    def __init__(self, X, T, B, X_flat,
                 JB_cub, Jx0_cub, vol_cub,
                 SB, contactI, X_bar_contact, M_contact,
                 BMB, BTfg, M_n, f_g,
                 mu, lam, K_contact, p_floor, n_floor, h, newton_iters=5):
        self.X, self.T = X, T
        self.n, self.dim = X.shape
        self.B, self.X_flat = B, X_flat
        self.r = B.shape[1]

        # Elastic cubature. ``mu`` / ``lam`` are aliased onto the cubature-sized
        # arrays so the TutorialUI material slider's ``s.mu[:] = value``
        # broadcast writes straight through; the energy methods read from
        # ``self.mu_cub`` / ``self.lam_cub`` directly.
        k_e = vol_cub.shape[0]
        self.JB_cub, self.Jx0_cub, self.vol_cub = JB_cub, Jx0_cub, vol_cub
        self.mu_cub  = np.full((k_e, 1), float(mu))
        self.lam_cub = np.full((k_e, 1), float(lam))
        self.mu  = self.mu_cub
        self.lam = self.lam_cub

        # Contact cubature
        self.SB, self.contactI = SB, contactI
        self.X_bar_contact, self.M_contact = X_bar_contact, M_contact
        self.K_contact = float(K_contact)
        self.p_floor, self.n_floor = p_floor, n_floor

        # Reduced quantities for kinetic / gravity
        self.BMB, self.BTfg = BMB, BTfg
        self.M_n, self.f_g = M_n, f_g

        self.h = float(h)

        # State: z is per-bone-affine displacement; rest is zero.
        self.z      = np.zeros((self.r, 1))
        self.z_curr = np.zeros((self.r, 1))
        self.z_prev = np.zeros((self.r, 1))

        # Visualization state: U is always derived from z.
        self.U = X.copy()

        # Handle penalty is set in full-space by MouseHandle3D; we project
        # in-line inside energy/gradient/hessian.
        D = self.n * self.dim
        self.Q_h = sp.sparse.csc_matrix((D, D))
        self.b_h = np.zeros((D, 1))

        self._solver = NewtonSolver(
            self.energy, self.gradient, self.hessian,
            NewtonSolverParams(max_iter=newton_iters, do_line_search=True),
        )

    # -- helpers --------------------------------------------------------------
    def _x_flat(self, zc):
        return self.X_flat + self.B @ zc

    # -- potential / kinetic --------------------------------------------------
    def energy(self, z):
        zc = z.reshape(-1, 1)
        zn = z.reshape(-1, self.dim)                                 # (r/d, d)

        E_el = float(macklin_mueller_neo_hookean_energy_u(
            zn, self.JB_cub, self.Jx0_cub,
            self.mu_cub, self.lam_cub, self.vol_cub))

        xn_contact = self.X_bar_contact + (self.SB @ zc).reshape(-1, self.dim)
        E_floor = float(contact_springs_plane_energy(
            xn_contact, self.K_contact, self.p_floor, self.n_floor,
            M=self.M_contact))

        xc = self._x_flat(zc)
        E_h = (0.5 * float((xc.T @ (self.Q_h @ xc))[0, 0])
               + float((self.b_h.T @ xc)[0, 0]))

        E_grav = -float((self.BTfg.T @ zc)[0, 0])                    # const dropped

        E_kin = float(kinetic_energy_be(
            zc, self.z_curr, self.z_prev, self.BMB, self.h))

        return E_el + E_floor + E_h + E_grav + E_kin

    def gradient(self, z):
        zc = z.reshape(-1, 1)
        zn = z.reshape(-1, self.dim)

        g_el = macklin_mueller_neo_hookean_gradient_u(
            zn, self.JB_cub, self.Jx0_cub,
            self.mu_cub, self.lam_cub, self.vol_cub)

        xn_contact = self.X_bar_contact + (self.SB @ zc).reshape(-1, self.dim)
        g_floor_x = contact_springs_plane_gradient(
            xn_contact, self.K_contact, self.p_floor, self.n_floor,
            M=self.M_contact)
        g_floor = self.SB.T @ g_floor_x

        xc = self._x_flat(zc)
        g_h = self.B.T @ (self.Q_h @ xc + self.b_h)

        g_grav = -self.BTfg

        g_kin = kinetic_gradient_be(
            zc, self.z_curr, self.z_prev, self.BMB, self.h)

        return g_el + g_floor + g_h + g_grav + g_kin

    def hessian(self, z):
        zc = z.reshape(-1, 1)
        zn = z.reshape(-1, self.dim)

        H_el = macklin_mueller_neo_hookean_hessian_u(
            zn, self.JB_cub, self.Jx0_cub,
            self.mu_cub, self.lam_cub, self.vol_cub, psd=True)

        xn_contact = self.X_bar_contact + (self.SB @ zc).reshape(-1, self.dim)
        H_floor_x = contact_springs_plane_hessian(
            xn_contact, self.K_contact, self.p_floor, self.n_floor,
            M=self.M_contact)
        H_floor = self.SB.T @ H_floor_x @ self.SB

        H_h = self.B.T @ self.Q_h @ self.B

        H_kin = kinetic_hessian_be(self.BMB, self.h)

        # Everything is dense (r, r); convert sparse pieces.
        return (np.asarray(H_el) + np.asarray(H_floor)
                + np.asarray(H_h.todense() if sp.sparse.issparse(H_h) else H_h)
                + np.asarray(H_kin.todense() if sp.sparse.issparse(H_kin) else H_kin))

    def step(self):
        z_next = self._solver.solve(self.z.flatten().reshape(-1, 1))
        z_next = z_next.reshape(-1, 1)
        self.z_prev[:] = self.z_curr
        self.z_curr[:] = z_next
        self.z[:] = z_next
        self.U[:] = (self.X_flat + self.B @ self.z).reshape(self.n, self.dim)


# ============================================================================
# Build sims, wire viewer + handle + UI
# ============================================================================
subspace_sim = ElasticSimBESubspace(
    X=X, T=T, B=B, X_flat=X_flat,
    JB_cub=JB_cub, Jx0_cub=Jx0_cub, vol_cub=vol_cub,
    SB=SB, contactI=contactI, X_bar_contact=X_bar_contact, M_contact=M_contact,
    BMB=BMB, BTfg=BTfg, M_n=M_n, f_g=f_g,
    mu=mu0, lam=lam0,
    K_contact=1e5, p_floor=floor_p, n_floor=floor_n, h=0.02,
)
full_sim = ElasticSimBE(
    X=X, T=T, J=J, vol=vol, M=M, M_n=M_n, f_g=f_g,
    mu=mu0, lam=lam0,
    K_contact=1e5, p_floor=floor_p, n_floor=floor_n, h=0.02,
)
sims = {
    "Subspace BE": subspace_sim,
    "Full-space BE": full_sim,
}

viewer = Viewer3D(X, T, floor_y=float(floor_p[1]))
sel_pc, target_pc = viewer.add_handle_markers()
handle = MouseHandle3D(sims, sel_pc, target_pc, K_handle=5e9)
ui     = TutorialUI(sims, handle, show_contact_K=True, show_handle_mode=True)


# ----------------------------------------------------------------------------
# Visualizations for the cubature samples.  Both update each frame and can be
# toggled in polyscope's structure panel.
# ----------------------------------------------------------------------------
def _tet_centroids(U_now):
    return U_now[T[elasticI]].mean(axis=1)            # (k_e, 3)

cub_pc = ps.register_point_cloud(
    "elastic cubature tets",
    _tet_centroids(X),
    radius=0.008, material="flat",
    color=(0.95, 0.55, 0.15),
    enabled=True,
)
con_pc = ps.register_point_cloud(
    "contact cubature verts",
    X[contactI],
    radius=0.010, material="flat",
    color=(0.20, 0.55, 0.95),
    enabled=True,
)


# ----------------------------------------------------------------------------
# On integrator switch, reset the subspace's z (its U state is rebuilt from z).
# ----------------------------------------------------------------------------
def _on_switch():
    subspace_sim.z[:]      = 0.0
    subspace_sim.z_curr[:] = 0.0
    subspace_sim.z_prev[:] = 0.0
    subspace_sim.U[:]      = subspace_sim.X
ui.on_switch(_on_switch)


def _on_reset():
    subspace_sim.z[:]      = 0.0
    subspace_sim.z_curr[:] = 0.0
    subspace_sim.z_prev[:] = 0.0
ui.on_reset(_on_reset)


def callback():
    ui.draw()
    if ui.handle_enabled:
        handle.update()
    ui.sim.step()
    viewer.refresh(ui.sim.U)
    cub_pc.update_point_positions(_tet_centroids(ui.sim.U))
    con_pc.update_point_positions(ui.sim.U[contactI])
    handle.refresh_markers()


viewer.show(callback)
