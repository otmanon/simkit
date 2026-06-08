"""Tutorial 013 - Fast Complementary Dynamics.

A 2D beam driven by an animator rig (a single global affine handle, ``J @ p``).
You grab the rig with the mouse and drag it around; the rig handles the
low-frequency "art-directed" motion, and the simulation adds *complementary*
secondary jiggle ``B @ z`` on top -- the squash/stretch/wobble the animator did
not author.

The trick (Benchekroun et al., "Fast Complementary Dynamics via Skinning
Eigenmodes", SIGGRAPH 2023) is that the simulation subspace ``B`` is built
ORTHOGONAL to the rig ``J`` via a momentum-leaking constraint, so ``B @ z``
never fights or duplicates the rig motion -- it is purely complementary.

Like tutorial 007, the simulator lives in a small *local* class (the no-class
rule is about the simkit library, not example scripts). Its ``step()`` is two
closures handed to ``simkit.solvers.block_coord``:

    local step  : per-cluster best-fit rotation (polar SVD)
    global step : one Cholesky back-solve in the reduced subspace ``B``

Run from ``examples/interactive_demos`` (so ``utils`` is importable)::

    python 013_fast_complementary_dynamics.py
"""
import numpy as np
import scipy as sp
import scipy.sparse as sps

from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.deformation_jacobian import deformation_jacobian
from simkit.diffuse_field import diffuse_field
from simkit.lbs_jacobian import lbs_jacobian
from simkit.lbs_weight_space_constraint import lbs_weight_space_constraint
from simkit.massmatrix import massmatrix
from simkit.orthonormalize import orthonormalize
from simkit.polar_svd import polar_svd
from simkit.project_into_subspace import project_into_subspace
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.solvers import block_coord
from simkit.spectral_cubature import spectral_cubature
from simkit.volume import volume
from simkit.ympr_to_lame import ympr_to_lame


# ============================================================================
# Local Fast-CoDy simulator: state + precomputed operators.
#
# Rig:   x_rig = J @ p          (single global affine handle, 6 DOF in 2D)
# Sim:   x     = J @ p + B @ z  (B orthogonal to J => z is complementary)
# ============================================================================

class CoDySim:
    """Reduced Fast-CoDy beam. Drives the rig ``p`` toward ``p_target`` each
    frame and solves for the complementary subspace coords ``z``.

    Exposes ``U`` (current full vertex positions, n x dim) so the viewer can
    refresh, and ``n`` / ``dim`` so the viewer scaffolding can size things.
    """

    def __init__(self, X, T, num_modes=10, num_clusters=20,
                 ym=1e5, rho=1e3, h=0.01, max_iter=10):
        n, dim = X.shape
        self.X, self.T = X, T
        self.n, self.dim = n, dim
        self.h = float(h)
        self.max_iter = int(max_iter)

        Mv = sps.kron(massmatrix(X, T, rho=rho), sps.identity(dim)).tocsc()
        G = deformation_jacobian(X, T)
        mu, _lam = ympr_to_lame(ym, 0)
        vol = volume(X, T)
        AMu = sps.diags((mu * vol).flatten())
        AMue = sps.kron(AMu, sps.identity(dim * dim))
        L = (G.T @ AMue @ G).tocsc()

        # ---- rig: single global affine handle ----------------------------
        J = lbs_jacobian(X, np.ones((n, 1)))

        # ---- momentum-leaking orthogonality constraint -------------------
        # weights: 1 in the interior, ~0 near the boundary. The rig is allowed
        # to "leak" momentum at the boundary, which is what makes B orthogonal
        # to J in a physically meaningful (mass-weighted) way.
        try:
            import igl
            bI = np.unique(igl.boundary_facets(T)[0])
        except Exception:
            # fall back to outer-rim vertices if igl is unavailable
            bI = np.unique(T[_boundary_edges(T)])
        d = diffuse_field(X, T, bI, np.ones((bI.shape[0], 1)), dt=1,
                          normalize=True)
        De = sps.kron(sps.diags((1 - d).flatten()), sps.identity(dim))
        O = De @ Mv @ J
        Aeq = lbs_weight_space_constraint(X, T, O.T)

        # ---- rig-orthogonal skinning eigenmodes + cubature ---------------
        W, _E, B = skinning_eigenmodes(X, T, num_modes, Aeq=Aeq)
        B = orthonormalize(B, M=Mv)
        _cI, _cW, labels = spectral_cubature(X, T, W, num_clusters,
                                             return_labels=True)
        P, _Pm = cluster_grouping_matrices(np.asarray(labels).flatten(), X, T)
        PAMue = sps.kron(P @ AMu, sps.identity(dim * dim))

        # ---- precomputed operators (the FastCoDySim "system") ------------
        self.J, self.B = J, B
        self.AMuPGB = (PAMue @ G) @ B
        self.AMuPGJ = (PAMue @ G) @ J
        self.BMB = B.T @ Mv @ B
        self.BMJ = B.T @ Mv @ J
        self.BLJ = B.T @ L @ J
        Hsys = B.T @ L @ B + self.BMB / self.h ** 2
        self.chol = sp.linalg.cho_factor(Hsys)

        # rest rig pose (closest affine fit to the identity map)
        JMJ = J.T @ Mv @ J
        JMy = J.T @ Mv @ X.reshape(-1, 1)
        self.p0 = project_into_subspace(X.reshape(-1, 1), J, M=Mv,
                                        BMB=JMJ, BMy=JMy)

        # ---- state -------------------------------------------------------
        self.z = np.zeros((B.shape[1], 1))
        self.z_dot = np.zeros_like(self.z)
        self.p = self.p0.copy()
        self.p_dot = np.zeros_like(self.p)
        self.p_target = self.p0.copy()       # mouse writes the translation here
        self.U = self._positions()

    # -- the two FastCoDy block-coordinate sub-steps ----------------------
    def _local(self, zv, p_next):
        c = self.AMuPGB @ zv + self.AMuPGJ @ p_next
        return polar_svd(c.reshape(-1, self.dim, self.dim))[0].reshape(-1, 1)

    def _global(self, r, y, q, p_next):
        rhs = (self.BMB @ y / self.h ** 2 - self.BMJ @ q / self.h ** 2
               + self.AMuPGB.T @ r - self.BLJ @ p_next)
        return sp.linalg.cho_solve(self.chol, rhs)

    def step(self):
        h = self.h
        p_next = self.p_target.copy()
        y = self.z + h * self.z_dot
        q = p_next - (self.p + h * self.p_dot)

        z_next = block_coord(
            self.z.copy(),
            lambda zv, r: self._global(r, y, q, p_next),
            lambda zv: self._local(zv, p_next),
            tolerance=0.0, max_iter=self.max_iter)

        self.z_dot = (z_next - self.z) / h
        self.p_dot = (p_next - self.p) / h
        self.z = z_next
        self.p = p_next.copy()
        self.U = self._positions()

    def reset(self):
        self.z[:] = 0.0
        self.z_dot[:] = 0.0
        self.p = self.p0.copy()
        self.p_dot[:] = 0.0
        self.p_target = self.p0.copy()
        self.U = self._positions()

    def _positions(self):
        return (self.J @ self.p + self.B @ self.z).reshape(self.n, self.dim)


def _boundary_edges(T):
    """Indices into T of boundary vertices (igl-free fallback)."""
    from collections import Counter
    edges = []
    for tri in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.append(tuple(sorted((int(tri[a]), int(tri[b])))))
    counts = Counter(edges)
    bv = np.unique([v for e, c in counts.items() if c == 1 for v in e])
    # map back to (rows,cols) of T touching a boundary vertex (shape-compatible)
    return np.isin(T, bv)


# ============================================================================
# Rig mouse handle: drag the global rig translation toward the cursor.
#
# Unlike the soft-pin MouseHandle2D, the rig is animator-controlled directly,
# so we steer p_target's translation block. The last two entries of p are the
# rig translation in (x, y) for an lbs_jacobian with a single bone.
# ============================================================================

class RigHandle2D:
    def __init__(self, sim, sel_pc=None):
        self.sim = sim
        self.sel_pc = sel_pc
        self.active = False
        self.status = ("left-click + drag anywhere to steer the rig; "
                       "release to let go")

    def update(self):
        import polyscope.imgui as psim
        from utils import screen_to_world_2d

        win_pos = psim.GetMousePos()
        if psim.IsMouseClicked(0):
            self.active = True
            if self.sel_pc is not None:
                self.sel_pc.set_enabled(True)
        if self.active and psim.IsMouseDown(0):
            world = screen_to_world_2d(win_pos)[:2].astype(float)
            # write the rig translation block (last `dim` entries of p)
            self.sim.p_target[-self.sim.dim:, 0] = world
        if psim.IsMouseReleased(0):
            self.active = False
            # ease the rig back to rest so the beam settles
            self.sim.p_target = self.sim.p0.copy()
            if self.sel_pc is not None:
                self.sel_pc.set_enabled(False)

    def refresh_markers(self):
        if self.sel_pc is None:
            return
        # show the rig's authored center (translation maps the origin)
        center = (self.sim.J @ self.sim.p).reshape(self.sim.n, self.sim.dim).mean(0)
        self.sel_pc.update_point_positions(center.reshape(1, self.sim.dim))


# ============================================================================
# Main viewer wiring (guarded so importing the file never opens a window).
# ============================================================================

def _make_beam(nx=60, ny=6, length=4.0, thickness=0.4):
    from utils import triangulated_grid
    return triangulated_grid(nx=nx, ny=ny, width=length, height=thickness)


def main():
    import polyscope.imgui as psim
    from utils import RollingPlot, Viewer2D

    X, T = _make_beam()
    sim = CoDySim(X, T, num_modes=10, num_clusters=20,
                  ym=1e5, rho=1e3, h=0.02, max_iter=10)

    viewer = Viewer2D(X, T, camera_distance=6.0)
    sel_pc, _ = viewer.add_handle_markers()
    handle = RigHandle2D(sim, sel_pc)

    sec_plot = RollingPlot("complementary motion  ||B z||", height=120.0)

    def callback():
        if psim.Button("Reset"):
            sim.reset()
            sec_plot.clear()
        psim.Text(handle.status)
        psim.Text(f"vertices: {sim.n}   triangles: {sim.T.shape[0]}   "
                  f"subspace dim: {sim.B.shape[1]}")

        handle.update()
        sim.step()
        viewer.refresh(sim.U)
        handle.refresh_markers()

        sec_plot.push(float(np.linalg.norm(sim.B @ sim.z)))
        sec_plot.draw()

    viewer.show(callback)


if __name__ == "__main__":
    main()
