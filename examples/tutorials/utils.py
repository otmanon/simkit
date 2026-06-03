"""Shared scaffolding for the SimKit tutorials.

The interesting code -- the simulator, with its energy / gradient / hessian /
step -- lives inside each tutorial file. This module is just the support layer:

* Mesh generators (``triangulated_grid``, ``tetrahedralized_grid``,
  ``ball_mesh_2d``) and small numeric helpers (``lame_from_E_nu``,
  ``screen_to_world_2d``).
* Polyscope wrappers (``Viewer2D``, ``Viewer3D``) that just register a mesh and
  expose pin / handle marker point clouds.
* A 3D mouse handle (``MouseHandle3D``) that picks the nearest vertex on left-
  click, drags it on a camera-facing plane, and writes the resulting soft-pin
  ``(Q_h, b_h)`` matrices directly onto the sim object(s).
* A reusable imgui control panel (``TutorialUI3D``) that draws the standard
  reset / integrator / dt / material / contact-K / handle-mode controls and
  propagates their values to every sim it owns.
"""
from __future__ import annotations

import numpy as np
import scipy as sp

# Polyscope is only needed by the *interactive* tutorials. The offline
# notebook tutorials import this module purely for the mesh generators,
# material conversions, and the matplotlib helpers at the bottom of the file,
# and may run headless (CI, nbconvert) where polyscope cannot open a window.
# So we import it lazily and leave ``ps`` / ``psim`` as ``None`` if absent;
# anything that actually touches polyscope will raise a clear error on use.
try:
    import polyscope as ps
    import polyscope.imgui as psim
except Exception:  # pragma: no cover - headless / not installed
    ps = None
    psim = None

from simkit.dirichlet_penalty import dirichlet_penalty


# =============================================================================
# Mesh generators
# =============================================================================

def triangulated_grid(nx, ny, width=2.0, height=1.0):
    """Right-triangulated rectangular grid in the xy-plane, centered on origin."""
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    X = np.stack([XX.ravel(), YY.ravel()], axis=1)

    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
    v00 = (j * nx + i).ravel()
    v01 = (j * nx + i + 1).ravel()
    v10 = ((j + 1) * nx + i).ravel()
    v11 = ((j + 1) * nx + i + 1).ravel()
    T = np.stack([
        np.stack([v00, v01, v11], axis=1),
        np.stack([v00, v11, v10], axis=1),
    ], axis=1).reshape(-1, 3)
    return X, T


def tetrahedralized_grid(nx, ny, nz, width=1.0, height=1.0, depth=1.0):
    """Tet-meshed rectangular brick (5-tet-per-hex, parity-flipped to match faces)."""
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    pattern_even = np.array([
        [0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6],
    ])
    pattern_odd = np.array([
        [0, 1, 2, 5], [0, 2, 3, 7], [0, 5, 7, 4], [2, 5, 6, 7], [0, 2, 7, 5],
    ])

    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    ii = ii.ravel(); jj = jj.ravel(); kk = kk.ravel()

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    corners = np.stack([
        vid(ii,     jj,     kk    ),
        vid(ii + 1, jj,     kk    ),
        vid(ii + 1, jj + 1, kk    ),
        vid(ii,     jj + 1, kk    ),
        vid(ii,     jj,     kk + 1),
        vid(ii + 1, jj,     kk + 1),
        vid(ii + 1, jj + 1, kk + 1),
        vid(ii,     jj + 1, kk + 1),
    ], axis=1)

    even_mask = ((ii + jj + kk) % 2) == 0
    tets = np.empty((corners.shape[0], 5, 4), dtype=corners.dtype)
    tets[even_mask] = corners[even_mask][:, pattern_even]
    tets[~even_mask] = corners[~even_mask][:, pattern_odd]
    return X, tets.reshape(-1, 4)


def ball_mesh_2d(radius=0.15, n_segments=48):
    """Triangulated 2D disk: 1 center + n_segments boundary vertices."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    boundary = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    X = np.vstack([np.zeros((1, 2)), boundary])
    rim = np.arange(1, n_segments + 1)
    T = np.stack([np.zeros(n_segments, dtype=int), rim, np.roll(rim, -1)], axis=1)
    return X, T


# =============================================================================
# Material conversions
# =============================================================================

def lame_from_E_nu(E, nu):
    """Young's modulus + Poisson ratio -> (mu, lambda) Lame parameters."""
    mu_s = E / (2.0 * (1.0 + nu))
    lam_s = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu_s, lam_s


# =============================================================================
# 2D screen-space picking
# =============================================================================

def screen_to_world_2d(win_pos):
    """Cursor pixel -> world point on the z=0 plane (used in the 2D demos)."""
    W, H = ps.get_window_size()
    u = win_pos[0] / W
    v = win_pos[1] / H
    params = ps.get_view_camera_parameters()
    ul, ur, ll, lr = params.generate_camera_ray_corners()
    pos = params.get_position()
    top = (1 - u) * np.array(ul) + u * np.array(ur)
    bot = (1 - u) * np.array(ll) + u * np.array(lr)
    ray_dir = (1 - v) * top + v * bot
    t = -pos[2] / ray_dir[2]
    return (pos + t * ray_dir)[:2]


# =============================================================================
# Color palette (shared across tutorials)
# =============================================================================

LIGHT_GREEN = np.array([153, 216, 201]) / 255
BLUE        = np.array([0.2, 0.4, 0.85])
RED         = np.array([0.85, 0.2, 0.2])
GRAY        = np.array([0.4, 0.4, 0.4])
BLACK       = np.array([0.0, 0.0, 0.0])
BALL_COLOR  = np.array([0.95, 0.45, 0.35])


# =============================================================================
# RollingPlot - one rolling-window line plot in the imgui sidebar
# =============================================================================

class RollingPlot:
    """Append a sample each frame; draw via ``imgui.PlotLines``."""

    def __init__(self, label, length=200, height=120.0, fmt="{:.3f}"):
        self.label = label
        self.length = length
        self.height = height
        self.fmt = fmt
        self.values = []

    def push(self, v):
        self.values.append(float(v))
        del self.values[: -self.length]

    def clear(self):
        self.values.clear()

    def draw(self):
        last = self.values[-1] if self.values else 0.0
        psim.PlotLines(
            f"{self.label} (last {self.length})",
            self.values,
            overlay_text=f"{self.label}: {self.fmt.format(last)}",
            graph_size=(0.0, self.height),
        )


# =============================================================================
# Viewer2D - polyscope wrapper for 2D scenes
# =============================================================================

def init_2d_scene(camera_distance=5.0, own_mouse=True):
    """Bare ``ps.init`` + 2D camera setup. Use when you want to manage your own
    ``register_*`` calls; otherwise use :class:`Viewer2D`.
    """
    ps.init()
    ps.remove_all_structures()
    ps.look_at(np.array([0, 0, camera_distance]), np.array([0, 0, 0]))
    ps.set_ground_plane_mode("none")
    if own_mouse:
        ps.set_do_default_mouse_interaction(False)


class Viewer2D:
    """Set up polyscope for a 2D simulation tutorial.

    Auto-registers a surface mesh + vertex point cloud and exposes helpers for
    pin markers, handle markers, custom point clouds and ``ps.show``. Call
    ``refresh(U)`` from your callback after the sim step.
    """

    def __init__(self, X, T, camera_distance=5.0, point_radius=0.012,
                 edge_width=2, mesh_color=None, own_mouse=True):
        self.dim = X.shape[1]
        init_2d_scene(camera_distance=camera_distance, own_mouse=own_mouse)
        color = LIGHT_GREEN if mesh_color is None else mesh_color
        self.mesh = ps.register_surface_mesh(
            "mesh", X, T, material="flat", color=color, edge_width=edge_width)
        self.pc = ps.register_point_cloud(
            "vertices", X, radius=point_radius, material="flat", color=BLACK)

    def refresh(self, U):
        self.mesh.update_vertex_positions(U)
        self.pc.update_point_positions(U)

    def add_pin_markers(self, X_pin, name="pinned", color=None, radius=0.022):
        return ps.register_point_cloud(
            name, X_pin, radius=radius, material="flat",
            color=BLUE if color is None else color)

    def add_handle_markers(self, name="handle", color=None, radius=0.028):
        c = RED if color is None else color
        zero = np.zeros((1, self.dim))
        sel = ps.register_point_cloud(
            f"{name} selected", zero, radius=radius, material="flat",
            color=c, enabled=False)
        tgt = ps.register_point_cloud(
            f"{name} target", zero, radius=radius, material="flat",
            color=c, enabled=False)
        return sel, tgt

    def add_floor_line(self, y, x_range=(-3.0, 3.0), color=None):
        nodes = np.array([[x_range[0], y, -0.005], [x_range[1], y, -0.005]])
        edges = np.array([[0, 1]])
        return ps.register_curve_network(
            "floor", nodes, edges, material="flat",
            color=GRAY if color is None else color, radius=0.006)

    def add_ball(self, ball_X, ball_T, center, name="ball", color=None):
        return ps.register_surface_mesh(
            name, ball_X + np.asarray(center)[None, :], ball_T, material="flat",
            color=BALL_COLOR if color is None else color, edge_width=1)

    def show(self, callback):
        ps.set_user_callback(callback)
        ps.show()


# =============================================================================
# Viewer3D - polyscope wrapper for 3D scenes
# =============================================================================

class Viewer3D:
    """Polyscope wrapper for a 3D tutorial. Registers a volume mesh."""

    def __init__(self, X, T, floor_y=None, mesh_color=None, own_mouse=True,
                 camera_eye=(2.0, 1.2, 3.5), camera_target=(0.0, 0.0, 0.0)):
        ps.init()
        ps.remove_all_structures()
        ps.set_up_dir("y_up")
        ps.set_front_dir("z_front")
        ps.set_ground_plane_mode("tile_reflection" if floor_y is not None else "none")
        ps.look_at(np.asarray(camera_eye, dtype=float),
                   np.asarray(camera_target, dtype=float))
        if own_mouse:
            ps.set_do_default_mouse_interaction(False)

        color = LIGHT_GREEN if mesh_color is None else mesh_color
        self.mesh = ps.register_volume_mesh(
            "block", X, tets=T, color=color, interior_color=color,
            edge_width=1.0, material="flat")

        if floor_y is not None:
            bb_lo = np.array([
                float(X[:, 0].min()) - 1.0, floor_y, float(X[:, 2].min()) - 1.0])
            bb_hi = np.array([
                float(X[:, 0].max()) + 1.0,
                float(X[:, 1].max()) + 1.0,
                float(X[:, 2].max()) + 1.0])
            ps.set_automatically_compute_scene_extents(False)
            ps.set_bounding_box(bb_lo, bb_hi)

    def refresh(self, U):
        self.mesh.update_vertex_positions(U)

    def add_handle_markers(self, name="handle", color=None, radius=0.012):
        c = RED if color is None else color
        zero = np.zeros((1, 3))
        sel = ps.register_point_cloud(
            f"{name} selected", zero, radius=radius, material="flat",
            color=c, enabled=False)
        tgt = ps.register_point_cloud(
            f"{name} target", zero, radius=radius, material="flat",
            color=c, enabled=False)
        return sel, tgt

    def show(self, callback):
        ps.set_user_callback(callback)
        ps.show()


# =============================================================================
# Mouse handles - pick a vertex, drag, write soft-pin (Q_h, b_h) onto every
# sim they manage so any of them can be stepped without further plumbing.
# =============================================================================

class MouseHandle2D:
    """2D click-and-drag handle. Picks the nearest vertex within ``pick_radius``
    and follows the cursor on the z=0 plane. Writes ``Q_h`` / ``b_h`` onto every
    sim it owns; release zeros them again.
    """

    def __init__(self, sims, sel_pc=None, target_pc=None,
                 K_handle=1e4, pick_radius=0.15):
        self.sims = sims if isinstance(sims, dict) else {"_": sims}
        self.sim = next(iter(self.sims.values()))
        self.sel_pc = sel_pc
        self.target_pc = target_pc
        self.K_handle = float(K_handle)
        self.pick_radius = float(pick_radius)
        self.n = self.sim.n
        self.dim = self.sim.dim
        self.D = self.n * self.dim
        self.idx = None
        self.target = None
        self.status = "left-click a vertex to grab; drag to move; release to drop"

    def update(self):
        win_pos = psim.GetMousePos()
        if psim.IsMouseClicked(0):
            pos = screen_to_world_2d(win_pos)
            dist = np.linalg.norm(self.sim.U - pos.reshape(-1, 2), axis=1)
            nearest = int(np.argmin(dist))
            if dist[nearest] < self.pick_radius:
                self.idx = nearest
                self.target = self.sim.U[nearest].copy()
                self._write_handle()
                self._set_markers_enabled(True)
                self.status = f"grabbed vertex {nearest}"
        if self.idx is not None and psim.IsMouseDown(0):
            self.target = screen_to_world_2d(win_pos)[: self.dim].astype(float).copy()
            self._write_handle()
        if psim.IsMouseReleased(0):
            if self.idx is not None:
                self.status = "released - click another vertex to grab again"
            self._clear()

    def refresh_markers(self):
        if self.idx is None:
            return
        if self.sel_pc is not None:
            self.sel_pc.update_point_positions(
                self.sim.U[self.idx].reshape(1, self.dim))
        if self.target_pc is not None:
            self.target_pc.update_point_positions(
                self.target.reshape(1, self.dim))

    def _write_handle(self):
        bI = np.array([self.idx])
        y = self.target.reshape(1, self.dim)
        Q, b = dirichlet_penalty(bI, y, self.n, self.K_handle)
        for s in self.sims.values():
            s.Q_h = Q
            s.b_h = b

    def _clear(self):
        self.idx = None
        self.target = None
        self._set_markers_enabled(False)
        Q0 = sp.sparse.csc_matrix((self.D, self.D))
        b0 = np.zeros((self.D, 1))
        for s in self.sims.values():
            s.Q_h = Q0
            s.b_h = b0

    def _set_markers_enabled(self, on):
        if self.sel_pc is not None:
            self.sel_pc.set_enabled(on)
        if self.target_pc is not None:
            self.target_pc.set_enabled(on)


# -----------------------------------------------------------------------------
# 3D handle: drags on a camera-facing plane through the picked point.
# -----------------------------------------------------------------------------

def _is_finite(p):
    return p is not None and np.all(np.isfinite(p))


def _pick_world_point(win_pos):
    try:
        p = ps.screen_coords_to_world_position(np.asarray(win_pos, dtype=float))
    except Exception:
        return None
    p = np.asarray(p, dtype=float)
    return p if _is_finite(p) else None


def _cursor_ray(win_pos):
    params = ps.get_view_camera_parameters()
    cam_pos = np.asarray(params.get_position(), dtype=float)
    ray_dir = np.asarray(
        ps.screen_coords_to_world_ray(np.asarray(win_pos, dtype=float)), dtype=float)
    nrm = np.linalg.norm(ray_dir)
    if nrm > 0:
        ray_dir = ray_dir / nrm
    return cam_pos, ray_dir


def _ray_plane(ray_o, ray_d, plane_p, plane_n):
    denom = float(np.dot(ray_d, plane_n))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(plane_p - ray_o, plane_n) / denom)
    return ray_o + t * ray_d


class MouseHandle3D:
    """3D click-and-drag handle.

    Accepts either a single sim or a dict ``{name: sim}``. On left-click it
    finds the nearest vertex of ``self.sim.U``; while held, it intersects the
    cursor ray with a camera-facing plane through the picked point and writes
    the corresponding soft-pin matrices ``sim.Q_h`` / ``sim.b_h`` onto every
    sim it owns. Release zeros them again.

    ``self.sim`` is the picking source; ``TutorialUI3D`` re-points it when the
    user switches integrators so the picked vertex is always read off the
    just-stepped state.
    """

    def __init__(self, sims, sel_pc=None, target_pc=None, K_handle=1e4):
        self.sims = sims if isinstance(sims, dict) else {"_": sims}
        self.sim = next(iter(self.sims.values()))
        self.sel_pc = sel_pc
        self.target_pc = target_pc
        self.K_handle = float(K_handle)
        self.n = self.sim.n
        self.dim = self.sim.dim
        self.D = self.n * self.dim

        self.idx = None
        self.target = None
        self._plane_p = None
        self._plane_n = None
        self.status = "left-click a vertex to grab; drag to move; release to drop"

    # ---- public API used by tutorials -------------------------------------
    def update(self):
        win_pos = psim.GetMousePos()
        if psim.IsMouseClicked(0):
            hit = _pick_world_point(win_pos)
            if hit is None:
                self._clear()
                self.status = "click landed on empty space"
            else:
                d = np.linalg.norm(self.sim.U - hit.reshape(1, self.dim), axis=1)
                nearest = int(np.argmin(d))
                self.idx = nearest
                self.target = self.sim.U[nearest].copy()
                self._write_handle()
                self._plane_p = hit.copy()
                self._plane_n = np.asarray(
                    ps.get_view_camera_parameters().get_look_dir(), dtype=float)
                self._set_markers_enabled(True)
                self.status = f"grabbed vertex {nearest}"
        if self.idx is not None and psim.IsMouseDown(0):
            ray_o, ray_d = _cursor_ray(win_pos)
            new_target = _ray_plane(ray_o, ray_d, self._plane_p, self._plane_n)
            if _is_finite(new_target):
                self.target = np.asarray(new_target, dtype=float).copy()
                self._write_handle()
        if psim.IsMouseReleased(0):
            if self.idx is not None:
                self.status = "released - click another vertex to grab again"
            self._clear()

    def refresh_markers(self):
        if self.idx is None:
            return
        if self.sel_pc is not None:
            self.sel_pc.update_point_positions(
                self.sim.U[self.idx].reshape(1, self.dim))
        if self.target_pc is not None:
            self.target_pc.update_point_positions(
                self.target.reshape(1, self.dim))

    # ---- internals --------------------------------------------------------
    def _write_handle(self):
        bI = np.array([self.idx])
        y = self.target.reshape(1, self.dim)
        Q, b = dirichlet_penalty(bI, y, self.n, self.K_handle)
        for s in self.sims.values():
            s.Q_h = Q
            s.b_h = b

    def _clear(self):
        self.idx = None
        self.target = None
        self._plane_p = None
        self._plane_n = None
        self._set_markers_enabled(False)
        Q0 = sp.sparse.csc_matrix((self.D, self.D))
        b0 = np.zeros((self.D, 1))
        for s in self.sims.values():
            s.Q_h = Q0
            s.b_h = b0

    def _set_markers_enabled(self, on):
        if self.sel_pc is not None:
            self.sel_pc.set_enabled(on)
        if self.target_pc is not None:
            self.target_pc.set_enabled(on)


# =============================================================================
# TutorialUI - configurable imgui control panel.
# =============================================================================

class TutorialUI:
    """Imgui control panel that owns slider state and propagates changes.

    Pass a dict ``{name: sim}`` of one sim per integrator. ``draw()`` emits the
    enabled controls (reset / integrator / dt / E + nu / K_contact / handle-vs-
    camera) and writes their values to every sim. ``self.sim`` is the active
    integrator; the tutorial callback steps that one.

    Flags
    -----
    show_integrator : bool
        Combo to pick the active integrator. State is copied from the previous
        active sim on switch (history slots are reset to the handed-off pose).
    show_dt : bool
        ``dt (h)`` slider; mirrored onto every sim's ``.h``.
    show_material : bool
        ``log10 E`` and ``log10 (0.5 - nu)`` sliders; mirrored onto each
        sim's ``.mu`` / ``.lam`` arrays.
    show_contact_K : bool
        ``log10 K_contact`` slider; mirrored onto each sim's ``.K_contact``.
    show_handle_mode : bool
        3D only: checkbox to swap between handle-grab and orbit-camera mouse
        ownership.
    """

    def __init__(self, sims, handle=None, *,
                 show_integrator=True, show_dt=True, show_material=True,
                 show_contact_K=False, show_handle_mode=False,
                 log_E=5.0, log_nu_compl=-1.0, log_K_contact=5.0,
                 h=0.02, dt_range=(0.001, 0.05), handle_enabled=True):
        self.sims = sims
        self.names = list(sims.keys())
        self.handle = handle
        self.active_idx = 0

        self.show_integrator = show_integrator
        self.show_dt = show_dt
        self.show_material = show_material
        self.show_contact_K = show_contact_K
        self.show_handle_mode = show_handle_mode

        self.log_E = log_E
        self.log_nu_compl = log_nu_compl
        self.log_K_contact = log_K_contact
        self.h = h
        self.dt_min, self.dt_max = dt_range
        self.handle_enabled = handle_enabled
        self.status = ""
        self.reset_hooks = []   # extra cleanups to run on Reset (plot.clear, etc.)
        self.switch_hooks = []  # extra cleanups to run on integrator switch

        # push initial slider values onto sims so they match the UI state
        if show_material:
            self._apply_material()
        if show_contact_K:
            self._apply_contact_K()
        if show_dt:
            self._apply_h()

    @property
    def sim(self):
        return self.sims[self.names[self.active_idx]]

    def on_reset(self, fn):
        """Register an extra callback invoked when the Reset button is clicked."""
        self.reset_hooks.append(fn)

    def on_switch(self, fn):
        """Register an extra callback invoked when the integrator changes."""
        self.switch_hooks.append(fn)

    def draw(self):
        if psim.Button("Reset"):
            self._reset_all()

        if self.show_handle_mode and self.handle is not None:
            psim.SameLine()
            changed_mode, self.handle_enabled = psim.Checkbox(
                "Click-to-drag handle (uncheck for camera)", self.handle_enabled)
            if changed_mode:
                ps.set_do_default_mouse_interaction(not self.handle_enabled)
                self.handle._clear()
                self.status = ("handle mode: click a vertex to grab"
                               if self.handle_enabled
                               else "camera mode: polyscope owns the mouse")

        if self.show_integrator:
            old_idx = self.active_idx
            changed_int, self.active_idx = psim.Combo(
                "Integrator", self.active_idx, self.names)
            if changed_int:
                self._switch_integrator(old_idx)

        if self.show_dt:
            changed_h, self.h = psim.SliderFloat(
                "dt (h)", self.h, v_min=self.dt_min, v_max=self.dt_max)
            if changed_h:
                self._apply_h()

        if self.show_material:
            E_val = 10.0 ** self.log_E
            nu_val = 0.5 - 10.0 ** self.log_nu_compl
            changed_E, self.log_E = psim.SliderFloat(
                f"log10 Young's E  (E = {E_val:.2e})",
                self.log_E, v_min=2.0, v_max=8.0)
            changed_nu, self.log_nu_compl = psim.SliderFloat(
                f"log10 (0.5 - nu)  (nu = {nu_val:.4f})",
                self.log_nu_compl, v_min=-4.0, v_max=-0.31)
            if changed_E or changed_nu:
                self._apply_material()

        if self.show_contact_K:
            K_val = 10.0 ** self.log_K_contact
            changed_K, self.log_K_contact = psim.SliderFloat(
                f"log10 contact penalty  (K = {K_val:.2e})",
                self.log_K_contact, v_min=2.0, v_max=9.0)
            if changed_K:
                self._apply_contact_K()

        msg = self.status or (self.handle.status if self.handle else "")
        if msg:
            psim.Text(msg)

    # ------- writes that fan out to every sim -------------------------------
    def _apply_h(self):
        for s in self.sims.values():
            s.h = float(self.h)

    def _apply_material(self):
        E_val = 10.0 ** self.log_E
        nu_val = 0.5 - 10.0 ** self.log_nu_compl
        mu, lam = lame_from_E_nu(E_val, nu_val)
        for s in self.sims.values():
            if hasattr(s, "mu") and hasattr(s.mu, "__setitem__"):
                s.mu[:] = mu
                s.lam[:] = lam

    def _apply_contact_K(self):
        K = 10.0 ** self.log_K_contact
        for s in self.sims.values():
            if hasattr(s, "K_contact"):
                s.K_contact = float(K)

    # ------- integrator switch + reset -------------------------------------
    def _switch_integrator(self, old_idx):
        old = self.sims[self.names[old_idx]]
        new = self.sim
        new.U[:] = old.U
        # reset history: every prev slot starts at the just-handed-off pose
        for attr in ("U_prev", "U_prev2", "U_prev3"):
            if hasattr(new, attr):
                getattr(new, attr)[:] = old.U
        if hasattr(new, "V"):
            new.V[:] = 0.0
        if self.handle is not None:
            self.handle.sim = new
            self.handle._clear()
        for fn in self.switch_hooks:
            fn()

    def _reset_all(self):
        for s in self.sims.values():
            s.U[:] = s.X
            for attr in ("U_prev", "U_prev2", "U_prev3"):
                if hasattr(s, attr):
                    getattr(s, attr)[:] = s.X
            if hasattr(s, "V"):
                s.V[:] = 0.0
        if self.handle is not None:
            self.handle._clear()
        for fn in self.reset_hooks:
            fn()
        self.status = "scene reset"


# =============================================================================
# =============================================================================
# Matplotlib helpers for the OFFLINE notebook tutorials.
#
# Everything below is pure matplotlib -- no polyscope, no imgui. The notebooks
# keep the *physics* visible (deformation gradients, energies, Newton / GD
# loops) and lean on these helpers for the drawing / animation plumbing so the
# narrative isn't drowned in plotting boilerplate.
#
# Conventions
# -----------
# * A "state" is an (n, 2) array of deformed vertex positions.
# * "states" is a list/array of such frames, one per animation step.
# * Material colors are shared across every plot so the same energy always has
#   the same color.
# =============================================================================
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Embed reasonably long inline animations in notebooks.
plt.rcParams["animation.embed_limit"] = 100.0

# ---- shared colors ----------------------------------------------------------
MAT_COLORS = {
    "Linear":      "#d62728",   # red    - linear elasticity
    "ARAP":        "#1f77b4",   # blue   - as-rigid-as-possible
    "Neo-Hookean": "#2ca02c",   # green  - (stable) neo-hookean
}
SOLVER_COLORS = {
    "Newton":           "#1f77b4",
    "Gradient Descent": "#d62728",
}
MESH_FACE = "#9ecae1"   # light blue fill
MESH_EDGE = "#08519c"   # dark blue edges
TRI_FACE  = "#a1d99b"   # light green fill
TRI_EDGE  = "#00441b"   # dark green edges
REST_EDGE = "#bdbdbd"   # gray ghost of the rest shape
HANDLE_C  = "#e6550d"   # orange handle marker
PIN_C     = "#3182bd"   # blue pin marker


# ---- axis setup -------------------------------------------------------------

def setup_axes(ax, xlim=None, ylim=None, title=None, equal=True, grid=True):
    """Standard 2D scene axis: equal aspect, light grid, optional limits."""
    if equal:
        ax.set_aspect("equal")
    if grid:
        ax.grid(True, color="0.9", linewidth=0.8, zorder=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)
    return ax


def auto_limits(states, pad=0.4):
    """Bounding box over every frame, padded -- so the camera never clips."""
    allpts = np.concatenate([np.asarray(s) for s in states], axis=0)
    lo = allpts.min(axis=0)
    hi = allpts.max(axis=0)
    return (float(lo[0] - pad), float(hi[0] + pad)), (float(lo[1] - pad), float(hi[1] + pad))


# ---- artists (each has .update(U)) ------------------------------------------

class TriangleArtist:
    """A single filled triangle + its vertices, optionally ghosting the rest
    shape. ``update(U)`` moves it to a new deformed state."""

    def __init__(self, ax, U, facecolor=TRI_FACE, edgecolor=TRI_EDGE, lw=2.5,
                 rest=None, vertex_color=TRI_EDGE, vertex_size=70, zorder=2):
        U = np.asarray(U, dtype=float)
        self.rest_poly = None
        if rest is not None:
            self.rest_poly = Polygon(np.asarray(rest), closed=True, fill=False,
                                     edgecolor=REST_EDGE, lw=1.5, linestyle="--",
                                     zorder=zorder)
            ax.add_patch(self.rest_poly)
        self.poly = Polygon(U, closed=True, facecolor=facecolor,
                            edgecolor=edgecolor, lw=lw, zorder=zorder + 1, alpha=0.9)
        ax.add_patch(self.poly)
        self.verts = ax.scatter(U[:, 0], U[:, 1], s=vertex_size, color=vertex_color,
                                zorder=zorder + 2)

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.poly.set_xy(U)
        self.verts.set_offsets(U)


class PolyMeshArtist:
    """A triangulated mesh drawn as a ``PolyCollection``. ``update(U)`` moves
    every triangle. Used for the cantilever-beam tutorials."""

    def __init__(self, ax, U, T, facecolor=MESH_FACE, edgecolor=MESH_EDGE,
                 lw=1.0, zorder=2, alpha=0.95):
        self.T = np.asarray(T)
        U = np.asarray(U, dtype=float)
        self.coll = PolyCollection([U[f] for f in self.T], facecolors=facecolor,
                                   edgecolors=edgecolor, linewidths=lw,
                                   zorder=zorder, alpha=alpha)
        ax.add_collection(self.coll)

    def update(self, U):
        U = np.asarray(U, dtype=float)
        self.coll.set_verts([U[f] for f in self.T])


def format_F(F):
    """Pretty 2x2 deformation-gradient string for an in-figure text box."""
    F = np.asarray(F).reshape(2, 2)
    return ("$F$ =\n"
            f"[{F[0,0]:+5.2f}  {F[0,1]:+5.2f}]\n"
            f"[{F[1,0]:+5.2f}  {F[1,1]:+5.2f}]\n"
            f"det $F$ = {np.linalg.det(F):+5.2f}")


def text_box(ax, s, loc="upper left"):
    """Monospace text box pinned to a corner of the axes (data-independent)."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
                   family="monospace", fontsize=11,
                   bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))


# ---- progressive line plot (the "plot drawn as the scene happens") ----------

class TracePlot:
    """One or more curves revealed progressively. Pre-computes the full series,
    then ``update(i)`` shows samples ``0..i`` plus a moving marker at ``i``.
    """

    def __init__(self, ax, xs, series, colors=None, xlabel="", ylabel="",
                 logy=False, title=None, lw=2.0, ylim=None):
        self.ax = ax
        self.xs = np.asarray(xs, dtype=float)
        self.series = {k: np.asarray(v, dtype=float) for k, v in series.items()}
        colors = colors or {}
        self.lines, self.marks = {}, {}
        for name, ys in self.series.items():
            c = colors.get(name)
            (ln,) = ax.plot([], [], color=c, lw=lw, label=name, zorder=2)
            (mk,) = ax.plot([], [], "o", color=ln.get_color(), ms=7, zorder=3)
            self.lines[name] = ln
            self.marks[name] = mk
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, color="0.9", linewidth=0.8)
        if ylim is not None:                 # explicit range (e.g. a flat-zero curve)
            xmn, xmx = float(self.xs.min()), float(self.xs.max())
            self.ax.set_xlim(xmn, xmx + 1e-12 if xmx == xmn else xmx)
            self.ax.set_ylim(*ylim)
        else:
            self._set_limits(logy)
        if len(self.series) > 1:
            ax.legend(loc="best", fontsize=9)

    def _set_limits(self, logy):
        xmn, xmx = float(self.xs.min()), float(self.xs.max())
        self.ax.set_xlim(xmn, xmx + 1e-12 if xmx == xmn else xmx)
        allv = np.concatenate([v[np.isfinite(v)] for v in self.series.values()])
        if logy:
            pos = allv[allv > 0]
            lo = pos.min() / 3 if pos.size else 1e-8
            hi = allv.max() * 3 if allv.size else 1.0
        else:
            span = (allv.max() - allv.min()) if allv.size else 1.0
            span = span if span > 0 else 1.0
            lo, hi = allv.min() - 0.08 * span, allv.max() + 0.08 * span
        self.ax.set_ylim(lo, hi)

    def update(self, i):
        j = i + 1
        for name, ys in self.series.items():
            self.lines[name].set_data(self.xs[:j], ys[:j])
            self.marks[name].set_data([self.xs[i]], [ys[i]])


# ---- saving / embedding -----------------------------------------------------

def save_anim(anim, path, fps=20):
    """Save to .mp4 (ffmpeg) when possible, else fall back to .gif (pillow).
    Returns the path actually written."""
    path = str(path)
    if path.lower().endswith(".mp4"):
        try:
            anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400))
            return path
        except Exception:
            path = path[:-4] + ".gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    return path


def show_anim(anim, fps=15, width=600):
    """Inline display as a small **palette-optimized GIF** (base64 ``<img>``).

    Far lighter than ``to_jshtml`` (which embeds one PNG per frame) and, being a
    plain image, it renders *everywhere* -- including GitHub's notebook viewer.
    The full-quality ``.mp4`` is written separately by :func:`save_anim`; this
    only affects the inline preview. Pair with ``plt.close(fig)``.
    """
    import base64, os, shutil, subprocess, tempfile
    from IPython.display import HTML

    tmp = tempfile.mkdtemp()
    mp4, gif = os.path.join(tmp, "a.mp4"), os.path.join(tmp, "a.gif")
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=fps))            # render frames once
        ffmpeg = (shutil.which("ffmpeg")
                  or plt.rcParams.get("animation.ffmpeg_path") or "ffmpeg")
        # downscale + single-pass palette for a compact, clean GIF
        vf = (f"fps={fps},scale={width}:-1:flags=lanczos,"
              "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse")
        subprocess.run([ffmpeg, "-y", "-i", mp4, "-vf", vf, gif],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = open(gif, "rb").read()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    b64 = base64.b64encode(data).decode("ascii")
    return HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%;"/>')


# ---- high-level one-call animations -----------------------------------------

def animate_deformation(states, F_list, rest, *, lims=None, fps=20,
                        title="Deformation gradient", interval=None):
    """Triangle deforming with a live 2x2 ``F`` read-out (tutorial 1).

    ``states[i]`` is the deformed triangle, ``F_list[i]`` its deformation
    gradient. The rest shape is ghosted so translations are obvious.
    Returns ``(fig, anim)``.
    """
    if lims is None:
        xlim, ylim = auto_limits(list(states) + [rest])
    else:
        xlim, ylim = lims
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    setup_axes(ax, xlim, ylim, title=title)
    tri = TriangleArtist(ax, states[0], rest=rest)
    txt = text_box(ax, format_F(F_list[0]))

    def update(i):
        tri.update(states[i])
        txt.set_text(format_F(F_list[i]))
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_scene_energy(states, xs, series, *, scene="triangle", T=None,
                         rest=None, lims=None, xlabel="", ylabel="energy",
                         colors=None, logy=False, fps=20, title=None,
                         scene_title="", interval=None, energy_ylim=None):
    """Left: a deforming scene. Right: energy curve(s) traced in lock-step
    (tutorial 2). ``scene`` is ``"triangle"`` or ``"mesh"`` (needs ``T``).
    Pass ``energy_ylim`` to fix the energy-panel y-range (e.g. a flat-zero
    curve that would otherwise auto-zoom into numerical noise).
    Returns ``(fig, anim)``.
    """
    colors = colors or MAT_COLORS
    if lims is None:
        xlim, ylim = auto_limits(states)
    else:
        xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    setup_axes(axL, xlim, ylim, title=scene_title)
    if scene == "mesh":
        art = PolyMeshArtist(axL, states[0], T)
    else:
        art = TriangleArtist(axL, states[0], rest=rest)
    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel,
                      ylabel=ylabel, logy=logy, title=title, ylim=energy_ylim)

    def update(i):
        art.update(states[i])
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_mesh(states, T, *, lims=None, fps=20, title="", pin_pts=None,
                 handle_traj=None, target_pts=None, interval=None,
                 figsize=(7, 4)):
    """A deforming triangle mesh, with optional pinned-vertex markers and
    moving handle / target markers (tutorials 4-6). Returns ``(fig, anim)``.

    ``handle_traj[i]`` (if given) is the handle position(s) at frame ``i`` and
    ``target_pts[i]`` the goal(s); each may be a single ``(2,)`` point or an
    ``(k, 2)`` array of several points (e.g. a whole pinned edge).
    """
    if lims is None:
        xlim, ylim = auto_limits(states)
    else:
        xlim, ylim = lims
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title)
    mesh = PolyMeshArtist(ax, states[0], T)
    if pin_pts is not None and len(pin_pts):
        ax.scatter(np.asarray(pin_pts)[:, 0], np.asarray(pin_pts)[:, 1],
                   s=55, color=PIN_C, marker="s", zorder=5, label="pinned")
    hdl = tgt = None
    if handle_traj is not None:
        (hdl,) = ax.plot([], [], "o", color=HANDLE_C, ms=11, zorder=6, label="handle")
    if target_pts is not None:
        (tgt,) = ax.plot([], [], "x", color="0.3", ms=10, mew=2.5, zorder=6,
                         label="target")
    if pin_pts is not None or handle_traj is not None:
        ax.legend(loc="upper right", fontsize=9)

    def update(i):
        mesh.update(states[i])
        if hdl is not None:
            p = np.atleast_2d(handle_traj[i])
            hdl.set_data(p[:, 0], p[:, 1])
        if tgt is not None:
            p = np.atleast_2d(target_pts[i])
            tgt.set_data(p[:, 0], p[:, 1])
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


# ---- static plots -----------------------------------------------------------

def line_plot(xs, series, *, xlabel="", ylabel="", colors=None, logy=False,
              title=None, figsize=(6.5, 4.5), markers=False, ax=None, ylim=None):
    """Quick multi-series line plot with the shared material colors. Returns
    ``(fig, ax)``. ``series`` maps a label to a y-array over ``xs``. Pass an
    explicit ``ylim`` to stop matplotlib from zooming into numerical noise
    (e.g. a curve that is constant zero)."""
    colors = colors or MAT_COLORS
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    style = "-o" if markers else "-"
    for name, ys in series.items():
        ax.plot(xs, ys, style, color=colors.get(name), lw=2, ms=4, label=name)
    if logy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, color="0.9", linewidth=0.8)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=9)
    return fig, ax


def deformation_panels(cases, rest, *, lims=None, ncols=4, figsize=None):
    """Row of static triangle panels, each captioned with its name and ``F``
    (tutorial 1 summary). ``cases`` is a list of ``(name, U, F)``."""
    n = len(cases)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if lims is None:
        xlim, ylim = auto_limits([U for _, U, _ in cases] + [rest])
    else:
        xlim, ylim = lims
    figsize = figsize or (3.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for k, (name, U, F) in enumerate(cases):
        ax = axes[k // ncols][k % ncols]
        setup_axes(ax, xlim, ylim, title=name)
        TriangleArtist(ax, U, rest=rest)
        text_box(ax, format_F(F), loc="lower right")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    return fig, axes


def convergence_plot(metrics, *, title="", figsize=(13, 4)):
    """Three side-by-side semilog panels -- energy gap, gradient norm, Newton
    decrement -- one line per solver (tutorial 4).

    ``metrics`` maps a solver name to a dict with keys ``energy`` (per-iter
    objective), ``grad`` (gradient norm), and optionally ``decrement``.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # shared optimum so the energy gap is comparable across solvers
    e_min = min(np.min(m["energy"]) for m in metrics.values())
    panels = [
        ("energy gap  $E_k - E^\\star$", "energy"),
        ("gradient norm  $\\|\\nabla E_k\\|$", "grad"),
        ("Newton decrement  $\\lambda_k$", "decrement"),
    ]
    for ax, (lbl, key) in zip(axes, panels):
        for name, m in metrics.items():
            if key not in m:
                continue
            y = np.asarray(m[key], dtype=float)
            if key == "energy":
                y = y - e_min + 1e-16
            it = np.arange(len(y))
            ax.semilogy(it, np.maximum(y, 1e-16), "-o", ms=3,
                        color=SOLVER_COLORS.get(name), label=name)
        ax.set_xlabel("iteration")
        ax.set_title(lbl)
        ax.grid(True, which="both", color="0.9", linewidth=0.7)
        ax.legend(fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Helpers for the dynamics / contact / complexity tutorials (6-9)
# =============================================================================

from matplotlib.patches import Circle

INTEGRATOR_COLORS = {
    "Forward Euler":  "#d62728",
    "Backward Euler": "#1f77b4",
    "BDF2":           "#2ca02c",
}
ENERGY_COLORS = {
    "elastic":   "#2ca02c",
    "kinetic":   "#1f77b4",
    "contact":   "#d62728",
    "potential": "#9467bd",
    "total":     "#000000",
}
BALL_FACE = "#f1a340"
BALL_EDGE = "#b35806"


def loglog_plot(xs, series, *, xlabel="", ylabel="", colors=None, title=None,
                ref_slopes=None, figsize=(6.5, 5), markers=True, ax=None):
    """Log-log plot, optionally overlaying dashed reference-slope guide lines.

    ``ref_slopes`` is a dict ``{label: slope}``; each becomes a dashed guide of
    that slope anchored near the data, to eyeball convergence order. Returns
    ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = colors or {}
    xs = np.asarray(xs, dtype=float)
    style = "-o" if markers else "-"
    for name, ys in series.items():
        ax.loglog(xs, np.asarray(ys, dtype=float), style, color=colors.get(name),
                  lw=2, ms=5, label=name)
    if ref_slopes:
        allv = np.concatenate([np.asarray(v, float) for v in series.values()])
        anchor = np.exp(np.mean(np.log(allv[allv > 0]))) if np.any(allv > 0) else 1.0
        xmid = np.exp(np.mean(np.log(xs)))
        for lbl, p in ref_slopes.items():
            guide = anchor * (xs / xmid) ** p
            ax.loglog(xs, guide, "--", color="0.5", lw=1.3, label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", color="0.9", linewidth=0.7)
    ax.legend(fontsize=9)
    return fig, ax


def _draw_floor(ax, floor_y, xlim, ylim):
    ax.axhline(floor_y, color="0.3", lw=2.0, zorder=1)
    ax.fill_between(list(xlim), ylim[0], floor_y, color="0.88", zorder=1)


def animate_dynamics(states, T, xs, series, *, lims, xlabel="time (s)",
                     ylabel="energy", colors=None, fps=20, title=None,
                     scene_title="", ball_centers=None, ball_radius=None,
                     sdf=False, floor_y=None, floor_sdf=False,
                     mesh_face=MESH_FACE, mesh_edge=MESH_EDGE,
                     figsize=(11, 5), interval=None):
    """Left: a deforming mesh with optional moving ball (and its signed-distance
    background) and/or a static floor (optionally with its own signed-distance
    background, ``floor_sdf``). Right: one or more energy curves traced over time
    in lock-step (tutorials 8-9). Returns ``(fig, anim)``."""
    colors = colors or ENERGY_COLORS
    xlim, ylim = lims
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    setup_axes(axL, xlim, ylim, title=scene_title)

    gx = np.linspace(xlim[0], xlim[1], 140)
    gy = np.linspace(ylim[0], ylim[1], 140)
    GX, GY = np.meshgrid(gx, gy)
    sdf_state = {"cs": None}

    def draw_sdf(c):
        if sdf_state["cs"] is not None:
            sdf_state["cs"].remove()
        D = np.sqrt((GX - c[0]) ** 2 + (GY - c[1]) ** 2) - ball_radius
        vmax = float(np.abs(D).max())
        sdf_state["cs"] = axL.contourf(GX, GY, D, levels=24, cmap="coolwarm",
                                       alpha=0.45, zorder=0, vmin=-vmax, vmax=vmax)

    if floor_sdf and floor_y is not None:
        # static signed distance to the ground plane: phi = y - floor_y (blue below)
        D = GY - floor_y
        vmax = float(np.abs(D).max())
        axL.contourf(GX, GY, D, levels=24, cmap="coolwarm", alpha=0.45, zorder=0,
                     vmin=-vmax, vmax=vmax)
        axL.axhline(floor_y, color="0.3", lw=2.0, zorder=1)
    elif floor_y is not None:
        _draw_floor(axL, floor_y, xlim, ylim)
    mesh = PolyMeshArtist(axL, states[0], T, zorder=3, facecolor=mesh_face,
                          edgecolor=mesh_edge)
    ball = None
    if ball_centers is not None:
        if sdf:
            draw_sdf(ball_centers[0])
        ball = Circle(tuple(ball_centers[0]), ball_radius, facecolor=BALL_FACE,
                      edgecolor=BALL_EDGE, lw=1.5, zorder=4, alpha=0.9)
        axL.add_patch(ball)

    trace = TracePlot(axR, xs, series, colors=colors, xlabel=xlabel,
                      ylabel=ylabel, title=title)

    def update(i):
        mesh.update(states[i])
        if ball is not None:
            if sdf:
                draw_sdf(ball_centers[i])
            ball.center = (float(ball_centers[i][0]), float(ball_centers[i][1]))
        trace.update(i)
        return ()

    anim = FuncAnimation(fig, update, frames=len(states),
                         interval=interval or 1000 / fps, blit=False)
    return fig, anim


def animate_meshes_grid(panels, *, lims=None, fps=20, figsize=None,
                        suptitle=None, floor_y=None, interval=None):
    """Several deforming meshes side by side, played in sync (tutorials 6, 9).

    ``panels`` is a list of dicts, each ``{"states": [...], "T": T, "title": str}``.
    Panels may have different meshes and different frame counts (shorter ones
    hold on their last frame). Returns ``(fig, anim)``.
    """
    npan = len(panels)
    if lims is None:
        allstates = [s for p in panels for s in p["states"]]
        xlim, ylim = auto_limits(allstates)
    else:
        xlim, ylim = lims
    figsize = figsize or (4.6 * npan, 4.2)
    fig, axes = plt.subplots(1, npan, figsize=figsize, squeeze=False)
    axes = axes[0]
    arts = []
    nframes = max(len(p["states"]) for p in panels)
    for ax, p in zip(axes, panels):
        setup_axes(ax, xlim, ylim, title=p.get("title", ""))
        if floor_y is not None:
            _draw_floor(ax, floor_y, xlim, ylim)
        arts.append(PolyMeshArtist(ax, p["states"][0], p["T"]))

    def update(i):
        for art, p in zip(arts, panels):
            art.update(p["states"][min(i, len(p["states"]) - 1)])
        return ()

    anim = FuncAnimation(fig, update, frames=nframes,
                         interval=interval or 1000 / fps, blit=False)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, anim


def plot_ball_sdf(center, radius, lims, *, U=None, T=None, figsize=(6, 5),
                  title="Signed distance to the ball"):
    """Static filled-contour plot of the signed distance field
    ``phi(x) = |x - center| - radius`` (negative inside the ball), with the
    ball surface as the zero level set. Optionally overlays a mesh. Returns
    ``(fig, ax)``."""
    xlim, ylim = lims
    fig, ax = plt.subplots(figsize=figsize)
    setup_axes(ax, xlim, ylim, title=title)
    gx = np.linspace(xlim[0], xlim[1], 220)
    gy = np.linspace(ylim[0], ylim[1], 220)
    GX, GY = np.meshgrid(gx, gy)
    D = np.sqrt((GX - center[0]) ** 2 + (GY - center[1]) ** 2) - radius
    vmax = float(np.abs(D).max())
    cs = ax.contourf(GX, GY, D, levels=30, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                     zorder=0, alpha=0.85)
    ax.contour(GX, GY, D, levels=[0.0], colors="k", linewidths=1.5, zorder=1)
    fig.colorbar(cs, ax=ax, label=r"signed distance  $\phi(x)=\|x-c\|-r$")
    if U is not None and T is not None:
        PolyMeshArtist(ax, U, T, zorder=3, alpha=0.85)
    ax.add_patch(Circle(tuple(center), radius, fill=False, ec=BALL_EDGE,
                        lw=1.5, zorder=4))
    return fig, ax


