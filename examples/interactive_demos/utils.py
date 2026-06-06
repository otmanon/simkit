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
import polyscope as ps
import polyscope.imgui as psim
import scipy as sp

from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.pairwise_distance import pairwise_distance


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
            "vertices", X, radius=point_radius, material="flat", color=BLACK, enabled=False)

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
                 K_handle=1e4, pick_radius=0.15, selection_radius=0.1):
        self.sims = sims if isinstance(sims, dict) else {"_": sims}
        self.sim = next(iter(self.sims.values()))
        self.sel_pc = sel_pc
        self.target_pc = target_pc
        self.K_handle = float(K_handle)
        self.pick_radius = float(pick_radius)
        self.selection_radius = selection_radius
        self.n = self.sim.n
        self.dim = self.sim.dim
        self.D = self.n * self.dim
        self.idx0 = None
        self.P0 = None
        self.P = None
        self.idx = None
        self.target = None
        self.target0 = None
        self.target_displacement = None
        self.status = "left-click a vertex to grab; drag to move; release to drop"

    def update(self):
        win_pos = psim.GetMousePos()
        if psim.IsMouseClicked(0):
            pos = screen_to_world_2d(win_pos)
            dist = np.linalg.norm(self.sim.U - pos.reshape(-1, 2), axis=1)
            nearest = int(np.argmin(dist))
            if dist[nearest] < self.pick_radius:
                self.idx0 = nearest
                self.target0 = self.sim.U[self.idx0].copy()
                distance = pairwise_distance(self.sim.U, pos.reshape(1, -1)).flatten()
                self.idx = np.where(distance < self.selection_radius)[0]
                self.target_displacement = np.zeros((1, self.dim))
                self.P0 = self.sim.U[self.idx].copy()
                self.P = self.P0.copy()
                self.target = self.target0
        
                self._write_handle()
                self._set_markers_enabled(True)
                self.status = f"grabbed vertex {nearest}"
        if self.idx is not None and psim.IsMouseDown(0):
            self.target = screen_to_world_2d(win_pos)[: self.dim].astype(float).copy()
            self.target_displacement = self.target - self.target0
            self.P = self.P0 + self.target_displacement
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
                self.sim.U[self.idx0].reshape(-1, self.dim))
        if self.target_pc is not None:
            self.target_pc.update_point_positions(
                self.target.reshape(-1, self.dim))

    def _write_handle(self):
        bI = np.array([self.idx]).flatten()
        y = self.P.reshape(-1, self.dim)
        Q, b = dirichlet_penalty(bI, y, self.n, self.K_handle)
        for s in self.sims.values():
            s.Q_h = Q
            s.b_h = b

    def _clear(self):
        self.idx = None
        self.target = None
        self.target0 = None
        self.target_displacement = None

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

    def draw(self):
        psim.Text("Handle UI")
        pass
        # self.K_handle, changed_K = psim.SliderFloat(
        #     "K_handle", self.K_handle, v_min=1e3, v_max=1e6, log_scale=True)
        # self.selection_radius, changed_radius = psim.SliderFloat(
        #     "selection radius", self.selection_radius, v_min=0.01, v_max=0.5)
        psim.Text("Handle UI")
        changed, self.K_handle = psim.SliderFloat("K_handle", self.K_handle, v_min=1e3, v_max=1e6, power=10.0)
        changed, self.selection_radius = psim.SliderFloat("selection radius", self.selection_radius,  v_min=0.01, v_max=1.0)


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
                self.target.reshape(-1, self.dim))

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

    def draw(self):
        psim.Test("Handle UI")
        psim.SliderFloat("K_handle", self.K_handle, v_min=1e3, v_max=1e6, log_scale=True)
        psim.SliderFloat("selection radius", self.selection_radius, v_min=0.01, v_max=0.5)
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

        if self.handle is not None and self.handle_enabled:
            self.handle.draw()

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


