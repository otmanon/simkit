"""All renders + the closing video (pyvista off-screen, composed with matplotlib).

    python sdm_render.py            # needs output/sdm_tets.npz and output/sdm_sim.npz
    python sdm_render.py --no-video

Writes to ``output/renders/``:
  01_parts_layout.png        CSG parts coloured by kind (+ exploded view)
  02_unified_surface.png     the unified genus-0 surface, several angles
  03_tet_materials.png       tet mesh coloured by Young's modulus, several angles
  04_tet_cutaway.png         sections through a finger and the thumb: flexures, pads
  05_tendons_pins.png        tendon springs (lines), anchors, pinned wrist vertices
  06_static_poses.png        static equilibria at a = 0, 0.33, 0.67, 1 (two views)
  07_static_appearance.png   same, robot-like appearance, palm view
  08_closing_curves.png      joint angles vs a, fingertip travel vs time
  closing_dynamic.mp4        backward-Euler closing, appearance + von Mises
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from sdm_geometry import OUT, HandParams, build_parts, manifold_to_VF, read_obj

pv.OFF_SCREEN = True
REN = os.path.join(OUT, "renders")

KIND_COL = {"palm": "#8a8d91", "link": "#3b3f45", "flexure": "#f28e2b", "pad": "#76b7e0"}
KIND_LAB = {"palm": "palm + wrist (stiff polyurethane)", "link": "phalanges (stiff polyurethane)",
            "flexure": "flexure joints (stiff elastomer, 6 MPa)", "pad": "fingertip + palm pads (softer elastomer)"}
# robot-like appearance: dark urethane blocks, amber flexures, skin-tone pads
APPEAR = {"palm": "#4a4d52", "link": "#2e3136", "flexure": "#e8a33d", "pad": "#e9c2a6"}
E_PALETTE = ["#3b4cc0", "#f2b134", "#d7301f"]      # stiff, flexure, pad (by E rank, high->low)

# camera directions (from the focal point towards the camera) and up vectors;
# the palmar face looks along -y, fingers point +z, thumb on -x
VIEWS = {
    "palm side": ((0.0, -1.0, 0.05), (0, 0, 1)),
    "three-quarter (palm side)": ((0.75, -1.0, 0.35), (0, 0, 1)),
    "three-quarter (thumb side)": ((-0.9, -1.0, 0.3), (0, 0, 1)),
    "side (little finger)": ((1.0, 0.0, 0.05), (0, 0, 1)),
    "back of hand": ((-0.25, 1.0, 0.2), (0, 0, 1)),
    "from above": ((0.0, -0.35, 1.0), (0, 1, 0)),
}


def tet_grid(X, T, **cell_data):
    cells = np.hstack([np.full((len(T), 1), 4), T]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(T), pv.CellType.TETRA), np.asarray(X, float))
    for k, v in cell_data.items():
        g.cell_data[k] = v
    return g


def polydata(V, F):
    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())


def shot(add, view, size=(900, 1000), zoom=1.35, focus=None, bounds_mesh=None):
    d, up = VIEWS[view] if isinstance(view, str) else view
    pl = pv.Plotter(window_size=size, off_screen=True)
    pl.set_background("white")
    add(pl)
    c = np.array(focus) if focus is not None else np.array(pl.bounds).reshape(3, 2).mean(1)
    d = np.asarray(d, float) / np.linalg.norm(d)
    pl.camera_position = [tuple(c + d), tuple(c), up]
    pl.reset_camera(bounds=bounds_mesh.bounds if bounds_mesh is not None else None)
    pl.camera.focal_point = tuple(c)
    pl.camera.position = tuple(c + d * np.linalg.norm(np.array(pl.camera.position) - c))
    pl.camera.zoom(zoom)
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def grid_figure(imgs, ncols, title, path, legend=None, legend_title=None, cell=(5.2, 5.6),
                legend_rows=1, dpi=115, colorbars=None):
    nrows = int(np.ceil(len(imgs) / ncols))
    leg_h = 0.55 + 0.32 * legend_rows if legend else 0.0
    if colorbars:
        leg_h = 1.1
    head = 0.45 + 0.32 * (title.count("\n") + 1)
    fig = plt.figure(figsize=(cell[0] * ncols, cell[1] * nrows + head + leg_h))
    top = 1 - head / fig.get_figheight()
    bot = leg_h / fig.get_figheight()
    for i, (name, img) in enumerate(imgs):
        ch = (top - bot) / nrows
        title_h = (0.25 + 0.25 * (name.count("\n") + 1)) / fig.get_figheight()
        ax = fig.add_axes([(i % ncols) / ncols, bot + (nrows - 1 - i // ncols) * ch,
                           1 / ncols, ch - title_h])
        ax.imshow(img)
        ax.set_title(name, fontsize=13)
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=1 - 0.25 / fig.get_figheight(), va="top")
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 4 if legend_rows == 1 else 2),
                   fontsize=11.5, frameon=False, title=legend_title, title_fontsize=12)
    for i, (cmap, norm, label) in enumerate(colorbars or []):
        n = len(colorbars)
        cax = fig.add_axes([0.08 + i / n * 0.9, 0.55 / fig.get_figheight(), 0.9 / n - 0.1,
                            0.18 / fig.get_figheight()])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                     label=label)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(OUT)))


class Scene:
    def __init__(self):
        self.tets = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
        self.sim = dict(np.load(os.path.join(OUT, "sdm_sim.npz")))
        self.rep_geo = json.load(open(os.path.join(OUT, "geometry_report.json")))
        self.rep_tet = json.load(open(os.path.join(OUT, "tet_report.json")))
        self.rep_sim = json.load(open(os.path.join(OUT, "sim_report.json")))
        t = self.tets
        self.X, self.T = t["X"], t["T"]
        self.kinds = np.array([str(k) for k in t["part_kind"]])
        self.kind_idx = np.array([list(KIND_COL).index(k) for k in self.kinds[t["part"]]], float)
        E_levels = np.unique(t["E"])[::-1]                   # high -> low
        self.E_levels = E_levels
        self.E_rank = np.array([np.where(E_levels == e)[0][0] for e in t["E"]], float)
        self.grid = tet_grid(self.X, self.T, kind=self.kind_idx, E=self.E_rank)

    def surface(self, x=None):
        g = self.grid.copy()
        if x is not None:
            g.points = np.asarray(x, float).reshape(-1, 3)
        return g.extract_surface(algorithm="dataset_surface")

    def kind_opts(self, palette=KIND_COL, **kw):
        cols = list(palette.values())
        return dict(scalars="kind", cmap=cols, clim=(-0.5, len(cols) - 0.5), n_colors=len(cols),
                    show_scalar_bar=False, **kw)

    def E_opts(self, **kw):
        return dict(scalars="E", cmap=E_PALETTE, clim=(-0.5, 2.5), n_colors=3,
                    show_scalar_bar=False, **kw)

    def E_legend(self):
        mats = [str(m) for m in self.tets["material_names"]]
        out = []
        for i, e in enumerate(self.E_levels):
            m = [k for k, v in self.rep_tet["materials"].items() if v["E"] == e][0]
            v = self.rep_tet["materials"][m]
            es = f"{e / 1e9:.1f} GPa" if e >= 1e8 else f"{e / 1e6:.1f} MPa"
            out.append(Patch(color=E_PALETTE[i], label=f"E = {es}, ν = {v['nu']:.2f}, "
                                                        f"ρ = {v['rho']:.0f} kg/m³ -- {m}"))
        return out


def kind_legend(palette=KIND_COL):
    return [Patch(color=c, label=KIND_LAB[k]) for k, c in palette.items()]


# --------------------------------------------------------------------- figures
def fig_parts(S):
    parts = build_parts(HandParams(), pads=True)
    meshes = {n: (polydata(*manifold_to_VF(M)), k) for n, (M, k) in parts.items()}

    def add(explode=0.0):
        def f(pl):
            for n, (m, k) in meshes.items():
                mm = m.copy()
                if explode:
                    c = np.array(mm.center)
                    off = np.array([c[0] * 0.35, -0.03 if k == "pad" else (0.012 if k == "flexure" else 0), 0])
                    if k != "palm":
                        off[2] = (c[2] - 0.095) * 0.35 if c[2] > 0.095 else 0.0
                    if n.startswith("thumb"):
                        off[0] = (c[0] + 0.045) * 0.5
                        off[2] = (c[2] - 0.018) * 0.5
                    mm = mm.translate(off)
                pl.add_mesh(mm, color=KIND_COL[k], show_edges=True, edge_color="#202020",
                            line_width=0.6, specular=0.3)
        return f
    imgs = [(v, shot(add(), v)) for v in ("palm side", "three-quarter (palm side)", "back of hand")]
    imgs.append(("exploded (palm side, three-quarter)", shot(add(0.35), "three-quarter (palm side)")))
    n = len(parts)
    cnt = {k: sum(1 for _, kk in parts.values() if kk == k) for k in KIND_COL}
    grid_figure(imgs, 4, f"CSG parts: {n} rectangular prisms  (palm + wrist {cnt['palm']}, "
                         f"phalanges {cnt['link']}, flexure slabs {cnt['flexure']}, pads {cnt['pad']})",
                os.path.join(REN, "01_parts_layout.png"), legend=kind_legend(), cell=(4.6, 5.4))


def fig_surface(S):
    t_np = S.rep_geo["surface_nopads"]
    t_p = S.rep_geo["surface_pads"]
    V, F = read_obj(os.path.join(OUT, "sdm_hand.obj"))
    V0, F0 = read_obj(os.path.join(OUT, "sdm_hand_nopads.obj"))
    m, m0 = polydata(V, F), polydata(V0, F0)

    def add(mesh, col):
        return lambda pl: pl.add_mesh(mesh, color=col, show_edges=True, edge_color="#333333",
                                      line_width=0.5, specular=0.3, smooth_shading=False)
    imgs = [(f"with pads: {v}", shot(add(m, "#c9ced6"), v))
            for v in ("palm side", "three-quarter (palm side)", "three-quarter (thumb side)",
                      "back of hand", "side (little finger)")]
    imgs.append(("without pads: three-quarter", shot(add(m0, "#d8d2c4"), "three-quarter (palm side)")))
    fmt = lambda t: (f"V={t['V']}, E={t['E']}, F={t['F']}, χ = V-E+F = {t['chi']}, "
                     f"{t['components']} component, {t['boundary_loops']} boundary loops "
                     f"→ genus {t['genus']:g}")
    grid_figure(imgs, 3, "Unified surface (manifold3d union): one closed oriented 2-manifold, genus 0\n"
                         f"with pads: {fmt(t_p)}\nwithout pads: {fmt(t_np)}",
                os.path.join(REN, "02_unified_surface.png"), cell=(5.0, 5.4))


def fig_tets(S):
    surf = S.surface()
    add = lambda pl: pl.add_mesh(surf, **S.E_opts(show_edges=True, edge_color="#222222", line_width=0.3))
    imgs = [(v, shot(add, v)) for v in ("palm side", "three-quarter (palm side)",
                                        "three-quarter (thumb side)", "back of hand",
                                        "side (little finger)", "from above")]
    tb, tb0 = S.rep_tet["tet_boundary_pads"], S.rep_tet["tet_boundary_nopads"]
    lc = S.rep_tet["label_counts"]
    grid_figure(imgs, 3, f"Tet mesh ({len(S.X)} vertices, {len(S.T)} tets; TetGen), per-tet Young's modulus "
                         f"from winding-number labels\ntets: palm {lc['palm']['tets']}, phalanges "
                         f"{lc['link']['tets']}, flexures {lc['flexure']['tets']}, pads {lc['pad']['tets']}"
                         f"   |   tet boundary: χ = {tb['chi']}, genus {tb['genus']:g} "
                         f"(without pads: χ = {tb0['chi']}, genus {tb0['genus']:g})",
                os.path.join(REN, "03_tet_materials.png"), legend=S.E_legend(), legend_rows=3,
                legend_title="Young's modulus (discrete, per tet)", cell=(5.0, 5.4))


def fig_cutaway(S):
    p = HandParams()
    xc_mid = p.fingers["middle"][0]
    cut_x = S.grid.clip(normal=(1, 0, 0), origin=(xc_mid, 0, 0), crinkle=True, invert=False)
    cut_x = cut_x.extract_cells(np.where(cut_x.cell_centers().points[:, 0] > xc_mid - 0.03)[0])
    cut_y = S.grid.clip(normal=(0, 1, 0), origin=(0, 0.0185, 0), crinkle=True, invert=True)
    # a section through the thumb, normal to its flexion axis
    from sdm_geometry import thumb_transform
    A = thumb_transform(p)
    cut_t = S.grid.clip(normal=tuple(A[:3, 0]), origin=tuple(A[:3, 3]), crinkle=True, invert=False)
    cut_t = cut_t.extract_cells(np.where(cut_t.cell_centers().points[:, 0] < -0.045)[0])
    opts = S.E_opts(show_edges=True, edge_color="#202020", line_width=0.35)
    imgs = [("section x = middle finger centre\n(flexures on the dorsal side, pad at the tip)",
             shot(lambda pl: pl.add_mesh(cut_x, **opts), ((1, -0.12, 0.05), (0, 0, 1)), zoom=1.3)),
            ("section y = 18.5 mm (flexure mid-plane), palmar half\nviewed from the back (cut face)",
             shot(lambda pl: pl.add_mesh(cut_y, **opts), ((0, 1, 0.0), (0, 0, 1)))),
            ("section through the thumb\n(normal to its flexion axis)",
             shot(lambda pl: pl.add_mesh(cut_t, **opts), (tuple(A[:3, 0] * -1 + np.array([0, -0.2, 0])), (0, 0, 1)),
                  zoom=1.0))]
    grid_figure(imgs, 3, "Cut-aways: soft flexure slabs between the stiff blocks, soft pads on the palmar faces",
                os.path.join(REN, "04_tet_cutaway.png"), legend=S.E_legend(), legend_rows=3,
                legend_title="Young's modulus", cell=(5.4, 6.2))


def add_tendons(pl, x, S, color="#d62728", radius=0.0009):
    Xm = np.asarray(x, float).reshape(-1, 3)
    for i, j in S.sim["E_t"]:
        pl.add_mesh(pv.Tube(pointa=Xm[i], pointb=Xm[j], radius=radius), color=color)
    pl.add_mesh(pv.PolyData(Xm[S.sim["E_t"].ravel()]), color="#ffd92f", point_size=11,
                render_points_as_spheres=True)


def fig_tendons(S):
    surf = S.surface()
    Xp = S.X[S.sim["pinned"]]

    def add(pl):
        pl.add_mesh(surf, **S.kind_opts(opacity=0.35))
        add_tendons(pl, S.X, S)
        pl.add_mesh(pv.PolyData(Xp), color="#1f77b4", point_size=13, render_points_as_spheres=True)
    imgs = [(v, shot(add, v)) for v in ("palm side", "three-quarter (palm side)", "side (little finger)")]
    # close-up of one finger
    p = HandParams()
    foc = (p.fingers["index"][0], 0.01, 0.14)

    xi, wi = p.fingers["index"][:2]
    C = S.grid.cell_centers().points
    sub = S.grid.extract_cells(np.where((C[:, 0] > xi) & (C[:, 0] < xi + wi) & (C[:, 2] > 0.075))[0])
    ks = [k for k, (i, j) in enumerate(S.sim["E_t"]) if abs(S.X[i, 0] - xi) < 1e-3]

    def add_zoom(pl):
        pl.add_mesh(sub.extract_surface(algorithm="dataset_surface"),
                    **S.kind_opts(show_edges=True, edge_color="#555555", line_width=0.3))
        Xm = S.X
        for k in ks:
            i, j = S.sim["E_t"][k]
            pl.add_mesh(pv.Tube(pointa=Xm[i] - [0.0006, 0, 0], pointb=Xm[j] - [0.0006, 0, 0],
                                radius=0.0006), color="#d62728")
    imgs.append(("close-up: index finger, section at its centre", shot(add_zoom, ((-1, -0.25, 0.05), (0, 0, 1)),
                                                                        zoom=0.95)))
    leg = kind_legend() + [
        Line2D([0], [0], color="#d62728", lw=3, label=f"tendon springs ({len(S.sim['E_t'])}, one per joint, palmar side)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffd92f", markersize=11, label="tendon anchors (stiff vertices)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=11,
               label=f"pinned: wrist bottom face ({len(Xp)} vertices)")]
    grid_figure(imgs, 4, "Actuation and boundary conditions: mass-spring tendons across every flexure, pinned wrist",
                os.path.join(REN, "05_tendons_pins.png"), legend=leg, legend_rows=4, cell=(4.6, 5.4))


def static_ids(S, targets=(0.0, 1 / 3, 2 / 3, 1.0)):
    a = S.sim["static_a"]
    return [int(np.argmin(np.abs(a - t))) for t in targets]


def fig_static(S):
    ids = static_ids(S)
    xs = S.sim["static_x"]
    union_mesh = S.surface(xs[0])
    stat = S.rep_sim["statics"]
    views = ["three-quarter (palm side)", "side (little finger)", "palm side"]
    imgs = []
    for v in views:
        for k in ids:
            surf = S.surface(xs[k])

            def add(pl, surf=surf, k=k):
                pl.add_mesh(surf, **S.E_opts())
                add_tendons(pl, xs[k], S, radius=0.0007)
            tip = np.mean(list(stat[k]["tip_disp_mm"].values()))
            imgs.append((f"a = {S.sim['static_a'][k]:.2f} -- {v}\nmean fingertip travel {tip:.0f} mm",
                         shot(add, v, bounds_mesh=union_mesh, size=(800, 900))))
    leg = S.E_legend() + [Line2D([0], [0], color="#d62728", lw=3, label="tendon springs")]
    grid_figure(imgs, 4, "Static equilibria under the single actuation parameter a "
                         "(continuation + Newton, SimKit stable Neo-Hookean + mass-spring tendons)",
                os.path.join(REN, "06_static_poses.png"), legend=leg, legend_rows=4, cell=(4.4, 5.0))
    imgs = []
    for v in ("three-quarter (palm side)", "three-quarter (thumb side)"):
        for k in ids:
            surf = S.surface(xs[k])
            imgs.append((f"a = {S.sim['static_a'][k]:.2f} -- {v}",
                         shot(lambda pl, surf=surf: pl.add_mesh(surf, **S.kind_opts(
                             palette=APPEAR, smooth_shading=False, specular=0.5, specular_power=20)),
                              v, bounds_mesh=union_mesh, size=(800, 900))))
    grid_figure(imgs, 4, "Closing sequence -- appearance (dark urethane blocks, amber flexures, skin-tone pads)",
                os.path.join(REN, "07_static_appearance.png"), legend=kind_legend(APPEAR), legend_rows=2,
                cell=(4.4, 5.0))


def fig_curves(S):
    stat = S.rep_sim["statics"]
    names = [t["joint"] for t in S.rep_sim["tendons"]]
    a = np.array([s["a"] for s in stat])
    ang = np.array([s["joint_deg"] for s in stat])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = {"index": "#4e79a7", "middle": "#f28e2b", "ring": "#59a14f", "little": "#e15759", "thumb": "#9c6ade"}
    ls = ["-", "--", ":"]
    for j, n in enumerate(names):
        f, i = n.split("_")[0], int(n[-1])
        axes[0].plot(a, ang[:, j], ls[i], color=cols[f], lw=2,
                     label=f"{f} joint {i}" if f in ("index", "thumb") or i == 0 else None)
    axes[0].set(xlabel="actuation a", ylabel="joint flexion [deg]", title="static joint flexion vs a")
    axes[0].legend(fontsize=8.5, ncol=2)
    axes[0].grid(alpha=0.3)
    fingers = list(stat[0]["tip_disp_mm"])
    for f in fingers:
        axes[1].plot(a, [s["tip_disp_mm"][f] for s in stat], color=cols[f], lw=2, label=f)
    axes[1].set(xlabel="actuation a", ylabel="fingertip travel [mm]", title="static fingertip travel vs a")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    if "dyn_t" in S.sim:
        t, d = S.sim["dyn_t"], S.sim["dyn_tip_disp"] * 1e3
        for i, f in enumerate(fingers):
            axes[2].plot(t, d[:, i], color=cols[f], lw=2, label=f)
        ax2 = axes[2].twinx()
        ax2.plot(t, S.sim["dyn_a"], "k--", lw=1.2)
        ax2.set_ylabel("a(t) (dashed)")
        axes[2].set(xlabel="time [s]", ylabel="fingertip travel [mm]",
                    title=f"backward Euler (h = {S.rep_sim['h']:.4f} s)")
        axes[2].legend(loc="center right")
        axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(REN, "08_closing_curves.png")
    fig.savefig(path, dpi=115)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.dirname(OUT)))


def video(S, fps=30):
    """Side-by-side mp4: appearance (three-quarter) | displacement magnitude (side)."""
    import imageio.v2 as imageio
    frames = S.sim["dyn_x"]
    vm = S.sim["dyn_vm"]
    ts, a_t = S.sim["dyn_t"], S.sim["dyn_a"]
    surf0 = S.grid.extract_surface(algorithm="dataset_surface")
    pid = surf0.point_data["vtkOriginalPointIds"]
    X0 = S.X
    disp = [np.linalg.norm(f.reshape(-1, 3) - X0, axis=1) * 1e3 for f in frames]
    dmax = float(np.ceil(max(d.max() for d in disp) / 10) * 10)
    bounds_mesh = S.surface(frames[0])
    size = (760, 860)

    def make(view, mode):
        pl = pv.Plotter(window_size=size, off_screen=True)
        pl.set_background("white")
        s = surf0.copy()
        if mode == "appearance":
            pl.add_mesh(s, **S.kind_opts(palette=APPEAR, specular=0.5, specular_power=20))
        else:
            s.point_data["disp"] = disp[0][pid]
            pl.add_mesh(s, scalars="disp", cmap="viridis", clim=(0, dmax),
                        scalar_bar_args=dict(title="displacement [mm]", vertical=True, position_x=0.84,
                                             position_y=0.2, width=0.08, height=0.6, fmt="%.0f",
                                             title_font_size=16, label_font_size=13, color="black"))
        d, up = VIEWS[view]
        c = np.array(bounds_mesh.bounds).reshape(3, 2).mean(1)
        d = np.asarray(d, float) / np.linalg.norm(d)
        pl.camera_position = [tuple(c + d), tuple(c), up]
        pl.reset_camera(bounds=bounds_mesh.bounds)
        pl.camera.zoom(1.2)
        pl.enable_anti_aliasing("ssaa")
        txt = pl.add_text("", position="upper_left", font_size=12, color="black")
        return pl, s, txt
    pls = [make("three-quarter (palm side)", "appearance"), make("side (little finger)", "disp")]
    path = os.path.join(REN, "closing_dynamic.mp4")
    w = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, quality=8)
    for k in range(len(frames)):
        Xk = frames[k].reshape(-1, 3).astype(float)
        imgs = []
        for i, (pl, s, txt) in enumerate(pls):
            s.points = Xk[pid]
            if i == 1:
                s.point_data["disp"] = disp[k][pid]
            txt.SetText(2, (f"backward Euler   t = {ts[k]:.3f} s   a = {a_t[k]:.2f}" if i == 0 else
                            "displacement magnitude"))
            pl.render()
            imgs.append(pl.screenshot(return_img=True))
        w.append_data(np.hstack(imgs))
    w.close()
    for pl, _, _ in pls:
        pl.close()
    # still montage of the video: displacement (top) and von Mises (bottom)
    ks = np.linspace(0, len(frames) - 1, 5).astype(int)
    lo, hi = 1e3, 1e6
    imgs = []
    for row in ("disp", "vm"):
        for k in ks:
            surf = S.surface(frames[k])
            if row == "disp":
                surf.point_data["c"] = disp[k][surf.point_data["vtkOriginalPointIds"]]
                opts = dict(scalars="c", cmap="viridis", clim=(0, dmax))
            else:
                surf.cell_data["c"] = np.clip(vm[k], lo, hi)[surf.cell_data["vtkOriginalCellIds"]]
                opts = dict(scalars="c", cmap="plasma", log_scale=True, clim=(lo, hi))
            imgs.append((f"t = {ts[k]:.2f} s, a = {a_t[k]:.2f}"
                         + (" (displacement)" if row == "disp" else " (von Mises)"),
                         shot(lambda pl, surf=surf, opts=opts: pl.add_mesh(surf, show_scalar_bar=False, **opts),
                              "three-quarter (palm side)", bounds_mesh=bounds_mesh, size=(700, 800))))
    cbs = [("viridis", matplotlib.colors.Normalize(0, dmax), "displacement magnitude [mm] (top row)"),
           ("plasma", matplotlib.colors.LogNorm(lo, hi), "von Mises stress [Pa] (bottom row; clipped)")]
    grid_figure(imgs, 5, "Dynamic closing (backward Euler, h = 1/60 s, a(t) ramped over 1.2 s)",
                os.path.join(REN, "09_dynamic_frames.png"), cell=(3.9, 4.5), colorbars=cbs)
    print("wrote", os.path.relpath(path, os.path.dirname(OUT)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(REN, exist_ok=True)
    S = Scene()
    figs = dict(parts=fig_parts, surface=fig_surface, tets=fig_tets, cutaway=fig_cutaway,
                tendons=fig_tendons, static=fig_static, curves=fig_curves)
    for name, f in figs.items():
        if args.only is None or name in args.only:
            f(S)
    if not args.no_video and "dyn_x" in S.sim and (args.only is None or "video" in args.only):
        video(S)
