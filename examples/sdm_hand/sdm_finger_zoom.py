"""Close-ups of each finger at the closed grasp (reduced model, fine hand x_f = B x),
with the object translucent and the fine surface vertices in contact marked.

    python sdm_finger_zoom.py      # 33_finger_zoom.png, finger_zoom_<kind>.mp4
"""
import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sdm_reduced as R
from sdm_cup import Cup
from sdm_render import KIND_COL, KIND_LAB
from sdm_coarse_render import kind_opts, ORDER
from sdm_reduced_render import Fine
from sdm_ball_render import object_mesh, OBJECTS, BALL_COL

pv.OFF_SCREEN = True
REN = os.path.join(R.OUT, "renders")
TAG = "1200_palmar_mixw3"
CASES = [("ball", f"{TAG}_kpin100000_ball_fric1e+10"), ("cup", f"{TAG}_kpin100000_cup_fric1e+10")]
FINGERS = ["index", "middle", "ring", "little", "thumb"]


def sdf_obj(kind):
    o = OBJECTS[kind]
    return R.Ball(o["center"], o["R"]) if kind == "ball" else Cup(center=o["center"], R=o["R"])


def state(F, kind, ftag):
    _, P = R.load_level(TAG)
    d = np.load(os.path.join(R.OUT, f"reduced_sim_{ftag}.npz"))
    xf = P @ d["static_x"][-1].reshape(-1, 3).astype(float)
    surf = F.at(xf)
    obj = sdf_obj(kind)
    touch = obj.sdf(surf.points) < 0
    return xf, surf, surf.points[touch], d


def finger_focus(F, xf, finger):
    names = list(F.sc["part_names"])
    ids = [names.index(n) for n in names if n.startswith(finger + "_")]
    v = np.unique(F.sc["T"][np.isin(F.sc["part"], ids)])
    return xf[v].mean(0), xf[v]


def finger_surface(F, surf, finger):
    """The part of the (posed) fine surface that belongs to one finger."""
    names = list(F.sc["part_names"])
    ids = np.array([names.index(n) for n in names if n.startswith(finger + "_")])
    cell_part = F.sc["part"][surf.cell_data["vtkOriginalCellIds"]]
    return surf.extract_cells(np.nonzero(np.isin(cell_part, ids))[0]).extract_surface(algorithm="dataset_surface")


def hinge_axis(finger):
    from sdm_geometry import joints
    return [j for j in joints() if j["name"] == f"{finger}_flex1"][0]["A"][:3, 0]


def closeup(pl, F, surf, touch, bm, finger, xf):
    fs = finger_surface(F, surf, finger)
    pl.add_mesh(surf, color="#bdbab2", opacity=0.12)
    pl.add_mesh(fs, **kind_opts(0.0) | dict(show_edges=False))
    pl.add_mesh(bm, color=BALL_COL, opacity=0.22, smooth_shading=True)
    focus, pts = finger_focus(F, xf, finger)
    if len(touch):
        near = np.linalg.norm(touch[:, None] - pts[None], axis=2).min(1) < 1e-9
        if near.any():
            pl.add_mesh(pv.PolyData(touch[near]), color="#d62728", point_size=12, render_points_as_spheres=True)
    ax = hinge_axis(finger)
    side = np.sign((focus - F.X.mean(0)) @ ax) or 1.0
    d = side * ax + np.array([0, -0.35, 0.15])
    d /= np.linalg.norm(d)
    ext = np.linalg.norm(pts - focus, axis=1).max()
    pl.camera_position = [tuple(focus + d * ext * 3.2), tuple(focus), (0, 0, 1)]
    pl.camera.view_angle = 30
    return int(near.sum()) if len(touch) else 0


def plotter(size):
    pl = pv.Plotter(window_size=size, off_screen=True)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    return pl


def add_scene(pl, surf, touch, bm):
    pl.add_mesh(surf, **kind_opts(0.0) | dict(show_edges=False))
    pl.add_mesh(bm, color=BALL_COL, opacity=0.35, smooth_shading=True)
    if len(touch):
        pl.add_mesh(pv.PolyData(touch), color="#d62728", point_size=7, render_points_as_spheres=True)


def cam_for(focus, pts, xf_all, view_dir, dist_scale=2.6):
    d = np.asarray(view_dir, float)
    d /= np.linalg.norm(d)
    ext = np.linalg.norm(pts - focus, axis=1).max()
    return focus + d * ext * dist_scale, focus


def view_dir_for(finger, focus, hand_c):
    # look at the finger from the palm side, a bit from outside the hand
    out = focus - hand_c
    out[2] = 0
    out = out / (np.linalg.norm(out) + 1e-12)
    return np.array([0, -1.0, 0.25]) + 0.9 * out


def main():
    F = Fine()
    hand_c = F.X.mean(0)
    fig, ax = plt.subplots(len(CASES), len(FINGERS) + 1, figsize=(3.6 * (len(FINGERS) + 1), 4.0 * len(CASES)))
    for i, (kind, ftag) in enumerate(CASES):
        xf, surf, touch, d = state(F, kind, ftag)
        bm = object_mesh(kind)
        pl = plotter((600, 640))
        add_scene(pl, surf, touch, bm)
        dv = np.array([0.75, -1.0, 0.35]); dv /= np.linalg.norm(dv)
        c = xf.mean(0)
        pl.camera_position = [tuple(c + dv), tuple(c), (0, 0, 1)]
        pl.reset_camera(bounds=surf.merge(bm).bounds)
        pl.camera.zoom(1.2)
        ax[i, 0].imshow(pl.screenshot(return_img=True))
        pl.close()
        ax[i, 0].set_title(f"{kind}: whole hand\n{len(touch)} fine vertices touching, "
                           f"{d['contact_force'][-1]:.0f} N", fontsize=11)
        for j, fg in enumerate(FINGERS):
            pl = plotter((600, 640))
            nt = closeup(pl, F, surf, touch, bm, fg, xf)
            ax[i, j + 1].imshow(pl.screenshot(return_img=True))
            pl.close()
            ax[i, j + 1].set_title(f"{fg} finger, side view ({nt} vertices touching)", fontsize=11)
        for a in ax[i]:
            a.axis("off")
    fig.legend(handles=[Patch(color=KIND_COL[k], label=KIND_LAB[k]) for k in ORDER] +
               [Patch(color=BALL_COL, label="rigid object (analytic SDF)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9,
                       label="fine surface vertex in contact")], loc="lower center", ncol=6, frameon=False, fontsize=10.5)
    fig.suptitle("Closed grasps, one finger at a time (reduced 3,600 DOFs, contact + friction on the fine surface, "
                 "k_f = 1e10)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    fig.savefig(os.path.join(REN, "33_finger_zoom.png"), dpi=80)
    print("wrote renders/33_finger_zoom.png")
    for kind, ftag in CASES:
        tour(F, kind, ftag, hand_c)


def tour(F, kind, ftag, hand_c, fps=24):
    """Whole hand, then each finger in turn (side view, that finger highlighted), with a
    camera move from the whole-hand view into each close-up."""
    import imageio.v2 as imageio
    xf, surf, touch, d = state(F, kind, ftag)
    bm = object_mesh(kind)
    w = imageio.get_writer(os.path.join(REN, f"finger_zoom_{kind}.mp4"), fps=fps, codec="libx264",
                           macro_block_size=1, quality=8)
    smooth = lambda t: t * t * (3 - 2 * t)
    ref = plotter((900, 900))
    add_scene(ref, surf, touch, bm)
    dv = np.array([0.75, -1.0, 0.35]); dv /= np.linalg.norm(dv)
    c = xf.mean(0)
    ref.camera_position = [tuple(c + dv), tuple(c), (0, 0, 1)]
    ref.reset_camera(bounds=surf.merge(bm).bounds)
    ref.camera.zoom(1.2)
    p0, f0, va0 = np.array(ref.camera.position), np.array(ref.camera.focal_point), ref.camera.view_angle
    ref.close()
    for fg in FINGERS:
        pl = plotter((900, 900))
        closeup(pl, F, surf, touch, bm, fg, xf)
        p1, f1 = np.array(pl.camera.position), np.array(pl.camera.focal_point)
        txt = pl.add_text(f"{kind} grasp (reduced, friction): {fg} finger", position="upper_left",
                          font_size=12, color="black")
        for t in list(np.linspace(0, 1, 36)) + [1.0] * 30 + list(np.linspace(1, 0, 24)):
            s_ = smooth(t)
            pl.camera_position = [tuple((1 - s_) * p0 + s_ * p1), tuple((1 - s_) * f0 + s_ * f1), (0, 0, 1)]
            pl.camera.view_angle = (1 - s_) * va0 + s_ * 30
            pl.render()
            w.append_data(pl.screenshot(return_img=True))
        pl.close()
    w.close()
    print(f"wrote renders/finger_zoom_{kind}.mp4")


def _old_tour(F, kind, ftag, hand_c, fps=24):
    """Camera flies from the whole hand to each finger in turn and back."""
    import imageio.v2 as imageio
    xf, surf, touch, d = state(F, kind, ftag)
    bm = object_mesh(kind)
    pl = plotter((900, 900))
    add_scene(pl, surf, touch, bm)
    txt = pl.add_text("", position="upper_left", font_size=12, color="black")
    dv = np.array([0.75, -1.0, 0.35]); dv /= np.linalg.norm(dv)
    c = xf.mean(0)
    pl.camera_position = [tuple(c + dv), tuple(c), (0, 0, 1)]
    pl.reset_camera(bounds=surf.merge(bm).bounds)
    pl.camera.zoom(1.2)
    home = (np.array(pl.camera.position), np.array(pl.camera.focal_point))
    shots = [("whole hand", home)]
    for fg in FINGERS:
        focus, pts = finger_focus(F, xf, fg)
        shots.append((f"{fg} finger", cam_for(focus, pts, xf, view_dir_for(fg, focus, hand_c))))
    w = imageio.get_writer(os.path.join(REN, f"finger_zoom_{kind}.mp4"), fps=fps, codec="libx264",
                           macro_block_size=1, quality=8)
    smooth = lambda t: t * t * (3 - 2 * t)

    def frame(pos, foc, label):
        pl.camera_position = [tuple(pos), tuple(foc), (0, 0, 1)]
        pl.camera.view_angle = 30
        txt.SetText(2, f"{kind} grasp (reduced, friction): {label}")
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    for k in range(1, len(shots)):
        (l0, (p0, f0)), (l1, (p1, f1)) = shots[0], shots[k]
        for t in np.linspace(0, 1, 30):
            s = smooth(t)
            frame((1 - s) * p0 + s * p1, (1 - s) * f0 + s * f1, l1)
        for _ in range(30):
            frame(p1, f1, l1)
        for t in np.linspace(0, 1, 24):
            s = smooth(t)
            frame((1 - s) * p1 + s * p0, (1 - s) * f1 + s * f0, "whole hand")
    w.close()
    pl.close()
    print(f"wrote renders/finger_zoom_{kind}.mp4")


if __name__ == "__main__":
    main()
