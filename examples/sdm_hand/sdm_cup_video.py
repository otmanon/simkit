"""Cup grasp animations (reduced model, friction): the whole closing, then a profile
close-up of one finger closing onto the cup.

    python sdm_cup_video.py [finger]      # cup_grasp_full.mp4, cup_grasp_profile_<finger>.mp4
"""
import os
import sys

import numpy as np
import pyvista as pv
import imageio.v2 as imageio

import sdm_reduced as R
from sdm_coarse_render import kind_opts
from sdm_reduced_render import Fine
from sdm_pca_render import interp
from sdm_ball_render import object_mesh, BALL_COL
from sdm_finger_zoom import sdf_obj, finger_surface, finger_focus, hinge_axis, TAG

pv.OFF_SCREEN = True
REN = os.path.join(R.OUT, "renders")
CUP = os.environ.get("CUP", "cup_yaw")
FTAG = f"{TAG}_kpin100000_cup_yaw-15_fric1e+10" if CUP == "cup_yaw" else f"{TAG}_kpin100000_cup_fric1e+10"


def main(finger="index"):
    F = Fine()
    _, P = R.load_level(TAG)
    d = dict(np.load(os.path.join(R.OUT, f"reduced_sim_{FTAG}.npz")))
    obj = sdf_obj(CUP)
    bm = object_mesh(CUP)
    A = np.concatenate([np.linspace(0, 1, 121), np.ones(36)])
    xfs = [P @ interp(d, a) for a in A]
    force = [np.interp(a, d["static_a"], d["contact_force"]) for a in A]

    # 1) whole grasp
    pl = pv.Plotter(window_size=(1000, 1000), off_screen=True)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    s = F.at(F.X)
    pl.add_mesh(s, **kind_opts(0.0) | dict(show_edges=False))
    pl.add_mesh(bm, color=BALL_COL, opacity=0.45, smooth_shading=True)
    touch = pv.PolyData(np.zeros((1, 3)))
    pl.add_mesh(touch, color="#d62728", point_size=6, render_points_as_spheres=True)
    txt = pl.add_text("", position="upper_left", font_size=13, color="black")
    b = s.merge(bm).bounds
    dv = np.array([0.75, -1.0, 0.35]); dv /= np.linalg.norm(dv)
    c = np.array(b).reshape(3, 2).mean(1)
    pl.camera_position = [tuple(c + dv), tuple(c), (0, 0, 1)]
    pl.reset_camera(bounds=b)
    pl.camera.zoom(1.15)
    w = imageio.get_writer(os.path.join(REN, f"{CUP}_grasp_full.mp4"), fps=30, codec="libx264", macro_block_size=1,
                           quality=8)
    for a, xf, f in zip(A, xfs, force):
        s.points = xf[F.pid]
        tp = s.points[obj.sdf(s.points) < 0]
        touch.points = tp if len(tp) else np.full((1, 3), 1e3)
        txt.SetText(2, f"hyper-reduced hand (3,600 DOFs) grasping a rigid cup\na = {a:.2f}   contact {f:.0f} N   "
                       f"{len(tp)} fine vertices touching")
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    w.close()
    pl.close()
    print(f"wrote renders/{CUP}_grasp_full.mp4")

    # 2) profile close-up of one finger
    pl = pv.Plotter(window_size=(1000, 1000), off_screen=True)
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    s = F.at(F.X)
    pl.add_mesh(s, color="#bdbab2", opacity=0.12)
    names = list(F.sc["part_names"])
    ids = np.array([names.index(n) for n in names if n.startswith(finger + "_")])
    cell_part = F.sc["part"][s.cell_data["vtkOriginalCellIds"]]
    base = s.copy()
    base.point_data["sid"] = np.arange(base.n_points)            # index into s's points
    fs = base.extract_cells(np.nonzero(np.isin(cell_part, ids))[0])
    fpid = np.asarray(fs.point_data["sid"])
    pl.add_mesh(fs, **kind_opts(0.0) | dict(show_edges=False))
    pl.add_mesh(bm, color=BALL_COL, opacity=0.25, smooth_shading=True)
    touch = pv.PolyData(np.zeros((1, 3)))
    pl.add_mesh(touch, color="#d62728", point_size=11, render_points_as_spheres=True)
    txt = pl.add_text("", position="upper_left", font_size=13, color="black")
    pts_all = np.vstack([finger_focus(F, xf, finger)[1] for xf in (xfs[0], xfs[len(xfs) // 2], xfs[-1])])
    focus = pts_all.mean(0)
    ax = hinge_axis(finger)
    side = np.sign((focus - F.X.mean(0)) @ ax) or 1.0
    dv = side * ax
    pl.camera_position = [tuple(focus + dv), tuple(focus), (0, 0, 1)]
    lo, hi = pts_all.min(0) - 0.01, pts_all.max(0) + 0.01
    pl.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    pl.camera.zoom(1.05)
    fv = np.unique(F.sc["T"][np.isin(F.sc["part"], [names.index(n) for n in names if n.startswith(finger + "_")])])
    w = imageio.get_writer(os.path.join(REN, f"{CUP}_grasp_profile_{finger}.mp4"), fps=30, codec="libx264",
                           macro_block_size=1, quality=8)
    for a, xf in zip(A, xfs):
        s.points = xf[F.pid]
        fs.points = s.points[fpid]
        fp = xf[fv]
        tp = fp[obj.sdf(fp) < 0]
        touch.points = tp if len(tp) else np.full((1, 3), 1e3)
        txt.SetText(2, f"{finger} finger, profile   a = {a:.2f}   {len(tp)} vertices touching")
        pl.render()
        w.append_data(pl.screenshot(return_img=True))
    w.close()
    pl.close()
    print(f"wrote renders/{CUP}_grasp_profile_{finger}.mp4")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "index")
