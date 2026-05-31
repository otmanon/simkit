"""Tutorial 001 — Deformation gradient of a single triangle.

Drag a vertex of the triangle and watch the 2x2 deformation gradient ``F`` and
the neo-Hookean energy update live. Rigid motions keep ``F`` close to a
rotation and the energy near zero; stretches grow the energy.
"""
import numpy as np
import polyscope.imgui as psim

import simkit.energies as energies
from simkit import deformation_gradient

from utils import RollingPlot, Viewer2D, screen_to_world_2d


X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3.0 / 4.0)]])
T = np.array([[0, 1, 2]])
U = X.copy()

selected = None
energy_plot = RollingPlot("energy", length=200, height=200.0)

viewer = Viewer2D(X, T, point_radius=0.04, edge_width=3)


def callback():
    global selected

    if psim.Button("Reset"):
        U[:] = X
        selected = None
        energy_plot.clear()

    win_pos = psim.GetMousePos()
    if psim.IsMouseClicked(0):
        pos = screen_to_world_2d(win_pos)
        dist = np.linalg.norm(U - pos.reshape(-1, 2), axis=1)
        nearest = int(np.argmin(dist))
        if dist[nearest] < 0.15:
            selected = nearest
    if selected is not None and psim.IsMouseDown(0):
        U[selected] = screen_to_world_2d(win_pos)[:2]
    if psim.IsMouseReleased(0):
        selected = None

    # core of this tutorial: F is the per-element deformation gradient and the
    # neo-Hookean elastic energy is a scalar function of F.
    F = deformation_gradient(X, T, U).reshape((2, 2))
    E = energies.neo_hookean_energy_element_F(F, mu=1, lam=1).item()
    energy_plot.push(E)

    viewer.refresh(U)

    psim.Text(
        "left-click a vertex and drag; release to drop"
        if selected is None else f"dragging vertex {selected}"
    )
    psim.Text(f"F:\n[{F[0,0]:6.1f}  {F[0,1]:6.1f}]\n[{F[1,0]:6.1f}  {F[1,1]:6.1f}]")
    energy_plot.draw()


viewer.show(callback)
