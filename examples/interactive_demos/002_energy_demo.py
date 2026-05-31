"""Tutorial 002 — Picking and dragging a vertex (warmup).

Same single-triangle scene as 001, but stripped down to just the picking layer:
right-click to grab the nearest vertex; hold ``Space`` and move the mouse to
drag it. Lives as a standalone file so the picking flow stays readable before
later tutorials layer dynamics, solvers, and contact on top.
"""
import numpy as np
import polyscope.imgui as psim

from utils import Viewer2D, screen_to_world_2d


X = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, np.sqrt(3.0 / 4.0)]])
T = np.array([[0, 1, 2]])
U = X.copy()

selected = None
viewer = Viewer2D(X, T, point_radius=0.04, edge_width=3)


def callback():
    global selected

    win_pos = psim.GetMousePos()
    if psim.IsMouseClicked(1):
        pos = screen_to_world_2d(win_pos)
        selected = int(np.argmin(np.linalg.norm(X - pos.reshape(-1, 2), axis=1)))

    if selected is not None and psim.IsKeyDown(psim.ImGuiKey_Space):
        U[selected] = screen_to_world_2d(win_pos)[:2]

    viewer.refresh(U)
    psim.Text(
        "right-click to pick the nearest vertex; hold Space + move mouse to drag"
        if selected is None
        else f"vertex {selected} selected — hold Space to drag"
    )


viewer.show(callback)
