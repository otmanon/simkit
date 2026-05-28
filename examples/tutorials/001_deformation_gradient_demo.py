import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from simkit import deformation_gradient
from utils import screen_to_world_2d


import simkit.energies as energies

X = np.array([[-0.5, 0], [0.5, 0], [0, np.sqrt(3/4)]])
T = np.array([[0, 1, 2]])
U = X.copy()

light_green = np.array([153,216,201])/255
black = np.array([0.0, 0, 0])

selected_index = None
P = None

ENERGY_HISTORY_LEN = 200
energy_history = []


def callback():
    global  selected_index, P

    if psim.Button("Reset"):
        U[:] = X
        selected_index = None
        energy_history.clear()
        pc.update_point_positions(U)
        mesh.update_vertex_positions(U)

    # get window pos
    win_pos = psim.GetMousePos()

    # left-click near a vertex to grab it
    if psim.IsMouseClicked(0):
        pos = screen_to_world_2d(win_pos)
        distance = np.linalg.norm(U - pos.reshape(-1, 2), axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] < 0.15:
            selected_index = nearest

    # while held: drag the selected vertex
    if selected_index is not None and psim.IsMouseDown(0):
        pos = screen_to_world_2d(win_pos)
        U[selected_index] = pos[:2]
        pc.update_point_positions(U)
        mesh.update_vertex_positions(U)

    # release to drop
    if psim.IsMouseReleased(0):
        selected_index = None

    if selected_index is not None:
        psim.Text("dragging vertex " + str(selected_index))
    else:
        psim.Text("left-click a vertex and drag to move it; release to drop")
    
    F = deformation_gradient(X, T, U).reshape((2, 2))
    # Format the 2x2 matrix with alignment and 2 decimal points
    formatted_F = (
        f"F:\n"
        f"[{F[0,0]:6.1f}  {F[0,1]:6.1f}]\n"
        f"[{F[1,0]:6.1f}  {F[1,1]:6.1f}]"
    )
    
    e = energies.neo_hookean_energy_element_F(F, mu=1, lam=1).item()

    energy_history.append(float(e))
    del energy_history[:-ENERGY_HISTORY_LEN]

    formatted_e = (
        f"E:\n"
        f"[{e:6.1f}]"
    )


    psim.Text(formatted_F)
    
    
    e_min = min(energy_history)
    e_max = max(energy_history)
    psim.Text(f"max: {e_max:.3f}")
    psim.PlotLines(
        f"energy (last {ENERGY_HISTORY_LEN})",
        energy_history,
        overlay_text=f"energy: {e:.3f}",
        graph_size=(0.0, 200.0),

    )
    psim.Text(f"min: {e_min:.3f}")
    
    
ps.init()
ps.remove_all_structures()
# ps.set_view_projection_mode("orthographic")
ps.look_at(np.array([0, 0, 5]), np.array([0, 0, 0]))
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", U, T, material='flat', color=light_green, edge_width=3)
pc = ps.register_point_cloud("vertices", U, radius=0.04, material="flat", color=black)
ps.set_do_default_mouse_interaction(False)
ps.set_user_callback(callback)
ps.show()


