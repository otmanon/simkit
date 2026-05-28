import polyscope as ps
import polyscope.imgui as psim
import numpy as np

from simkit import deformation_gradient

X = np.array([[-0.5, 0], [0.5, 0], [0, np.sqrt(3/4)]])
T = np.array([[0, 1, 2]])
U = X.copy()

light_green = np.array([153,216,201])/255
black = np.array([0.0, 0, 0])


    
d = None
selected = True
selected_index = None

P = None


def screen_to_world_2d(win_pos):
    W, H = ps.get_window_size()
    u = win_pos[0] / W            # 0..1 left->right
    v = win_pos[1] / H            # 0..1 top->bottom
    params = ps.get_view_camera_parameters()
    ul, ur, ll, lr = params.generate_camera_ray_corners()
    pos = params.get_position()
    # in ortho the rays are parallel; corners give the world rectangle directions.
    # bilinear interp of the corner ray endpoints at the z=0 plane:
    top = (1 - u) * np.array(ul) + u * np.array(ur)
    bot = (1 - u) * np.array(ll) + u * np.array(lr)
    ray_dir = (1 - v) * top + v * bot
    # intersect ray (pos + t*ray_dir) with z=0
    t = -pos[2] / ray_dir[2]
    world = pos + t * ray_dir
    return world[:2]

def callback():
    global pc, d, selected_index, selected, P

    # get window pos
    win_pos = psim.GetMousePos()    

    # if right mouse button is clicked, place a point on the mesh.
    if psim.IsMouseClicked(1):     
        pos = screen_to_world_2d(win_pos)
        
        distance = np.linalg.norm(X - pos.reshape(-1, 2), axis=1)
        selected_index = np.argmin(distance)



    # if point being moved exists, and space is being held down, move the point by dragging mouse around
    if selected is not None and psim.IsKeyDown((psim.ImGuiKey_Space)): 
        # pos = ps.query_pick_at_screen_coords(win_pos)
        # pick_result = ps.pick(screen_coords=win_pos)
        pos = screen_to_world_2d(win_pos)
        print(pos)
        U[selected_index] = pos[:2]
        pc.update_point_positions(U)
        mesh.update_vertex_positions(U)


    


    
    
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


