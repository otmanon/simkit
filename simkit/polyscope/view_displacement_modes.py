from pathlib import Path
import os

import numpy as np
import polyscope as ps

from ..filesystem.video_from_image_dir import video_from_image_dir


def view_displacement_modes(X, T, W, a=0.1, period=24, path=None, edge_width=1, fps=120):




    dirstem = None
    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    ps.init()
    ps.set_give_focus_on_show(True)
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")
    dt = T.shape[1]
    dim = X.shape[1]
    if dt == 1:
        geo = ps.register_point_cloud("geo", X)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T)
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T, edge_width=edge_width)
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T, edge_width=edge_width)

    dw = W.shape[1]

    # for i in range(dw):
    #     Wi = W[:, i]
    #     U =  Wi.reshape((-1, dim))
    #     Umax = np.linalg.norm(W, axis=1)

    #     geo.add_vector_quantity("eigenvector " + str(i), U)

    #     ps.frame_tick()
    i = 0
    j = 0
    
    if path is not None:
        ps.screenshot(dirstem + "/" + str(0).zfill(4) + ".png", transparent_bg=False)
    count = 1

    ps.set_give_focus_on_show(True)
    while True:
        if path is not None:
            ps.screenshot(dirstem + "/" + str(count).zfill(4) + ".png", transparent_bg=False)
        if j < W.shape[1]:
            Wi = W[:,j]
            U = Wi.reshape((-1, dim))
            # Umax = np.linalg.norm(W, axis=1)
            D = a * U * np.sin(2 * np.pi * i / period)

            if i == period:
                i = 0
                j +=1

            
            geo.update_vertex_positions(X + D)

        

            i += 1
            count+=1 
            ps.frame_tick()
        else:
            break


    # ps.set_user_callback(callback)
    # ps.show()

    if path is not None:
        video_from_image_dir(dirstem, path, fps=fps)
    return
