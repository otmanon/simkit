import time
import os
import numpy as np

import polyscope as ps


def view_scalar_modes(X, T, W, period=1 / 10, cmap="coolwarm", name="modes", dir=None, normalize=True, vminmax=None, eye_pos=None, eye_target=None):

    ps.init()

    ps.remove_all_structures()
    ps.set_give_focus_on_show(True)
    if eye_pos is not None and eye_target is not None:
        ps.look_at(eye_pos, eye_target)   
    dt = T.shape[1]

    if dt == 1:
        geo = ps.register_point_cloud("geo", X)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T)
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T)
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T)

    ps.set_ground_plane_mode("none")

    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    dw = W.shape[1]

    if vminmax is not None:
        if isinstance(vminmax, np.ndarray):
            if vminmax.shape[0] > 1:
                vminmax_mat = vminmax.copy()
    for i in range(dw):
        Wi = W[:, i]
        # if vminmax is None:
        if normalize:
            wmax = np.abs(Wi).max()
            vminmax = [-wmax, wmax]
        #     else:
        #         vminmax= None
        # else:
        #     if isinstance(vminmax, np.ndarray):
        #         if vminmax.shape[0] > 1:
        #             vminmax = vminmax_mat[i, :]
        
        geo.add_scalar_quantity(
            name+ " " + str(i), Wi, enabled=True, cmap=cmap,  vminmax=vminmax,
        )


        time.sleep(period)

        ps.frame_tick()

        if dir is not None:
            ps.screenshot(dir + "./" + str(i).zfill(4) + ".png") 

    # ps.show()
    return
