import time
import os
import numpy as np
import igl

import polyscope as ps


def view_scalar_fields(X, T, W, colormap_path=None, cmap="coolwarm",   dir=None, normalize=True,  eye_pos=None, look_at=None, material="clay", outline_width=None):

    ps.init()

    ps.remove_all_structures()
    ps.set_give_focus_on_show(True)
    ps.set_SSAA_factor(4)
    if eye_pos is not None and look_at is not None:
        ps.look_at(eye_pos, look_at)   
    dt = T.shape[1]

    if dt == 1:
        geo = ps.register_point_cloud("geo", X, material=material)
    elif dt == 2:
        geo = ps.register_curve_network("geo", X, T, material=material)
    elif dt == 3:
        geo = ps.register_surface_mesh("geo", X, T, material=material)
        
        if outline_width is not None:
            E = igl.boundary_facets(T)[0]
            V, E , _, _= igl.remove_unreferenced(X, E)
            edge = ps.register_curve_network("edge", V, E, material=material, radius=outline_width, color=[0.0, 0.0, 0.0])
            # ps.show()
    elif dt == 4:
        geo = ps.register_volume_mesh("geo", X, T, material=material)

    # if colormap_dir is not None:
    #     ps.load_color_map(colormap_dir, colormap_dir)
    #     cmap = colormap_dir
    ps.set_ground_plane_mode("none")

    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    dw = W.shape[1]

    if colormap_path is not None:
        cmap = colormap_path
        ps.load_color_map(cmap, colormap_path)
    vminmax = None
    for i in range(dw):
        Wi = W[:, i]
        if normalize:
            wmax = np.abs(Wi).max()
            vminmax = [-wmax, wmax]

        geo.add_scalar_quantity("mesh" + str(i).zfill(3), Wi, 
                            enabled=True, cmap=cmap,  vminmax=vminmax)
        
        if dir is not None:
            ps.screenshot(dir + "./" + str(i).zfill(4) + ".png") 

    if dir is None:
        ps.show()

    return
