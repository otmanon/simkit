from pathlib import Path
import polyscope as ps
import os
import numpy as np
from PIL import Image

from simkit.filesystem.mp4_to_gif import mp4_to_gif

from ..filesystem.video_from_image_dir import video_from_image_dir

def view_animation(X, T, U, pI=None, path=None, ground_plane='none', 
                   eye_pos=None, eye_target=None, fps=60, radius=0.05,
                   texture=None, uv=None, material='clay', Ps=None):


    dirstem = None
    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    ps.init()
    ps.remove_all_structures()
    
    ps.set_SSAA_factor(4)
    if eye_pos is not None and eye_target is not None:
        ps.look_at(eye_pos, eye_target)

    if ground_plane is not None:
        ps.set_ground_plane_mode(ground_plane)

    if T.shape[1] == 1:
        geo = ps.register_point_cloud("geo", X, material=material)
    elif T.shape[1] == 2:
        geo = ps.register_curve_network("geo", X, T, material=material)
    elif T.shape[1] == 3:
        geo = ps.register_surface_mesh("geo", X, T, material=material)
        
        if texture is not None and uv is not None:
            geo.add_parameterization_quantity("test_param",  uv,
                                            defined_on='vertices')
            
            geo.add_color_quantity("test_vals", texture,
                                        defined_on='texture', param_name="test_param",
                                            enabled=True)
        
    elif T.shape[1] == 4:
        geo = ps.register_volume_mesh("geo", X, T)

    if pI is not None:
        pc = ps.register_point_cloud("pI", X[pI, :].reshape((-1, X.shape[1])), radius=radius, color=[0.0, 0.0, 0.0], material=material)

    if Ps is not None:
        pc2 = ps.register_point_cloud("pI", Ps[:, 0].reshape((-1, X.shape[1])), radius=radius, color=[0.0, 0.0, 0.0], material=material)
        
    for i in range(U.shape[1]):
        ps.frame_tick()
        if path is not None:
            ps.screenshot(dirstem + "/" + str(i + 1).zfill(4) + ".png", transparent_bg=True)

        geo.update_vertex_positions( X + U[:, i].reshape(X.shape))

        if pI is not None:
            pc.update_point_positions(  (X  + (U[:, i]).reshape(-1, X.shape[1]))[pI].reshape((-1, X.shape[1])))
            
        if Ps is not None:
            pc2.update_point_positions(Ps[:, i].reshape(-1, X.shape[1]))

    if path is not None:
        video_from_image_dir(dirstem, path, fps=fps)
        mp4_to_gif(path, path.replace(".mp4", ".gif"))