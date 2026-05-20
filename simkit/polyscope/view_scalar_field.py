import polyscope as ps
import numpy as np

def view_scalar_field(X, T, phi, path=None, colormap_path=None, cmap="coolwarm", look_at=None, eye_pos=None, normalize_vmax=False):

    ps.init()    
    ps.remove_all_structures()
    ps.set_SSAA_factor(4)
    ps.set_ground_plane_mode("none")
    ps.set_automatically_compute_scene_extents(False)
    if look_at is not None and eye_pos is not None:
        ps.look_at( eye_pos, look_at)
    
    vmax = np.max(np.abs(phi))

    if normalize_vmax:
        vmin = - vmax
    else:
        vmin = np.min(phi)

    dt = T.shape[1]
    if dt == 1:
        mesh = ps.register_point_cloud("geo", X)
    elif dt == 2:
        mesh = ps.register_curve_network("geo", X, T)
    elif dt == 3:
        mesh = ps.register_surface_mesh("geo", X, T)
    elif dt == 4:
        mesh = ps.register_volume_mesh("geo", X, T)

    if colormap_path is not None:
        ps.load_color_map("custom", colormap_path)
        mesh.add_scalar_quantity("phi", phi, cmap='custom', vminmax=[vmin, vmax], enabled=True)
    else:
        mesh.add_scalar_quantity("phi", phi, cmap='coolwarm', vminmax=[vmin, vmax], enabled=True)

    if path is not None:
        ps.screenshot(path, transparent_bg=True)
    else:
        ps.show()