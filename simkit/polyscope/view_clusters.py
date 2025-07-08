import polyscope as ps
import numpy as np



def view_clusters(X, T, l, pI=None, path=None, eye_pos=None, eye_target=None,):


    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")

    if T.shape[1] == 1:
        mesh = ps.register_point_cloud("mesh", X)
    elif T.shape[1] == 2:
        mesh = ps.register_curve_network("mesh", X, T)
    elif T.shape[1] == 3:
        mesh = ps.register_surface_mesh("mesh", X, T)
    elif T.shape[1] == 4:
        mesh = ps.register_volume_mesh("mesh", X, T)


    if l.shape[0] == T.shape[0]:
        if T.shape[1] == 1:
            mesh.add_scalar_quantity("cluster", l.astype(np.float64), cmap='rainbow', enabled=True)
        elif T.shape[1] == 2:
            mesh.add_scalar_quantity("cluster", l.astype(np.float64), defined_on='edges', cmap='rainbow', enabled=True)
        elif T.shape[1] == 3:
            mesh.add_scalar_quantity("cluster", l.astype(np.float64),defined_on='faces', cmap='rainbow', enabled=True)
        elif T.shape[1] == 4:
            mesh.add_scalar_quantity("cluster", l.astype(np.float64), defined_on='cells', cmap='rainbow', enabled=True)

    else:
        mesh.add_scalar_quantity("cluster", l.astype(np.float64), cmap='rainbow', enabled=True)

    if pI is not None:
        ps.register_point_cloud("centroids", X[pI, :], radius=0.02)

    if eye_pos is not None and eye_target is not None:
        ps.look_at(eye_pos, eye_target)
    ps.frame_tick()
    if path is not None:
        ps.screenshot(path, transparent_bg=False)
    else:
        ps.show()

    return
