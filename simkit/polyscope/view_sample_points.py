import polyscope as ps


def view_sample_points(X, T, sample_points, path=None, eye_pos=None, eye_target=None):
    """
    Visualize a mesh (X, T) and a list of sample point positions using polyscope.

    Args:
        X (np.ndarray): Vertex positions of the mesh (N, 3).
        T (np.ndarray): Mesh topology (indices into X).
        sample_points (np.ndarray): Array of sample point positions (M, 3).
        eye_pos (np.ndarray, optional): Camera eye position (3,).
        eye_target (np.ndarray, optional): Camera target position (3,).
    """
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
    else:
        raise ValueError("Unsupported mesh topology shape: {}".format(T.shape[1]))

    ps.register_point_cloud("sample_points", sample_points, radius=0.02)

    if eye_pos is not None and eye_target is not None:
        ps.look_at(eye_pos, eye_target)

    ps.frame_tick()
    
    if path is not None:
        ps.screenshot(path, transparent_bg=False)
    else:
        ps.show()