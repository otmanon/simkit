import numpy as np
import polyscope as ps


def triangulated_grid(nx, ny, width=2.0, height=1.0):
    """Right-triangulated rectangular grid in the xy-plane, centered on the origin.

    Returns
    -------
    X : (nx*ny, 2) vertex positions
    T : (2*(nx-1)*(ny-1), 3) triangle indices, CCW
    """
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    X = np.stack([XX.ravel(), YY.ravel()], axis=1)

    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
    v00 = (j * nx + i).ravel()
    v01 = (j * nx + i + 1).ravel()
    v10 = ((j + 1) * nx + i).ravel()
    v11 = ((j + 1) * nx + i + 1).ravel()
    T = np.stack([
        np.stack([v00, v01, v11], axis=1),
        np.stack([v00, v11, v10], axis=1),
    ], axis=1).reshape(-1, 3)
    return X, T


def tetrahedralized_grid(nx, ny, nz, width=1.0, height=1.0, depth=1.0):
    """Tet-meshed rectangular brick (5-tet-per-hex, parity-flipped to match faces).

    Each hex is split into 5 tets; the diagonal direction alternates by
    ``(i+j+k) % 2`` so neighbouring hexes share matching face diagonals. All
    tets are emitted with positive signed volume (compatible with
    ``simkit.volume``).

    Parameters
    ----------
    nx, ny, nz : int
        Vertices per axis (must be >= 2).
    width, height, depth : float
        Extents along x, y, z. Centered on the origin.

    Returns
    -------
    X : (nx*ny*nz, 3) vertex positions
    T : (5*(nx-1)*(ny-1)*(nz-1), 4) tetrahedron indices, all positively oriented
    """
    xs = np.linspace(-width / 2.0, width / 2.0, nx)
    ys = np.linspace(-height / 2.0, height / 2.0, ny)
    zs = np.linspace(-depth / 2.0, depth / 2.0, nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    # Corner indices of a unit hex at (i,j,k):
    # v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
    # v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
    # Two parity-flipped 5-tet patterns; all positively oriented.
    pattern_even = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ])
    pattern_odd = np.array([
        [0, 1, 2, 5],
        [0, 2, 3, 7],
        [0, 5, 7, 4],
        [2, 5, 6, 7],
        [0, 2, 7, 5],
    ])

    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
    )
    ii = ii.ravel()
    jj = jj.ravel()
    kk = kk.ravel()

    def vid(i, j, k):
        return (i * ny + j) * nz + k

    # Per-cell 8 hex corners: shape (Ncells, 8)
    corners = np.stack([
        vid(ii,     jj,     kk    ),
        vid(ii + 1, jj,     kk    ),
        vid(ii + 1, jj + 1, kk    ),
        vid(ii,     jj + 1, kk    ),
        vid(ii,     jj,     kk + 1),
        vid(ii + 1, jj,     kk + 1),
        vid(ii + 1, jj + 1, kk + 1),
        vid(ii,     jj + 1, kk + 1),
    ], axis=1)

    # Per-cell 5 tets, ordering preserved (cell-major).
    even_mask = ((ii + jj + kk) % 2) == 0
    tets = np.empty((corners.shape[0], 5, 4), dtype=corners.dtype)
    tets[even_mask] = corners[even_mask][:, pattern_even]
    tets[~even_mask] = corners[~even_mask][:, pattern_odd]
    T = tets.reshape(-1, 4)
    return X, T


def ball_mesh_2d(radius=0.15, n_segments=48):
    """Triangulated 2D disk centered at the origin: 1 center + n_segments boundary.

    Returns
    -------
    X : (n_segments + 1, 2) vertex positions
    T : (n_segments, 3) triangle indices (fan), CCW
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    boundary = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    X = np.vstack([np.zeros((1, 2)), boundary])
    rim = np.arange(1, n_segments + 1)
    T = np.stack([np.zeros(n_segments, dtype=int), rim, np.roll(rim, -1)], axis=1)
    return X, T


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