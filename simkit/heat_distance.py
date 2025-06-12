import numpy as np
import scipy.sparse as sp
import igl

"""
This module extends the heat-based geodesics from
https://github.com/libigl/libigl/blob/main/include/igl/heat_geodesics.h
to support input Laplacians, allowing for stiffness-weighted Laplacians (i.e. compliance distances)
"""
class HeatGeodesicsData:
    """
    Container class for precomputed data used in the heat method.

    Attributes:
    - Grad: Discrete gradient operator
    - M: Mass matrix
    - Div: Discrete divergence operator
    - Neumann / Dirichlet / Poisson: Data for solving corresponding linear systems
    - b: List of boundary vertex indices (for Dirichlet BCs)
    - ng: Number of gradient components per simplex (2 in 2D, 3 in 3D)
    """
    def __init__(self):
        self.Grad = None
        self.M = None
        self.Div = None
        self.Neumann = igl.pyigl_core.min_quad_with_fixed_data()
        self.Dirichlet = igl.pyigl_core.min_quad_with_fixed_data()
        self.Poisson = igl.pyigl_core.min_quad_with_fixed_data()
        self.b = None
        self.ng = 0

def heat_distance_precompute(V, F, L=None, t=None, data=None):
    """
    Precompute the necessary matrices for heat-based geodesic distance computation.

    Parameters:
    - V: Vertex positions (n x 3)
    - F: Faces or elements (m x 3 or m x 4)
    - L: (Optional) custom Laplacian matrix. Defaults to cotangent Laplacian
    - t: (Optional) time step for the heat method. Defaults to avg_edge_length^2
    - data: (Optional) existing HeatGeodesicsData object to reuse

    Returns:
    - data: Filled-in HeatGeodesicsData object
    """
    if data is None:
        data = HeatGeodesicsData()
    if t is None:
        h = igl.avg_edge_length(V, F)
        t = h * h
    if L is None:
        L = igl.cotmatrix(V, F)

    M = igl.massmatrix(V, F)
    if F.shape[1] == 3:
        dblA = igl.doublearea(V, F) * 0.5
        data.Grad = igl.grad(V, F) * 0.5 # triangle grad is doubled, so half it
    else:
        dblA = igl.volume(V, F)
        data.Grad = igl.grad(V, F)
    data.M = M

    data.ng = data.Grad.shape[0] // F.shape[0]

    diag = sp.diags(np.tile(dblA, (1, data.ng)).flatten())
    data.Div = -data.Grad.T @ diag

    Q = M - t * L
    O,_,_ = igl.boundary_facets(F)
    data.b = np.unique(O.flatten())

    # create empty sparse matrix
    Aeq = sp.csc_matrix((0, Q.shape[1]))
    igl.min_quad_with_fixed_precompute(Q,  np.array([], dtype=np.int64), Aeq, True, data.Neumann)
    if data.b.size > 0:
        igl.min_quad_with_fixed_precompute(Q, data.b, Aeq, True, data.Dirichlet)

    L *= -0.5
    Aeq = sp.csc_matrix(M.diagonal())
    igl.min_quad_with_fixed_precompute(L, np.array([], dtype=np.int64), Aeq, True, data.Poisson)
    return data

def heat_distance_solve(data, gamma, mode='average'):
    """
    Solve for geodesic distances from a set of source vertices using the heat method.

    Parameters:
    - data: Precomputed HeatGeodesicsData object
    - gamma: List or array of source vertex indices
    - mode: Solve mode; one of ['neumann', 'dirichlet', 'average']

    Returns:
    - D: Approximate geodesic distance vector (n x 1)
    """
    n = data.Grad.shape[1]
    u0 = np.zeros((n,1))
    u0[gamma] = 1

    # Create empty matrices 0x0
    Beq = np.zeros((0,0))
    Yn = np.zeros((0, 1))
    Yp = np.ones((0, 1))

    if mode == 'neumann':
        u = igl.min_quad_with_fixed_solve(data.Neumann, data.M @ u0, Yn, Beq)
    elif mode == 'dirichlet':
        u = igl.min_quad_with_fixed_solve(data.Dirichlet, data.M @ u0, np.zeros((len(data.b), 1)), Beq)
    elif mode == 'average':
        u = igl.min_quad_with_fixed_solve(data.Neumann, data.M @ u0, Yn, Beq)
        uD = igl.min_quad_with_fixed_solve(data.Dirichlet, data.M @ u0, np.zeros((len(data.b), 1)), Beq)
        u = 0.5 * (u + uD)
    else:
        raise ValueError("Invalid mode. Choose from 'neumann', 'dirichlet', or 'average'.")
    grad_u = data.Grad @ u

    # Stable normalization
    m = data.Grad.shape[0] // data.ng
    grad_u = grad_u.reshape(data.ng, m)

    ma = np.max(np.abs(grad_u), axis=0)
    ma_safe = np.where(ma == 0, 1.0, ma)
    scaled = grad_u / ma_safe
    norm = np.linalg.norm(scaled, axis=0) * ma

    # Handle cases where ma or norm is zero or norm is NaN
    mask = (ma == 0) | (norm == 0) | np.isnan(norm)
    grad_u[:, mask] = 0

    # Normalize the gradients for non-masked columns
    grad_u[:, ~mask] /= norm[~mask]

    # Reshape back to the original shape
    grad_u = grad_u.reshape(-1, 1)

    div_X = -data.Div @ grad_u
    Beq = np.zeros((1, 1)) # constrain constant to be 0
    D = igl.min_quad_with_fixed_solve(data.Poisson, -div_X, Yp, Beq)
    D -= np.mean(D[gamma])
    if np.mean(D) < 0:
        D = -D
    return D
