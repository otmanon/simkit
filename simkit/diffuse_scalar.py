import numpy as np
import scipy as sp


from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix


def diffuse_scalar(X, T, bI, y, h=None, ord=1, mu=1):
    """
    
    """

    L = dirichlet_laplacian(X, T, mu=mu)
    M = massmatrix(X,  T)

    if ord ==1:
        H = L 
    elif ord == 2:
        Mi = sp.sparse.diags(1 / M.diagonal())
        H = L.T @ Mi @ L 

    if h is not None:
        H = H + M / h**2
        
    W = np.zeros((X.shape[0], y.shape[1]))

    aI = np.setdiff1d(np.arange(X.shape[0]), bI)

    Hii = H[aI, :][:, aI]
    Hbi = H[aI, :][:, bI]

    Wii = sp.sparse.linalg.spsolve(Hii, -Hbi @ y)

    if Wii.ndim == 1:
        Wii = Wii.reshape(-1, 1)
        
    W[aI, :] = Wii

    W[bI, :] = y

    # import polyscope as ps
    # ps.init()
    # mesh = ps.register_surface_mesh("mesh", X, T)
    # for i in range(W.shape[1]):
    #     mesh.add_scalar_quantity("handle : " + str(i), W[:, i], enabled=True, cmap="reds")
    # ps.set_ground_plane_mode("none")
    # ps.show()
    return W