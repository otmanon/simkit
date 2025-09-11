import scipy as sp
import numpy as np



from .normals import normals
from .massmatrix import massmatrix
from .volume import volume
from .simplex_vertex_averaging_matrix import simplex_vertex_averaging_matrix

def normal_force_matrix(X, E, mass_weigh=True):

    dim = X.shape[1]
    Me = massmatrix(X=X, T=E)
    Mee = sp.sparse.kron(Me, sp.sparse.identity(dim))
    N = normals(X, E)[:, :, None]
    Nmat = sp.sparse.block_diag(N)
    vol = volume(X, E)
    Av = simplex_vertex_averaging_matrix(E, X.shape[0], vol)
    Ave = sp.sparse.kron(Av, sp.sparse.identity(dim))
    D = Mee @ (Ave @ Nmat).toarray()
    # sigma_F_sqrt = -(D *1e4) #.sum(axis=1).reshape(-1, 1)
    
    return D