
import numpy as np

import scipy as sp

import simkit as sk

from .interweaving_matrix import interweaving_matrix

def interweaving_matrix(t, d):
    
    
    ii = np.arange(t*d)
    
    i = ii.reshape(t, d)
    j = ii.reshape(t, d, order='F')
    v = np.ones(ii.shape)
    M = sp.sparse.csc_matrix((v, (i.flatten(), j.flatten())), shape=(t*d, t*d))
    return M

def membrane_deformation_gradient(Y, X, T):
    X0 = X[T[:, 0]]
    X1 = X[T[:, 1]]
    X2 = X[T[:, 2]]
    
    V1 = X1 - X0
    V2 = X2 - X0
    
    dim = 2
    dt = 3
    U1 = V1 / np.linalg.norm(V1, axis=1)[:, None]
    N = np.cross(V1, V2)
    N = N / np.linalg.norm(N, axis=1)[:, None]
    U2 = np.cross(N, U1)
    
    # edge vectors in the local basis
    E2D = np.array([[ (U1*V1).sum(axis=1), (U1*V2).sum(axis=1)], 
              [(U2*V1).sum(axis=1), (U2*V2).sum(axis=1)]]).transpose(2, 0, 1)
   
    E2Di = np.linalg.inv(E2D)
    
    v1 = (Y[T[:, 1]] - Y[T[:, 0]])[:, :, None ]
    v2 = (Y[T[:, 2]] - Y[T[:, 0]])[:, :, None ]
    G = np.concatenate([v1, v2], axis=2)
    F = G @ E2Di
    return F

  
    
def membrane_deformation_jacobian(X, T):
    
    assert X.shape[1] == 3
    assert T.shape[1] == 3
    X0 = X[T[:, 0]]
    X1 = X[T[:, 1]]
    X2 = X[T[:, 2]]
    
    V1 = X1 - X0
    V2 = X2 - X0
    
    dim = 2
    dt = 3
    U1 = V1 / np.linalg.norm(V1, axis=1)[:, None]
    N = np.cross(V1, V2)
    N = N / np.linalg.norm(N, axis=1)[:, None]
    U2 = np.cross(N, U1)
    
    
    # edge vectors in the local basis
    E2D = np.array([[ (U1*V1).sum(axis=1), (U1*V2).sum(axis=1)], 
              [(U2*V1).sum(axis=1), (U2*V2).sum(axis=1)]]).transpose(2, 0, 1)
   
    E2Di = np.linalg.inv(E2D)
    
    H = np.array([[-1, -1],
                    [1, 0],
                    [0, 1]])
    D = (H[None, :, :] @ E2Di).transpose(0, 2, 1)
    # D = D.reshape(-1, dim*(dim+1))
    # D = D.reshape(-1, dim+1, dim).transpose(0, 2, 1)
    G = np.kron(D, np.eye(dim+1))
    M = interweaving_matrix(3, 2).toarray()
    G = M[None, :, :] @ G
    Q = sp.sparse.block_diag(G)
    S = sk.simplex_vertex_map(T)
    
    # nt = T.shape[0]
    # De = np.zeros((nt, (dim+1)*dim,  dt*dim))
   
    # for i in range(dim):
    #     Di = np.zeros((nt, dim, dt*dim))
    #     Ii = np.arange(dt)*dim + i
    #     Di[:, :, Ii] = D
        
    #     De[:, dim*i:dim*(i + 1), :] = Di      
  
    Se = sp.sparse.kron(S, sp.sparse.identity(dim+1))
    J = Q @ Se
    return J




