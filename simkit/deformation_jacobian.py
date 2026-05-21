import scipy as sp
import numpy as np


from .simplex_vertex_map import simplex_vertex_map

def deformation_jacobian(X : np.array, T : np.array):
    """
    Linear mapping between positions and deformation gradients, assuming x has been flattened with default order="C".

    Parameters
    ----------
    V : (n, dim) array
        The vertices of the mesh

    T : (t, 3|4) array
        Simplex indices

    Returns
    -------
    J : (d*d*t, d*n) sparse matrix
        The deformation Jacobian matrix

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 3)
    ```

    In the above, F will have the form:
    F = [[Fxx Fxy Fxz]]
        [[Fyx Fyy Fyz]]
        [[Fzx Fzy Fzz]]
    """
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]    
    dim = X.shape[1]

    if dim == 1:
        H = np.array([[-1],
                    [1]])
    if dim == 2:
        H = np.array([[-1, -1],
                        [1, 0],
                        [0, 1]])
    if dim == 3:
        H = np.array([[-1, -1, -1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    De = np.zeros((nt, dim*dim,  dt*dim))
   
    for i in range(dim):
        Di = np.zeros((nt, dim, dt*dim))
        Ii = np.arange(dt)*dim + i
        Di[:, :, Ii] = D
        
        De[:, dim*i:dim*(i + 1), :] = Di      
      

    Ii = np.arange(dim*dim*nt).reshape(nt, dim*dim, 1)
    Ii = np.repeat(Ii, dim*dt, axis=2 )

    Ji = np.arange(dim*dt*nt).reshape(nt, 1, dim*dt)
    Ji = np.repeat(Ji, dim*dim, axis=1 )

    dims = np.prod(De.shape).item()
    H = sp.sparse.csc_matrix((De.flatten(), (Ii.flatten(), Ji.flatten())), (nt*dim*dim, nt*dim*dt))

    S = simplex_vertex_map(T)
    Se = sp.sparse.kron(S, sp.sparse.identity(dim))
    J = H @ Se
    return J




def membrane_deformation_jacobian(X : np.array, T : np.array):
    '''
    Linear mapping between positions and deformation gradients, assuming x has been flattened with default order="C":

    Parameters
    ----------
    V : (n, dim) array
        The vertices of the mesh

    T : (t, 3) array
        Triangle indices

    Returns
    -------
    J : ((d-1)*d*t, d*n) sparse matrix
        The deformation Jacobian matrix

    Example
    -------
    ```python
    x = X.reshape(-1, 1)
    f = J @ x
    F = f.reshape(-1, 3, 2)
    ```

    In the above, F will have the form:
    F = [[Fxu Fxv]]
        [[Fyu Fyv]]
        [[Fzu Fzv]]
    '''
    dt = T.shape[-1]
    T = T.reshape(-1, dt)
    nt = T.shape[0]    
    dim = X.shape[1]

    if dim == 1:
        H = np.array([[-1],
                    [1]])
    if dim == 2:
        H = np.array([[-1, -1],
                        [1, 0],
                        [0, 1]])
    if dim == 3:
        H = np.array([[-1, -1, -1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    XT = X[T].transpose(0, 2, 1)
    XH = XT @ H
    XHi = np.linalg.inv(XH)
    D = (H @ XHi).transpose(0, 2, 1)

    De = np.zeros((nt, dim*dim,  dt*dim))
   
    for i in range(dim):
        Di = np.zeros((nt, dim, dt*dim))
        Ii = np.arange(dt)*dim + i
        Di[:, :, Ii] = D
        
        De[:, dim*i:dim*(i + 1), :] = Di      
      

    Ii = np.arange(dim*dim*nt).reshape(nt, dim*dim, 1)
    Ii = np.repeat(Ii, dim*dt, axis=2 )

    Ji = np.arange(dim*dt*nt).reshape(nt, 1, dim*dt)
    Ji = np.repeat(Ji, dim*dim, axis=1 )

    dims = np.prod(De.shape).item()
    H = sp.sparse.csc_matrix((De.flatten(), (Ii.flatten(), Ji.flatten())), (nt*dim*dim, nt*dim*dt))

    S = simplex_vertex_map(T)
    Se = sp.sparse.kron(S, sp.sparse.identity(dim))
    J = H @ Se
    return J


