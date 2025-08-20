import scipy as sp
import numpy as np

def symmetric_stretch_map(t, dim):
    """
        [S0  S1  S2] = [s0  s3  s4]
    S = [S3  S4  S5] = [s3  s1  s5]
        [S6  S7  S8] = [s4  s5  s2]

    vec(S) = [s0, s1, s2, s3, s4, s5, s6, s7, s8]^T

    s = [s0, s4, s8, s1, s2, s3]^T

    [S0]   [1 0 0 0 0 0]
    |S1|   |0 0 0 1 0 0| [s0]
    |S2|   |0 0 0 0 1 0| |s1|
    |S3| = |0 0 0 1 0 0| |s2|
    |S4|   |0 1 0 0 0 0| |s3|
    |S5|   |0 0 0 0 0 1| |s4|
    |S6|   |0 0 0 0 1 0| [s5]
    |S7|   |0 0 0 0 0 1|
    [S8]   [0 0 1 0 0 0]      

    
    [S0]   [1 0 0]
    |S1| = |0 0 1] [s0]
    |S2|   |0 0 1] |s1|
    [S3]   [0 1 0] [s2]

    [s0] =  [1  0  0  0 ] [S0]   
    |s1| =  |0  0  0  1 ] |S1|  
    [s2] =  [0  1/2 1/2  0] |S2|   
                          [S3]   

    """
    # if dim == 2:
    #     I = np.array([0, 3, 1, 2])
    #     J = np.array([0, 1, 2, 2])
    #     v = np.ones((4))
    #     S = sp.sparse.csc_matrix((v, (I, J)), shape=(dim*dim, 3))
    #     # Ii = np.array([0, 1, 2, 2])
    #     # Ji = np.array([0, 3, 1, 2])
    #     vi = np.array([1, 1/2, 1/2, 1])
    #     Si = sp.sparse.csc_matrix((vi, (J, I)), shape=(3, dim*dim))
    # elif dim == 3:
    #     I = [0, 4, 8, 1, 2, 3, 5, 6, 7]
    #     J = [0, 1, 2, 3, 4, 3, 5, 4, 5]
    #     v = np.ones((9, ))
    #     S = sp.sparse.csc_matrix((v, (I, J)), shape=(dim*dim, 6))
    #     vi = [1, 1, 1, 1/2, 1/2, 1/2,  1/2, 1/2, 1/2]
    #     Si = sp.sparse.csc_matrix((vi, (J, I)), shape=(6, dim*dim))
    # else:
    #     raise ValueError("dim must be 2 or 3")
    
    SI = np.arange(dim*dim, dtype=int).reshape(dim, dim)
    SJ = -np.ones((dim, dim), dtype=int)
    SV = np.zeros((dim, dim), dtype=float)
    SVi = np.zeros((dim, dim), dtype=float)
    counter = 0
    for i in range(dim):
        SJ[i, i] = counter
        counter += 1
        SV[i, i] = 1
        SVi[i, i] = 1

    for i in range(dim):
        for j in range(dim):
            if i==j:
                continue
            else:
                if j > i: # upper triangular
                    SJ[i, j] = counter
                    counter += 1
                elif j < i: # lower triangular
                    SJ[i, j] = SJ[j, i]
                SV[i, j] = 1
                SVi[i, j] = 1/2
    

    # dim * dim - (dim-1)
    # dim  + (dim-1) + (dim - 2)) 
    S = sp.sparse.csc_matrix((SV.flatten(), (SI.flatten(), SJ.flatten())), shape=(dim*dim, SJ.max() + 1 ))
    Si = sp.sparse.csc_matrix((SVi.flatten(), (SJ.flatten(), SI.flatten())), shape=(SJ.max()+1, dim*dim))

    Se = sp.sparse.kron(sp.sparse.identity(t), S)
    Sei = sp.sparse.kron(sp.sparse.identity(t), Si)
    return Se, Sei