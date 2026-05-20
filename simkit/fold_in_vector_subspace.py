import numpy as np

def fold_in_vector_subspace(B, dim):

    n = B.shape[0] // dim

    I = np.repeat(np.arange(n)[:, None], dim, axis=1)* dim + np.arange(dim)[None, :]

    Ws = np.zeros((n, dim*B.shape[1]))


    for i in range(dim):
        ii = I[:, i]
        Bi = B[ii, :]
        
        Ws[:, i*B.shape[1]:(i+1)*B.shape[1]] = Bi


    return Ws

        

