
import numpy as np
import scipy as sp
def interweaving_matrix(t, d):
    
    
    ii = np.arange(t*d)
    
    i = ii.reshape(t, d)
    j = ii.reshape(t, d, order='F')
    v = np.ones(ii.shape)
    M = sp.sparse.csc_matrix((v, (i.flatten(), j.flatten())), shape=(t*d, t*d))
    return M