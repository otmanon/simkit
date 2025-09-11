import numpy as np
import scipy as sp

import igl

from .energies.linear_elasticity_hessian import linear_elasticity_hessian
from .eigs import eigs
from .fold_vector_hessian import fold_vector_hessian
from simkit.massmatrix import massmatrix

def substructuring(X, T, labels, return_components=False):
    """ 
    Performs substructuring on a mesh X, T, with substructures defined
    by per-simplex clusters labels
    
    Returns:
    Xexp: the exploded mesh
    Texp: the exploded mesh connectivity
    E: the exploded mesh interface edges

    Xcomp: the exploded mesh components
    Tcomp: the exploded mesh components connectivity
    """
    
    dim = X.shape[1]
    # split mesh
    Xexp = np.zeros((0, dim))
    Texp = np.zeros((0, dim+1), dtype=int)
    Xcomp = []
    Tcomp = []
    k = labels.max() +1
    
    inverse_map = np.zeros(0, dtype=int)
    forward_map = np.zeros(0, dtype=int)
    for i in range(k):
        Ti = T[labels == i]
        Xi, Ti, IM, J = igl.remove_unreferenced(X, Ti)
        
        Texp = np.concatenate((Texp, Ti + Xexp.shape[0]))
        Xexp = np.concatenate((Xexp, Xi))
        
        inverse_map = np.concatenate((inverse_map, IM))
        forward_map = np.concatenate((forward_map, J))
        Xcomp.append(Xi)
        Tcomp.append(Ti)
            
    _unique, _uniqueI, unique_inv, uniqueC= np.unique(Xexp, axis=0, return_index=True, return_inverse=True, return_counts=True)

    v = np.ones(unique_inv.shape[0], dtype=int)
    A = sp.sparse.csc_matrix((v, (unique_inv, np.arange(unique_inv.shape[0]))))  
    at_interface = uniqueC > 1
    at_interfaceI = np.where(at_interface)[0]
    Ainterface = A[at_interfaceI]

    uniqueCinterface = uniqueC[at_interface]
    max_count = uniqueCinterface.max()

    E = np.zeros((0, 2), dtype=int)# interface edges
    for count in range(2, max_count+1):
        at_count_interface = uniqueCinterface == count
        at_count_interfaceI = np.where(at_count_interface)[0]
        
        # exploded edge 1
        # exploded edge 2
        # add a bunch of pairwise edges
        [ii, jj] = Ainterface[at_count_interfaceI].nonzero()
        Ect = jj.reshape( -1, count)
        
        for e in range(count-1):
            Ec = Ect[:, [e, e+1]]
            E = np.concatenate((E, Ec), axis=0)
    
    if return_components:
        return Xexp, Texp, E, forward_map, inverse_map, Xcomp, Tcomp
    else:
        return Xexp, Texp, E, forward_map, inverse_map


        
        
def substructured_skinning_eigenmodes(Xcomp, Tcomp, m, n, return_components=False):

    # now build skinning eigenmodes for each component
    Wcomp = []
    k = len(Xcomp)
    W = np.zeros((n, m*k))
    ind = 0
    for i in range(k):
        Xi = Xcomp[i]
        Ti = Tcomp[i]
        dim = Xi.shape[1]
        Hi = linear_elasticity_hessian(X=Xi, T=Ti)
        Mi = massmatrix(X=Xi, T=Ti)
        Li = fold_vector_hessian(Hi, dim)
        [Di, Wi] = eigs(Li, m, M=Mi)
        Wcomp.append(Wi)
        
        W[ind:ind+Xcomp[i].shape[0], i*m:(i+1)*m] = Wi
        ind += Xcomp[i].shape[0]
        
    if return_components:
        return W, Wcomp
    else:
        return W
    