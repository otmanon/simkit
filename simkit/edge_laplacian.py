import scipy as sp
import numpy as np

from .edge_lengths import edge_lengths
from .edge_gradient import edge_gradient



def edge_laplacian(X, E):
    l = edge_lengths(X, E)
    li = 1 / l
    
    A = sp.sparse.diags(l)
    G = edge_gradient(X, E)
    L = G.T @ A @ G
    return L