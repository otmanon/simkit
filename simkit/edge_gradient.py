import scipy as sp
import numpy as np

from simkit.edge_lengths import edge_lengths



def edge_gradient(X, E):
    l = edge_lengths(X, E)
    li = 1 / l
    I = np.repeat(np.arange((E.shape[0]))[:, None], 2, axis=1)
    J = E
    V = np.hstack([-li[:, None], li[:, None]])
    G = sp.sparse.csc_matrix((V.flatten(), (I.flatten(), J.flatten())), shape=(E.shape[0], X.shape[0]))
    return G