
import numpy as np


def average_onto_simplex(A, T):
    """
    Average the node values A onto the simplices T to which th enodes belong.
    
    A (n, d) array of node values
    T (t, s) array of simplex indices
    
    Returns
    -------
    At (t, d) array of simplex values
    
    """
    
    At = np.zeros((T.shape[0], A.shape[1]))
    for td in range(T.shape[1]):
        At += (A[T[:, td], :])/T.shape[1]

    return At