import numpy as np

def closed_polyline(V):
    E = np.hstack((np.arange(V.shape[0])[:, None], np.arange(V.shape[0])[:, None] + 1))
    E[-1, -1] = 0
    return E
