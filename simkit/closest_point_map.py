
import numpy as np
import igl
import scipy as sp


def closest_point_map(Y, X):
    _sqrD, I, _bary = igl.point_mesh_squared_distance(Y, X, np.arange(X.shape[0])[:, None])
    jj = I
    ii = np.arange(jj.shape[0])
    val = np.ones(I.shape[0])
    M_tex = sp.sparse.csc_matrix((val, (ii, jj)), (Y.shape[0], X.shape[0]))
    return M_tex
    