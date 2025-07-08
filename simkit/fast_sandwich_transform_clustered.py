import os

import scipy as sp
import numpy as np


'''
Quickly evaluate A R B, where R is a block diagonal rotation marix
 parameterized by very few parameters and changes frequently, and A and B are constant

'''

class fast_sandwich_transform_clustered:


    '''
    Inputs:
    A - m x 9n matrix, where columns are ordered in the xx yy zz fashion
    B - 9n x m matrix, where columns are ordered in the xx yy zz fashion

    Assume the flattening of these n 3x3 matrices occurs as follows:
    first tet:
    f11
    f12
    f13
    f21
    f22
    f23
    f31
    f32
    f33
    ---
    next tet
    '''
    def __init__(s, A, B, l, read_cache=False, cache_dir=None, dim=3):

        s.dim = dim
        n = A.shape[1] // (dim * dim)

        # this is not necessarily true for each tet

        A = A
        B = B

        num_clusters = l.max() + 1
        s.num_clusters = num_clusters

        Re = np.zeros(( dim, dim, dim, dim))
        for i in range(dim):
            for j in range(dim):
                Re[i, j, i, j] = 1

        def clustered_ARBs(c):
            dim = s.dim
            ARBs = np.zeros((A.shape[0], B.shape[1], dim, dim))
            ti = np.where(l == c)[0]  # tet indices that belong to cluster c
            num_t = ti.shape[0]
            v = np.ones(ti.shape[0])
            tie = (np.tile(ti[:, None], (1, dim*dim))*dim*dim + np.arange(dim*dim)[None, :]).flatten()

            Ati = A[:, tie]
            # # Atir = Ati.reshape((Ati.shape[0], -1, 1,  3, 3))
            Bti = B[tie, :]
            # # Btir = Bti.reshape(( Bti.shape[1],-1, 1,  3, 3))

            off = np.arange(num_t)
            for c in range(dim*dim):
                i = c // dim
                j = c % dim
                II = dim*dim * off[:, None] + dim*i  + np.arange(dim)# np.array([0, 1, 2]) #np.array([i, i + 3 * 1, i + 3 * 2])
                JJ = dim*dim * off[:, None] + dim*j + np.arange(dim)# np.array([0, 1, 2]) #np.array([j, j + 3 * 1, j + 3 * 2])

                VV = np.tile(v[:, None], (1, dim))
                S = sp.sparse.csc_matrix((VV.flatten(), (II.flatten(), JJ.flatten())), shape=(dim*dim * num_t, dim*dim * num_t))
                ARBs[:, :, i, j] = Ati @ (S @ Bti)
            return ARBs

        if cache_dir is not None and read_cache:
            try:
                s.ARBs = np.load(cache_dir + "/ARBs.npy")
                print("Loaded ARBs from cache " + cache_dir)
            except:
                vfunc = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
                o = vfunc(np.arange(num_clusters))
                s.ARBs = np.stack(o).transpose(1, 2, 0,  3, 4)
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_dir + "/ARBs.npy", s.ARBs)
        else:
            vfunc = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
            o = vfunc(np.arange(num_clusters))
            s.ARBs = np.stack(o).transpose(1, 2, 0, 3, 4)

            # vfunc1 = np.vectorize(clustered_ARBs, otypes=[np.ndarray])
            # o1 = vfunc1(np.arange(num_clusters))
            # s.ARBs1 = np.stack(o1).transpose(1, 2, 0, 3, 4)
            if cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_dir + "/ARBs.npy", s.ARBs)

    '''
    Inputs :
    r - 3 x 3 rotation matrix
    '''
    def __call__(s, r):
        ARB = s.eval(r)
        return ARB


    def eval(s, r):
        r = r.reshape((-1, s.dim, s.dim))

        assert(r.shape[0] == s.num_clusters and \
               "FST set up for " + str(s.num_clusters) + " rotation clusters, but only got " + str(r.shape[0]) )
        # r2 = r[:,  :, :]

        prod = s.ARBs * r
        ARB = np.sum(prod, axis=(-3, -2, -1))
        return ARB
