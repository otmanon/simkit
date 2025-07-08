import numpy as np
import scipy as sp
import warnings
import os
import igl

from ..polyscope.view_clusters import view_clusters


from .Subspace import Subspace, SubspaceParams


from ..polyscope.view_scalar_modes import view_scalar_modes
from ..polyscope.view_displacement_modes import view_displacement_modes


from ..farthest_point_sampling import farthest_point_sampling
from ..selection_matrix import selection_matrix
from ..massmatrix import massmatrix
from ..linear_modal_analysis import linear_modal_analysis
from ..orthonormalize import orthonormalize
from ..spectral_cubature import spectral_cubature


class ModalAnalysisParams(SubspaceParams):
    def __init__(self, m, k, c, cache_dir = None, read_cache=False):
        self.m = m  
        self.k = k
        self.c = c
        self.read_from_cache = read_cache
        self.name = "modal_analysis_m"+str(m)+"_k"+str(k)+"_c"+str(c)
        self.cache_dir = cache_dir + "/" + self.name + "/"
        pass

class ModalAnalysis():
    """
    Modal analysis class
    """
    def __init__(self,  X, T, params: ModalAnalysisParams):
        
        self.params = params
        cache_dir = params.cache_dir

        self.X = X
        self.T = T
        well_read = False
        if params.read_from_cache:
            try:
                [E, B, cI, cW, labels, conI, SB] = self.read_subspace(cache_dir)
                well_read = True
            except:
                warnings.warn("Warning : Couldn't read subspace from cache. Recomputing from scratch...")

        if not well_read:
            [ E, B, cI, cW, labels, conI, SB] = self.compute_subspace(X, T, params)
            if params.cache_dir is not None:
                os.makedirs(cache_dir, exist_ok=True)
                self.save_subspace(cache_dir, E, B, cI, cW, labels, conI, SB)

        self.E = E
        self.B = B
        self.cI = cI
        self.cW = cW
        self.labels = labels
        self.conI = conI
        self.SB = SB
        return


    def read_subspace(self, cache_dir):
        E = np.load(cache_dir + "E.npy")
        B = np.load(cache_dir + "B.npy")
        cI = np.load(cache_dir + "cI.npy")
        cW = np.load(cache_dir + "cW.npy")
        labels = np.load(cache_dir + "labels.npy")
        conI = np.load(cache_dir + "conI.npy")
        SB = np.load(cache_dir + "SB.npy")
        return  E, B, cI, cW, labels, conI, SB
    
    def save_subspace(self, cache_dir,  E, B, cI, cW, labels, conI, SB):
        np.save(cache_dir + "E.npy", E)
        np.save(cache_dir + "B.npy", B)
        np.save(cache_dir + "cI.npy", cI)
        np.save(cache_dir + "cW.npy", cW)
        np.save(cache_dir + "labels.npy", labels)
        np.save(cache_dir + "conI.npy", conI)
        np.save(cache_dir + "SB.npy", SB)
        return

    def compute_subspace(self, X, T, params):
                
        dim = X.shape[1]
        [ E,  B] = linear_modal_analysis(X, T, k=params.m)

        # concatenate
        B = np.concatenate([B, X.reshape(-1, 1)], axis=1)
        # I = np.repeat((np.arange(X.shape[0]), 1))

        W = np.zeros((X.shape[0], 10 * dim))
        for i in range(dim):
            I = np.arange(X.shape[0])* dim + i

            Wi = B[I, :10]
            W[:, i *10: (i+1)*10] = Wi

        B = orthonormalize(B, M=sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))
        [cI, cW, labels] = spectral_cubature(X, T, W, params.k, return_labels=True)

        # view_clusters(X, T, labels)
        n = X.shape[0]
        faceI = np.unique(igl.boundary_facets(T))
        sI = farthest_point_sampling(X[faceI], params.c)
        conI = faceI[sI]
        S = selection_matrix(conI, n)
        Se = sp.sparse.kron(S, sp.sparse.identity(dim))
        SB = Se @ B

        return  E, B, cI, cW, labels, conI, SB

    def vis_subspace(self, eye_pos=None, eye_target=None):
        
        # view_displacement_modes(self.X, self.T, self.B, dir = self.params.cache_dir + "/B/", normalize=True, eye_pos=eye_pos, eye_target=eye_target)
        # view_scalar_modes(self.X, self.T, self.Wd, dir = self.params.cache_dir + "/Wd/", normalize=True, eye_pos=eye_pos, eye_target=eye_target)
        # view_scalar_modes(self.X, self.T, self.W, dir = self.params.cache_dir + "/W/", normalize=True, eye_pos=eye_pos, eye_target=eye_target)

        # view_clusters(self.X, igl.boundary_facets(self.T), self.labels, path = self.params.cache_dir + "/labels.png", eye_pos=eye_pos, eye_target=eye_target)
        return