import numpy as np
from .deformation_jacobian import deformation_jacobian
from .vectorized_transpose import vectorized_transpose

class clustered_plastic_stretch_tensor():

    def __init__(self, X, T, l, B, D, w=None):

        dim = X.shape[1]
        if w is None:
            w = np.ones((T.shape[0], 1))

        J = deformation_jacobian(X, T)
        t = T.shape[0]
        JB = (J @ B).reshape(t, dim, dim, -1)
        JD = (J @ D).reshape(t, dim, dim, -1)
        # BD = np.einsum('...abc,...ibk->...acik', JB, JD)  we used to form this but

        cBD = np.zeros( (l.max() + 1,) + (JB.shape[1] , JB.shape[3] )  + (JD.shape[1] , JD.shape[3] ))
        
        # can't this be written as an einsum? I'm sure it can but it'd have to be a sparse einsum.
        # we have to sum up all BD's , JBs, and JDs based on their clustering
        for i, c in enumerate(np.unique(l)):
            cI = np.where(l == c)[0]
            wcI =  w[cI].reshape(-1, 1, 1, 1, 1)#.transpose(1, 2, 3, 4, 0)

            JBc = JB[cI, :, :, :]
            JDc = JD[cI, :, :, :]
            wJB = w[cI].reshape(-1, 1, 1, 1) * JBc
            cBD[i]= np.einsum('pabc,pibk->acik', wJB, JDc)

        self.cBD = cBD

    def __call__(self, z, a):
        a = a.reshape(-1)
        z = z.reshape(-1)
        BDa = np.einsum('...acik,k->...aci', self.cBD, a)
        FYT = np.einsum('...aci,c->...ai', BDa, z)
        return FYT
