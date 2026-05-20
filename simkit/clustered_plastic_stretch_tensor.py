import numpy as np
from .deformation_jacobian import deformation_jacobian
from .vectorized_transpose import vectorized_transpose

class clustered_plastic_stretch_tensor():
    """
    Helper class for building the clustered plastic stretch tensor,
    this is a precomputed tensor used by Actuators A La Mode. 

    Specifically sometimes you want to solve
    
    argmin_\{a, R\} \sum_{i=1}^c ||F(z) - R Y(a)||^2
    st R is a rotation matrix and a is a vector of actuation amplitudes.
    
    Where x = Bz is the simulation configuration, expressed in a low dimensional space {B}
    
    and y = D a is the actuation configuration, expressed in a low dimensional space {D}
    
    
    The energy is minimized with Local-Gloabl solver, first solving for z (fixing R), then R (fixing z).
    
    When minimizing for R, the generalized procesutes problem shows we need to do a polar deocmposition on evaluate FY^T.
    
    Computing FY^T is bilinear in z, and a, and so a tensor can be precomputed to expose the rotations.
    
    This is that tensor.
    """

    def __init__(self, X, T, l, B, D, w=None):
        """
        Initialize the clustered plastic stretch tensor. Precomputes the tensor for the given actuation subspace and configuration subspace.
        
        Parameters
        ----------
        X (n, d) array of vertex positions
        T (t, s) array of simplex indices
        l (t,) array of cluster labels defining groupings for which the rotation is evaluated.
        B (n, m) array of basis functions for the configuration subspace
        D (n, m) array of displacement functions for the actuation subspace
        w (t, 1) array of weights (default is uniform weights)
        """
        
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
        """
        Evaluate the clustered plastic stretch tensor for the given configuration and actuation.
        
        Parameters
        ----------
        z (m, 1) array of configuration
        a (m, 1) array of actuation
        """
        a = a.reshape(-1)
        z = z.reshape(-1)
        BDa = np.einsum('...acik,k->...aci', self.cBD, a)
        FYT = np.einsum('...aci,c->...ai', BDa, z)
        return FYT
