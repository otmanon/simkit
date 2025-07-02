import numpy as np

from .deformation_jacobian import deformation_jacobian

def limit_actuation_dirichlet_energy(X, T, D, max_s):

    J = deformation_jacobian(X, T)
    dim = X.shape[1]
    JD =  (J @ D).reshape(T.shape[0], dim, dim, D.shape[1])
    dirichlet_energy_density =  np.sqrt(np.sum(JD**2, axis=(1, 2)))
    a =  max_s / np.max(dirichlet_energy_density, axis=0)
    a = a.reshape(-1,)
    return a
