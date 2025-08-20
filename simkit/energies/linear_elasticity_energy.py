import numpy as np
import scipy as sp


from ..vectorized_trace import vectorized_trace
from ..vectorized_transpose import vectorized_transpose

from ..polar_svd import polar_svd


def linear_elasticity_energy_F(F, mu, lam, vol):
    """
    Linear Elasticity from the Sifakis Notes
    https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
    """
    assert(F.ndim == 3)
    assert(F.shape[1] == F.shape[2])
    
    mu = np.array(mu).reshape(-1, 1)
    lam = np.array(lam).reshape(-1, 1)
    vol = np.array(vol).reshape(-1, 1)
    eps = (F + F.transpose(0, 2, 1))/2 - np.identity(F.shape[1])[None, :, :]
    density = mu.flatten()* np.sum(eps**2, axis=(1, 2)) + (lam.flatten()/2.0)* np.trace(eps, axis1=1, axis2=2)**2
    E = (density * vol.flatten()).sum()
    return E


def linear_elasticity_energy_density_F(F, mu, lam, vol):
    """
    Linear Elasticity from the Sifakis Notes
    https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
    """
    assert(F.ndim == 3)
    assert(F.shape[1] == F.shape[2])
    
    mu = np.array(mu).reshape(-1, 1)
    lam = np.array(lam).reshape(-1, 1)
    vol = np.array(vol).reshape(-1, 1)
    eps = (F + F.transpose(0, 2, 1))/2 - np.identity(F.shape[1])[None, :, :]
    density = mu.flatten()* np.sum(eps**2, axis=(1, 2)) + (lam.flatten()/2.0)* np.trace(eps, axis1=1, axis2=2)**2
    psi = (density * vol.flatten())
    return psi



def linear_elasticity_energy_df(f, mu, lam, vol, dim):
    """
    Linear Elasticity from the Sifakis Notes
    https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
    """

    nt = f.shape[0]//(dim*dim)

    T = vectorized_transpose(nt, dim)
    Tr = vectorized_trace(nt, dim)

    i = np.repeat(np.identity(dim)[None, :, :], nt, axis=0).flatten()[:, None]
    eps = (f + T @ f)/2 - i


    density = mu * Tr @ (eps**2) + (lam/2.0)* (Tr @ eps)**2

    E = (density.flatten() * vol.flatten()).sum()

    return E