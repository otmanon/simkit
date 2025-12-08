import numpy as np
import scipy as sp

from ..deformation_jacobian import deformation_jacobian
from ..volume import volume
from ..vectorized_trace import vectorized_trace
from ..vectorized_transpose import vectorized_transpose

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





def linear_elasticity_gradient_dF(F, mu, lam, vol):
    """
    Linear Elasticity from the Sifakis Notes
    https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
    """

    mu = np.array(mu)
    lam = np.array(lam)
    vol = np.array(vol)
    assert(F.ndim == 3)
    assert(F.shape[1] == F.shape[2])

    # dtracedeps = np.repeat(np.identity(F.shape[1])[None, :], F.shape[0], axis=0)
    I = np.identity(F.shape[1])[None, :, :]
    dPsidF = mu.reshape(-1, 1, 1) * (F + F.transpose(0, 2, 1) - 2*I ) + lam.reshape(-1, 1, 1) * np.trace(F - I, axis1=1, axis2=2)[:, None, None] * np.identity(F.shape[1])[None, :, :]
    P = (dPsidF * vol.flatten()[:, None, None])

    return P



def linear_elasticity_gradient_dx(F, mu, lam, vol, J):
    P = linear_elasticity_gradient_dF(F, mu, lam, vol)

    g = J.T @ P.reshape(-1, 1)
    return g




def linear_elasticity_hessian(**kwargs):
    if 'mu' in kwargs:
        mu = kwargs['mu']
    else :
        mu = 1

    if 'lam' in kwargs:
        lam = kwargs['lam']
    else :
        lam = 1

    if 'X'  in kwargs and 'T' in kwargs:
        X = kwargs['X']
        T = kwargs['T']

        if 'U' in kwargs:
            U = kwargs['U']
        else:
            U = kwargs['X'].copy()
        
        if 'J' in kwargs:
            J = kwargs['J']
        else:        
            J = deformation_jacobian(X, T)

        if 'vol' in kwargs:
            vol = kwargs['vol']
        else:
            vol = volume(X, T)

        return linear_elasticity_hessian_d2x(U,mu, lam, vol, J=J)
    else:
        ValueError("X and T are required")


def linear_elasticity_hessian_d2F(F, mu, lam, vol):
    
    mu = np.array(mu)
    lam = np.array(lam)
    vol = np.array(vol)
    assert(F.ndim == 3)
    assert(F.shape[1] == F.shape[2])

    dim = F.shape[1]

    I = np.identity(dim*dim)
    T = np.asarray(vectorized_transpose(1, F.shape[1]).toarray())
    Tr = np.asarray(vectorized_trace(1, F.shape[1]).toarray())


    H1 = np.repeat((I + T)[None, :, :], F.shape[0], axis=0)
    H2 = np.repeat((Tr.T @ Tr)[None, :, :], F.shape[0], axis=0)

    H = (mu.reshape(-1, 1, 1) * H1 + lam.reshape(-1, 1, 1) * H2)*vol.flatten()[:, None, None]

    return H


def linear_elasticity_hessian_d2x(U, mu,  lam, vol, J):

    dim = U.shape[1]
    F = (J @ U.reshape(-1, 1)).reshape(-1, dim, dim)
    d2psidF2 = linear_elasticity_hessian_d2F(F, mu=mu, lam=lam, vol=vol)
    H = sp.sparse.block_diag(d2psidF2)  # block diagonal hessian matrix
    Q = J.transpose() @ H @ J
    return Q
