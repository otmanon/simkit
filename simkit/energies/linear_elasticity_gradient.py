import numpy as np



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