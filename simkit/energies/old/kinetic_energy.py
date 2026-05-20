import numpy as np



def kinetic_energy(x : np.ndarray, y : np.ndarray, M, h : float):
    """
    Computes the kinetic energy of the elastic system

    Parameters
    ----------
    x : (n*d, 1) numpy array
        positions of the elastic system

    y : (n*d, 1) numpy array
        inertial target positions (2 x_curr -  x_prev or equivalently x_curr + h * x_dot_curr)

    Returns
    -------
    k : float
        Kinetic energy of the elastic system
    """
    d = x - y
    g = 0.5 * d.T @ M @ d * (1/ (h**2))
    return g



class KineticEnergyZPrecomp():
    def __init__(self, B,  M):
        self.BMB = B.T @ M @ B
    
        
def kinetic_energy_z(z : np.ndarray, y : np.ndarray, h, precomp : KineticEnergyZPrecomp):
    """
    Computes the kinetic energy of the reduced elastic system where the reduced dofs z are the reduced positions x = Bz

    Parameters
    ----------
    z : (r, 1) numpy array
        reduced positions of the elastic system

    y : (n*d, 1) numpy array
        inertial target positions (2 x_curr -  x_prev or equivalently x_curr + h * x_dot_curr)

    Returns
    -------
    k : float
        Kinetic energy of the elastic system
    """
    
    d = z - y
    E = 0.5 * d.T @ precomp.BMB @ d * (1/ (h**2))

    return E