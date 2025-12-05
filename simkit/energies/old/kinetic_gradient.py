import numpy as np

def kinetic_gradient(x : np.ndarray, y : np.ndarray, M, h : float):
    d = x - y
    g = M @ d * (1/ (h**2))
    return g


from .kinetic_energy import KineticEnergyZPrecomp

def kinetic_gradient_z(z : np.ndarray, y : np.ndarray, h, precomp):
    d = z - y
    g = precomp.BMB @ d * (1/ (h**2))
    return g