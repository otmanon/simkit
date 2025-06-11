import numpy as np

def kinetic_hessian(M, h : float):
   
    H = M *(1/ (h**2))
    return H


from .kinetic_energy import KineticEnergyZPrecomp

def kinetic_hessian_z(h, precomp : KineticEnergyZPrecomp):
    H = precomp.BMB * (1/ (h**2))
    return H
