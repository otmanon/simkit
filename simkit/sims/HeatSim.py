



import numpy as np
import scipy as sp
import os 
import igl
import gpytoolbox as gpy

from simkit.uniform_line import uniform_line
from simkit.gaussian_rbf import gaussian_rbf
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.random_impulse_vibes import random_impulse_vibes
from simkit.subspace_corrolation import subspace_corrolation
from simkit.eigs import eigs
from simkit.selection_matrix import selection_matrix
from simkit.dirichlet_penalty import dirichlet_penalty

from simkit.quadratic_energy import quadratic_energy
from simkit.quadratic_gradient import quadratic_gradient
from simkit.quadratic_hessian import quadratic_hessian

from simkit.project_into_subspace import project_into_subspace
    
class HeatSimParams():

    def __init__(self, sigma, dt, rho=1, Q0=None, b0=None):

        self.sigma = sigma
        self.dt = dt

        self.Q0 = Q0
        self.b0 = b0
        self.rho = rho
        
class HeatSim():
    
        def __init__(self, X, T, p : HeatSimParams, B=None):
            self.X = X
            self.T = T
            n = X.shape[0]

            self.p = p
            if B is None:
                B = sp.sparse.identity(n)

            self.B = B 
            
            self.L = B.T @  dirichlet_laplacian(X, T, mu=p.sigma) @ B
            self.M = B.T @ massmatrix(X,  T, p.rho) @ B

            if p.Q0 is None:
                self.Q = sp.sparse.csc_matrix((B.shape[1], B.shape[1]))
            else:
                self.Q = B.T @ self.p.Q0 @ B

            if p.b0 is None:
                self.b = np.zeros((B.shape[1], 1))
            else:
                self.b = B.T @ self.p.b0

            self.sim_params = p

        def energy(self, u):
            # smoothness potential energy
            smooth = 0.5 * u.T @ self.L @ u

            # kinetic energy
            d = u - self.y
            kinetic = 0.5 * d.T @ self.M @ d / self.p.dt

            # constraint energy
            quadratic = quadratic_energy(u, self.Q, self.b)

            energy = smooth + kinetic + quadratic    
            return energy
        
        
        def gradient(self, u):
            # smoothness potential energy gradient
            smooth = self.L @ u

            # kinetic energy gradient
            d = u - self.y
            kinetic = self.M @ d / self.p.dt

            # constraint energy gradient
            quadratic = quadratic_gradient(u, self.Q, self.b)

            gradient = smooth + kinetic + quadratic
            return gradient
        
        def hessian(self, u):
            # smoothness potential energy hessian
            smooth = self.L 
            
            # kinetic energy hessian
            kinetic = self.M / self.p.dt

            # constraint energy hessian
            quadratic = quadratic_hessian(self.Q)


            hessian = smooth + kinetic + quadratic

            return hessian 
        
        def step(self, u, Q_ext=None, b_ext=None):

            self.y = u.copy()
            # add to current Q_ext
            if Q_ext is not None:
                if self.p.Q0 is None:
                    self.Q = Q_ext
                else:
                    self.Q = Q_ext + self.p.Q0
                self.Q = self.B.T @ self.Q @ self.B

            # same for b_ext
            if b_ext is not None:
                if self.p.b0 is None:
                    self.b = b_ext
                else:
                    self.b = b_ext + self.p.b0
                
                self.b = self.B.T @ self.b
        
            H = self.hessian(u)
            g = self.gradient(u)


            if sp.sparse.issparse(H):
                d = sp.sparse.linalg.spsolve(H, -g)
            else:
                d = np.linalg.solve(H, -g)
            if d.ndim == 1:
                d = d[:, None]

            if u.ndim == 1:
                u = u[:, None]
            u += d
    
            return u
        
        def zero_state(self):
            u = project_into_subspace(np.zeros(self.X.shape[0]), self.B, M=massmatrix(self.X, self.T))
            return u


