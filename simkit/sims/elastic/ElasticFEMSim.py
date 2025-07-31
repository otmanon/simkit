import igl
import numpy as np
import scipy as sp

import simkit
from simkit.solvers import NewtonSolver, NewtonSolverParams
from simkit.energies import kinetic_energy, kinetic_gradient, kinetic_hessian
from simkit.energies import elastic_energy_x, elastic_gradient_dx, elastic_hessian_d2x
from simkit.energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from simkit.sims.Sim import *
from simkit.sims.State import State

class ElasticFEMState(State):

    def __init__(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot
        return


class ElasticFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0, g=0,  material='arap',
                 solver_p : NewtonSolverParams  = NewtonSolverParams(), f_ext = None, Q=None, b=None):
        """
        Parameters of the pinned pendulum simulation


        argmin_x  K(x) + V(x)  = L(x)



        K(x) = 1/2 h^2 (x - y)^T M (x - y)
        y = x_curr + h x_dot_curr

        V(x) = 1/ 2 \sum_t  || F_t x_t - R ||^2



        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = material
        self.solver_p = solver_p

        self.f_ext = f_ext
        self.Q = Q
        self.b = b
        return


class ElasticFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, p : ElasticFEMSimParams = ElasticFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        dim = X.shape[1]
        self.dim = dim
        self.p = p
        self.mu, self.lam = simkit.ympr_to_lame(self.p.ym, self.p.pr)

        # preprocess some quantities
        self.X = X
        self.T = T
        x = X.reshape(-1, 1)
        self.x = x
        self.x_dot = None
        self.x0 = None

        M = simkit.massmatrix(self.X, self.T, rho=self.p.rho)
        self.M = M
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        self.Mv = Mv

        # elastic energy, gradient and hessian
        self.J = simkit.deformation_jacobian(self.X, self.T)
        self.vol = simkit.volume(self.X, self.T)
        
   

        if p.Q is None:
            self.Q = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
        else:
            assert(p.Q.shape[0] == x.shape[0] and p.Q.shape[1] == x.shape[0])
            self.Q = self.p.Q
        if p.b is None:
            self.b = np.zeros((x.shape[0], 1))
        else:
            assert(p.b.shape[0] == x.shape[0])
            self.b = self.p.b.reshape(-1, 1)

        if p.f_ext is None:
            self.f_ext = np.zeros((x.shape[0], 1))
        else:
            assert(p.f_ext.shape[0] == x.shape[0])
            self.f_ext = self.p.f_ext.reshape(-1, 1)


        # should also build the solver parameters
        self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)


    def energy(self, x : np.ndarray):
        """
        Computes the energy of the elastic system

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        y : (n*d, 1) numpy array
            inertial target positions (2 x_curr -  x_prev)
        Returns
        -------
        total : float
            Total energy of the elastic system
        """
        
        dim = self.dim
        k = kinetic_energy(x, self.y, self.Mv, self.p.h)
        v = elastic_energy_x(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)
        quad = quadratic_energy(x, self.Q, self.b)
        total = k + v + quad
        return total


    def gradient(self, x : np.ndarray):
        """
        Computes the gradient of the energy of the elastic system

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        Returns
        -------
        g : (n*d, 1) numpy array
            Gradient of the energy of the  elastic system
        """

        dim = self.dim
        k =  kinetic_gradient(x, self.y, self.Mv, self.p.h)
        v = elastic_gradient_dx(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)
        quad = quadratic_gradient(x, self.Q, self.b)
        total = k + v + quad
        return total


    def hessian(self, x : np.ndarray):
        """
        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        Returns
        -------
        g : (n*d, n*d)  sparse matrix
            Hessian of the energy of the elastic system
        """

        dim = self.dim
        v = elastic_hessian_d2x(x.reshape(-1, dim), self.J, self.mu, self.lam, self.vol, self.p.material)
        k = kinetic_hessian(self.Mv, self.p.h)
        quad = quadratic_hessian(self.Q)
        total = v + k + quad
        return total


    def step(self, x : np.ndarray, x_dot : np.ndarray, Q_ext = None, b_ext = None):
        """
        Steps the simulation forward in time.

        Parameters
        ----------
        x : (n*d, 1) numpy array
            positions of the elastic system

        x_dot : (n*d, 1) numpy array
            velocity of the elastic system

        Returns
        -------
        x : (n*d, 1) numpy array
            Next positions of the pinned pendulum system
        """

        #info changes every timestep
        
        self.Q = Q_ext
        self.b = b_ext
        self.y = x + self.p.h * x_dot
        x0 = x.copy() # very important to copy this here so that x does not get over-written
        
        
        x_next = self.solver.solve(x0)
        return x_next





