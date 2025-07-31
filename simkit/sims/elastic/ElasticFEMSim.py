import igl
import numpy as np
import scipy as sp

from ...solvers import NewtonSolver, NewtonSolverParams
from ...energies import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from ...energies import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from ...energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from ...sims.Sim import *
from ...solvers import NewtonSolver, NewtonSolverParams
from ... import ympr_to_lame
from ... import project_into_subspace
from ... import volume
from ... import massmatrix
from ... import deformation_jacobian, quadratic_hessian, selection_matrix



from ..Sim import  *
from ..State import State

class ElasticROMFEMState(State):

    def __init__(self, z, z_dot):
        self.z = z
        self.z_dot = z_dot
        return


class ElasticFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,  material='fcr',
                 solver_p : NewtonSolverParams  = NewtonSolverParams(), Q0=None, b0=None):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = material
        self.solver_p = solver_p

        self.Q0 = Q0
        self.b0 = b0
        return


class ElasticFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix =None, cI : np.ndarray=None,  cW : np.ndarray=None, p : ElasticFEMSimParams = ElasticFEMSimParams(), x0=None):
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

        if B is None:
            B = sp.sparse.identity(X.shape[0] * X.shape[1])
        self.B = B

        # offset. final state is X = B @ z + x0
        if x0 is None:
            x0 = np.zeros((B.shape[0], 1))
        self.x0 = x0
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI
        
        [self.kin_pre, self.el_pre, self.mu, self.lam, self.vol, self.mu0, self.lam0, self.vol0, self.J, self.Mv] =  \
            self.initial_precomp(X, T, B, x0, cI, cW, p.rho, p.ym, p.pr, dim)


        if p.Q0 is None:
            self.Q0 = sp.sparse.csc_matrix((B.shape[0], B.shape[0]))
        else:
            self.Q0 = self.p.Q0

        self.BQ0B = B.T @ self.Q0 @ B

        if p.b0 is None:
            self.b0 = np.zeros((B.shape[0], 1))
        else:
            self.b0 = self.p.b0.reshape(-1, 1)
        self.Bb0 = B.T @ self.b0


        # should also build the solver parameters
        self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)

    def initial_precomp(self, X, T, B, x0, cI, cW, rho, ym, pr, dim):
        
        # kinetic energy precomp
        M = massmatrix(self.X, self.T, rho=rho)
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
        # elastic energy precomp
        vol0 = volume(X, T).reshape(-1, 1)
        if cW is not None:
            vol = cW.reshape(-1, 1)
        else:
            vol = vol0.copy()  
            
        ## ym, pr to lame parameters
        mu, lam = ympr_to_lame(ym, pr)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu
        if isinstance(lam, float):
            lam = np.ones((T.shape[0], 1)) * lam
        mu0= mu.copy()
        lam0 = lam.copy()
        mu = mu[cI]
        lam = lam[cI]

        ## selection matrix from cubature precomp.
        G = selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        J = deformation_jacobian(self.X, self.T)
        elastic_z_precomp = ElasticEnergyZPrecomp(B, x0, Ge, J, self.dim)


        self.dynamic_precomp_done = False
        return  kin_z_precomp, elastic_z_precomp, mu, lam, vol, mu0, lam0, vol0, J, Mv

    def dynamic_precomp(self, z, z_dot, BQB_ext=None, Bb_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.p.h * z_dot

        # add to current Q_ext
        if BQB_ext is not None:
            self.Q = BQB_ext + self.BQ0B
        else:
            self.Q = self.BQ0B

        # same for b
        if Bb_ext is not None:
            self.b = Bb_ext + self.Bb0
        else:
            self.b = self.Bb0
        
        self.dynamic_precomp_done = True

    def energy(self, z : np.ndarray):
        k = kinetic_energy_z(z, self.y ,self.p.h, self.kin_pre)
        v = elastic_energy_z(z,  self.mu, self.lam, self.vol, self.p.material,  self.el_pre)
        quad =  quadratic_energy(z, self.Q, self.b)
        total = k + v + quad
        return total

    def gradient(self, z : np.ndarray):
        k = kinetic_gradient_z(z, self.y, self.p.h, self.kin_pre)
        v = elastic_gradient_dz(z, self.mu, self.lam, self.vol, self.p.material, self.el_pre)
        quad = quadratic_gradient(z, self.Q, self.b)
        total = v  + k + quad 
        return total

    def hessian(self, z : np.ndarray):
        v = elastic_hessian_d2z(z, self.mu, self.lam, self.vol, self.p.material, self.el_pre)
        k = kinetic_hessian_z(self.p.h, self.kin_pre)
        quad = quadratic_hessian(self.Q)
        total = v + k + quad
        return total

    
    def step(self, z : np.ndarray, z_dot : np.ndarray, Q_ext=None, b_ext=None):
    
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)


        z0 = z.copy() # very important to copy this here so that x does not get over-written
        z_next = self.solver.solve(z0)
        # z_next = project_into_subspace(x_next, self.B)

        self.dynamic_precomp_done = False
        return z_next


    def rest_state(self):
        z = project_into_subspace( self.X.reshape(-1, 1), self.B, 
                        M=sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.identity(self.dim)), x0=self.x0)# 

        z_dot = np.zeros_like(z)
        return z, z_dot



