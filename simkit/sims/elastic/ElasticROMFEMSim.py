import igl
import numpy as np
import scipy as sp

from simkit.solvers import NewtonSolver, NewtonSolverParams
from simkit import ympr_to_lame
from simkit import volume
from simkit import massmatrix
from simkit import deformation_jacobian, selection_matrix
from simkit.energies import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from simkit.energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from simkit.energies import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp

from simkit.sims.Sim import *
from simkit.sims.State import State

class ElasticROMFEMState(State):

    def __init__(self, z, z_dot):
        self.z = z
        self.z_dot = z_dot
        return


class ElasticROMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,  material='arap',
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


class ElasticROMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, cI=None,  cW=None, p : ElasticROMFEMSimParams = ElasticROMFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim

        self.p = p

        
        # preprocess some quantities
        self.X = X
        self.T = T

        # i hate initializing state here. get rid of this.
        x = X.reshape(-1, 1)
        self.x = x
        self.x_dot = None
        self.x0 = None

        self.B = B
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI
        [self.kin_pre, self.el_pre, self.mu, self.lam, self.vol] =  \
            self.initial_precomp(X, T, B, cI, cW, p.rho, p.ym, p.pr, dim)


        if p.Q0 is None:
            self.Q = None
        else:
            self.Q = self.p.Q0
        if p.b0 is None:
            self.b = None 
        else:
            assert(p.b0.shape[0] == B.shape[-1])
            self.b = self.p.b0.reshape(-1, 1)


        # should also build the solver parameters
        if isinstance(p.solver_p, NewtonSolverParams):
            self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)
        else:
            # print error message and terminate programme
            assert(False, "Error: solver_p of type " + str(type(p.solver_p)) +
                          " is not a valid instance of NewtonSolverParams. Exiting.")
        return

    def initial_precomp(self, X, T, B,  cI, cW, rho, ym, pr, dim):
        
        # kinetic energy precomp
        M = massmatrix(self.X, self.T, rho=rho)
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
        
        # elastic energy precomp
        if cW is None:
            vol = volume(X, T)
        else:
            vol = cW.reshape(-1, 1)
            
        ## ym, pr to lame parameters
        mu, lam = ympr_to_lame(ym, pr)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu
        if isinstance(lam, float):
            lam = np.ones((T.shape[0], 1)) * lam

        mu = mu[cI]
        lam = lam[cI]

        ## selection matrix from cubature precomp.
        G = selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        J = deformation_jacobian(self.X, self.T)
        elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)


        self.dynamic_precomp_done = False
        return  kin_z_precomp, elastic_z_precomp, mu, lam, vol

    def dynamic_precomp(self, z, z_dot, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.p.h * z_dot

        # add to current Q_ext
        if Q_ext is not None:
            if self.p.Q0 is None:
                self.Q = Q_ext
            else:
                self.Q = Q_ext + self.p.Q0

        # same for b
        if b_ext is not None:
            if self.p.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.p.b0
        
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





