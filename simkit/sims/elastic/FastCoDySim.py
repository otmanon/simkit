import igl
import numpy as np
import scipy as sp

from simkit import polar_svd
from simkit import project_into_subspace
from simkit import ympr_to_lame
from simkit import volume
from simkit import massmatrix
from simkit import deformation_jacobian, selection_matrix
from simkit import cluster_grouping_matrices
from simkit.solvers import BlockCoordSolver, BlockCoordSolverParams
from simkit.energies import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from simkit.energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from simkit.energies import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp

from simkit.sims.Sim import *
from simkit.sims.State import State

class FastCoDyState(State):

    def __init__(self, z, p, z_dot, p_dot):
        self.z = z
        self.p = p

        self.z_dot = z_dot
        self.p_dot = p_dot
        return

class FastCoDySimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,
                 solver_params : BlockCoordSolverParams  = BlockCoordSolverParams(), Q0=None, b0=None, cache_dir=None, read_cache=False):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------

        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = 'arap'
        self.solver_p = solver_params
        self.Q0 = Q0
        self.b0 = b0
        self.cache_dir = cache_dir
        self.read_cache = read_cache
        return


class FastCoDySim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, J: np.ndarray, B : np.ndarray, labels=None, params : FastCoDySimParams = FastCoDySimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim

        self.params = params
        
        # preprocess some quantities
        self.X = X
        self.T = T

        # i hate initializing state here. get rid of this.
        x = X.reshape(-1, 1)
        self.x = x
        self.x_dot = None
        self.x0 = None

        self.B = B
        self.J = J
        ## cubature precomp
        if labels is None:
            labels = np.arange(0, T.shape[0])

        P, _Pm = cluster_grouping_matrices(labels, X, T)

        vol = volume(X, T)
 
        ## ym, pr to lame parameters
        mu, _lam = ympr_to_lame(self.params.ym, 0)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu

        AMu = sp.sparse.diags(mu.flatten() * vol.flatten() )
        
        AMue = sp.sparse.kron(AMu, sp.sparse.identity(dim*dim))

        PAMue = sp.sparse.kron(P @ AMu , sp.sparse.identity(dim*dim))
        G = deformation_jacobian(X, T)
                                                                                        
        self.AMuPGB = (  PAMue @ G) @ B
        self.AMuPGJ = (  PAMue @ G) @ J
        self.B = B
                                                                                                                                                                                                                                                                                                                                                                 
        L = G.T @ AMue @ G

        Mv = sp.sparse.kron(massmatrix(X, T, rho=params.rho), sp.sparse.identity(dim))
        self.BLB = B.T @ L @ B 
        self.BMB = B.T @ Mv @ B
        
        self.JMJ = J.T @ Mv @ J
        self.JMy = J.T @ Mv @ X.reshape(-1, 1)

        self.BMJ = B.T @ Mv @ J
        self.BLJ = B.T @ L @ J
        
    
        self.kin_pre = KineticEnergyZPrecomp(B, Mv)
        
        if params.Q0 is None:
            self.Q = np.zeros((B.shape[1], B.shape[1])) #sp.sparse.csc_matrix((B.shape[1], B.shape[1]))
        else:
            self.Q = self.params.Q0
        if params.b0 is None:
            self.b = np.zeros((B.shape[1], 1))
        else:
            assert(params.b0.shape[0] == B.shape[-1])
            self.b = self.params.b0.reshape(-1, 1)

        self.recompute_system()

        # should also build the solver parameters
        assert(isinstance(params.solver_p, BlockCoordSolverParams))
        self.solver = BlockCoordSolver(self.global_step, self.local_step, params.solver_p)
        
        return

    def recompute_system(self):
        self.H = self.BLB + self.BMB / self.params.h**2 + self.Q
        self.is_sparse = False
        if (sp.sparse.issparse(self.B)):
            self.is_sparse = True
            self.chol_H = sp.sparse.linalg.factorized(self.H)
        else:
            self.chol_H = sp.linalg.cho_factor(self.H)

        return
    
    def dynamic_precomp(self, z : np.ndarray, p : np.ndarray, z_dot : np.ndarray, p_dot : np.ndarray, p_next : np.ndarray, b_ext=None):
        """
        Computation done once every timestep and never again
        """

        self.p = p.copy()
        self.p_next = p_next.copy()

        h = self.params.h
        self.y = z + h * z_dot 
        self.q =   p_next - (p + h * p_dot)
        # same for b
        if b_ext is not None:
            if self.params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.params.b0
        
        self.dynamic_precomp_done = True

    def energy(self, z : np.ndarray):
        k = kinetic_energy_z(z, self.y ,self.params.h, self.kin_pre)

        r = self.local_step(z)
        
        v = 0.5 *  z.T @ self.BLB @ z + z.T @ self.AMuPGB.T @ r
        
        quad =  quadratic_energy(z, self.Q, self.b)
        
        total = k + v + quad
        return total

    # def gradient(self, z : np.ndarray):
    #     k = kinetic_gradient_z(z, self.y, self.params.h, self.kin_pre)
    #     v = elastic_gradient_dz(z, self.mu, self.lam, self.vol, self.params.material, self.el_pre)
    #     quad = quadratic_gradient(z, self.Q, self.b)
    #     total = v  + k + quad
    #     return total

    # def hessian(self, z : np.ndarray):
    #     v = elastic_hessian_d2z(z, self.mu, self.lam, self.vol, self.params.material, self.el_pre)
    #     k = kinetic_hessian_z(self.params.h, self.kin_pre)
    #     quad = quadratic_hessian(self.Q)
    #     total = v + k + quad
    #     return total


    def local_step(self, z : np.ndarray):
        c = self.AMuPGB @ z + self.AMuPGJ @ self.p
        C = c.reshape(-1, self.dim, self.dim)
        R = polar_svd(C)[0]
        r = R.reshape(-1, 1)
        return r
    
    def global_step(self, z : np.ndarray, r: np.ndarray):
        h = self.params.h
        k = self.BMB @ self.y / h**2 - self.BMJ @ self.q / h**2
        
        e =  self.AMuPGB.T @  r - self.BLJ @ self.p_next

        q = - self.b 

        rhs = k + e + q

        if self.is_sparse:
            z_next = self.chol_H(rhs)
        else:
            z_next = sp.linalg.cho_solve(self.chol_H, rhs)

        return z_next

    def step(self, z : np.ndarray,  p : np.ndarray,  z_dot : np.ndarray, p_dot : np.ndarray, p_next : np.ndarray, b_ext : np.ndarray =None):
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, p,  z_dot, p_dot, p_next, b_ext)
        
        z0 = z.copy() # very important to copy this here so that x does not get over-written
        z_next = self.solver.solve(z0)

        self.dynamic_precomp_done = False
        return z_next
    
    def rest_state(self):
        z = np.zeros((self.B.shape[1], 1))
        z_dot = np.zeros_like(z)

        p = project_into_subspace( self.X.reshape(-1, 1), self.J,
                        M=sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.identity(self.dim)), BMB=self.JMJ, BMy=self.JMy)# 

        p_dot = np.zeros_like(p)

        p_next = p.copy()
        return z, p, z_dot, p_dot, p_next




