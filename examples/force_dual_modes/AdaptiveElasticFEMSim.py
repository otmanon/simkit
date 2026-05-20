import igl
import numpy as np
import scipy as sp

from simkit.project_into_subspace import project_into_subspace
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSim, ElasticFEMSimParams

from simkit.solvers import NewtonSolver, NewtonSolverParams
from simkit import ympr_to_lame
from simkit.energies import elastic_energy_z, elastic_gradient_dz, elastic_hessian_d2z, ElasticEnergyZPrecomp
from simkit.energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from simkit.energies import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from simkit import volume
from simkit import massmatrix
from simkit import deformation_jacobian, selection_matrix




class AdaptiveElasticFEMSim(ElasticFEMSim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix =None, cI : np.ndarray=None,  cW : np.ndarray=None, sim_params : ElasticFEMSimParams = ElasticFEMSimParams(), q=None):
        super().__init__(X, T, B, cI, cW, sim_params, q)

    def update_subspace(self, B, cI, cW):

        self.mu = self.mu0[cI].reshape(-1, 1)
        self.lam = self.lam0[cI].reshape(-1, 1)

        self.vol = cW.reshape(-1, 1)
        self.cI = cI
        self.cW = cW
        self.B = B
        ## selection matrix from cubature precomp.
        G = selection_matrix(cI, self.T.shape[0])
        dim = self.X.shape[1]
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        J = deformation_jacobian(self.X, self.T)
        self.el_pre = ElasticEnergyZPrecomp(B, self.q, Ge, J, self.dim)

        M = massmatrix(self.X, self.T, rho=self.sim_params.rho)
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        self.kin_pre= KineticEnergyZPrecomp(B, Mv)
        

        self.BQ0B = B.T @ self.Q0 @ B
        self.Bb0 = B.T @ self.b0
        return
        

    # def initial_precomp(self, X, T, B, x0, cI, cW, rho, ym, pr, dim):
        
    #     # kinetic energy precomp
    #     M = massmatrix(self.X, self.T, rho=rho)
    #     Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
    #     kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
    #     # elastic energy precomp
    #     if cW is None:
    #         vol = volume(X, T)
    #     else:
    #         vol = cW.reshape(-1, 1)
            
    #     ## ym, pr to lame parameters
    #     mu0, lam0 = ympr_to_lame(ym, pr)
        
    #     if isinstance(mu0, float):
    #         mu0 = np.ones((T.shape[0], 1)) * mu0
    #     if isinstance(lam0, float):
    #         lam0 = np.ones((T.shape[0], 1)) * lam0
    #     mu0 = mu0.reshape(-1, 1)
    #     lam0 = lam0.reshape(-1, 1)   
    #     mu = mu0[cI]
    #     lam = lam0[cI]

    #     ## selection matrix from cubature precomp.
    #     G = selection_matrix(cI, T.shape[0])
    #     Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
    #     J = deformation_jacobian(self.X, self.T)
    #     elastic_z_precomp = ElasticEnergyZPrecomp(B, x0, Ge, J, self.dim)
    #     # GJ = Ge @ J

    #     self.dynamic_precomp_done = False
    #     return  kin_z_precomp, elastic_z_precomp, mu, lam, mu0, lam0, vol, J, Mv

    def dynamic_precomp(self, z, z_dot, BQB_ext=None, Bb_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.sim_params.h * z_dot

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
        k = kinetic_energy_z(z, self.y ,self.sim_params.h, self.kin_pre)
        v = elastic_energy_z(z,  self.mu, self.lam, self.vol, self.sim_params.material,  self.el_pre)
        quad =  quadratic_energy(z, self.Q, self.b)
        total = k + v + quad
        return total

    def gradient(self, z : np.ndarray):
        k = kinetic_gradient_z(z, self.y, self.sim_params.h, self.kin_pre)
        v = elastic_gradient_dz(z, self.mu, self.lam, self.vol, self.sim_params.material, self.el_pre)
        quad = quadratic_gradient(z, self.Q, self.b)
        total = v  + k + quad 
        return total

    def hessian(self, z : np.ndarray):
        v = elastic_hessian_d2z(z, self.mu, self.lam, self.vol, self.sim_params.material, self.el_pre)
        k = kinetic_hessian_z(self.sim_params.h, self.kin_pre)
        quad = quadratic_hessian(self.Q)
        total = v + k + quad
        return total

    
    def step(self, z : np.ndarray, z_dot : np.ndarray, BQB_ext=None, Bb_ext=None):
    
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, BQB_ext, Bb_ext)
        z0 = z.copy() # very important to copy this here so that x does not get over-written
        # z_next = project_into_subspace(x_next, self.B)
        z_next = self.solver.solve(z0)
        self.dynamic_precomp_done = False
        return z_next


    def rest_state(self):
        z = project_into_subspace( self.X.reshape(-1, 1) - self.q, self.B, 
                        M=sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.identity(self.dim)))# 

        z_dot = np.zeros_like(z)
        return z, z_dot



