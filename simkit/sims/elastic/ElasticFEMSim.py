
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
from ... import deformation_jacobian, selection_matrix
from ..Sim import  *



class ElasticFEMSimParams():
    def __init__(self, rho : float | np.ndarray = 1, 
                 h : float = 1e-2, 
                 ym : float | np.ndarray = 1,
                 pr : float | np.ndarray = 0, 
                 material : str =  'fcr',
                 solver_p : NewtonSolverParams  = NewtonSolverParams(),
                 Q0 : sp.sparse.csc_matrix | None = None,
                 b0 : np.ndarray | None = None):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------
        rho : float | (num_vertices, 1) np.ndarray
            Density of the material 
        h : float
            Time step
        ym : float | (num_simplex, 1) np.ndarray
            Young's modulus
        pr : float | ( num_simplex, 1) np.ndarray
            Poisson's ratio of the material 
        material : str
            Material type ('fcr' or 'arap' or 'neo-hookean' or 'linear-elasticity')
        solver_p : NewtonSolverParams
            Parameters of the Newton solver
        Q0 : (num_vertices*dim, num_vertices*dim) sp.sparse.csc_matrix
            Initial quadratic penalty quadratic term for an energy of 
            the form 0.5 * u.T @ Q0 @ u + b0.T @ u
            If None, then assumed to be zeros.
        b0 : (num_vertices*dim, 1) np.ndarray
            Initial quadratic penalty linear term for an energy of 
            the form 0.5 * u.T @ Q0 @ u + b0.T @ u
            If None, then assumed to be zeros.
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

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray | sp.sparse.csc_matrix =None, cI : np.ndarray=None,  cW : np.ndarray=None, sim_params : ElasticFEMSimParams = ElasticFEMSimParams(), x0=None, q=None):
        """

        Parameters
        ----------
        X : (num_vertices, (2|3)) np.ndarray
            Vertex rest positions
        T : (num_simplex, (3|4)) np.ndarray
        B : (num_vertices*dim, subspace_dim) np.ndarray | sp.sparse.csc_matrix | None
            Reduced space basis used to accelerate simulation. If provided, the simulation
            will solve for subspace quantities.
            If None, then assumed to be the identity matrix and 
            the simulation will solve for full space (per-vertex) quantities.
        cI : (num_cubature_points, 1) np.ndarray | None
            Indices of the cubature points. If None, then assumed to be all indices.
        cW : (num_cubature_points, 1) np.ndarray | None
            Weights of the cubature points. If None, then assumed to be per-tet volumes.
        p : ElasticFEMSimParams
            Parameters of the elastic system
        q : (num_vertices*dim, 1) np.ndarray | None
            Constant offset term applied to the subspace. 
            Final per-vertex positions can be obtained via x = B @ z + q.
            If q is None, then assumed to be all zeros.
        """

        dim = X.shape[1]
        self.dim = dim
        self.sim_params = sim_params
        self.mu, self.lam = ympr_to_lame(self.sim_params.ym, self.sim_params.pr)

        # preprocess some quantities
        self.X = X
        self.T = T

        if B is None:
            B = sp.sparse.identity(X.shape[0] * X.shape[1])
        self.B = B

        if q is None:
            q = np.zeros((B.shape[0], 1))
        self.q = q
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI
        
        [self.kin_pre, self.el_pre, self.mu, self.lam, self.vol, 
         self.mu0, self.lam0, self.vol0, self.J, self.Mv] =  \
            self.initial_precomp(X, T, B, q, cI, cW, sim_params.rho, sim_params.ym, sim_params.pr, dim)

        if sim_params.Q0 is None:
            self.Q0 = sp.sparse.csc_matrix((B.shape[0], B.shape[0]))
        else:
            self.Q0 = self.sim_params.Q0

        self.BQ0B = B.T @ self.Q0 @ B

        
        if sim_params.b0 is None:
            self.b0 = np.zeros((B.shape[0], 1))
        else:
            self.b0 = self.sim_params.b0.reshape(-1, 1)
        self.Bb0 = B.T @ self.b0


        # should also build the solver parameters
        self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, sim_params.solver_p)

    def initial_precomp(self, X, T, B, q, cI, cW, rho, ym, pr, dim):
        
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
        elastic_z_precomp = ElasticEnergyZPrecomp(B, q, Ge, J, self.dim)


        self.dynamic_precomp_done = False
        return  kin_z_precomp, elastic_z_precomp, mu, lam, vol, mu0, lam0, vol0, J, Mv

    def dynamic_precomp(self, z, z_dot, BQB_ext=None, Bb_ext=None):
        """
        Computation done once every timestep and never again. 
        Used for dynamics which involve external forces and hessians that 
        may change every timestep, but not every newton iteration
        """
        self.y = z + self.sim_params.h * z_dot

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

    
    def step(self, z : np.ndarray, z_dot : np.ndarray, Q_ext=None, b_ext=None, return_info=False):
        """
        Steps the simulation state forward in time.

        Parameters
        ----------
        z : (num_vertices*dim | subspace_dim, 1) np.ndarray
            Subspace state
        z_dot : (num_vertices*dim | subspace_dim, 1) np.ndarray
            Subspace velocity
        Q_ext : (num_vertices*dim, num_vertices*dim) sp.sparse.csc_matrix |  (subspace_dim, subspace_dim) np.ndarray | None
            External quadratic penalty term. If None, then assumed to be zero.
            If B was provided upon sim construction, then this Q_ext should be subspace_dim x subspace_dim.
            Otherwise, it should be num_vertices*dim x num_vertices*dim.
        b_ext : (num_vertices*dim, 1) np.ndarray | (subspace_dim, 1) np.ndarray | None
            External linear penalty term. If None, then assumed to be zero.
            If B was provided upon sim construction, then this b_ext should be subspace_dim x 1.
            Otherwise, it should be num_vertices*dim x 1.
            
        Returns
        -------
        z_next : (num_vertices*dim | subspace_dim, 1) np.ndarray
            Subspace state after one simulation time step
        """
        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)
        z0 = z.copy() 
        
        if return_info:
            z_next, info = self.solver.solve(z0, return_info=return_info)
            self.dynamic_precomp_done = False
            return z_next, info
        else:
            z_next = self.solver.solve(z0)
            self.dynamic_precomp_done = False
            return z_next

    def rest_state(self):
        """
        Returns the rest state of the system.
        
        Returns
        -------
        z : (num_vertices*dim | subspace_dim, 1) np.ndarray
            Subspace state at rest
        z_dot : (num_vertices*dim | subspace_dim, 1) np.ndarray
            Subspace velocity at rest (zeros)
        """
        M = sp.sparse.kron(massmatrix(self.X, self.T), sp.sparse.identity(self.dim))
        z = project_into_subspace(  self.X.reshape(-1, 1) - self.q,
                                    self.B, 
                                    M=M)

        z_dot = np.zeros_like(z)
        return z, z_dot



