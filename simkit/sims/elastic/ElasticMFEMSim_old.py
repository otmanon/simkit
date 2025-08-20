import igl
import numpy as np
import scipy as sp

from ... import symmetric_stretch_map
from ... import ympr_to_lame
from ... import stretch
from ...stretch_gradient import stretch_gradient_dx
from ... import volume
from ... import massmatrix
from ... import deformation_jacobian
from ... import backtracking_line_search
from ... import selection_matrix

from ..Sim import Sim
from ...solvers import NewtonSolver, NewtonSolverParams
from ...energies import elastic_energy_S, elastic_gradient_dS, elastic_hessian_d2S
from ...energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from ...energies import kinetic_energy, kinetic_gradient, kinetic_hessian
from ...stretch_gradient import stretch_gradient_dz
from ...energies import kinetic_hessian_z, elastic_hessian_d2S
from ...energies import kinetic_energy_z
from ...energies import kinetic_gradient_z, elastic_gradient_dS
from ...energies import KineticEnergyZPrecomp




class SQPMFEMSolverParams():

    def __init__(self, max_iter=100, tol=1e-6, do_line_search=True, verbose=False):
        self.do_line_search = do_line_search
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        return
    
class SQPMFEMSolver():

    def __init__(self, energy_func, grad_blocks_func, hess_blocks_func, params : SQPMFEMSolverParams = None):
        """
         SQP Solver for the MFEM System from https://www.dgp.toronto.edu/projects/subspace-mfem/ ,  Section 4.

         The full system looks like:
            [Hu    0    Gu] [du]   - [fu]
            [0     Hz   Gz] [dz] = - [fz]
            [Gz.T  0     0] [mu]   - [fmu]

        Where Gz is diagonal and easily invertible. Using this fact, we can rewrite the system into a small solve and a matrix mult

        (Hu + Gu Gz^-1 Hz Gz^-1 Gu.T) du = -fu + Gu Gz^-1 fz - Gu Gz^-1 Hz Gz^-1 fmu
        dz = -Gz^-1 (fz + Hz du)

        Parameters
        ----------
        energy_func : function
            Energy function to minimize
        grad_blocks_func : function
            Function that returns the blocks of the gradient: fu, fz, fmu
        hess_blocks_func : function
            Function that returns the important blocks of the hessian: Hu, Hz, Gu, Gzi
        params : SQPMFEMSolverParams
            Parameters for the solver

        """

        if params is None:
            params = SQPMFEMSolverParams()
        self.params = params
        self.energy_func = energy_func

        self.hess_blocks_func = hess_blocks_func
        self.grad_blocks_func = grad_blocks_func
        return


    def solve(self, p0):
        
        p = p0.copy()
        for i in range(self.params.max_iter):

            [H_u, H_z, G_u, G_zi] = self.hess_blocks_func(p)            
            
            [f_u, f_z, f_mu] = self.grad_blocks_func(p)

            #form K
            K = G_u @ G_zi @ H_z @ G_zi @ G_u.T 
            Q = H_u + K

            # form g_u
            g_u = -f_u + G_u @ G_zi @ (f_z - H_z @ G_zi @ f_mu)

            if isinstance(Q, sp.sparse.spmatrix):
                du =  sp.sparse.linalg.spsolve(Q, g_u).reshape(-1, 1)
                du = np.array(du)
            else:
                du = np.linalg.solve(Q, g_u)
            if du.ndim == 1:
                du = du.reshape(-1, 1)
        
            g_z = - (f_mu + G_u.T @ du)
            dz = G_zi @ g_z

            g = np.vstack([f_u, f_z])
            dp = np.vstack([du, dz])
            if self.params.do_line_search:
                energy_func = lambda z: self.energy_func(z)
                alpha, lx, ex = backtracking_line_search(energy_func, p, g, dp)
            else:
                alpha = 1.0

            p += alpha * dp
            if np.linalg.norm(g) < 1e-4:
                break

        return p


class ElasticMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0, g=0,  material='arap',
                 solver_p : SQPMFEMSolverParams  = SQPMFEMSolverParams(), f_ext = None, Q0=None, b0=None, gamma=1):
        """
        Parameters of the pinned pendulum simulation

        Parameters
        ----------
        rho : float | (num_vertices, 1) np.ndarray
            Density of the material
        h : float | (num_simplex, 1) np.ndarray
            Time step
        ym : float | (num_simplex, 1) np.ndarray
            Young's modulus
        pr : float | (num_simplex, 1) np.ndarray
            Poisson's ratio
        material : str
            Material type ('fcr' or 'arap' or 'neo-hookean' or 'linear-elasticity')
        solver_p : NewtonSolverParams
            Parameters for the Newton solver
        f_ext : (num_vertices*dim, 1) np.ndarray | None
            External force
        Q : (num_vertices*dim, num_vertices*dim) sp.sparse.csc_matrix | None
            Quadratic penalty term
        b : (num_vertices*dim, 1) np.ndarray | None
            Linear penalty term
        gamma : float 
            Merit function scaling factor
        """
        self.h = h
        self.rho = rho
        self.ym = ym
        self.pr = pr
        self.material = material
        self.solver_p = solver_p
        self.gamma = gamma
        self.f_ext = f_ext
        self.Q0 = Q0
        self.b0 = b0
        return


class ElasticMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B=None,  cI=None, cW=None, sim_params : ElasticMFEMSimParams = ElasticMFEMSimParams()):

        dim = X.shape[1]

        self.sim_params = sim_params
        self.mu, self.lam = ympr_to_lame(self.sim_params.ym, self.sim_params.pr)

        self.mu = np.array(self.mu).reshape(-1, 1)
        self.lam = np.array(self.lam).reshape(-1, 1)

        self.lam = self.lam.reshape(-1, 1)
        # preprocess some quantities
        self.X = X
        self.T = T
        x = X.reshape(-1, 1)

        [self.kin_pre, self.GJB, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy] =  \
                self.initial_precomp(X, T, B, cI, cW, p.rho, p.ym, p.pr, dim)
                
                
        M = massmatrix(self.X, self.T, rho=self.sim_params.rho)
        self.M = M
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        self.Mv = Mv
        
        B = sp.sparse.identity(X.shape[0]*dim)
        
        self.kin_pre = KineticEnergyZPrecomp(B, Mv)
        
        

        # elastic energy, gradient and hessian
        self.dim = dim
        self.J = deformation_jacobian(self.X, self.T)
        self.vol = volume(self.X, self.T)


        self.C, self.Ci = symmetric_stretch_map(T.shape[0], dim)
        self.nz = x.shape[0]
        self.nz = x.shape[0]
        
        self.ns = self.Ci.shape[0]
        
        if self.sim_params.Q0 is None:
            self.Q = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
        else:
            assert(self.sim_params.Q0.shape[0] == x.shape[0] and self.sim_params.Q0.shape[1] == x.shape[0])
            self.Q = self.sim_params.Q0
            
        if self.sim_params.b0 is None:
            self.b = np.zeros((x.shape[0], 1))
        else:
            assert(self.sim_params.b0.shape[0] == x.shape[0])
            self.b = self.sim_params.b0.reshape(-1, 1)

        self.solver = SQPMFEMSolver(self.energy, self.gradient_blocks, self.hessian_blocks, sim_params.solver_p)


        # should also build the solver parameters
        # self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, sim_params.solver_p)


    def initial_precomp(self, X, T, B,  cI, cW, rho, ym, pr, dim):
        
        # kinetic energy precomp
        M = massmatrix(X, T, rho=rho)
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
        J = deformation_jacobian(X, T)
        GJB = Ge @ J @ B
        # elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)
        
        C, Ci = symmetric_stretch_map(cI.shape[0], dim)

        MB = Mv @ B
        BMB = B.T @ MB
        BMy = MB.T @ X.reshape(-1, 1)

        return  kin_z_precomp, GJB, mu, lam, vol, C, Ci, BMB, BMy


    def energy(self, p : np.ndarray):

        dim = self.dim
        x = p[:self.nz]
        s = p[self.nz:self.nz + self.ns ]

        S = s.reshape(-1, dim * (dim + 1) // 2)
        F = (self.J @ x).reshape(-1, dim, dim)
        
        elastic = elastic_energy_S(S, self.mu, self.lam,  self.vol, self.sim_params.material)

        quad =  quadratic_energy(x, self.Q, self.b)

        kinetic = kinetic_energy(x, self.y, self.Mv, self.sim_params.h)

        constraint = (np.linalg.norm(stretch(F) - self.C @ s)) * self.sim_params.gamma # merit function

        total = elastic + quad + kinetic + constraint
        return total


    # def energy_sub(self, p : np.ndarray):
    #     dim = self.dim

    #     z, a = self.z_s_from_p(p)

    #     A = a.reshape(-1, dim * (dim + 1) // 2)
    #     F = (self.GJB @ z).reshape(-1, dim, dim) # deformation gradient at cubature tets
        
    #     elastic = elastic_energy_S(A, self.mu, self.lam,  self.vol, self.sim_params.material)

    #     quad =  quadratic_energy(z, self.Q, self.b)

    #     kinetic = kinetic_energy_z(z, self.y, self.sim_params.h, self.kin_pre)

    #     constraint = (np.linalg.norm(stretch(F) - self.C @ a)) * self.sim_params.gamma # merit function

    #     external = 0
    #     if self.ext_energy_func is not None:
    #         external = self.ext_energy_func(z)

    #     total = elastic + quad + kinetic + constraint + external
    #     return total



    def gradient(self, p : np.ndarray):
        # dim = self.dim
        # x = p[:self.nz]
        # s = p[self.nz:self.nz + self.ns ]
        # l = p[self.nz + self.ns:]

        # F = (self.J @ x).reshape(-1, dim, dim)
        # S = s.reshape(-1, dim * (dim + 1) // 2)
        # X = x.reshape(-1, dim)

        # print ('self Q shape', self.Q.shape, 'self b shape', self.b.shape)

        # g_x = kinetic_gradient(x, self.y, self.Mv, self.p.h) \
        #         + quadratic_gradient(x, self.Q, self.b) \
        #         + stretch_gradient_dx(X, self.J, Ci=self.Ci)  @ l 
        
        # g_s =  elastic_gradient_dS(S,  self.mu, self.lam, self.vol, self.p.material) - l 
    
        # g_l =    (self.Ci @ stretch(F) - s)
    
        # g = np.vstack([g_x, g_s, g_l])
        dim = self.dim

        z, s = self.z_s_from_p(p)

        F = (self.J @ z).reshape(-1, dim, dim)
        A = s.reshape(-1, dim * (dim + 1) // 2)
       
        g_x = kinetic_gradient_z(z, self.y, self.sim_params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        
        g_s = elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.sim_params.material)
    
        g_l = (self.Ci @ stretch(F) - s)

        g = np.vstack([g_x, g_s, g_l])
        return g

    def hessian(self, p : np.ndarray):
        
        # dim = self.dim
        # x = p[:self.nz]
        # s = p[self.nz:self.nz + self.ns]
     
        # S = (s).reshape(-1, dim * (dim + 1) // 2)

        # dim = self.dim
        # H_xx = kinetic_hessian(self.Mv, self.p.h)+ \
        #     quadratic_hessian(self.Q)

        # H_xs = sp.sparse.csc_matrix((x.shape[0], s.shape[0])) 

        # H_xl = stretch_gradient_dx(x.reshape(-1, self.dim), self.J, Ci=self.Ci) 

        # H_ss =  elastic_hessian_d2S(S, self.mu, self.lam, self.vol, self.p.material) 
    
        # H_sl = -sp.sparse.identity(s.shape[0])

        # H_ll =  sp.sparse.csc_matrix((s.shape[0], s.shape[0])) 

        # H = sp.sparse.bmat([[H_xx,    H_xs,   H_xl], 
        #                     [H_xs.T,  H_ss,   H_sl], 
        #                     [H_xl.T,  H_sl, H_ll]])

        dim = self.dim
        z, s = self.z_s_from_p(p)

        # F = (self.GJ @ z).reshape(-1, dim, dim)
        ss = s.reshape(-1, dim * (dim + 1) // 2)

        H_xx = kinetic_hessian_z(self.sim_params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)

 
        H_xs = sp.sparse.csc_matrix((z.shape[0], s.shape[0])) 

        H_xl = stretch_gradient_dz(z, self.J, Ci=self.Ci, dim=dim) 

        H_ss =  elastic_hessian_d2S(ss, self.mu, self.lam, self.vol, self.sim_params.material) 
    
        H_sl = -sp.sparse.identity(s.shape[0])

        H_ll =  sp.sparse.csc_matrix((s.shape[0], s.shape[0])) 

        H = sp.sparse.bmat([[H_xx,    H_xs,   H_xl], 
                            [H_xs.T,  H_ss,   H_sl], 
                            [H_xl.T,  H_sl, H_ll]])
        return H

    
    def gradient_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_s_from_p(p)
        F = (self.GJB @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        g_x = kinetic_gradient_z(z, self.y, self.sim_params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        


        g_s =  elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.sim_params.material)
        g_mu = (self.Ci @ stretch(F) - a)
        return g_x, g_s, g_mu


    def hessian_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_s_from_p(p)
        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_x = kinetic_hessian_z(self.sim_params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)
        

        G_x = stretch_gradient_dz(z, self.J, Ci=self.Ci, dim=dim) 

        H_s =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.sim_params.material) 
    
        G_s = -sp.sparse.identity(a.shape[0])
        G_si = G_s 

        return H_x, H_s, G_x, G_si


    def dynamic_precomp(self, x : np.ndarray, x_dot : np.ndarray, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = x + self.sim_params.h * x_dot

        # add to current Q_ext
        if Q_ext is not None:
            if self.sim_params.Q0 is None:
                self.Q = Q_ext
            else:
                self.Q = Q_ext + self.sim_params.Q0

        # same for b
        if b_ext is not None:
            if self.sim_params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.sim_params.b0
        
        return

    def step(self, z : np.ndarray, s : np.ndarray, z_dot : np.ndarray, Q_ext=None, b_ext=None):
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
        # print('self Q shape', self.Q.shape, 'self b shape', self.b.shape)
        # print('Q_ext shape', Q_ext.shape if Q_ext is not None else None, 'b_ext shape', b_ext.shape if b_ext is not None else None)

        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)

        # Se = self.C @ se
        p = np.vstack([z, s])
        p_next = self.solver.solve(p)
        z_next, s_next = self.z_s_from_p(p_next)
        return z_next, s_next


    def rest_state(self):
        """
        Returns the rest state of the system.
        """
        
        z = self.X.reshape(-1, 1)
        z_dot = np.zeros((z.shape[0], 1))
        
        s = np.zeros((self.T.shape[0] * self.dim * (self.dim + 1) // 2, 1))
        s[:, :self.dim] = 1.0
        s = s.reshape(-1,1)
        
        
        return z, s, z_dot


    def z_s_from_p( self, p):
        z = p[:self.nz]
        s = p[self.nz:self.nz + self.ns]
        return z, s