import warnings
import igl
import numpy as np
import scipy as sp
import os

import simkit
from simkit.solvers import Solver, SolverParams, NewtonSolver
from simkit.energies import elastic_energy_S, elastic_gradient_dS, elastic_hessian_d2S
from simkit.energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from simkit.energies import kinetic_energy_z, kinetic_gradient_z, kinetic_hessian_z, KineticEnergyZPrecomp
from simkit.sims.Sim import *

class SQPMFEMSolverParams(SolverParams):

    def __init__(self, max_iter=100, tol=1e-6, do_line_search=True, verbose=False):
        self.do_line_search = do_line_search
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        return
    
class SQPMFEMSolver(Solver):

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

    
    def solve(self, p0, return_info=False):
        
        p = p0.copy()
        
        if return_info:
            info = {'g' : [], 'dp' : [], 'alphas' : [], 'iters' : -1}
        for i in range(self.params.max_iter):

            [H_u, H_z, G_u, G_zi] = self.hess_blocks_func(p)            
            
            [f_u, f_z, f_mu] = self.grad_blocks_func(p)

            #form K
            K = G_u @ G_zi @ H_z @ G_zi @ G_u.T 
            Q = H_u + K

            # form g_u
            g_u =  G_u @ G_zi @ (f_z - H_z @ G_zi @ f_mu) - f_u 

            if isinstance(Q, sp.sparse.spmatrix):
                du =  sp.sparse.linalg.spsolve(Q, g_u).reshape(-1, 1)
                du = np.array(du)
            else:
                du = np.linalg.solve(Q, g_u)
            if du.ndim == 1:
                du = du.reshape(-1, 1)
            

            g_z = - (f_mu + G_u.T @ du)
            dz = G_zi @ g_z

            dmu = - G_zi @ (f_z + H_z @ dz)

            g = np.vstack([f_u, f_z, f_mu])
            dp = np.vstack([du, dz, dmu])
            
            if return_info:
                info['g'].append(g)
                info['dp'].append(dp)

            if self.params.do_line_search:
                energy_func = lambda z: self.energy_func(z)
                alpha, lx, ex = simkit.backtracking_line_search(energy_func, p, g, dp)
                if return_info:
                    info['alphas'].append(alpha)
            else:
                alpha = 1.0
                if return_info:
                    info['alphas'].append(alpha)

            p += alpha * dp
            if np.linalg.norm(g) < 1e-4:
                break

        if return_info:
            info['iters'] = i
            return p, info
        else:
            return p
    


class ElasticMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,  material='arap', 
                 solver_params : SQPMFEMSolverParams  = None,
                 Q0=None, b0=None,
                 read_cache=False, cache_dir=None):
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

        if solver_params is None:
            solver_params = SQPMFEMSolverParams()
        self.solver_p = solver_params
        self.Q0 = Q0
        self.b0 = b0
    
        
        self.read_cache = read_cache
        self.cache_dir = cache_dir
        return


class ElasticMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray = None, cI=None,  
                 cW=None, sim_params : 
                     ElasticMFEMSimParams = ElasticMFEMSimParams(), q=None):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim
        self.sim_params = sim_params

        # preprocess some quantities
        self.X = X
        self.T = T
        
        if B is None:
            B = sp.sparse.identity(X.shape[0] * X.shape[1])
        self.B = B
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI

        if q is None:
            q = np.zeros((X.shape[0]*dim, 1))
        self.q = q
        
        well_read = False
        if self.sim_params.read_cache:
            self.cache_dir = self.sim_params.cache_dir
            try:
                [self.kin_pre, self.GJB, self.GJq, self.mu, self.lam,
                 self.vol, self.C, self.Ci, self.BMB, self.BMy] = self.read_cache(self.cache_dir)
                well_read = True
            except:
                warnings.warn("Warning : Couldn't read promputations from cache. Recomputing from scratch...")
        if not well_read:
            [self.kin_pre, self.GJB, self.GJq, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy] =  \
                self.initial_precomp(X, T, B, q, cI, cW, sim_params.rho, sim_params.ym, sim_params.pr, dim)
            if self.sim_params.cache_dir is not None:
                os.makedirs(self.sim_params.cache_dir, exist_ok=True)
                self.save_cache(self.sim_params.cache_dir, self.kin_pre, self.GJB, self.GJq, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy)

        self.nz = B.shape[-1]
        self.na = self.Ci.shape[0]
    
        # should also build the solver parameters
        # TODO: do we want to support NewtonSolverParams as well?
        self.solver = SQPMFEMSolver(self.energy, self.gradient_blocks, self.hessian_blocks, sim_params.solver_p)
    
        # self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, sim_params.solver_p)
        
    # def read_cache(self, cache_dir):
    #     kin_pre = np.load(cache_dir + "kin_pre.npy", allow_pickle=True).item()
    #     GJB = np.load(cache_dir + "GJB.npy")
    #     GJq = np.load(cache_dir + "GJq.npy")
    #     mu = np.load(cache_dir + "mu.npy")
    #     lam = np.load(cache_dir + "lam.npy")
    #     vol = np.load(cache_dir + "vol.npy")
    #     C = np.load(cache_dir + "C.npy", allow_pickle=True).item()
    #     Ci = np.load(cache_dir + "Ci.npy", allow_pickle=True).item()
    #     BMB = np.load(cache_dir + "BMB.npy")
    #     BMy = np.load(cache_dir + "BMy.npy")
    #     return kin_pre, GJB, GJq, mu, lam, vol, C, Ci, BMB, BMy
    
    # def save_cache(self, cache_dir, kin_pre, GJB, GJq, mu, lam, vol, C, Ci, BMB, BMy):
    #     os.makedirs(cache_dir, exist_ok=True)
    #     np.save(cache_dir + "kin_pre.npy", kin_pre)
    #     np.save(cache_dir + "GJB.npy", GJB)
    #     np.save(cache_dir + "GJq.npy", GJq)
    #     np.save(cache_dir + "mu.npy", mu)
    #     np.save(cache_dir + "lam.npy", lam)
    #     np.save(cache_dir + "vol.npy", vol)
    #     np.save(cache_dir + "C.npy", C)
    #     np.save(cache_dir + "Ci.npy", Ci)
    #     np.save(cache_dir + "BMB.npy", self.BMB)
    #     np.save(cache_dir + "BMy.npy", self.BMy)
    #     return

    def initial_precomp(self, X, T, B,q,  cI, cW, rho, ym, pr, dim):
        
        # kinetic energy precomp
        M = simkit.massmatrix(self.X, self.T, rho=rho)
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        kin_z_precomp = KineticEnergyZPrecomp(B, Mv)
        
        
        # elastic energy precomp
        if cW is None:
            vol = simkit.volume(X, T)
        else:
            vol = cW.reshape(-1, 1)
            
        ## ym, pr to lame parameters
        mu, lam = simkit.ympr_to_lame(ym, pr)
        if isinstance(mu, float):
            mu = np.ones((T.shape[0], 1)) * mu
        if isinstance(lam, float):
            lam = np.ones((T.shape[0], 1)) * lam

        mu = mu[cI]
        lam = lam[cI]

        ## selection matrix from cubature precomp.
        G = simkit.selection_matrix(cI, T.shape[0])
        Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
        J = simkit.deformation_jacobian(self.X, self.T)
        GJB = Ge @ J @ B
        GJq = Ge @ J @ q
        # elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)
        
        C, Ci = simkit.symmetric_stretch_map(cI.shape[0], dim)

        MB = Mv @ B
        BMB = B.T @ MB
        BMy = MB.T @ X.reshape(-1, 1)

        return  kin_z_precomp, GJB, GJq, mu, lam, vol, C, Ci, BMB, BMy

    def dynamic_precomp(self, z, z_dot, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.sim_params.h * z_dot

        # add to current Q_ext
        if Q_ext is not None:
            if self.sim_params.Q0 is None:
                self.Q = Q_ext
            else:
                self.Q = Q_ext + self.sim_params.Q0
        else:
            if self.sim_params.Q0 is None:
                self.Q = sp.sparse.csc_matrix((z.shape[0], z.shape[0])) 
            else:
                self.Q = self.sim_params.Q0

        # same for b
        if b_ext is not None:
            if self.sim_params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.sim_params.b0
        else:
            if self.sim_params.b0 is None:
                self.b = np.zeros((z.shape[0], 1))
            else:
                self.b = self.sim_params.b0

    def energy(self, p : np.ndarray):
        dim = self.dim

        z, a = self.z_a_from_p(p)
        l = p[self.nz + self.na:]
        A = a.reshape(-1, dim * (dim + 1) // 2)
        F = (self.GJB @ z + self.GJq).reshape(-1, dim, dim) # deformation gradient at cubature tets
        
        elastic = elastic_energy_S(A, self.mu, self.lam,  self.vol, self.sim_params.material)

        quad =  quadratic_energy(z, self.Q, self.b)

        kinetic = kinetic_energy_z(z, self.y, self.sim_params.h, self.kin_pre)

        # c = simkit.stretch(F) - self.C @ a
        if self.dim == 2:
            w = np.kron(self.vol, np.array([[1, 1, 2]]).T)
        elif self.dim == 3:
            w = np.kron(self.vol, np.array([[1, 1, 1, 2, 2, 2]]).T)
 
        c = w * (self.Ci @ simkit.stretch(F) - a)
        constraint = l.T @ c
        # constraint = w.T @ c
        # constraint = (c.T @ c) * self.sim_params.gamma # merit function

        total = elastic + quad + kinetic + constraint 
        return total

    def gradient(self, p : np.ndarray):
        dim = self.dim

        z, a = self.z_a_from_p(p)

        F = (self.GJB @ z + self.GJq).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
       
        g_x = kinetic_gradient_z(z, self.y, self.sim_params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        
        g_s = elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.sim_params.material)
    
        g_l = (self.Ci @ simkit.stretch(F) - a)

    
        g = np.vstack([g_x, g_s, g_l])
        return g

    def hessian(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)

        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_xx = kinetic_hessian_z(self.sim_params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)

        H_xs = sp.sparse.csc_matrix((z.shape[0], a.shape[0])) 

        H_xl = simkit.stretch_gradient_dz(z, self.GJB, Ci=self.Ci, dim=dim, GJq=self.GJq) 

        H_ss =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.sim_params.material) 
    
        H_sl = -sp.sparse.identity(a.shape[0])

        H_ll =  sp.sparse.csc_matrix((a.shape[0], a.shape[0])) 

        H = sp.sparse.bmat([[H_xx,    H_xs,   H_xl], 
                            [H_xs.T,  H_ss,   H_sl], 
                            [H_xl.T,  H_sl, H_ll]])
        # H = H.toarray()
        return H
    

    def hessian_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)
        
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_x = kinetic_hessian_z(self.sim_params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)
        
        G_x = simkit.stretch_gradient_dz(z, self.GJB, Ci=self.Ci, dim=dim, GJq=self.GJq) 

        H_s =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.sim_params.material) 
    
        G_s = -sp.sparse.identity(a.shape[0])
        G_si = G_s 

        return H_x, H_s, G_x, G_si
    
    def gradient_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)
        F = (self.GJB @ z + self.GJq).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        g_x = kinetic_gradient_z(z, self.y, self.sim_params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        

        g_s =  elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.sim_params.material)
        g_mu = (self.Ci @ simkit.stretch(F) - a)
        return g_x, g_s, g_mu
    
    def step(self, z : np.ndarray, a : np.ndarray,  z_dot : np.ndarray, Q_ext=None, b_ext=None,  return_info=False):
    
        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)
        l = np.zeros((a.shape[0], 1))
        p = np.vstack([z, a, l])

        if return_info:
            p, info = self.solver.solve(p, return_info=return_info)
        else:
            p = self.solver.solve(p)
        
        z, a = self.z_a_from_p(p)


        if return_info:
            return z, a, info
        else:
            return z, a
        
        return z, a, info


    def z_a_from_p( self, p):
        z = p[:self.nz]
        a = p[self.nz:self.nz + self.na]
        return z, a
    
    def p_from_z_c(self, z, a):
        return np.concatenate([z, a])
    

    def rest_state(self):
        dim = self.X.shape[1]
        # position dofs init to rest position
        z = simkit.project_into_subspace( self.X.reshape(-1, 1) - self.q, self.B,
                                M=sp.sparse.kron(simkit.massmatrix(self.X, self.T), sp.sparse.identity(dim)))# 

        # velocity dofs init to zero 
        z_dot = np.zeros(z.shape)

        cc = int(dim * (dim+1)/2)
        # stretch dofs init to identity
        s = np.ones((self.cI.shape[0], cc))
        s[:, dim:] = 0
        s = s.reshape(-1, 1)

        return z, s, z_dot

