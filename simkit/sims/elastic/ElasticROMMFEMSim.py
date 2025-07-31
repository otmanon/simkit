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
                alpha, lx, ex = simkit.backtracking_line_search(energy_func, p, g, dp)
            else:
                alpha = 1.0

            p += alpha * dp
            if np.linalg.norm(g) < 1e-4:
                break

        return p
    

class ElasticROMMFEMState(State):

    def __init__(self, z, c, l, z_dot):
        self.z = z
        self.c = c
        self.l = l
        self.z_dot = z_dot
        return


class ElasticROMMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0,  material='arap', gamma=1,
                 solver_params : SQPMFEMSolverParams  = None, Q0=None, b0=None, read_cache=False, cache_dir=None):
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
        self.gamma = gamma
        
        self.read_cache = read_cache
        self.cache_dir = cache_dir
        return


class ElasticROMMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, cI=None,  cW=None, p : ElasticROMMFEMSimParams = ElasticROMMFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticFEMSimParams
            Parameters of the elastic system
        """
        
        dim = X.shape[1]
        self.dim = dim
        self.params = p

        # preprocess some quantities
        self.X = X
        self.T = T
        self.B = B
     
        ## cubature precomp
        if cI is not None:
            assert cW is not None
            assert cW.shape[0] == cI.shape[0]
        if cI is None:
            cI = np.arange(0, T.shape[0])

        self.cI = cI

        well_read = False
        if self.params.read_cache:
            self.cache_dir = self.params.cache_dir
            try:
                [self.kin_pre, self.GJB, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy] = self.read_cache(self.cache_dir)
                well_read = True
            except:
                warnings.warn("Warning : Couldn't read promputations from cache. Recomputing from scratch...")
        if not well_read:
            [self.kin_pre, self.GJB, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy] =  \
                self.initial_precomp(X, T, B, cI, cW, p.rho, p.ym, p.pr, dim)
            if self.params.cache_dir is not None:
                os.makedirs(self.params.cache_dir, exist_ok=True)
                self.save_cache(self.params.cache_dir, self.kin_pre, self.GJB, self.mu, self.lam, self.vol, self.C, self.Ci, self.BMB, self.BMy)

        # These get filled up if use calls step with their own energy/gradient/hessian that changes every timestep
        self.ext_energy_func = None
        self.ext_gradient_func = None
        self.ext_hessian_func = None

        self.nz = B.shape[-1]
        self.na = self.Ci.shape[0]
    
        # should also build the solver parameters
        # TODO: do we want to support NewtonSolverParams as well?
        self.solver = SQPMFEMSolver(self.energy, self.gradient_blocks, self.hessian_blocks, p.solver_p)
    
    def read_cache(self, cache_dir):
        kin_pre = np.load(cache_dir + "kin_pre.npy", allow_pickle=True).item()
        GJB = np.load(cache_dir + "GJB.npy")
        mu = np.load(cache_dir + "mu.npy")
        lam = np.load(cache_dir + "lam.npy")
        vol = np.load(cache_dir + "vol.npy")
        C = np.load(cache_dir + "C.npy", allow_pickle=True).item()
        Ci = np.load(cache_dir + "Ci.npy", allow_pickle=True).item()
        BMB = np.load(cache_dir + "BMB.npy")
        BMy = np.load(cache_dir + "BMy.npy")
        return kin_pre, GJB, mu, lam, vol, C, Ci, BMB, BMy
    
    def save_cache(self, cache_dir, kin_pre, GJB, mu, lam, vol, C, Ci, BMB, BMy):
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_dir + "kin_pre.npy", kin_pre)
        np.save(cache_dir + "GJB.npy", GJB)
        np.save(cache_dir + "mu.npy", mu)
        np.save(cache_dir + "lam.npy", lam)
        np.save(cache_dir + "vol.npy", vol)
        np.save(cache_dir + "C.npy", C)
        np.save(cache_dir + "Ci.npy", Ci)
        np.save(cache_dir + "BMB.npy", self.BMB)
        np.save(cache_dir + "BMy.npy", self.BMy)
        return

    def initial_precomp(self, X, T, B,  cI, cW, rho, ym, pr, dim):
        
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
        # elastic_z_precomp = ElasticEnergyZPrecomp(B, Ge, J, self.dim)
        
        C, Ci = simkit.symmetric_stretch_map(cI.shape[0], dim)

        MB = Mv @ B
        BMB = B.T @ MB
        BMy = MB.T @ X.reshape(-1, 1)

        return  kin_z_precomp, GJB, mu, lam, vol, C, Ci, BMB, BMy

    def dynamic_precomp(self, z, z_dot, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = z + self.params.h * z_dot

        # add to current Q_ext
        if Q_ext is not None:
            if self.params.Q0 is None:
                self.Q = Q_ext
            else:
                self.Q = Q_ext + self.params.Q0
        else:
            if self.params.Q0 is None:
                self.Q = sp.sparse.csc_matrix((z.shape[0], z.shape[0])) 
            else:
                self.Q = self.params.Q0

        # same for b
        if b_ext is not None:
            if self.params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.params.b0
        else:
            if self.params.b0 is None:
                self.b = np.zeros((z.shape[0], 1))
            else:
                self.b = self.params.b0

    def energy(self, p : np.ndarray):
        dim = self.dim

        z, a = self.z_a_from_p(p)

        A = a.reshape(-1, dim * (dim + 1) // 2)
        F = (self.GJB @ z).reshape(-1, dim, dim) # deformation gradient at cubature tets
        
        elastic = elastic_energy_S(A, self.mu, self.lam,  self.vol, self.params.material)

        quad =  quadratic_energy(z, self.Q, self.b)

        kinetic = kinetic_energy_z(z, self.y, self.params.h, self.kin_pre)

        constraint = (np.linalg.norm(simkit.stretch(F) - self.C @ a)) * self.params.gamma # merit function

        external = 0
        if self.ext_energy_func is not None:
            external = self.ext_energy_func(z)

        total = elastic + quad + kinetic + constraint + external
        return total

    def gradient(self, p : np.ndarray):
        dim = self.dim

        z, a = self.z_a_from_p(p)

        F = (self.GJB @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
       
        g_x = kinetic_gradient_z(z, self.y, self.params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        
        g_s = elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.params.material)
    
        g_l = (self.Ci @ simkit.stretch(F) - a)

        if self.ext_gradient_func is not None:
            g_x += self.ext_gradient_func(z)

        g = np.vstack([g_x, g_s, g_l])
        return g

    def hessian(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)

        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_xx = kinetic_hessian_z(self.params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)

        if self.ext_hessian_func is not None:
            H_xx += self.ext_hessian_func(z)

        H_xs = sp.sparse.csc_matrix((z.shape[0], a.shape[0])) 

        H_xl = simkit.stretch_gradient_dz(z, self.GJB, Ci=self.Ci, dim=dim) 

        H_ss =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.params.material) 
    
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
        # F = (self.GJ @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)

        H_x = kinetic_hessian_z(self.params.h, self.kin_pre)+ \
            quadratic_hessian(self.Q)
        
        
        if self.ext_hessian_func is not None:
            H_x += self.ext_hessian_func(z)

        G_x = simkit.stretch_gradient_dz(z, self.GJB, Ci=self.Ci, dim=dim) 

        H_s =  elastic_hessian_d2S(A, self.mu, self.lam, self.vol, self.params.material) 
    
        G_s = -sp.sparse.identity(a.shape[0])
        G_si = G_s 

        return H_x, H_s, G_x, G_si
    
    def gradient_blocks(self, p : np.ndarray):
        dim = self.dim
        z, a = self.z_a_from_p(p)
        F = (self.GJB @ z).reshape(-1, dim, dim)
        A = a.reshape(-1, dim * (dim + 1) // 2)
        g_x = kinetic_gradient_z(z, self.y, self.params.h, self.kin_pre) \
                + quadratic_gradient(z, self.Q, self.b) 
        
        if self.ext_gradient_func is not None:
            g_x += self.ext_gradient_func(z)

        g_s =  elastic_gradient_dS(A,  self.mu, self.lam, self.vol, self.params.material)
        g_mu = (self.Ci @ simkit.stretch(F) - a)
        return g_x, g_s, g_mu
    
    def step(self, z : np.ndarray, a : np.ndarray,  z_dot : np.ndarray, Q_ext=None, b_ext=None, 
             ext_energy_func=None, ext_gradient_func=None, ext_hessian_func=None):
    

        self.ext_energy_func = ext_energy_func
        self.ext_gradient_func = ext_gradient_func
        self.ext_hessian_func = ext_hessian_func
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, Q_ext, b_ext)


        p = np.vstack([z, a])

        p = self.solver.solve(p)
        
        z, a = self.z_a_from_p(p)

        self.ext_energy_func = None
        self.ext_gradient_func = None
        self.ext_hessian_func = None
        return z, a


    def z_a_from_p( self, p):
        z = p[:self.nz]
        a = p[self.nz:self.nz + self.na]
        return z, a
    
    def p_from_z_c(self, z, a):
        return np.concatenate([z, a])
    

    def rest_state(self):
        dim = self.X.shape[1]
        # position dofs init to rest position
        z = simkit.project_into_subspace( self.X.reshape(-1, 1), self.B,
                                M=sp.sparse.kron(simkit.massmatrix(self.X, self.T), sp.sparse.identity(dim)), BMB=self.BMB, BMy=self.BMy)# 

        # velocity dofs init to zero 
        z_dot = np.zeros(z.shape)

        cc = int(dim * (dim+1)/2)
        # stretch dofs init to identity
        s = np.ones((self.cI.shape[0], cc))
        s[:, dim:] = 0
        s = s.reshape(-1, 1)

        return z, s, z_dot

