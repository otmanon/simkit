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
from ..Sim import Sim
from ..State import State
from ...solvers import NewtonSolver, NewtonSolverParams
from ...energies import elastic_energy_S, elastic_gradient_dS, elastic_hessian_d2S
from ...energies import quadratic_energy, quadratic_gradient, quadratic_hessian
from ...energies import kinetic_energy, kinetic_gradient, kinetic_hessian

class ElasticMFEMState(State):

    def __init__(self, x, s, l, x_dot):
        self.x = x
        self.s = s
        self.l = l
        self.x_dot = x_dot
        return



class ElasticMFEMSimParams():
    def __init__(self, rho=1, h=1e-2, ym=1, pr=0, g=0,  material='arap',
                 solver_p : NewtonSolverParams  = NewtonSolverParams(), f_ext = None, Q=None, b=None, gamma=None):
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

        if gamma is None:
            gamma = 1
        self.gamma = gamma

        self.f_ext = f_ext
        self.Q0 = Q
        self.b0 = b

        return


class ElasticMFEMSim(Sim):

    def __init__(self, X : np.ndarray, T : np.ndarray, p : ElasticMFEMSimParams = ElasticMFEMSimParams()):
        """

        Parameters
        ----------
        p : ElasticMFEMSimParams
            Parameters of the elastic system
        """
        dim = X.shape[1]

        self.p = p
        self.mu, self.lam = ympr_to_lame(self.p.ym, self.p.pr)

        self.mu = np.array(self.mu).reshape(-1, 1)
        self.lam = np.array(self.lam).reshape(-1, 1)

        self.lam = self.lam.reshape(-1, 1)
        # preprocess some quantities
        self.X = X
        self.T = T
        x = X.reshape(-1, 1)



        M = massmatrix(self.X, self.T, rho=self.p.rho)
        self.M = M
        Mv = sp.sparse.kron( M, sp.sparse.identity(dim))# sp.sparse.block_diag([M for i in range(dim)])
        self.Mv = Mv

        
        # elastic energy, gradient and hessian
        self.dim = dim
        self.J = deformation_jacobian(self.X, self.T)
        self.vol = volume(self.X, self.T)

        self.C, self.Ci = symmetric_stretch_map(T.shape[0], dim)

        self.nx = x.shape[0]
        self.ns = self.Ci.shape[0]
        
        if p.Q0 is None:
            self.Q = sp.sparse.csc_matrix((x.shape[0], x.shape[0]))
        else:
            assert(p.Q0.shape[0] == x.shape[0] and p.Q0.shape[1] == x.shape[0])
            self.Q = self.p.Q0
        if p.b0 is None:
            self.b = np.zeros((x.shape[0], 1))
        else:
            assert(p.b0.shape[0] == x.shape[0])
            self.b = self.p.b0.reshape(-1, 1)

        print ('self Q shape', self.Q.shape, 'self b shape', self.b.shape)

        # should also build the solver parameters
        self.solver = NewtonSolver(self.energy, self.gradient, self.hessian, p.solver_p)

    def energy(self, p : np.ndarray):
        """
        E = E_elastic(s) + E_quad(x) + E_kinetic(x) +  l^T (s(x) - s) 
        
        """

        dim = self.dim
        x = p[:self.nx]
        s = p[self.nx:self.nx + self.ns ]
        l = p[self.nx + self.ns:]

        S = s.reshape(-1, dim * (dim + 1) // 2)
        F = (self.J @ x).reshape(-1, dim, dim)
        
        elastic = elastic_energy_S(S, self.mu, self.lam,  self.vol, self.p.material)

        quad =  quadratic_energy(x, self.Q, self.b)

        kinetic = kinetic_energy(x, self.y, self.Mv, self.p.h)

        constraint = (np.linalg.norm(stretch(F) - self.C @ s)) * self.p.gamma # merit function

        total = elastic + quad + kinetic + constraint
        return total


    def gradient(self, p : np.ndarray):
        """
            [g_x]
        g = |g_s|
            [g_l]
        g_x =  dE_quad/dx + dE_kinetic/dx + ds/dx * l
        g_s = dE_elastic/ds - l
        g_l = s(x) - s
        """

        dim = self.dim
        x = p[:self.nx]
        s = p[self.nx:self.nx + self.ns ]
        l = p[self.nx + self.ns:]

        F = (self.J @ x).reshape(-1, dim, dim)
        S = s.reshape(-1, dim * (dim + 1) // 2)
        X = x.reshape(-1, dim)

        print ('self Q shape', self.Q.shape, 'self b shape', self.b.shape)

        g_x = kinetic_gradient(x, self.y, self.Mv, self.p.h) \
                + quadratic_gradient(x, self.Q, self.b) \
                + stretch_gradient_dx(X, self.J, Ci=self.Ci)  @ l 
        
        g_s =  elastic_gradient_dS(S,  self.mu, self.lam, self.vol, self.p.material) - l 
    
        g_l =    (self.Ci @ stretch(F) - s)
    
        g = np.vstack([g_x, g_s, g_l])
        return g

    def hessian(self, p : np.ndarray):
        """
            [H_xx  H_xs  H_xl]
        H = |H_sx  H_ss  H_sl|
            [H_sl  H_ss  H_ll]
        
        H_xx = d2E_quad/dx2 + d2E_kinetic/dx2  + (missing d2sdx2 we usually leave out)
        H_xs = Hsx = 0
        H_xl = H_lx = ds/dx
        H_ss = d2E_elastic/ds2
        H_sl = H_ls = -I
        H_ll = 0 
        """
        dim = self.dim
        x = p[:self.nx]
        s = p[self.nx:self.nx + self.ns]
     
        S = (s).reshape(-1, dim * (dim + 1) // 2)

        dim = self.dim
        H_xx = kinetic_hessian(self.Mv, self.p.h)+ \
            quadratic_hessian(self.Q)

        H_xs = sp.sparse.csc_matrix((x.shape[0], s.shape[0])) 

        H_xl = stretch_gradient_dx(x.reshape(-1, self.dim), self.J, Ci=self.Ci) 

        H_ss =  elastic_hessian_d2S(S, self.mu, self.lam, self.vol, self.p.material) 
    
        H_sl = -sp.sparse.identity(s.shape[0])

        H_ll =  sp.sparse.csc_matrix((s.shape[0], s.shape[0])) 

        H = sp.sparse.bmat([[H_xx,    H_xs,   H_xl], 
                            [H_xs.T,  H_ss,   H_sl], 
                            [H_xl.T,  H_sl, H_ll]])


        return H


    def dynamic_precomp(self, x : np.ndarray, x_dot : np.ndarray, Q_ext=None, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.y = x + self.p.h * x_dot

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
        
        return

    def step(self, x : np.ndarray, s : np.ndarray, l : np.ndarray, x_dot : np.ndarray, Q_ext=None, b_ext=None):
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
        print('self Q shape', self.Q.shape, 'self b shape', self.b.shape)
        print('Q_ext shape', Q_ext.shape if Q_ext is not None else None, 'b_ext shape', b_ext.shape if b_ext is not None else None)

        self.dynamic_precomp(x, x_dot, Q_ext, b_ext)

        # Se = self.C @ se
        p = np.vstack([x, s])
        p_next = self.solver.solve(p)
        x_next = p_next[:self.nx]
        s_next = p_next[self.nx:self.nx + self.ns]
        l_next = p_next[self.nx + self.ns:]
        return x_next, s_next, l_next





