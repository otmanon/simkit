import numpy as np
import scipy as sp

from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.pairwise_displacement import pairwise_displacement
from simkit.polar_svd import polar_svd
from simkit.project_into_subspace import project_into_subspace
from simkit.selection_matrix import selection_matrix

from ...clustered_plastic_stretch_tensor import clustered_plastic_stretch_tensor
from ...fast_sandwich_transform_clustered import fast_sandwich_transform_clustered
from ...solvers import BlockCoordSolver, BlockCoordSolverParams
from ... import ympr_to_lame
from ... import volume
from ... import massmatrix
from ... import deformation_jacobian


class ModalMuscleSimParams():
    def __init__(self, rho=1, h=1e-2, mu=1, gamma = 1,
                 solver_params : BlockCoordSolverParams  = BlockCoordSolverParams(), Q0=None, b0=None, cache_dir=None, read_cache=False, alpha=0.8, contact=False):

        self.h = h
        self.rho = rho
        self.mu = mu
        self.gamma = gamma 
        self.material = 'arap'
        self.solver_p = solver_params
        self.Q0 = Q0
        self.b0 = b0
        self.cache_dir = cache_dir
        self.read_cache = read_cache

        self.alpha = alpha
        self.contact = contact
        
        return


class ModalMuscleSim():

    def __init__(self, X : np.ndarray, T : np.ndarray, B : np.ndarray, D : np.ndarray, l=None, d=None, cI=None, plane_normal=None, plane_pos = None, params : ModalMuscleSimParams = ModalMuscleSimParams()):

        
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

        self.D = D

        ## cubature precomp
        if l is None:
            l = np.arange(0, T.shape[0])
        self.l = l.astype(int)

        if d is None:
            d = np.zeros((T.shape[0]))
        d = d.astype(int)

        P, Pm = cluster_grouping_matrices(l, X, T)

        
        vol = volume(X, T)

        mu = self.params.mu
        if isinstance(mu, float) or isinstance(mu, int):
            mu = np.ones((T.shape[0], 1)) * mu

        A = sp.sparse.diags(vol.flatten())
        Mu = sp.sparse.diags(mu.flatten())
        
        AMue = sp.sparse.kron(A @ Mu, sp.sparse.identity(dim*dim))

        PAMue = sp.sparse.kron(P @ (A @ Mu) , sp.sparse.identity(dim*dim))
        J = deformation_jacobian(X, T)
                                              
        self.num_passive_clusters = l.max() + 1                                        
        self.AMuPJB = (PAMue @ J) @ B
        self.B = B
                                                                                                                                                                                                                                                                                                                                                                 
        L_passive = J.T @ AMue @ J

        self.AMuPJB_full =  PAMue @ J @ B

        Mv = sp.sparse.kron(massmatrix(X, T, rho=params.rho), sp.sparse.identity(dim))
        self.BLB_passive = B.T @ L_passive @ B 
        self.BMB = B.T @ Mv @ B
        self.BMy = B.T @ Mv @ X.reshape(-1, 1)


        ######### active muscle force 
        gamma = self.params.gamma
        if isinstance(gamma, float) or isinstance(gamma, int):
            gamma = np.ones((T.shape[0], 1)) * gamma
        Gamma = sp.sparse.diags(gamma.flatten())
        AGammae = sp.sparse.kron(A @ Gamma, sp.sparse.identity(dim*dim))
        

        L_active = J.T @ (AGammae @ J)
        JD = J @ self.D
        self.BLB_active = self.B.T @ L_active @ self.B
        JAgamma = J.T @ AGammae
        BJAgamma = B.T @ JAgamma
        
        self.DMD = self.D.T @ Mv @ self.D
        self.DMy = self.D.T @ Mv @ X.reshape(-1, 1)

        self.num_active_clusters = d.max() + 1
        self.K = clustered_plastic_stretch_tensor(X, T,d, B, D, w=(vol*gamma).reshape(-1, 1))
        self.fst_BJAgamma_JD = fast_sandwich_transform_clustered(BJAgamma, JD, d,dim=dim)


        ######### contact forces
        if params.contact:
            if plane_normal is None:
                plane_normal = np.zeros((dim, 1))
                plane_normal[1] = 1
            if plane_pos is None:
                plane_pos = np.zeros((dim, 1))
            
            self.plane_pos = plane_pos
            self.plane_normal = plane_normal

            # build orthogonal tangent vector to the plane normal
            tangent = np.random.rand(dim, 1)
            tangent = tangent - tangent.T @ plane_normal * plane_normal
            tangent = tangent / np.linalg.norm(tangent)

            if self.dim == 3:
                tangent_2 = np.cross(plane_normal.flatten(), tangent.flatten()).reshape(-1, 1)
                tangent = np.hstack([tangent, tangent_2])

            contact_frame = np.vstack([plane_normal.T, tangent.T])

            cI = np.unique(cI)
            self.cI = cI
            S = selection_matrix(cI, X.shape[0])
            Se = sp.sparse.kron(S, sp.sparse.identity(dim))
            self.Je = Se @ B  # contact jacobian (from reduced positions to per-contact point positions)
            
            Sc = sp.sparse.kron(S, contact_frame)
            self.Jc = Sc @ B # contact jacobian (from reduced positions to per-contact velocities)


            self.num_contact_points = cI.shape[0] + 1
            

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

        self.H = self.BLB_passive + self.BLB_active + self.BMB / self.params.h**2 + self.Q


        self.chol_H = sp.linalg.cho_factor(self.H)

        if self.params.contact:
            JeQi = sp.linalg.cho_solve(self.chol_H, self.Je.T).T
            JcQi = sp.linalg.cho_solve(self.chol_H, self.Jc.T).T
            self.JeQi =  JeQi
            self.DQi = JeQi
            self.JcQi =  JcQi

        return
    
    def dynamic_precomp(self, z, z_dot, a : np.ndarray, b_ext=None):
        """
        Computation done once every timestep and never again
        """
        self.a = a.copy()
        
        self.z_curr = z.copy()
        self.z_prev = z - self.params.h * z_dot
        self.y = z + self.params.h * z_dot

        # same for b
        if b_ext is not None:
            if self.params.b0 is None:
                self.b = b_ext
            else:
                self.b = b_ext + self.params.b0
        
        self.dynamic_precomp_done = True


    def global_step(self, z : np.ndarray, local_vars: np.ndarray):

        k = self.BMB @ self.y / self.params.h**2
        
        passive_dofs = self.num_passive_clusters * (self.dim**2)
        r_passive = local_vars[:passive_dofs]
        e_passive =  self.AMuPJB.T @  r_passive

        active_dofs = self.num_active_clusters * (self.dim**2)
        r_active = local_vars[passive_dofs:active_dofs + passive_dofs]
        R_active = r_active.reshape(-1, self.dim, self.dim)
        e_active = self.fst_BJAgamma_JD(R_active) @ self.a


        f = k + e_passive + e_active - self.b
        if self.params.contact:
            f_contact = self.contact_projection(z, f)            
            f = f + f_contact
            
        z_next = sp.linalg.cho_solve(self.chol_H, f)

        return z_next
    
    def local_step(self, z : np.ndarray):
        r_passive = self.passive_local_step(z)

        r_active = self.active_local_step(z)

        r = np.vstack([r_passive, r_active])

  
        return r
    
    def passive_local_step(self, z : np.ndarray):
        c = self.AMuPJB @ z
        C = c.reshape(-1, self.dim, self.dim)
        
        R = polar_svd(C)[0]
    
        r = R.reshape(-1, 1)
        
        return r

    def active_local_step(self, z : np.ndarray,):
        C = self.K(z, self.a)
        R = polar_svd( C)[0]
        r = R.reshape(-1, 1)
        return r
    
    def contact_projection(self, z : np.ndarray, f:np.ndarray):

        z_dot_tent = (sp.linalg.cho_solve(self.chol_H, f)   - self.z_curr) / self.params.h
        P = (self.Je @ (z) ).reshape(-1, self.dim) # contact positions

        # D = pairwise_displacement(P, self.plane_pos.T)
        offset = P[:, 1]

        under_ground_plane = (offset < self.plane_pos[1]).flatten() 

        # z_dot = np.random.rand(*z_dot.shape)
        local_vel = (self.Je @ z_dot_tent).reshape(-1, self.dim)

        getting_closer = local_vel[:, 1] < 0

        nI = under_ground_plane * getting_closer
        contacting_indices = np.where(nI)[0]

        num_contacts = (contacting_indices).shape[0]

        if num_contacts > 0:
            
            cI = np.repeat(contacting_indices[:, None], self.dim, axis=1)*self.dim + np.arange(self.dim)
            cI = cI.flatten()

            JeI = self.Je[cI, :]
            L = self.JeQi[cI, :]

            vel_in_contact = local_vel[contacting_indices, :]

            vel_t = vel_in_contact[:, 0]

            v = np.zeros((num_contacts, self.dim))
            v[:, 0] = (1.0 - self.params.alpha) * vel_t

            p = (JeI @ self.z_curr).reshape(-1, self.dim)
            local_f = (L @ f).reshape(-1, self.dim)
            
            b = (v*self.params.h + p - local_f).reshape(-1, 1)

            if L.shape[0] >= L.shape[1]:
                LTL= L.T @ L
                # overconstrained, do least squares
                c = np.linalg.solve(LTL, L.T @  b) # @ z_tent))
            else:
                LLT = L @ L.T
                c = L.T @ np.linalg.solve(LLT,  b) 

        else:
            c = np.zeros_like(f)
        
        return c


    def step(self, z : np.ndarray, z_dot : np.ndarray, a : np.ndarray, b_ext=None):
        # call this to set up inertia forces that change each timestep.
        self.dynamic_precomp(z, z_dot, a, b_ext)

        z0 = z.copy() # very important to copy this here so that x does not get over-written
        z_next = self.solver.solve(z0)
        # z_next = project_into_subspace(x_next, self.B)

        self.dynamic_precomp_done = False
        return z_next
    
    def rest_state(self):

        Mv = sp.sparse.kron(massmatrix(self.X, self.T, rho=self.params.rho), sp.sparse.identity(self.dim))
        z = project_into_subspace( self.X.reshape(-1, 1), self.B,
                        M=Mv, BMB=self.BMB, BMy=self.BMy)# 

        z_dot = np.zeros_like(z)
        
        a = project_into_subspace( self.X.reshape(-1, 1), self.D,
                        M=Mv, BMB=self.DMD, BMy=self.DMy)# 

        return z, z_dot, a




