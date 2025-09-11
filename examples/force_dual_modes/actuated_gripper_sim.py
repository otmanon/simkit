import os
import sys
import igl
import numpy as np
import scipy as sp
from PIL import Image



from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.filesystem.compute_with_cache_check import compute_with_cache_check
from simkit.fold_in_vector_subspace import fold_in_vector_subspace
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_dual_modes import force_dual_modes_sqrt
from simkit.hessian_blocks import hessian_blocks
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.massmatrix import massmatrix
from simkit.normal_force_matrix import normal_force_matrix
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_animation import view_animation
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSimParams
from simkit.sims.elastic.PneumaticActuatorFEMSim import PneumaticActuatorFEMSim
from simkit.spectral_cubature import spectral_cubature
from simkit.volume import volume
from simkit.von_tycowics_expansion import von_tycowicz_expansion
from simkit.ympr_to_lame import ympr_to_lame


sys.path.append(os.path.dirname(__file__) + "/../../../")


def compute_subspace(method):
    if method == "force_dual_eigs":
        B = np.zeros((X.shape[0]*X.shape[1], 0))  
        for i, E in enumerate(Es):
            D = normal_force_matrix(X, E)               
            D = D.sum(axis=1).reshape(-1, 1)                    
            sigma_F_sqrt = D.sum(axis=1).reshape(-1, 1)
            Msqrt_ie = sp.sparse.kron(Msqrt_i, sp.sparse.identity(dim))
            Bi, Di= force_dual_modes_sqrt(H, sigma_F_sqrt, m, M_sqrt=Msqrt_ie)
            B = np.hstack([B, Bi])
        
    elif method == "eigs":
        E, B = eigs(H, k=m, M=Mext)
        
    elif method == "force_dual_von_tycowicz":
        B = np.zeros((X.shape[0]*X.shape[1], 0))  
        for i, E in enumerate(Es):
            D = normal_force_matrix(X, E)               
            D = D.sum(axis=1).reshape(-1, 1)                    
            sigma_F_sqrt = D.sum(axis=1).reshape(-1, 1)
            Msqrt_ie = sp.sparse.kron(Msqrt_i, sp.sparse.identity(dim))
            Msqrt_e = sp.sparse.kron(Msqrt, sp.sparse.identity(dim))
            Bi, Di= force_dual_modes_sqrt(H, sigma_F_sqrt, m, M_sqrt=Msqrt_e)
            B = np.hstack([B, Bi])

    elif method == "force_dual_von_tycowicz_combined":
        
        Sigma_Fs_sqrt = np.zeros((X.shape[0]*dim, 0))
        for i, E in enumerate(Es):
            D = normal_force_matrix(X, E)               
            D = D.sum(axis=1).reshape(-1, 1)                    
            Sigma_F_sqrt = D #.sum(axis=1).reshape(-1, 1)
            Sigma_Fs_sqrt = np.hstack([Sigma_Fs_sqrt, Sigma_F_sqrt])
        
        # D = np.abs(D).sum(axis=1).reshape(-1, 1)             
        # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
        Me_sqrt = sp.sparse.kron(Msqrt, sp.sparse.identity(dim))
        B, Dei= force_dual_modes_sqrt(H, Sigma_Fs_sqrt, 12, M_sqrt=Me_sqrt)
        B = von_tycowicz_expansion(B, dim)
        # view_scalar_fields(X, T, D.reshape(-1, 1))
        # W = np.hstack([W, np.ones((X.shape[0], 1))])
        B = orthonormalize(B, Mext, 1e-6)
        
        # view_displacement_modes(X, T, B, period=100)
        
        
                
    elif method == "force_dual_skinning_eigenmodes":
        W = np.zeros((X.shape[0], 0))  
        # Ls = hessian_blocks(H, dim)
        
        L = fold_vector_hessian(H, dim)
        Ls = [L]
        for L in Ls:
            for i, E in enumerate(Es):
                D = normal_force_matrix(X, E)  
                D = fold_in_vector_subspace(D, dim)
                
                # D = np.abs(D).sum(axis=1).reshape(-1, 1)             
                # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
                Wi, Dei= force_dual_modes_sqrt(L, D, 2, M_sqrt=Msqrt)
                W = np.hstack([W, Wi])
        
        # view_scalar_fields(X, T, D.reshape(-1, 1))
        # W = np.hstack([W, np.ones((X.shape[0], 1))])
        B = lbs_jacobian(X, W)
        B = orthonormalize(B, Mext, 1e-6)
      
    elif method == "skinning_eigenmodes":
        W = np.zeros((X.shape[0], 0))  
        # Ls = hessian_blocks(H, dim)
        
        L = fold_vector_hessian(H, dim)
        Ls = [L]
        for L in Ls:
            E,Wi = eigs(L, 24, M=Msqrt)
            W  = np.hstack([W, Wi])    # D = np.abs(D).sum(axis=1).reshape(-1, 1)             

        # view_scalar_fields(X, T, D.reshape(-1, 1))
        # W = np.hstack([W, np.ones((X.shape[0], 1))])
        B = lbs_jacobian(X, W)
        B = orthonormalize(B, Mext, 1e-6)
    
    

    elif method == "force_dual_skinning_eigenmodes_combined":
        
        # L = fold_vector_hessian(H, dim)
        # Ls = hessian_blocks(H, dim)
        L = fold_vector_hessian(H, dim)
        Ls = [L]
        W = np.zeros((X.shape[0], 0))
        for L in Ls:
            Sigma_Fs_sqrt = np.zeros((X.shape[0]*dim, 0))
            for i, E in enumerate(Es):
                D = normal_force_matrix(X, E)               
                D = D.sum(axis=1).reshape(-1, 1)                    
                Sigma_F_sqrt = D #.sum(axis=1).reshape(-1, 1)
                Sigma_Fs_sqrt = np.hstack([Sigma_Fs_sqrt, Sigma_F_sqrt])
            
            Di = fold_in_vector_subspace(Sigma_Fs_sqrt, dim)
        # D = np.abs(D).sum(axis=1).reshape(-1, 1)             
        # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
            Wi, Dei= force_dual_modes_sqrt(L, Di, 6, M_sqrt=Msqrt)
            W = np.hstack([W, Wi])
            
        # view_scalar_fields(X, T, D)
        # view_scalar_fields(X, T, W)
        # W = np.hstack([W, np.ones((X.shape[0], 1))])
        B = lbs_jacobian(X, W)
        # B = np.kron(W, np.identity(dim))
        B = orthonormalize(B, Mext, 1e-6)
    elif method == "skinning_eigenmodes_combined":
        
        # L = fold_vector_hessian(H, dim)
        Ls = hessian_blocks(H, dim)
        L = fold_vector_hessian(H, dim)
        Ls = [L]
        W = np.zeros((X.shape[0], 0))
        for L in Ls:
            
           E,Wi = eigs(L, 6, M=Msqrt)
        # D = np.abs(D).sum(axis=1).reshape(-1, 1)             
        # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
           W = np.hstack([W, Wi])
            
        # view_scalar_fields(X, T, D)
        # view_scalar_fields(X, T, W)
        # W = np.hstack([W, np.ones((X.shape[0], 1))])
        B = lbs_jacobian(X, W)
        # B = np.kron(W, np.identity(dim))
        B = orthonormalize(B, Mext, 1e-6)
    # elif method == "force_dual_skinning_eigenmodes":
    #     W = np.zeros((X.shape[0], 0))  
    #     L = fold_vector_hessian(H, dim)
        
        
    #     for i, E in enumerate(Es):
    #         D = normal_force_matrix(X, E)  
    #         D = np.abs(D).sum(axis=1).reshape(-1, 1)             
    #         D = fold_in_vector_subspace(D, dim)
    #         # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
    #         Wi, Dei= force_dual_modes_sqrt(L, D, m)#, M_sqrt=Msqrt_i)
    #         W = np.hstack([W, Wi])
    #     # W = np.hstack([W, np.ones((X.shape[0], 1))])
    #     B = lbs_jacobian(X, W)
    #     B = orthonormalize(B, Mext, 1e-6)
        
        # view_displacement_modes(X, T, B, period=100)
        # B = von_tycowicz_expansion(B, dim)  
    elif method == "von_tycowicz":
        E, B = eigs(H, k=m, M=Mext)
        #
        
    B = orthonormalize(B, Mext, 1e-6) 
    
    # view_displacement_modes(X, T, B2, period=100)
    # [cI, cW] = spectral_cubature(X, T, B, k=600)
    
    cI = np.arange(T.shape[0])
    vol = volume(X, T)
    cW = vol
        
    return B, cI, cW


directory = os.path.dirname(__file__)


colormap_dir = directory  + "/../../../data/colormaps/"
force_edge_path = directory + "/../../../data/2d/gripper/gripper_force_Es.npy"
data_dir = directory + "/../../../data/2d/gripper/"
mesh_file = data_dir + "gripper.obj"
materials_path = data_dir + "gripper_materials.npy"
pinned_vertices_path = data_dir + "gripper_pinned_vertices.npy"
uv_path = data_dir + "gripper_uv.npy"
texture_path = data_dir + "gripper.png"
k_handle = 1e8
read_subspace_cache = False
read_sim_cache = False

amplitude = 1.0e8
m = 2
k = 100
gamma_pin = 1e8

num_timesteps = 400
period = 100
eye_pos = [0, -0.15, 4.]
look_at = [0, -0.15, 0]
methods = ["skinning_eigenmodes", "force_dual_skinning_eigenmodes"]#, "force_dual_von_tycowicz", "von_tycowicz"]

X, _, _, T, _, _ = igl.readOBJ(mesh_file)
X = X[:, :2]
X = normalize_and_center(X)
F = T

dim = X.shape[1]

pinned = np.load(pinned_vertices_path)
materials = np.load(materials_path)
ym = materials[:, [0]]
pr = materials[:, [1]]
mu, lam =  ympr_to_lame(ym, pr)
H_elastic = linear_elasticity_hessian(X=X, T=T, lam=lam, mu=mu)
H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0], 1e8)
H = H_elastic + H_pin
M = massmatrix(X=X, T=T)


mass = M.diagonal()
Mi = sp.sparse.diags(1.0/M.diagonal())
Msqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
Msqrt_i = sp.sparse.diags(1/Msqrt.diagonal())
Mext = sp.sparse.kron(M, sp.sparse.identity(dim))
Es = np.load(force_edge_path, allow_pickle=True)[1:]
# Es = [ np.vstack(Es[0:6]),  np.vstack(Es[6:12])]
# Es = [ np.vstack(Es)]
# Es 
import polyscope as ps
uv = np.load(uv_path)
texture = np.array(Image.open(texture_path))[:, :, 0:3]/255


for method in methods:
    results_dir = directory + "/results_sim/" + method + "/"
    os.makedirs(results_dir, exist_ok=True)
    cache_path = results_dir + "subspace.npz"
    
    subspace_func = lambda :compute_subspace(method)
    [B, cI, cW] = compute_with_cache_check(subspace_func, results_dir + "subspace.npy", read_subspace_cache)
    
    # view_displacement_modes(X, T, B, period=100)
    # import polyscope as ps
    # ps.show()
    def compute_sim(num_timesteps):
        params = ElasticFEMSimParams(h=1e-2, rho=1e3, ym=ym, pr=pr, material='neo-hookean', Q0=H_pin, b0=0*b_pin)
        params.solver_p.max_iter = 1
        sim = PneumaticActuatorFEMSim(np.vstack(Es), X, T, B=B, cI=cI, cW=cW, p = params, x0=X.reshape(-1, 1))
        
        Z = np.zeros((B.shape[1], num_timesteps))
        z, z_dot = sim.rest_state()
        
        actuation = np.zeros((sim.E.shape[0], 1))
        for i in range(num_timesteps):
            
            # should smoothly go from 0 to ampltide  from 0 to num_timesteps//3
            # should hold amplitude for num_timesteps//3 to 2*num_timesteps//3
            # should smoothly go from amplitude to 0 from 2*num_timesteps//3 to num_timesteps
            if i < num_timesteps//5:
                a = amplitude  * i / (num_timesteps // 5)
            elif i < 2*num_timesteps//5:
                a = amplitude
            elif i < 4*num_timesteps//5:
                a = amplitude  * (1 - ((i - 2*num_timesteps//5) / (2* num_timesteps // 5)))
            elif i < 5*num_timesteps//5:
                a = 0
            # print(a)
            actuation[:] = a
            
            z_next = sim.step(z, z_dot, actuation)
            z_dot = (z_next - z) / sim.p.h
            z = z_next.copy()
            Z[:, [i]] = z
            
        return Z
    
    Z = compute_sim(num_timesteps)
    
    U = B @ Z 

    view_animation(X, T, U, texture=texture, uv=uv,
                   eye_pos=eye_pos, eye_target=look_at,
                   path= results_dir + "/" + method + "_actuation.mp4", material='flat')

            
