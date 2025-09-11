from copy import deepcopy
import os
from PIL import Image
import igl
import numpy as np
import scipy as sp

from simkit import common_selections, volume
from simkit.apps.configs import *
from simkit.average_onto_simplex import average_onto_simplex
from simkit.low_rank_covariance_folding import low_rank_covariance_folding
from simkit.pairwise_distance import pairwise_distance
from simkit.blender.render_face_labels import render_face_scalars
from simkit.covariance_sampling import correlation_sampling, covariance_sampling, greedy_maxmin_correlation_sampling
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.equality_matrix import edge_equality_matrix, equality_matrix
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.fold_in_vector_subspace import fold_in_vector_subspace
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_covariances_from_clusters import force_covariances_from_clusters
from simkit.force_dual_modes import force_dual_modes_diagonal, force_dual_modes_sqrt
from simkit.gravity_force import gravity_force
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.modal_moment_fitting import modal_moment_fitting
from simkit.normal_force_matrix import normal_force_matrix
from simkit.normalize_and_center import normalize_and_center
from simkit.normals import normals
from simkit.orthonormalize import orthonormalize
from simkit.pairwise_distance import pairwise_distance
from simkit.polyscope.view_animation import view_animation
from simkit.polyscope.view_clusters import view_clusters
from simkit.polyscope.view_cubature import view_cubature
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.rotation_strain_coordinates import rotation_strain_coordinates
from simkit.simplex_vertex_averaging_matrix import simplex_vertex_averaging_matrix
from simkit.sims.elastic.AdaptiveClickFEMSim import AdaptiveClickFEMSim
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSim, ElasticFEMSimParams
from simkit.sims.elastic.PneumaticActuatorFEMSim import PneumaticActuatorFEMSim
from simkit.sims.elastic.simulate import HistoryCallback, simulate
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.spectral_clustering import spectral_clustering
from simkit.massmatrix import massmatrix
from simkit.spectral_cubature import spectral_cubature
# from simkit.stvk_modal_derivatives import stvk_modal_derivatives
from simkit.substructuring import substructured_skinning_eigenmodes, substructuring
from simkit.worlds.pneumatic_actuator_world import pneumatic_actuator_offline_world, pneumatic_actuator_world
from simkit.worlds.handle_world import handle_world
from simkit.ympr_to_lame import ympr_to_lame
from simkit.limit_actuation_dirichlet_energy import limit_actuation_dirichlet_energy
            

config = GripperConfig()
config.m = 5
config.k = 400
# config.sub_k = 48
# config.mesh_file = config.data_dir + "/" + config.character_name + ".obj"
config.colormap_dir = os.path.dirname(__file__) + "/../../../data/colormaps/"
config.force_edge_path = os.path.dirname(__file__) + "/../../../data/2d/gripper/gripper_force_Es.npy"
# config.texture_path = config.data_dir + "/" + config.character_name + "_stiff_walls_combined.png"
       
config.labels_path = None
config.k_handle = 1e8
config.read_cache = False
config.name = "full"

config1 = deepcopy(config)
config1.name = 'eigs'
config2 = deepcopy(config)
config2.name = 'skinning-eigenmodes'
config3 = deepcopy(config)
config3.name = 'force-dual-modes'
config4 = deepcopy(config)
config4.name = 'force-dual-skinning-eigenmodes'
# config4.name = 'stvk_modal_derivatives'
configs = [config4, config2, config3]

directory = os.path.dirname(__file__)
for config in configs:
    if config.dim == 2:
        X, _, _, T, _, _ = igl.readOBJ(config.mesh_file)
        X = X[:, :2]
        F = T
    else:
        [X, T, F] = igl.read_mesh(config.mesh_file)
        F = igl.boundary_facets(T)
        
    config.result_dir = directory + "/results/" + config.character_name + "_" + config.name + "/"
    config.cache_dir = config.result_dir +  "/cache/" 
    
    config.video_path = config.result_dir + "/" + config.character_name + "_" + \
                config.name + ".mp4"
    X = normalize_and_center(X)
    m = config.m
    k = config.k
    dim = X.shape[1]
    pinned = np.load(config.pinned_vertices_path)
    materials = np.load(config.materials_path)
    ym = materials[:, [0]]
    pr = materials[:, [1]]
    if config.read_cache:
        B = np.load(config.cache_dir + "/B.npy")
        cI = np.load(config.cache_dir + "/cI.npy")
        cW = np.load(config.cache_dir + "/cW.npy")
        Xsim = np.load(config.cache_dir + "/Xsim.npy")
        Tsim = np.load(config.cache_dir + "/Tsim.npy")
        W = np.load(config.cache_dir + "/W.npy")
        
        Es = np.load(config.cache_dir + "/Es.npy")
        
        if config.name == "force-dual-modes":
            sigma_F_list = np.load(config.cache_dir + "/sigma_F_list.npy", allow_pickle=True) 
    else:
        
        mu, lam =  ympr_to_lame(ym, pr)
        H_elastic = linear_elasticity_hessian(X=X, T=T, lam=lam, mu=mu)
        H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0], 1e8)
        H = H_elastic + H_pin
        L = fold_vector_hessian(H, dim, full=True)
        M = massmatrix(X=X, T=T)
        mass = M.diagonal()
        Mi = sp.sparse.diags(1.0/M.diagonal())
        Msqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
        Msqrt_i = sp.sparse.diags(1/Msqrt.diagonal())
        Mext = sp.sparse.kron(M, sp.sparse.identity(dim))
        Es = np.load(config.force_edge_path, allow_pickle=True)[1:]
        # Es = [ np.vstack(Es[:6]), np.vstack(Es[6:])]
        # Es = [ np.vstack(Es)]
        amplitude = 1e8
        if config.name == "force-dual-modes":
            
            B = np.zeros((X.shape[0]*X.shape[1], 0))  
            for i, E in enumerate(Es):
                D = normal_force_matrix(X, E)               
                D = D.sum(axis=1).reshape(-1, 1)
                     
                sigma_F_sqrt = D.sum(axis=1).reshape(-1, 1)
                Msqrt_ie = sp.sparse.kron(Msqrt_i, sp.sparse.identity(dim))
                Bi, Di= force_dual_modes_sqrt(H, sigma_F_sqrt, m, M_sqrt=Msqrt_ie)
                
                B = np.hstack([B, Bi])
            
            
            
            # B = orthonormalize(B, Mext, 1e-6)
            W = fold_in_vector_subspace(B, dim)
            # cI, cW = spectral_cubature(X, T, W, k)
            cI = np.arange(T.shape[0])
            vol = volume(X, T)
            cW = vol
            
            from scipy.io import savemat


            # Save the array to a .mat file
            savemat(directory + "/B.mat", {"B": B})
            savemat(directory + "/X.mat", {"X": X})
            savemat(directory + "/T.mat", {"T": T})
            savemat(directory + "/ym.mat", {"ym": ym})

            a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
            # view_displacement_modes(X, T, B @ np.diag(a), a=1.0, path = config.result_dir + "/force_dual_modes.mp4")
        if config.name == 'force-dual-skinning-eigenmodes':
            W = np.zeros((X.shape[0], 0))  
            for i, E in enumerate(Es):
                D = normal_force_matrix(X, E)  
                D = np.abs(D).sum(axis=1).reshape(-1, 1)             
                # D = fold_in_vector_subspace(D, dim)
                
                L_force_dual = low_rank_covariance_folding(H, D, dim=2)
                Wi = np.linalg.svd(L_force_dual, full_matrices=False)[0][:, :m]
                # D = np.linalg.norm(D, axis=1).reshape(-1, 1)
                # Wi, Dei= force_dual_modes_sqrt(L, D, m)#, M_sqrt=Msqrt_i)
                W = np.hstack([W, Wi])
            # W = np.hstack([W, np.ones((X.shape[0], 1))])
            B = lbs_jacobian(X, W)
            B = orthonormalize(B, Mext, 1e-6)
            # cI, cW = spectral_cubature(X, T, W, k)
            cI = np.arange(T.shape[0])
            vol = volume(X, T)
            cW = vol
        
            # view_scalar_fields(X, T, W,dir = config.result_dir + "/force_dual_skinning_eigenmodes/")
        if config.name == "eigs":
            D, B = eigs(H, m, M=Mext)
            W = fold_in_vector_subspace(B, dim)
            cI, cW = spectral_cubature(X, T, B, k)
            a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
            # view_displacement_modes(X, T, B @ np.diag(a), a=1.0, path = config.result_dir + "/eigs.mp4")
        if config.name == "skinning-eigenmodes":
            D, W = eigs(L, m, M=M)
            B = lbs_jacobian(X, W)
            B = orthonormalize(B, Mext, 1e-6)
            # cI, cW = spectral_cubature(X, T, W, k)
            cI = np.arange(T.shape[0])
            vol = volume(X, T)
            cW = vol
            # view_scalar_fields(X, T, W,dir = config.result_dir + "/skinning_eigs/")
        if config.name == "full":
            amplitude = 4e7
            B = sp.sparse.identity(H.shape[0])
            W = sp.sparse.identity(X.shape[0])
            cI = np.arange(T.shape[0])
            vol =  volume(X, T)
            cW = vol
        if config.name == "stvk_modal_derivatives":
            B = stvk_modal_derivatives(X, T, 3)
            cI = np.arange(T.shape[0])
            vol =  volume(X, T)
            cW = vol
        
        
        Xsim , Tsim = X, T
        # B = lbs_jacobian(Xsim, W)
        # B = orthonormalize(B, Me, 1e-6)
        
        # [cI2] =greedy_maxmin_correlation_sampling(H, k*sub_k)
        # save it 
        os.makedirs(config.cache_dir, exist_ok=True)
        np.save(config.cache_dir + "/B.npy", B)
        # np.save(config.cache_dir + "/W.npy", W)
        np.save(config.cache_dir + "/cI.npy", cI)
        np.save(config.cache_dir + "/cW.npy", cW)
        np.save(config.cache_dir + "/Xsim.npy", Xsim)
        np.save(config.cache_dir + "/Tsim.npy", Tsim)
        np.save(config.cache_dir + "/Es.npy", np.array(Es, dtype=object))
        if config.name == "force-dual-modes":
            np.save(config.cache_dir + "/sigma_F_sqrt.npy", sigma_F_sqrt)
        
    H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0], 1e8)
    g = gravity_force(Xsim, Tsim, -0, rho=1e0).reshape(-1, 1)
    b = -g + b_pin
    sim_params = ElasticFEMSimParams(rho=1e0, ym=ym, pr=pr, 
                                    material='neo-hookean', Q0=H_pin , b0=b)
    sim_params.solver_p.max_iter = 1
    x0 = Xsim.reshape(-1, 1)
        
    if config.texture_path is not None and config.uv_path is not None:
            uv = np.load(config.uv_path)
            texture = np.array(Image.open(config.texture_path))[:, :, 0:3]/255
    
    
    E = np.vstack(Es)
    import polyscope as ps
    sim = PneumaticActuatorFEMSim(E,  Xsim, Tsim, B=B, cI=cI, cW=cW, x0 = x0, p=sim_params)
 
    max_steps = 300

    def actuation_func(count):
        offset = max_steps / 4
        start_time = max_steps / 2 - offset
        end_time = max_steps / 2 + offset
        pause = ( count > start_time) * ( count < end_time)
        if not pause:
            a = amplitude * ((np.cos((count * 2 * np.pi)/ max_steps + np.pi)) + 1)*0.5
        else:
            a = amplitude * ((np.cos((start_time * 2 * np.pi)/ max_steps + np.pi)) + 1)*0.5
        a = a * np.ones((sim.BD.shape[1], 1))
        return a
    
    do_rs = True
    x, rs_pre = rotation_strain_coordinates(Xsim, Tsim, Xsim.reshape(-1, 1), pinned=pinned, return_pre=True)
    pneumatic_actuator_offline_world(Xsim, Tsim, sim, uv=uv, texture=texture, max_steps=300,
                             actuation_func=actuation_func, do_rs=False, rs_pre=rs_pre)#, path=config.result_dir + "/video.mp4")

  