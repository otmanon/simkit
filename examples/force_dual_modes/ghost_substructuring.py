from copy import deepcopy
import os
from PIL import Image
import igl
import numpy as np
import scipy as sp



from simkit import common_selections
from simkit.apps.configs import *
from simkit.average_onto_simplex import average_onto_simplex
from simkit.blender.render_face_labels import render_face_scalars
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.equality_matrix import edge_equality_matrix, equality_matrix
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_covariances_from_clusters import force_covariances_from_clusters
from simkit.force_dual_modes import force_dual_modes_diagonal
from simkit.gravity_force import gravity_force
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
from simkit.pairwise_distance import pairwise_distance
from simkit.polyscope.view_animation import view_animation
from simkit.polyscope.view_clusters import view_clusters
from simkit.polyscope.view_cubature import view_cubature
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.sims.elastic.AdaptiveClickFEMSim import AdaptiveClickFEMSim
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSim, ElasticFEMSimParams
from simkit.sims.elastic.simulate import HistoryCallback, simulate
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.spectral_clustering import spectral_clustering
from simkit.massmatrix import massmatrix
from simkit.spectral_cubature import spectral_cubature
from simkit.substructuring import substructured_skinning_eigenmodes, substructuring
from simkit.worlds.handle_world import handle_world

config = JesterConfig()
config.num_substructures = 4
config.sub_m = 4
config.sub_k = 50
config.colormap_dir = os.path.dirname(__file__) + "/../../../data/colormaps/"

config.num_timesteps = 300
config.handle_pos = np.array([0.8, 0.18, 0.0])
config.actuation_dir = np.array([0.05, 1, 0])
config.actuation_amplitude = 0.8
config.actuation_period = 100

config.k_handle = 1e8


config.read_cache = False
config1 = deepcopy(config)
config1.name = "substructuring"
config1.k_coupling = 1e8

config2 = deepcopy(config1)
config2.k_coupling = 1e5


config3 = deepcopy(config)
config3.name = "force-dual-modes" 
config3.read_cache = True
configs = [config3] #config2, config3]

directory = os.path.dirname(__file__)
for config in configs:
    if config.dim == 2:
        X, _, _, T, _, _ = igl.read_obj(config.mesh_file)
        X = X[:, :2]
        F = T
    else:
        [X, T, F] = igl.read_mesh(config.mesh_file)
        F = igl.boundary_facets(T)
        
    config.result_dir = directory + "/results/" + config.character_name + "_" + config.name + "/"
    config.cache_dir = config.result_dir +  "/cache/" 
    
    if config.name == "substructuring":
        config.video_path = config.result_dir + "/" + config.character_name + "_" + \
                    config.name + "_" + '{:.2E}'.format(config.k_coupling) + ".mp4"
    elif config.name == "force-dual-modes":
        config.video_path = config.result_dir + "/" + config.character_name + "_" + \
                    config.name + ".mp4"
    X = normalize_and_center(X)

    sub_m = config.sub_m
    sub_k = config.sub_k
    k = config.num_substructures
    dim = X.shape[1]

    if config.read_cache:
        B = np.load(config.cache_dir + "/B.npy")
        cI = np.load(config.cache_dir + "/cI.npy")
        cW = np.load(config.cache_dir + "/cW.npy")
        labels = np.load(config.cache_dir + "/labels.npy")
        Xsim = np.load(config.cache_dir + "/Xsim.npy")
        Tsim = np.load(config.cache_dir + "/Tsim.npy")
        
        if config.name == "force-dual-modes":
            sigma_F_list = np.load(config.cache_dir + "/sigma_F_list.npy", allow_pickle=True) 
        if config.name == "substructuring":
            E = np.load(config.cache_dir + "/E.npy")
            forward_map = np.load(config.cache_dir + "/forward_map.npy")
            inverse_map = np.load(config.cache_dir + "/inverse_map.npy")
    else:
        H_elastic = linear_elasticity_hessian(X=X, T=T)
        # pinned = np.load(config.pinned_vertices_path)
        pinned = common_selections.center_indices(X, 0.15)[1]
        H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0], 1e8)
        H = H_elastic + H_pin
        L = fold_vector_hessian(H, dim)
        M = massmatrix(X=X, T=T)
        Mi = sp.sparse.diags(1.0/M.diagonal())
        Msqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
        [D, W] = eigs(L, config.num_substructures, M=M)
        Wt = average_onto_simplex(W, T)
        labels, means = spectral_clustering(Wt, config.num_substructures)    
        # build clusters
        if config.name == "force-dual-modes":
            sigma_F_list = force_covariances_from_clusters(X, T, labels)
            D, W = force_dual_modes_diagonal(L, sigma_F_list, sub_m, M=M )
            Xsim , Tsim = X, T
        elif config.name == "substructuring":
            [Xsim, Tsim, E, forward_map, inverse_map, Xcomp, Tcomp] = substructuring(X, T, labels, return_components=True)
            labelsexp = np.zeros(Tsim.shape[0])
            ind = 0
            for i in range(k):
                labelsexp[ind : ind + Tcomp[i].shape[0]] = i
                ind += Tcomp[i].shape[0]
            labels = labelsexp
 
            W = substructured_skinning_eigenmodes(Xcomp, Tcomp, sub_m, Xsim.shape[0])
            
        B = lbs_jacobian(Xsim, W)
        M = massmatrix(Xsim, Tsim)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        B = orthonormalize(B, Me, 1e-5)
        cI, cW = spectral_cubature(Xsim, Tsim, W, k * config.sub_k * (dim)*(dim + 1) )

        # save it 
        os.makedirs(config.cache_dir, exist_ok=True)
        np.save(config.cache_dir + "/B.npy", B)
        np.save(config.cache_dir + "/cI.npy", cI)
        np.save(config.cache_dir + "/cW.npy", cW)
        np.save(config.cache_dir + "/labels.npy", labels)
        np.save(config.cache_dir + "/Xsim.npy", Xsim)
        np.save(config.cache_dir + "/Tsim.npy", Tsim)
        if config.name == "force-dual-modes":
            np.save(config.cache_dir + "/sigma_F_list.npy", sigma_F_list)
        if config.name == "substructuring":
            np.save(config.cache_dir + "/E.npy", E)
            np.save(config.cache_dir + "/forward_map.npy", forward_map)
            np.save(config.cache_dir + "/inverse_map.npy", inverse_map)
        
        
    # view_cubature(X, T, cI, cW)
    pinned = common_selections.center_indices(Xsim, 0.15)[1]
    H_pin, b_pin = dirichlet_penalty(pinned, Xsim[pinned]*0, Xsim.shape[0], 1e8)
    
    if config.name == "substructuring":
        AE = edge_equality_matrix(E, Xsim.shape[0])
        H_couple = sp.sparse.kron(AE.T @ AE * config.k_coupling, sp.sparse.identity(dim))
    else:
        H_couple = sp.sparse.csc_matrix((Xsim.shape[0]*dim, Xsim.shape[0]*dim))
    g = gravity_force(Xsim, Tsim, -0, rho=1e3).reshape(-1, 1)
    b = -g + b_pin
    sim_params = ElasticFEMSimParams(rho=1e3, ym=5e5, pr=0.45, 
                                    material='neo-hookean', Q0=H_pin + H_couple, b0=b)
    sim_params.solver_p.max_iter = 1
    x0 = Xsim.reshape(-1, 1)
        
    if config.texture_path is not None and config.uv_path is not None:
            uv = np.load(config.uv_path)
            texture = np.array(Image.open(config.texture_path))[:, :, 0:3]/255
            
            if config.name == "substructuring":
                uv = uv[forward_map]
            

    
    sim = AdaptiveClickFEMSim(config.k_handle, Xsim, Tsim, B=B, cI=cI, cW=cW, x0 = x0, p=sim_params)
    handle_world(Xsim, Tsim, sim, uv=uv, texture=texture)

