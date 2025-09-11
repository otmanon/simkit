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
from simkit.covariance_sampling import correlation_sampling, covariance_sampling, greedy_maxmin_correlation_sampling
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.equality_matrix import edge_equality_matrix, equality_matrix
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_covariances_from_clusters import force_covariances_from_clusters
from simkit.force_dual_modes import force_dual_modes_diagonal
from simkit.gravity_force import gravity_force
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.modal_moment_fitting import modal_moment_fitting
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
from simkit.ympr_to_lame import ympr_to_lame


def simulate(sim,vI, bc_func, num_timesteps=240):
    z, z_dot = sim.rest_state()
    
    sim.add_handle(vI, bc_func(0))
    Zs =np.zeros((sim.B.shape[1], num_timesteps))
    for i in range(num_timesteps):
        if sim.num_handles > 0 :
            sim.update_handle_position(0, bc_func(i))

        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z)/ sim.p.h
        z = z_next.copy()
        Zs[:, i] = z.flatten()

    return Zs
    
    
#     return Z

directory = os.path.dirname(__file__)

character_name = "chicken_wing"

data_dir = directory + "/../../../data/2d/" + character_name + "/"

m =4 
k = 200
character_name = "chicken_wing"
colormap_dir = os.path.dirname(__file__) + "/../../../data/colormaps/"

mode_colormap = colormap_dir + "RdBu_21.png"
load_colormap = colormap_dir + "Purples_21.png"
mesh_file = data_dir + "/chicken_wing.obj"
materials_path = data_dir + "/chicken_wing_materials.npy"
pinned_vertices_path = data_dir + "/chicken_wing_pinned_vertices.npy"
distribution_path = data_dir + "/chicken_wing_distribution.npy"
texture_path = data_dir + "/chicken_wing.png"
uv_path = data_dir + "/chicken_wing_uv.npy"

labels_path =  None #config.data_dir + "/jester_labels.npy"
k_handle = 1e14
read_subspace_cache = True

dim = 2
X, _, _, T, _, _ = igl.readOBJ(mesh_file)
X = X[:, :dim]
X = normalize_and_center(X)
F = T
materials = np.load(materials_path)
ym = materials[:, [0]]
pr = materials[:, [1]]
pinned = np.load(pinned_vertices_path)
distribution = np.load(distribution_path)
distribution = np.clip(distribution, 0.01, 1)
mu, lam =  ympr_to_lame(ym, pr)
H_elastic = linear_elasticity_hessian(X=X, T=T, lam=lam, mu=mu)
H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0],  1)
H = H_elastic + H_pin
L = fold_vector_hessian(H, dim)
M = massmatrix(X=X, T=T)
H_pin, b_pin = dirichlet_penalty(pinned, X[pinned]*0, X.shape[0], k_handle)

# distributions = np.load(data_dir + "/distributions.npy")
Mi = sp.sparse.diags(1.0/M.diagonal())
Msqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
Me = sp.sparse.kron(M, sp.sparse.identity(dim))
methods = ["force-dual-modes", "direct",   "eigs"  ,   "bbw" , "bh" ]


chicken_wing_dance = True


def compute_subspace(method, m, k):
    
    if method == "direct":
        W = distribution
        
    if method == "force-dual-modes":
        
        Ws = []
        
        for i in range(distribution.shape[1]):
                
            Sigma_F = sp.sparse.diags((distribution[:, i]).flatten())
            Sigma_Fe = sp.sparse.kron(Sigma_F, sp.sparse.identity(dim))
            Sigma_Fe_inv = sp.sparse.diags(1.0/Sigma_Fe.diagonal())
            
            Q = (H + H_pin).T @ Sigma_Fe_inv @ (H + H_pin)
        
            L = fold_vector_hessian(Q, dim)
            E, Wi = eigs(L, k=m//2, M=M) 
            
            
            Ws.append(Wi)
        W = np.hstack(Ws)
        # view_scalar_fields(X, T, distribution, normalize=False, colormap_dir=load_colormap,  dir=directory + "/results/" + character_name + "_" + method + "_load", material="flat", outline_width=0.005)
    if method == "eigs":
        

        Q = (H + H_pin)

        L = fold_vector_hessian(Q, dim)
        E, W = eigs(L, k=m, M=M) 
        
        # Wt = average_onto_simplex(W, T)
        # labels_initial = spectral_clustering(Wt, 3)[0]
        # import polyscope as ps
        # ps.init()
        # mesh = ps.register_surface_mesh("initial_labels", X, T)
        # mesh.add_scalar_quantity("labels", labels_initial, enabled=True, defined_on='faces', cmap='rainbow')
        # ps.show()
        # sigma_F_list = force_covariances_from_clusters(X, T, labels_initial, var= 1e4)#config.k_handle)#config.k_handle) 
        # sigma_F_list = [sigma_F_list[0], sigma_F_list[1]]
        # D, W= force_dual_modes_diagonal(L, sigma_F_list, m, M=Mi , return_components=False)
        # cI, cW = spectral_cubature(Xsim, Tsim, Wcomp, k * sub_m * (dim)*(dim + 1))
    if method == "bbw":
        # bI = farthest_point_sampling(X, m)
        bI = np.array([3855, 3982,  4004, 3526,  3870])
        W = igl.bbw(X, T, bI, bc=np.identity(bI.shape[0]), partition_unity=True)
        # cI, cW = spectral_cubature(Xsim, Tsim, W, k * sub_m * (dim)*(dim + 1), D=1/D)
    if method == "bh":
        # bI = farthest_point_sampling(X, m)
        bI = np.array([3855, 3982,  4004, 3526,  3870])
        W = igl.biharmonic_coordinates(np.append(X, np.zeros((X.shape[0], 1)), axis=1), T, bI.reshape(-1, 1).tolist(),k=2)
        # cI, cW = spectral_cubature(Xsim, Tsim, W, k * sub_m * (dim)*(dim + 1), D=1/D)

    B = lbs_jacobian(X, W)
    Wt = average_onto_simplex(X, T)
    # [cI, cW, labels] = spectral_cubature(X, T, W, k, return_labels=True)
    
    cI =np.arange(T.shape[0])
    cW = np.ones((T.shape[0], 1))
    labels = np.ones((T.shape[0], 1))
    B = orthonormalize(B, Me, 1e-6)
    
    view_scalar_fields(X, T, W, colormap_dir=mode_colormap,  dir=directory + "/results/" + character_name + "_" + method + "_subspace/", material="flat",  outline_width=0.005)
    return B, W, labels, cI, cW

for method in methods:
    result_dir = directory + "/results/" + character_name + "_" + method + "/"

    subspace_cache_dir = result_dir +  "/cache/" 
    dim = X.shape[1]


    [B, W, labels, cI, cW] = compute_subspace(method, m, k)

    # view_scalar_fields(X, T, W)
    sim_params = ElasticFEMSimParams(rho=1e3, ym=ym, pr=pr, 
                                    material='neo-hookean', Q0=H_pin , b0=b_pin)
    sim_params.solver_p.max_iter = 2
    x0 = X.reshape(-1, 1)
        
    if texture_path is not None and uv_path is not None:
            uv = np.load(uv_path)
            texture = np.array(Image.open(texture_path))[:, :, 0:3]/255

    sim = AdaptiveClickFEMSim(k_handle, X, T, B=B, cI=cI, cW=cW, x0 = x0, p=sim_params)


    vI = [3950]
    
    start_pos = X[vI]
    end_pos = X[vI] + np.array([-0.5, -0.5])
    num_timesteps = 180
    def bc_func(i):
        if i < num_timesteps/2:
            return start_pos + (end_pos - start_pos) * i / ( num_timesteps / 2)
        else:
            return end_pos + (start_pos - end_pos) * (i - num_timesteps/2) / (num_timesteps / 2)
    
    Zs = simulate( sim,vI, bc_func, num_timesteps=num_timesteps)
    
    Ps = np.zeros((dim, num_timesteps))
    for i in range(num_timesteps):
        Ps[:, i] = bc_func(i)
        
    Us = B @ Zs
    # view_animation( X, T, Us, Ps=Ps,texture=texture, uv=uv, material='flat',  path=directory + "/results/" + character_name + "_" + method + "_animation.mp4", radius=0.01)
    # offline_app = AdaptiveSubspaceApp(config)
    # import polyscope as ps
    # ps.init()
    # mesh = ps.register_surface_mesh(character_name, X, T)
    # mesh.add_scalar_quantity("labels", labels, enabled=True, defined_on='faces', cmap='rainbow')
    # pc = ps.register_point_cloud("cubature", average_onto_simplex(X, T[cI]), radius=0.01)
    # ps.show()
    # handle_world(X, T, sim, uv=uv, texture=texture, material="flat")

