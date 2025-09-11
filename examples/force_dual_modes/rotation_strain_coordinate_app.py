from copy import deepcopy
import os
from PIL import Image
import igl
import numpy as np
import scipy as sp

from simkit import common_selections, volume
from simkit.apps.configs import *
from simkit.average_onto_simplex import average_onto_simplex
from simkit.blender.render_mesh import render_mesh
from simkit.blender.render_vector_field_on_mesh import render_vector_field_on_mesh
from simkit.deformation_jacobian import deformation_jacobian
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.TriangleMesh import TriangleMesh
from simkit.matplotlib.VectorField import VectorField
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
config.m = 4
config.k = 400

config.colormap_dir = os.path.dirname(__file__) + "/../../../data/colormaps/"
config.force_edge_path = os.path.dirname(__file__) + "/../../../data/2d/gripper/gripper_force_Es.npy"

    
config.labels_path = None
config.k_handle = 1e8
config.read_cache = False   
config.name = "full"

video_from_image_dir
config1 = deepcopy(config)
config1.actuation_max = -2
config1.name = 'eigs'

config2 = deepcopy(config)
config2.name = 'force-dual-modes-combined'
config2.actuation_max = 10
configs = [ config1,  config1]
directory = os.path.dirname(__file__)
for config in configs:
    if config.dim == 2:
        X, _, _, T, _, _ = igl.readOBJ(config.mesh_file)
        X = X[:, :2]
        F = T
    else:
        [X, T, F] = igl.read(config.mesh_file)
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
    Es = np.load(config.force_edge_path, allow_pickle=True)[1:]
    # Es = [ np.vstack(Es[:6]), np.vstack(Es[6:])]
    # Es = [ np.vstack(Es)]
    amplitude = 1e8
    if config.name == "force-dual-modes":
        
        B = np.zeros((X.shape[0]*X.shape[1], 0))  
        
        for i, E in enumerate(Es):
            D = normal_force_matrix(X, E)               
            D = D.sum(axis=1).reshape(-1, 1)                    
            Sigma_F_sqrt = D.sum(axis=1).reshape(-1, 1)
            
            
        Msqrt_ie = sp.sparse.kron(Msqrt_i, sp.sparse.identity(dim))
        Bi, Di= force_dual_modes_sqrt(H, Sigma_F_sqrt, m, M_sqrt=Msqrt_ie)
        B = np.hstack([B, Bi])
        
        # Ee = Es[0]
        # bc = average_onto_simplex(X, Ee)
        dd = Sigma_F_sqrt.reshape(-1, dim)
        dnorm = np.linalg.norm(dd, axis=1)
        dd = dd[dnorm > 1e-6]
        xx = X[dnorm > 1e-6]
        
        dd = dd / dnorm[dnorm>1e-6, None]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 12))
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.6, 1.2)
        TriangleMesh(X, T, outlinewidth=2)
        # PointCloud(xx)
        
        scale2 = 0.25
        VectorField(xx[::5], dd[::5],color=[50/255, 50/255, 179/255, 1], scale=75, width=0.002)
        # plt.show()
        plt.tight_layout()
        plt.savefig(config.result_dir + '/force_prior.png', dpi=300)
        # plt.show()
        
        # render_vector_field_on_mesh(X, T, xx, dd, path=config.result_dir + "/force_prior.png")
        W = fold_in_vector_subspace(B, dim)
        cI = np.arange(T.shape[0])
        vol = volume(X, T)
        cW = vol
        a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
        # view_displacement_modes(X, T, B @ np.diag(a), a=1.0, path = config.result_dir + "/force_dual_modes.mp4")
    if config.name == "force-dual-modes-combined":
        B = np.zeros((X.shape[0]*X.shape[1], 0))  
        
        Sigma_Fs_sqrt = np.zeros((X.shape[0]*dim, 0))
        for i, E in enumerate(Es):
            D = normal_force_matrix(X, E)               
            D = D.sum(axis=1).reshape(-1, 1)                    
            Sigma_F_sqrt = D #.sum(axis=1).reshape(-1, 1)
            Sigma_Fs_sqrt = np.hstack([Sigma_Fs_sqrt, Sigma_F_sqrt])

        Msqrt_ie = sp.sparse.kron(Msqrt_i, sp.sparse.identity(dim))
        B, Di= force_dual_modes_sqrt(H, Sigma_Fs_sqrt, m, M_sqrt=Msqrt_ie)
        # B = np.hstack([B, Bi])
        
        a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
        view_displacement_modes(X, T, B @ np.diag(a), a=1.0)#, path = config.result_dir + "/force_dual_modes.mp4")
        # Ee = Es[0]
        # bc = average_onto_simplex(X, Ee)
        dd = Sigma_F_sqrt.reshape(-1, dim)
        dnorm = np.linalg.norm(dd, axis=1)
        dd = dd[dnorm > 1e-6]
        xx = X[dnorm > 1e-6]
        
        dd = dd / dnorm[dnorm>1e-6, None]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 12))
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.6, 1.2)
        TriangleMesh(X, T, outlinewidth=2)
        # PointCloud(xx)
        
        scale2 = 0.25
        VectorField(xx[::5], dd[::5],color=[50/255, 50/255, 179/255, 1], scale=75, width=0.002)
        # plt.show()
        plt.tight_layout()
        plt.savefig(config.result_dir + '/force_prior.png', dpi=300)
        plt.show()
        
        # render_vector_field_on_mesh(X, T, xx, dd, path=config.result_dir + "/force_prior.png")
        W = fold_in_vector_subspace(B, dim)
        cI = np.arange(T.shape[0])
        vol = volume(X, T)
        cW = vol
        a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
        # view_displacement_modes(X, T, B @ np.diag(a), a=1.0, path = config.result_dir + "/force_dual_modes.mp4")

    if config.name == "eigs":
        D, B = eigs(H, m, M=Mext)
        W = fold_in_vector_subspace(B, dim)
        cI, cW = spectral_cubature(X, T, B, k)
        a = limit_actuation_dirichlet_energy(X, T, B, 2.0)
        
        dd = np.random.randn(X.shape[0], dim)
        dnorm = np.linalg.norm(dd, axis=1)
        dd = dd / dnorm[:, None]
        xx = X.copy()
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 12))
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.6, 1.2)
        TriangleMesh(X, T, outlinewidth=2, facecolors=[224/255, 162/255, 166/255, 1], edgecolors=[224/255, 162/255, 166/255, 1])
        # PointCloud(xx)
        
        scale2 = 0.25
        VectorField(xx[::8], dd[::8],color=[237/255, 28/255, 36/255, 1], scale=75, width=0.002)
        # plt.show()
        plt.tight_layout()
        plt.savefig(config.result_dir + '/force_prior_eigs.png', dpi=300)
        
        # view_displacement_modes(X, T, B @ np.diag(a), a=1.0, path = config.result_dir + "/eigs.mp4")

    modes = [0, 1, 2, 3]
    for mode in modes:
            
        mode = 0
        num_steps = 300
        period = 100
        amplitude = -limit_actuation_dirichlet_energy(X, T, B, config.actuation_max)[mode]
    

        import polyscope as ps
        uv = np.load(config.uv_path)
        texture = np.array(Image.open(config.texture_path))[:, :, 0:3]/255
            
            
        U = np.zeros((X.shape[0]*dim, num_steps))
        U_rs = np.zeros((X.shape[0]*dim, num_steps))
        for i in range(num_steps):  
            a = amplitude * (np.sin(2* i/period * np.pi - np.pi/2) + 1.0) * 0.5
            u = B[:, [mode]] @ np.array(a).reshape(-1, 1)
            U[:, i] = u.flatten()
            
        pre = None
        for i in range(num_steps):
            a = amplitude * (np.sin( 2 * i/period * np.pi - np.pi/2) + 1.0) * 0.5
            u = B[:, [mode]] @ np.array(a).reshape(-1, 1)
            u_rs, pre= rotation_strain_coordinates(X, T, u, pinned=pinned, pre=pre, return_pre=True)
            U_rs[:, i] = u_rs.flatten()

        
        # view_displacement_modes(X, T, U[:, 0:1], a=1.0, path = config.result_dir + "/eigs.mp4")
        eye_pos = [0, -0.15, 4.]
        look_at = [0, -0.15, 0]

        # view_animation(X, T, U_rs, texture=texture, uv=uv,# path= config.result_dir + "/mode_rs_actuation.mp4",
        # eye_pos=eye_pos, eye_target=look_at, material='flat')
        
        
        # view_animation(X, T, U, texture=texture, uv=uv,# path= config.result_dir + "/mode_actuation.mp4", 
        # eye_pos=eye_pos, eye_target=look_at, material='flat')
    
    # ps.init()
    # ps.remove_all_structures()
    # ps.set_ground_plane_mode("none")
    # mesh = ps.register_surface_mesh("mesh", X + u.reshape(-1, dim) , T)
    # mesh.add_parameterization_quantity("test_param",  uv,
    #                             defined_on='vertices')
    # mesh.add_color_quantity("test_vals", texture,
    #                             defined_on='texture', param_name="test_param",
    #                                 enabled=True)
    
    # mesh2 = ps.register_surface_mesh("mesh2", X + u_rs, T)
    # mesh2.add_parameterization_quantity("test_param",  uv,
    #                                 defined_on='vertices')
    # mesh2.add_color_quantity("test_vals", texture,
    #                             defined_on='texture', param_name="test_param",
    #                                 enabled=True)
    # # ps.show()
    # # amplitude = -2.5e5


    
    # ps.show()

    
    