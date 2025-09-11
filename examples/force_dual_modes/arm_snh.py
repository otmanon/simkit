import numpy as np
import igl
import polyscope as ps
import scipy as sp
import gpytoolbox as gpt
import igl.triangle
import os 
from PIL import Image

from simkit import deformation_jacobian, massmatrix, volume, ympr_to_lame
from simkit.eigs import eigs
from simkit.emu_energy import emu_energy_F
from simkit.emu_gradient import emu_force_matrix, emu_gradient_dF
from simkit.emu_hessian import emu_hessian_d2F, emu_hessian_d2x
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.low_rank_covariance_folding import low_rank_covariance_folding
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
from simkit.pairwise_displacement import pairwise_displacement
from simkit.polyscope import view_displacement_modes, view_scalar_fields
from simkit.selection_matrix import selection_matrix
from simkit.shape_outlines import rectangle_outline
from simkit.sims.elastic import *
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.grad import grad
from simkit.gravity_force import gravity_force
from simkit.sims.elastic import ElasticROMMFEMSim, ElasticROMMFEMSimParams, ElasticFEMSim, ElasticFEMSimParams
from simkit.solvers.NewtonSolver import NewtonSolver
from simkit.stretch import stretch
from simkit.symmetric_stretch_map import symmetric_stretch_map
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.spectral_cubature import spectral_cubature
from simkit.average_onto_simplex import average_onto_simplex
from simkit.polyscope.view_clusters import view_clusters
from simkit.polyscope.view_cubature import view_cubature
from simkit.pairwise_distance import pairwise_distance
from simkit.umfpack_lu_solve import umfpack_lu_solve
from simkit.volume import volume
from simkit.project_into_subspace import project_into_subspace


# [X, _, _, T, _, _] = igl.read_obj("./data/2d/square/square.obj")
directory = os.path.dirname(__file__)

data_dir = directory + "/../../../data/2d/arm"
[X0, T] = igl.read_triangle_mesh(data_dir + "/arm.obj")
tex_path = data_dir + "/arm.png"
w_muscle = np.load(data_dir + "/arm_w_muscle.npy") 
w_bone_1 = np.load(data_dir + "/arm_w_bone_1.npy") 
w_bone_2 = np.load(data_dir + "/arm_w_bone_2.npy") 
X0 = X0[:, :2]
uv = X0 / [2360, 1640]
X = normalize_and_center(X0)
dim = X.shape[1]

# mode design
m = 4
k = 400

subspace_names = [ "skinning_eigenmodes", "full" ]#. "full"] #"full"
vis_modes = False
save_video = False

# simulation params
rho = 1e-3
ym = 1e6 * np.ones((T.shape[0], 1))
ym[ w_bone_1 > 0.5] = 1e11
ym[w_bone_2 > 0.5] = 1e11
pr = 0.49 * np.ones((T.shape[0], 1))

[mu, lam] = ympr_to_lame(ym, pr)
dt = 1e-2
bg =  - 0*gravity_force(X, T, rho=rho).reshape(-1, 1) # negative sign of force we desire bc the b's are gonna be in rhs of newton's method
eps = 0.1
pI = np.where(X[:, 1] > X[:, 1].max() - eps)[0]
pc0 = (X[pI, :])
[Q_pin, b_pin] = dirichlet_penalty(pI, pc0, X.shape[0],  1e8)
H_elastic = linear_elasticity_hessian(X=X, T=T, mu=mu, lam=lam) 

import polyscope as ps
ps.init()
ps.set_ground_plane_mode("none")
ps.remove_all_structures()
E = igl.boundary_facets(T)[0]
Vs, Es, _, _ = igl.remove_unreferenced(X, E)
mesh = ps.register_surface_mesh("mesh", X, T,  material="flat")
outline = ps.register_curve_network("outline", Vs, Es, radius=0.005, material="flat", color=[0.0, 0.0, 0.0])
ps.show()
ps.load_color_map("oranges2", directory + "/../../../data/colormaps/Oranges_11.png")
mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', cmap='oranges2', enabled=True)
ps.screenshot(directory + "/ym.png")
ps.show()
ps.load_color_map("blues2", directory + "/../../../data/colormaps/Purples_11.png")
mesh.add_scalar_quantity("w_muscle", ( w_muscle.flatten() > 0.1), defined_on='faces', cmap='blues2', enabled=True)
ps.screenshot(directory + "/w_bone_1.png")
# mesh.add_scalar_quantity("w_bone_2", w_bone_2.flatten(), defined_on='faces', cmap='blues', enabled=True)
ps.show()
# control params
amp =  1.5e6 #3e7#1e7 #
period = 100
phase = np.pi
offset = 1

# muscle parameters
mI = np.where(w_muscle > 0.5)[0]
num_tets = T.shape[0]
muscle_direction = np.zeros((num_tets, 2))
muscle_actuation = np.zeros((num_tets, 1)) 
# muscle_actuation[mI] = 1e7
muscle_direction[mI, 0] = 0.5
muscle_direction[mI, 1] = 1.5
muscle_direction[mI, :] = muscle_direction[mI, :] / np.linalg.norm(muscle_direction[mI, :], axis=1)[:, None]
K = emu_force_matrix(X, muscle_direction, vol=volume(X, T), J=deformation_jacobian(X, T)).tocsc()[mI, :]

M = massmatrix(X, T)
M_sqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
M_sqrt_inv = sp.sparse.diags(1.0/M_sqrt.diagonal())
# H_muscle = emu_hessian_d2x(X, muscle_direction, muscle_actuation,  vol=volume(X, T), J=deformation_jacobian(X, T))
for subspace_name in subspace_names:
    H = H_elastic  + Q_pin
    # build subspace from simulation
    if subspace_name =="skinning_eigenmodes":
        L_skinning_eigs = fold_vector_hessian(H, dim)
        [D, W] = eigs(L_skinning_eigs, m, M=massmatrix(X, T))
        view_scalar_fields(X, T, W=-W, dir = directory + "/modes_" + subspace_name + "/", material="flat", colormap_path=directory + "/../../../data/colormaps/RdBu_21.png", outline_width=0.005)
        cI, cW, l = spectral_cubature(X, T, W, k, return_labels=True)

    elif subspace_name  =="force_dual_eigenmodes":
        # fold in covariances for skinning eigenmodes
        L_force_dual =  low_rank_covariance_folding(H, K.T.todense(), dim=2)
        W = np.linalg.svd(M_sqrt @ L_force_dual, full_matrices=False)[0][:, :m]
        W = M_sqrt_inv @ W
        # W = orthonormalize(W, M=massmatrix(X, T))
        view_scalar_fields(X, T, W=-W, dir = directory + "/modes_" + subspace_name + "/", material="flat", colormap_path=directory + "/../../../data/colormaps/RdBu_21.png", outline_width=0.005)
        cI, cW, l = spectral_cubature(X, T, W, k, return_labels=True)

    if subspace_name != "full":
        if vis_modes:
            view_scalar_fields(X, T, W=W, dir = directory + "/modes_" + subspace_name + "/", material="flat", colormap_path=directory + "/../../../data/colormaps/RdBu_21.png", outline_width=0.005)
        W = np.concatenate([W, np.ones((X.shape[0],1))], axis=1)
        B = lbs_jacobian(X, W)
        [cI, cW, labels] = spectral_cubature(X, T, W, k, return_labels=True)
    else:
        B = sp.sparse.identity(X.shape[0]*dim).tocsc()
        cI = np.arange(T.shape[0])
        cW = volume(X, T)
        

    BQB_pin = B.T @ Q_pin @ B
    Bb_pin = B.T @ (b_pin + bg)

    sim_params = ElasticFEMSimParams()
    sim_params.material = "neo-hookean"
    sim_params.ym = ym
    sim_params.pr = pr
    sim_params.h = dt
    sim_params.rho = rho
    sim_params.Q0 = Q_pin
    sim_params.b0 = b_pin + bg
    sim_params.solver_p.max_iter= 3
    sim_params.solver_p.do_line_search = True
    sim = ElasticFEMSim(X, T, B, cI=cI, cW=cW, p=sim_params)

    # initialize sim state
    z = project_into_subspace(X.reshape(-1, 1), B)
    z_dot = np.zeros(z.shape)

    def new_hessian(z):
        F = (sim.el_pre.JB @ z).reshape(-1, 2, 2)   
        Q_muscle = emu_hessian_d2F(F, muscle_direction[sim.cI], muscle_actuation[sim.cI], sim.vol)
        H_muscle = sim.el_pre.JB.T @ sp.sparse.block_diag(Q_muscle) @ sim.el_pre.JB
        H_normal = sim.hessian(z)
        H = H_normal + H_muscle
        return H

    def new_gradient(z):
        F = (sim.el_pre.JB @ z).reshape(-1, 2, 2)   
        PK1 = emu_gradient_dF(F, muscle_direction[sim.cI], muscle_actuation[sim.cI], sim.vol)
        g_muscle = sim.el_pre.JB.T @ PK1.reshape(-1, 1)
        g_normal = sim.gradient(z)
        g = g_normal + g_muscle
        return g

    def new_energy(z):
        F = (sim.el_pre.JB @ z).reshape(-1, 2, 2)   
        E_muscle = emu_energy_F(F, muscle_direction[sim.cI], muscle_actuation[sim.cI], sim.vol)
        E_normal = sim.energy(z)
        E = E_normal + E_muscle 
        return E

    sim.solver = NewtonSolver(new_energy, new_gradient, new_hessian, sim_params.solver_p)
    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")
    mesh = ps.register_surface_mesh("mesh", X, T, edge_width=0, material="flat")
    mesh.add_parameterization_quantity("test_param",  uv,
                                      defined_on='vertices')
    vals = np.array(Image.open(tex_path))
    mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                                defined_on='texture', param_name="test_param",
                                    enabled=True)
    mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', cmap='blues')

    if save_video:
        result_dir = directory + "/sim_" + subspace_name + "/"
        os.makedirs(result_dir, exist_ok=True)


    def sim_many():
        global z, z_dot
        for i in range(20):
            z_next = sim.step(z,  z_dot)
            z_dot = (z_next - z) / sim_params.h    
            z = z_next.copy()
    num_trials = 10  
    import timeit      
    times_sim = np.array(timeit.repeat(sim_many, number=1, repeat=num_trials))
    print("--------------------------------")
    print("subspace: " + subspace_name)
    print("num verts: " + str(X.shape[0]))
    print("num tets: " + str(T.shape[0]))
    print("num sims: " + str(num_trials))
    print("time per sim: " + str(times_sim.mean()/20))
    print("Cubature points: " + str(cI.shape[0]))
    print("B shape: " + str(B.shape))
    
    z_dot = np.zeros(z.shape)
    if subspace_name != "full":
        def reconstruct():
            global z
            u = B @ z
            return u
        times_reconstruct = np.array(timeit.repeat(reconstruct, number=1, repeat=num_trials))
        print("time per reconstruct: " + str(times_reconstruct.mean()))
        total_time = times_reconstruct.mean() + times_sim.mean() / 20
        print("total time: " + str(total_time))
        print("FPS: " + str(1 / total_time))
# for i in range(4*period):
        
    #     muscle_actuation[mI] = amp* (np.cos( 2.0 * np.pi * i / (period) + phase) + offset)
      

    #     z_next = sim.step(z,  z_dot)
    #     z_dot = (z_next - z) / sim_params.h    
    #     z = z_next.copy()
    #     x = B @ z    

    #     mesh.update_vertex_positions(x.reshape(-1, 2))
    #     ps.frame_tick()

    #     if save_video:
    #         ps.screenshot(result_dir + "/" + str(i).zfill(4) + ".png", transparent_bg=True)

    # if save_video:
    #     video_from_image_dir(result_dir, result_dir + "/../sim_" + subspace_name + ".mp4", fps=30)
    #     mp4_to_gif(result_dir + "/../sim_" + subspace_name + ".mp4", result_dir + "/../sim_" + subspace_name + ".gif")
