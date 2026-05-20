import numpy as np
import igl
import polyscope as ps
import scipy as sp
from examples.force_dual_modes.force_dual_modes import force_dual_modes_sqrt
from simkit.limit_actuation_dirichlet_energy import limit_actuation_dirichlet_energy
from umfpack_lu_solve import umfpack_lu_solve
import igl.triangle
import os 
from PIL import Image

from simkit import deformation_jacobian, massmatrix, volume, ympr_to_lame
from simkit.eigs import eigs

from simkit.energies import *

import simkit as sk

# [X, _, _, T, _, _] = igl.read_obj("./data/2d/square/square.obj")
directory = os.path.dirname(__file__)
name = "arm"
data_dir = directory + "/../../data/2d/" + name + "/"
colormap_dir = directory + "/../../data/colormaps/"
[X0, T] = igl.read_triangle_mesh(data_dir + "/" + name + ".obj")

tex_path = data_dir + "/arm.png"
w_muscle = np.load(data_dir + "/arm_w_muscle.npy") 
w_bone_1 = np.load(data_dir + "/arm_w_bone_1.npy") 
w_bone_2 = np.load(data_dir + "/arm_w_bone_2.npy") 
X0 = X0[:, :2]
uv = X0 / [2360, 1640]
X = sk.normalize_and_center(X0)
dim = X.shape[1]

# mode design
m = 4

subspace_names = ["fdm-lma",  "lma"]
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
eps = 0.1
pI = np.where(X[:, 1] > X[:, 1].max() - eps)[0]
pc0 = (X[pI, :])
[Q_pin, b_pin] = sk.dirichlet_penalty(pI, pc0, X.shape[0],  1e8)
H_elastic = linear_elasticity_hessian(X=X, T=T, mu=mu, lam=lam) 
H = H_elastic  + Q_pin


# import polyscope as ps
# ps.init()
# ps.set_ground_plane_mode("none")
# ps.remove_all_structures()
# E = igl.boundary_facets(T)[0]
# Vs, Es, _, _ = igl.remove_unreferenced(X, E)
# mesh = ps.register_surface_mesh("mesh", X, T,  material="flat")
# outline = ps.register_curve_network("outline", Vs, Es, radius=0.005, material="flat", color=[0.0, 0.0, 0.0])
# ps.show()
# ps.load_color_map("oranges2", colormap_dir + "Oranges_11.png")
# mesh.add_scalar_quantity("ym", ym.flatten(), defined_on='faces', cmap='oranges2', enabled=True)
# # ps.screenshot(directory + "/ym.png")
# ps.show()
# ps.load_color_map("blues2", colormap_dir + "Purples_11.png")
# mesh.add_scalar_quantity("w_muscle", ( w_muscle.flatten() > 0.1), defined_on='faces', cmap='blues2', enabled=True)
# # ps.screenshot(directory + "/w_bone_1.png")
# ps.show()


# muscle parameters
mI = np.where(w_muscle > 0.5)[0]
num_tets = T.shape[0]
muscle_direction = np.zeros((num_tets, 2))
muscle_actuation = np.zeros((num_tets, 1)) 
# muscle_actuation[mI] = 1e7
muscle_direction[mI, 0] = 0.5
muscle_direction[mI, 1] = 1.5
muscle_direction[mI, :] = muscle_direction[mI, :] / np.linalg.norm(muscle_direction[mI, :], axis=1)[:, None]
K = sk.energies.emu_force_matrix(X, muscle_direction, vol=volume(X, T), J=deformation_jacobian(X, T)).tocsc()[mI, :]

M = massmatrix(X, T)
M_sqrt = sp.sparse.diags(np.sqrt(M.diagonal()))
M_sqrt_inv = sp.sparse.diags(1.0/M_sqrt.diagonal())

Me = sp.sparse.kron(M, sp.sparse.identity(dim))
# H_muscle = emu_hessian_d2x(X, muscle_direction, muscle_actuation,  vol=volume(X, T), J=deformation_jacobian(X, T))

M_sqrt_e = sp.sparse.kron(M, sp.sparse.identity(dim))

for subspace_name in subspace_names:
    # build subspace from simulation
    result_dir = directory + "/results/" + subspace_name + "/"
    os.makedirs(result_dir, exist_ok=True)
    if subspace_name == "lma":
        D, B = eigs(H, k=m, M=Me)
        
        A = limit_actuation_dirichlet_energy(X, T, B, 0.5)
        

        # sk.polyscope.view_displacement_modes(X, T, B * A, a=1,  material='flat', edge_width=0.000, uv=uv, texture_png=tex_path, period=120, 
        #                                      path=os.path.join(result_dir, 'eigs_modes.mp4'))
        # Sigma_F_sqrt = sp.sparse.identity(X.shape[0]*dim)
        Sigma_F_sqrt = M_sqrt_e.T
    if subspace_name == "fdm-lma":
        W = np.zeros((X.shape[0], 0))
        HiC = umfpack_lu_solve(H, K.T.todense())

        
        D, B = force_dual_modes_sqrt(H, K.T.todense(), m, M_sqrt=M_sqrt_e)
        A = limit_actuation_dirichlet_energy(X, T, B, 1.0)
        # sk.polyscope.view_displacement_modes(X, T, B * A, a=1,  material='flat', edge_width=0.000, uv=uv, texture_png=tex_path, period=120, 
        #                                      path=os.path.join(result_dir, 'fdm-lma_modes.mp4'))    
        
        Sigma_F_sqrt = K.T
        
    subsample = 2
    mu = np.zeros((Sigma_F_sqrt.shape[0], 1))
    num_samples = 5
    forces = sk.sample_gaussian_force(Sigma_F_sqrt, mu, num_samples)
    
    Forces = forces.reshape((X.shape[0], dim, num_samples))
    Forces_true = np.zeros((X.shape[0], dim, num_samples))
    Forces_true[::subsample, :, :] = Forces[::subsample, :, :]
    
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.remove_all_structures()
    mesh = ps.register_surface_mesh("mesh", X, T,  material="flat", color=np.array([252,174,145])/255)
    E = igl.boundary_facets(T)[0]
    Vs, Es, _, _ = igl.remove_unreferenced(X, E)
    outline = ps.register_curve_network("outline", Vs, Es, material='flat', color=np.array([0,0,0]))
    # if tex_path is not None and uv is not None:
    #     mesh.add_parameterization_quantity("test_param",  uv,
    #                                     defined_on='vertices')
        
    #     # if its a string then read the image with PIL. use it direclty
    #     if isinstance(tex_path, str):
    #         tex_path_2 = np.array(Image.open(tex_path)) / 255 /1.1
    #     else:
    #         tex_path_2 = tex_path.copy()
        
    #     mesh.add_color_quantity("test_vals", tex_path_2,
    #                                 defined_on='texture', param_name="test_param",
    #                                     enabled=True)
        
    for i in range(num_samples):
        mesh.add_vector_quantity("force_samples", Forces_true[:, :, i].reshape((-1, 2)), 
                                 defined_on='vertices', material='flat', radius=0.01, length=0.1, color=np.array([165,15,21])/255, enabled=True)
        ps.frame_tick()
        ps.screenshot(result_dir + "/force_samples_method_" + subspace_name + "_" + str(i) + ".png", transparent_bg=False)
        
    ps.show()