from code import interact
from copy import deepcopy
import os
from PIL import Image

from matplotlib.figure import Figure
import polyscope as ps
import polyscope.imgui as psim
import igl
import numpy as np
import gpytoolbox as gpt
from examples.localized_subspaces.contact.configs import *
from simkit import common_selections, contact_springs_sphere_gradient, contact_springs_sphere_hessian, massmatrix
from simkit.contact_springs_sphere_energy import contact_springs_sphere_energy
from simkit.diffuse_scalar import diffuse_scalar
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_dual_modes import force_dual_modes, force_dual_modes_diagonal, force_dual_modes_sqrt
from simkit.gravity_force import gravity_force
from simkit.lbs_jacobian import lbs_jacobian
from simkit.limit_actuation_dirichlet_energy import limit_actuation_dirichlet_energy
from simkit.low_rank_covariance_folding import low_rank_covariance_folding
from simkit.matplotlib.VisVectorLoadVariance import VisVectorLoadVariance
from simkit.matplotlib.VectorField import VectorField
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.TriangleMesh import TriangleMesh
from simkit.normal_force_matrix import normal_force_matrix
from simkit.normals import normals
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.remove_zero_cols import remove_zero_cols
from simkit.selection_matrix import selection_matrix
from simkit.shape_outlines import circle_outline
from simkit.simplex_vertex_averaging_matrix import simplex_vertex_averaging_matrix
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSim, ElasticFEMSimParams
from simkit.skinning_eigenmodes import skinning_eigenmodes

from simkit.spectral_cubature import spectral_cubature
from simkit.linear_elasticity_hessian import linear_elasticity_hessian

from simkit.average_onto_simplex import average_onto_simplex

import scipy as sp

from simkit.umfpack_lu_solve import umfpack_lu_solve
from simkit.volume import volume


# create Enum for subspace method
dir = os.path.dirname(__file__)

c1 =  Heart()
c1.subspace_method = Subspace.force_dual_modal_analysis
c2 = deepcopy(c1)

c2.subspace_method = Subspace.modal_analysis

c3 = deepcopy(c1)
c3.subspace_method = Subspace.greens_function

c4 = deepcopy(c1)
c4.subspace_method = Subspace.greens_function_thick
# c4 = deepcopy(c1)
# c4.subspace_method = Subspace.force_dual_skinning_eigenmodes

# c5 = deepcopy(c1)
# c5.subspace_method = Subspace.skinning_eigenmodes
cs = [c1, c2, c3, c4]
for c in cs:

    c.m  = 30
    # c.save_video = False
    result_dir = dir + "/results/" + c.character_name+ "/sim_" + Subspace.name(c.subspace_method) + "/"
    # contact_vertices = c.contact_vertices
    pinned_vertices = c.pinned_vertices
    X = c.X
    T = c.T
    E = c.E
    dim = c.dim
    sphere_radius = 0.2
    k_contact = 1e6
    speed = 0.02


    std = c.phi
    # D = sp.sparse.kron(sp.sparse.diags(std), sp.sparse.identity(dim))
    
    # D, is_all_zeros = remove_zero_cols(D.tocsc(), thresh=1e-1)
    # DA = D
    
    bc = average_onto_simplex(X, E)
    a = (bc[:, 0] < bc.mean(axis=0)[0])[:, None]
    
    
    D = normal_force_matrix(X, E)

    
    contact_vertices = np.unique(E)
    # DA = D @ Asqrt
    phi_t = a*1 #average_onto_simplex(c.phi, E)
    DA = D @ sp.sparse.diags(phi_t.flatten()*1)
    DA_sum_col = DA.sum(axis=0)
    is_all_zeros = (DA_sum_col == 0)
    DA = DA[:, ~is_all_zeros]
    
    M = sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim))
    H_elastic = linear_elasticity_hessian(X=X, T=T)
    H_pin, b_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices], X.shape[0], 1e9)
    x0 = np.zeros((X.size, 1))
    H = H_elastic + H_pin

    if c.subspace_method == Subspace.full:
        B = sp.sparse.identity(X.shape[0]*2).tocsc()
        cI = None
        cW = None
    else:
        if c.subspace_method == Subspace.skinning_eigenmodes:
            Ms = massmatrix(X, T)
            L = fold_vector_hessian(H, dim)
            W = eigs(L, c.m, M=Ms)[1]
            B = lbs_jacobian(X, W)
            b_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices]*0, X.shape[0], 1e8, only_b=True)[0]
            x0 = X.reshape(-1, 1)

        if c.subspace_method == Subspace.greens_function:
            Ms = massmatrix(X, T).tocsc()
            # fBi = farthest_point_sampling(c.X[contact_vertices], c.m)
            # fBi = np.random.choice(np.arange(contact_vertices.shape[0]), c.m, replace=False, p=c.phi.flatten()/c.phi.flatten().sum())
            
            E = igl.boundary_facets(T)[0]
            
            contact_vertices_right = np.unique(E[a.flatten()])
            fBi = farthest_point_sampling(c.X[contact_vertices_right], c.m)
            indices = contact_vertices_right[fBi]
            # indices = np.where(~is_all_zeros)[0]
            N = normals(X, E)
            
            # N_mat = sp.sparse.block_diag(N[:, :, None]).tocsc()
            # N_cI = N_mat[:, indices]
            vol = volume(X, E)
            Av = simplex_vertex_averaging_matrix(E, X.shape[0], vol)
            Nv = Av @ N
            Nv = Nv / (np.linalg.norm(Nv, axis=1))[:, None]
            Nv = np.nan_to_num(Nv, 0)
            
            Nv_mat = sp.sparse.block_diag(Nv[:, :, None]).tocsc()
            
            Nv_cI = Nv_mat[:, indices]
            # cI = indices[sI]
            # Mi = Ms[:, cI]
            # Mie = sp.sparse.kron(Mi, sp.sparse.identity(dim))
            # B = umfpack_lu_solve(H, Mie)
            B = sp.sparse.linalg.spsolve(H, Nv_cI.toarray())
            # W = skinning_eigenmodes(X, T, c.m)[0]
            b_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices]*0, X.shape[0], 1e8, only_b=True)[0]
            x0 = X.reshape(-1, 1)
        if c.subspace_method == Subspace.greens_function_thick:
            Ms = massmatrix(X, T).tocsc()
            # fBi = np.random.choice(np.arange(contact_vertices.shape[0]), c.m, replace=False, p=c.phi.flatten()/c.phi.flatten().sum())
            E = igl.boundary_facets(T)[0]
            
            contact_vertices_right = np.unique(E[a.flatten()])
            fBi = farthest_point_sampling(c.X[contact_vertices_right], c.m)
            indices = contact_vertices_right[fBi]
            # indices = np.where(~is_all_zeros)[0]
            N = normals(X, E)
            N_cI = N[indices, :]
            # N_mat = sp.sparse.block_diag(N[:, :, None]).tocsc()
            # N_cI = N_mat[:, indices]
            vol = volume(X, E)
            Av = simplex_vertex_averaging_matrix(E, X.shape[0], vol)
            Nv = Av @ N
            Nv = Nv / (np.linalg.norm(Nv, axis=1))[:, None]
            Nv = np.nan_to_num(Nv, 0)
            
            distances = pairwise_distance(X[indices], X)
            
            # sigmoid with sharp falloff according to radius
            sigmoid =1 -  1 / (1 + np.exp(-100 * (distances - sphere_radius)))
            
            on_surface = np.zeros((X.shape[0]), dtype=bool)
            on_surface[np.unique(E)] = True
            
            weight = sigmoid * on_surface[None, :]
            
            weights= np.repeat(weight[:, :, None ], dim, axis=2)
            D = Nv[None, :, :] * weights

            D = D.reshape(c.m, -1).T
            
            Mae = massmatrix(X, E)
            Mae = sp.sparse.kron(Mae, sp.sparse.identity(dim))
    
            
            # Nv_mat = sp.sparse.block_diag(Nv[:, :, None]).tocsc()
            Xs, Es, _, _ = igl.remove_unreferenced(X, E)
            # view_scalar_fields(Xs, Es, sigmoid.T)
            # Nv_cI = Nv_mat[:, indices]
            # cI = indices[sI]
            # Mi = Ms[:, cI]
            # Mie = sp.sparse.kron(Mi, sp.sparse.identity(dim))
            # B = umfpack_lu_solve(H, Mie)
            B = sp.sparse.linalg.spsolve(H, Mae @ D)
            

            # view_displacement_modes(X, T, B[:, :10], a=1, period=20)
            # view_scalar_fields(X, E,D)
            # W = skinning_eigenmodes(X, T, c.m)[0]
            b_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices]*0, X.shape[0], 1e8, only_b=True)[0]
            x0 = X.reshape(-1, 1)




            # if c.vis_subspace:
            #     # D2 = normal_force_matrix(X, E, mass_weigh=False)
            #     # D2A = D2 @ Asqrt    
            #     # VisVectorLoadVariance(X, T, D2A, samples=35, 
            #     #                         path = dir + "/results/load_variance_" + Subspace.name(c.subspace_method) +".png" )

            #     amps = limit_actuation_dirichlet_energy(X, T, B, 30)
            #     view_displacement_modes(X, T, B ,fps=30)#,  path = dir + "/results/modes_" + Subspace.name(c.subspace_method) + ".mp4")

        if c.subspace_method == Subspace.modal_analysis:
            B = eigs(H, c.m, M=M)[1]
            # W = skinning_eigenmodes(X, T, c.m)[0]
            x0 = X.reshape(-1, 1)
    
        if c.subspace_method == Subspace.force_dual_skinning_eigenmodes:
            Ms = massmatrix(X, T)
            HiD = low_rank_covariance_folding(H, DA.toarray(), dim)
            W = force_dual_modes(H , D.toarray(), c.m, M=Ms, HiSigma_F_sqrt=HiD)
            B = lbs_jacobian(X, W)
            x0 = X.reshape(-1, 1)


        if c.subspace_method == Subspace.force_dual_modal_analysis:              
            # a = X[:, 0] < X[:, 0].mean()
            
            # sigma_F = sp.sparse.diags( a.flatten()*1 + 1e-6)
            # sigma_F_e = sp.sparse.kron(sigma_F, sp.sparse.identity(dim))
            # sigma_F_e = sigma_F_e @ M
            # # Me = sp.sparse.kron(M, sp.sparse.identity(dim))
            # [D, B] = force_dual_modes_diagonal(H, sigma_F_e, c.m, M=M)
            B = force_dual_modes_sqrt(H, DA, c.m, M_sqrt=sp.sparse.diags(np.sqrt(M.diagonal())))[0]
        
            # W = skinning_eigenmodes(X, T, c.k)[0]
            x0 = X.reshape(-1, 1)
            
            B = orthonormalize(B, M, 1e-6)

            # if c.vis_subspace:
            #     # D2 = normal_force_matrix(X, E, mass_weigh=False)
            #     # D2A = D2 @ Asqrt    
            #     # VisVectorLoadVariance(X, T, D2A, samples=35, 
            #     #                       path = result_dir + "/../load_variance_" + Subspace.name(c.subspace_method) +".png" )

            #     amps = limit_actuation_dirichlet_energy(X, T, B, 10)
            #     view_displacement_modes(X, T, B @ np.diag(amps),fps=30,  path = dir + "/results/modes_" + Subspace.name(c.subspace_method) + ".mp4")
        W = skinning_eigenmodes(X, T, c.k)[0]
        cI, cW = spectral_cubature(X, T, W, c.k)
    # cI = None
    # cW = None
    # view_displacement_modes(X, T, B)

    g = gravity_force(X, T, a=-0, rho=1e-3).reshape(-1, 1)
    sim_params = ElasticFEMSimParams( h=1e-2, rho=1e-3, ym=1e6, pr=0.45,
                                    material='neo-hookean',  b0=(-g + b_pin), Q0=H_pin)

    sim = ElasticFEMSim(X, T, B=B, cI=cI, cW=cW, p=sim_params, x0=x0)
    # sim.p.solver_p.do_line_search = False

    z, z_dot = sim.rest_state()
    z = np.zeros_like(z)
    z_dot = np.zeros_like(z_dot)
    step = 0

    P_collider = c.P_collider0


    S = selection_matrix( c.bI, X.shape[0])
    Se = sp.sparse.kron(S, sp.sparse.identity(2))
    SB = Se@ B
    Sex0 = Se @ sim.x0
    def contact_energy(z):
        Xcon = (SB @ z + Sex0).reshape(-1, dim)
        energy_sphere = contact_springs_sphere_energy(Xcon, k_contact, P_collider[:, :dim], sphere_radius)
        energy_total = energy_sphere + sim.energy(z)
        return energy_total
    def contact_gradient(z):
        Xcon = (SB @ z + Sex0).reshape(-1, dim)
        gradient_sphere = contact_springs_sphere_gradient(Xcon, k_contact, P_collider[:, :dim],  sphere_radius)
        gradient_total = SB.T @ ( gradient_sphere ) + sim.gradient(z)
        return (gradient_total)
    def contact_hessian(z):
        Xcon = (SB @ z + Sex0).reshape(-1, dim)
        hessian_sphere = contact_springs_sphere_hessian(Xcon, k_contact, P_collider[:, :dim],  sphere_radius)
        hessian_total =  SB.T @ ( hessian_sphere) @ SB  + sim.hessian(z)
        return hessian_total
    sim.solver.energy_func = contact_energy
    sim.solver.gradient_func = contact_gradient
    sim.solver.hessian_func = contact_hessian

    ps.init()
    mesh = ps.register_surface_mesh("mesh", X, T, material='flat')
    # pinned = ps.register_point_cloud("pinned", X[c.pinned_vertices], radius=0.01)
    # contact = ps.register_point_cloud("contact", X[c.contact_vertices], radius=0.01)
    sphere_Vs, sphere_Es = circle_outline(radius=sphere_radius)
    import igl.triangle

    sphere_V, sphere_F , _, _, _= igl.triangle.triangulate(sphere_Vs, sphere_Es, flags="qpa0.01")
    sphere_V = np.hstack([sphere_V, np.zeros((sphere_V.shape[0], 1))])
    
    ps.set_ground_plane_mode("none")
    pc = ps.register_surface_mesh("collider", P_collider + sphere_V, sphere_F, color=np.array([255/255, 192/255, 203/255]), material='flat')#, radius=0.001)
    uv = np.load(c.uv_path)
    
    sphere_Vs = np.concatenate([sphere_Vs, 1e-8*np.ones((sphere_Vs.shape[0], 1))], axis=1)
    pc_outline = ps.register_curve_network("outline", sphere_Vs, sphere_Es, color=np.array([0, 0, 0]), material='flat', radius=0.001)

    mesh.add_parameterization_quantity("test_param",  uv,
                                      defined_on='vertices')
    vals = np.array(Image.open(c.tex_path))
    mesh.add_color_quantity("test_vals", vals[:, :, 0:3]/255,
                                defined_on='texture', param_name="test_param",
                                    enabled=True)
    
    D_collider = np.array([[0.0, 0, 0]])
    
    
    start_pos = np.array([[0.011, -1.5, 0]])
    end_pos = np.array([[-1.4, 0.8, 0]])
    # import polyscope as ps
    # ps.init()
    # mesh = ps.register_surface_mesh("mesh", X, T, material='flat')
    
    # ps.set_ground_plane_mode("none")
    # ps.show()

    
    c.direction = (end_pos - start_pos)
    c.P_collider0 = start_pos
    c.interactive = False
    c.save_video = True
    
    c.num_timesteps = 240
    c.period = 240
    
    D_collider =  0.5 * (np.sin(2 * np.pi * 0 / c.period - np.pi/2) + 1 ) * c.direction
    P_collider = c.P_collider0 + D_collider
    if c.interactive:
        def callback():
            global z, z_dot,  speed, D_collider, P_collider
            z_next = sim.step(z, z_dot)
            z_dot = (z_next - z)/sim.p.h
            z = z_next.copy()
            
            if psim.IsKeyPressed(psim.ImGuiKey_J):
                D_collider += speed * np.array([[-1, 0, 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_L):
                D_collider += speed * np.array([[1, 0, 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_I):
                D_collider += speed * np.array([[0, 1., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_K):
                D_collider += speed * np.array([[0, -1., 0]])
            if psim.IsKeyPressed(psim.ImGuiKey_A):
                speed *=2
            if psim.IsKeyPressed(psim.ImGuiKey_S):
                speed /=2
            P_collider = c.P_collider0 + D_collider
            mesh.update_vertex_positions((B @ z + x0).reshape(-1, 2))
            pc.update_vertex_positions(P_collider + sphere_V+ np.array([0, 0, 0.01]))
            pc_outline.update_node_positions(sphere_Vs + P_collider + np.array([0, 0, 0.01]))
        ps.set_user_callback(callback)
        ps.show()
    else:

        if c.save_video:
            video_name = "sim_" + Subspace.name(c.subspace_method)
            result_dir2 = result_dir + "/"
            os.makedirs(result_dir2, exist_ok=True)

        for i in range(c.num_timesteps + 1):
            z_next = sim.step(z, z_dot)
            z_dot = (z_next - z)/sim.p.h
            z = z_next.copy()
            
            D_collider =  0.5 * (np.sin(2 * np.pi * i / c.period - np.pi/2) + 1 ) * c.direction
            P_collider = c.P_collider0 + D_collider
            mesh.update_vertex_positions((B @ z + x0).reshape(-1, 2))
            pc.update_vertex_positions(P_collider + sphere_V + np.array([0, 0, 0.01]))
            
            pc_outline.update_node_positions(sphere_Vs + P_collider + np.array([0, 0, 0.01]))
            ps.frame_tick()
            if c.save_video:
                ps.screenshot(result_dir2 + "/" + str(i).zfill(4) + ".png", transparent_bg=True)
        if c.save_video:
            video_from_image_dir(result_dir2, result_dir2 + "/../" + video_name + ".mp4", fps=60)
            mp4_to_gif(result_dir2 + "/../" + video_name + ".mp4", result_dir2 + "/../" + video_name + ".gif")
    BC = average_onto_simplex(X, T)
    # ps.register_point_cloud("cI", BC[cI], radius=0.01)
    # ps.show()


