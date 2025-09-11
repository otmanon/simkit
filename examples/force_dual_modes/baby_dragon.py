


import igl
import numpy as np
import scipy as sp
import sys
import os
import timeit as timeit

from simkit.low_rank_covariance_folding import low_rank_covariance_folding
from simkit.polyscope.view_displacement_modes import view_displacement_modes

# from simkit.filesystem.video_from_image_dir import video_from_image_dir

sys.path.append(os.path.dirname(__file__) + "/../../../")
from simkit.blender.render_spring_ik_animation import render_spring_ik_animation
from simkit.blender.render_edges_on_transparent_mesh_animation import render_edges_on_transparent_mesh_animation
from simkit.orthonormalize import orthonormalize
# from simkit.rotation_strain_coordinates import rotation_strain_coordinates, subspace_rotation_strain_coordinates
from simkit.sims.elastic.ARAPSpringActuatorPDSim import ARAPSpringActuatorPDSim, ARAPSpringActuatorPDSimParams
from simkit.sims.elastic.MassSpringsSpringActuatorPDSim import MassSpringsSpringActuatorPDSimParams, MassSpringsSpringActuatorPDSim
from simkit.boxes_to_edges import boxes_to_edges
from simkit.closest_point_map import closest_point_map
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.edge_length_jacobian import edge_length_jacobian
from simkit.eigs import eigs
from simkit.filesystem.compute_with_cache_check import compute_with_cache_check
from simkit.fold_in_vector_subspace import fold_in_vector_subspace
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.force_dual_modes import force_dual_modes_sqrt
from simkit.gradient_cfd import gradient_cfd
from simkit.lbs_jacobian import lbs_jacobian
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
# from simkit.polyscope.animation_viewer import animation_viewer
# from simkit.polyscope.view_displacement_modes import view_displacement_modes
# from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.solvers.LevenbergMarquardtSolver import LevenbergMarquardtParams, LevenbergMarquardtSolver
from simkit.spectral_clustering import spectral_clustering
from simkit.spectral_cubature import spectral_cubature
from simkit.von_tycowics_expansion import von_tycowicz_expansion
from simkit.cluster_grouping_matrices import cluster_grouping_matrices
from simkit.edge_lengths import edge_lengths
from simkit.polar_svd import polar_svd
from simkit.project_into_subspace import project_into_subspace
from simkit.simplex_vertex_map import simplex_vertex_map
from simkit.solvers import BlockCoordSolver, BlockCoordSolverParams
from simkit.volume import volume
from simkit.massmatrix import massmatrix
from simkit.deformation_jacobian import deformation_jacobian
from simkit.quadratic_hessian import quadratic_hessian
from simkit.selection_matrix import selection_matrix



def compute_subspace():
    
    if method == "full":
        B = sp.sparse.identity(X.shape[0]*X.shape[1])
        W = sp.sparse.identity(X.shape[0])
        
        cI = np.arange(T.shape[0])
        cW = volume(X , T)
        labels = np.arange(T.shape[0])
        return B, W, cI, cW, labels
    
    if method == "eigs":
        M = massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        [eval, B] = eigs(H , k=m_disp, M = Me)
    # B = lbs_jacobian(X, W)
        B = np.concatenate((X.reshape(-1, 1), B), axis=1)
        
        W = np.zeros((X.shape[0], 1))
        B = orthonormalize(B, Me, 1e-6)
        cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
    elif method == "force_dual_von_tycowicz":
        J = edge_length_jacobian(X, E_actuated).toarray().T

        M = massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        
        B, e = force_dual_modes_sqrt(H, J, m_skin, M_sqrt = sp.sparse.diags(np.sqrt(Me.diagonal())))
        B = von_tycowicz_expansion(B, 3)
        # view_displacement_modes(X, F, B[:, :10], a=20, period=30)
        B = np.concatenate((X.reshape(-1, 1), B), axis=1)
    
        W = np.zeros((X.shape[0], 1))
        cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
        
    # elif method == "force_dual_eigs_separate":
    #     M = massmatrix(X, T)
    #     Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        
    #     Bs = []
    #     for i in range(len(muscle_names)):
    #         J = edge_length_jacobian(X, E_actuated_list[i]).toarray().T
    #         Bi, e = force_dual_modes_sqrt(H, J, m_disp_sub, M_sqrt = sp.sparse.diags(np.sqrt(Me.diagonal())))
            
    #         Bs.append(Bi)
    #     B = np.hstack(Bs)
    #     B = np.concatenate((X.reshape(-1, 1), B), axis=1)
    #     # view_displacement_modes(X, F, B[:, :10], a=1, period=30)
    
    #     W = np.zeros((X.shape[0], 1))
    #     cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
    elif method == "force_dual_eigs":
        M = massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        Es = np.vstack(E_actuated)
        J = edge_length_jacobian(X, E_actuated).toarray().T
        B, e = force_dual_modes_sqrt(H, J, m_disp, M_sqrt = sp.sparse.diags(np.sqrt(Me.diagonal())))
        # view_displacement_modes(X, F, B[:, :10], a=20, period=30)
        B = np.concatenate((X.reshape(-1, 1), B), axis=1)
    
        W = fold_in_vector_subspace(B, 3)
        # IU = data['IU']
        # # view_displacement_modes(X, T, IU)
        # # view_displacement_modes(X, T, Iphi[:, :10])
        cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
    
    # elif method == "force_dual_skinning_eigenmodes_separate":
    #     M = massmatrix(X, T)
    #     Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        
    #     L = fold_vector_hessian((H).tocsc(), 3)
    #     Ws = []
    #     for i in range(len(muscle_names)):
    #         J = edge_length_jacobian(X, E_actuated_list[i]).toarray().T
    #         Js = fold_in_vector_subspace(J, 3)
    #         Wi, Di = force_dual_modes_sqrt(L, Js, m_skin_sub,M_sqrt = sp.sparse.diags(np.sqrt(1/M.diagonal())))
    #         Ws.append(Wi)
    #     W = np.hstack(Ws)
    #     W = np.concatenate((np.ones((X.shape[0], 1)), W), axis=1)
    #     B = lbs_jacobian(X, W)
        
        B = orthonormalize(B, Me, 1e-6)
        cI, cW, labels = spectral_cubature(X, T, W, 30, return_labels=True)
    elif method == "force_dual_skinning_eigenmodes":
        J = edge_length_jacobian(X, E_actuated).toarray().T
        L = fold_vector_hessian((H).tocsc(), 3)
        Js = fold_in_vector_subspace(J, 3)
        M = massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        M_sqrt = sp.sparse.diags(np.sqrt(Me.diagonal()))
        M_sqrt_inv = sp.sparse.diags(1.0/M_sqrt.diagonal())
           # fold in covariances for skinning eigenmodes
        L_force_dual = low_rank_covariance_folding(H,J, dim=3)
        W = np.linalg.svd(M_sqrt @ L_force_dual, full_matrices=False)[0][:, :5]
        W = M_sqrt_inv @ W
        W, D = force_dual_modes_sqrt(L, Js, m_skin,M_sqrt = sp.sparse.diags(np.sqrt(M.diagonal())))
        
        W = np.concatenate((np.ones((X.shape[0], 1)), W), axis=1)
        B = lbs_jacobian(X, W)
        
        B = orthonormalize(B, Me, 1e-6)
        
        # view_scalar_fields(X, F, W)
        cI, cW, labels = spectral_cubature(X, T, W, 30, return_labels=True)
    elif method == "skinning_eigenmodes":
        L = fold_vector_hessian(H, dim)
        M = massmatrix(X, T)
        Me = sp.sparse.kron(M, sp.sparse.identity(dim))
        
        E, W = eigs(L, k=m_skin, M = M)
        W = np.concatenate((np.ones((X.shape[0], 1)), W), axis=1)
        B = lbs_jacobian(X, W)
        B = orthonormalize(B, Me, 1e-6)
        cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
        
    elif method == "force_dual_modal_derivatives":
                
        from scipy.io import loadmat
        data = loadmat(directory + "/" + muscle_name + "_data.mat")
        IU = data['IU']
        Iphi = data['Iphi']
        B = np.concatenate((IU, Iphi), axis=1)
        W = np.zeros((X.shape[0], 1))
        cI, cW, labels = spectral_cubature(X, T, B, 30, return_labels=True)
        
        
        
        
    return B, W, cI, cW, labels



def interactive_app(sim):
    z, z_dot = sim.rest_state()
    
    
    target_indices = np.zeros((0, 1))
    target_pos = np.zeros((0, 3))
    S = None
    Se = None
    SeB = None
    BSSB = None
    

    l0 = sim.l0.copy()
    a_prev = sim.l0.copy()
    a_min = a_prev.copy()*0.5
    a_max = a_prev.copy()*2
    
    def energy(a):
        sim.l0[control_edge_indices] = a
        z_next = sim.step(z, z_dot)
        reg_loss = 0.5 * np.linalg.norm(a - l0)**2
        
        matching_loss =0.5 * np.linalg.norm( (SeB @ z_next)  - target_pos.reshape(-1, 1))**2
        energy_loss = sim.energy_gradient(z_next)
        l = matching_loss + gamma_spring_reg * reg_loss + gamma_energy * energy_loss
        return l

    def gradient(a):
        nonlocal a_prev
        sim.l0[control_edge_indices] = a
        z_next = sim.step(z, z_dot)
        sim_dloss_dz =   BSSB @ z - SeB.T @ target_pos.reshape(-1, 1)
        
        # sim_dloss_dz_temp[target_indices, :] = 0

        if gamma_energy > 0:
            sim_dloss_dz += sim.energy_gradient(z_next) * gamma_energy
        
        # sim_dloss_dz[1:] = 0
        # sim_dloss_dz_temp = sim_dloss_dz.reshape(-1, 3).copy()
        # trans = sim_dloss_dz_temp[::4, :].copy()
        # sim_dloss_dz_temp[...] = 0
        # sim_dloss_dz_temp[::4, :] = trans
        # sim_dloss_dz = sim_dloss_dz_temp.reshape(-1, 1)
        
        # print("Sim grad norm : ", np.linalg.norm(sim.energy_gradient(z_next)))
        sim_dloss_da =  sim.backward(z_next,sim_dloss_dz)
                
        reg_dloss_da = gamma_spring_reg * (a - l0)
        reg_dot_dloss_da = gamma_spring_dot_reg * (a - a_prev)
        a_prev = a.copy()
        dloss_da =  gamma_spring_reg * reg_dloss_da + sim_dloss_da + reg_dot_dloss_da 
        # print("Final grad norm : ", np.linalg.norm(dloss_da))
        return dloss_da
            

    params = LevenbergMarquardtParams(max_iter=lm_iters, tolerance=lm_tol, do_line_search=lm_do_line_search, lam=lm_lam, alpha0=lm_alpha0)
    optimizer = LevenbergMarquardtSolver(energy, gradient, params)

    import polyscope as ps
    import polyscope.imgui as psim
    ps.init()
    mesh = ps.register_surface_mesh("dragon", sim.X, F, transparency=0.5)
    Xs, Es, II, JJ= igl.remove_unreferenced(sim.X, E_actuated)
    spring_ps = ps.register_curve_network("springs", X[JJ, :], Es)
    target_pc_ps = ps.register_point_cloud("target", np.array([[0, 0, 0]]), radius=0.02)
    curr_pc_ps = ps.register_point_cloud("current", np.array([[0, 0, 0]]), radius=0.02)


    # crate array for storing recording
    actuations_list = np.zeros((sim.l0.shape[0], 0))
    target_pos_list = np.zeros((dim, 0))
    target_indices_list = np.zeros((1, 0))
    
    
    pc = None
    d = 0
    
    index = 0
    def callback():
        nonlocal  z, z_dot,  pc, d, SeB, BSSB, target_pos, target_indices, optimizer, curr_pc_ps, target_pc_ps, index, actuations_list, target_pos_list, target_indices_list
        win_pos = psim.GetMousePos()    
        view = ps.get_camera_view_matrix()
        camera_pos = np.linalg.inv(view)[:3, 3]  # get camera position

        # if right mouse button is clicked, place a point on the mesh.
        if psim.IsMouseClicked(1):     
            pos = ps.screen_coords_to_world_position(win_pos).reshape(-1, 3)    # use polyscope to find intersection into scene
            d = np.linalg.norm(pos - camera_pos)    # remember depth for future dragging
            curr_pc_ps = ps.register_point_cloud("current", pos, radius=0.01)   #vis
            target_pc_ps = ps.register_point_cloud("target", target_pos.reshape(-1, dim), radius=0.02)
            
            U =  (sim.B @ z).reshape(-1, 3)
            [_sqrD, vI, _cp] = igl.point_mesh_squared_distance(pos,U, np.arange(sim.X.shape[0])[:, None])
            
            # target_indices = np.vstack((target_indices, vI))
            # target_pos = np.vstack((target_pos,  U[vI]))
            target_indices = np.vstack((vI))
            target_pos = np.vstack(( U[vI]))
            S = selection_matrix(target_indices.flatten(), X.shape[0])
            Se = sp.sparse.kron(S, sp.sparse.identity(dim))
            SeB = Se @ B
            BSSB = SeB.T @ SeB
            
            index = target_indices.shape[0] - 1
            params = LevenbergMarquardtParams(max_iter=lm_iters, tolerance=lm_tol, do_line_search=lm_do_line_search, lam=lm_lam, alpha0=lm_alpha0)
            optimizer = LevenbergMarquardtSolver(energy, gradient, params)
            target_pc_ps = ps.register_point_cloud("target",target_pos.reshape(-1, dim), radius=0.02)
            curr_pc_ps = ps.register_point_cloud("current", target_pos.reshape(-1, dim), radius=0.02)

            # selected_edge = np.zeros((E.shape[0], 1))
            # selected_edge[control_edge_indices] = 1
    
            # if point being moved exists, and space is being held down, move the point by dragging mouse around
        if  psim.IsKeyDown(psim.GetKeyIndex(psim.ImGuiKey_Space)): 
            ray = ps.screen_coords_to_world_ray(win_pos)   # shoot ray from camera into scene
            P = (camera_pos + d * ray).reshape(-1, 3)   # get final position
            
            target_pos[index, :] = P
            target_pc_ps.update_point_positions(target_pos.reshape(-1, dim))
        
        # if psim.IsMouseClicked(0) and psim.IsKeyDown(psim.GetKeyIndex(psim.ImGuiKey_ModShift)):     
        #     pos = ps.screen_coords_to_world_position(win_pos).reshape(-1, 3)    # use polyscope to find intersection into scene
        #     d = np.linalg.norm(pos - camera_pos)    # remember depth for future dragging
            
        #     index = np.argmin(np.linalg.norm(target_pos - pos, axis=1))
        # if psim.IsKeyDown(psim.GetKeyIndex(psim.ImGuiKey_X)):
        #     # delete target
        #     if target_indices.shape[0] > 1:
        #         target_indices = np.delete(target_indices, index, axis=0)
        #         target_pos = np.delete(target_pos, index, axis=0)
        #         S = selection_matrix(target_indices.flatten(), X.shape[0])
        #         Se = sp.sparse.kron(S, sp.sparse.identity(dim))
        #         SeB = Se @ B
        #         BSSB = SeB.T @ SeB
        #         index = target_indices.shape[0] - 1
        #         target_pc_ps = ps.register_point_cloud("target",target_pos.reshape(-1, dim), radius=0.02)
        #         curr_pc_ps = ps.register_point_cloud("current", target_pos.reshape(-1, dim), radius=0.02)
        #     else: 

        #         target_indices = np.zeros((0, 1))
        #         target_pos = np.zeros((0, 3))
        #         S = None
        #         Se = None
        #         SeB = None
        #         BSSB = None
        #         index = -1
                
        #     params = LevenbergMarquardtParams(max_iter=lm_iters, tolerance=lm_tol, do_line_search=lm_do_line_search, lam=lm_lam, alpha0=lm_alpha0)
        #     optimizer = LevenbergMarquardtSolver(energy, gradient, params)
                
                
        if target_indices.shape[0] > 0:
            sim.l0[control_edge_indices] = optimizer.solve(sim.l0[control_edge_indices].copy(), x_min=a_min, x_max=a_max)
        
        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z) / sim.params.h
        z = z_next.copy()
        
        # z_rs = subspace_rotation_strain_coordinates(X, T, B_se, B[:, 1:], z[1:],  labels=labels, pinned=pinned_vertices, pre=rs_pre)
        # z[1:] = z_rs
        # pc.update_point_positions(sim.sphere_center)
        # U = (sim.B @ z_rs).reshape(-1, 3)
        U = (B @ z).reshape(-1, 3) 
        spring_ps.update_node_positions(U[JJ, :])
        
        activity = np.abs( sim.l0 - l0 ) 
        spring_ps.add_scalar_quantity("activity", activity.flatten(), defined_on="edges", enabled=True, cmap="reds", vminmax=[0, 0.5])
        mesh.update_vertex_positions(U)
        
        if target_indices.shape[0] > 0:
            target_pc_ps.update_point_positions(target_pos.reshape(-1, dim))
            curr_pc_ps.update_point_positions((SeB @ z).reshape(-1, dim))

        actuations_list = np.hstack((actuations_list, sim.l0.reshape(-1, 1)))
        
        if target_indices.shape[0] > 0:
            target_pos_list = np.hstack((target_pos_list, target_pos.T))
            target_indices_list = np.hstack((target_indices_list, target_indices))
        else:
            target_pos_list = np.hstack((target_pos_list, np.zeros((dim, 1))))
            target_indices_list = np.hstack((target_indices_list, np.array([-1])[:,None]))
            
        if psim.Button("Save"):
            np.savez(recording_path,  actuations_list=actuations_list,
                     target_pos_list=target_pos_list, target_indices_list=target_indices_list
                     )
        
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(callback)
    ps.show()
    
def simulate(sim, num_timesteps, target_indices_func, target_pos_func):
       
        target_indices = target_indices_func(0)
        target_pos = target_pos_func(0)
        
        S = selection_matrix(target_indices, X.shape[0])
        Se = sp.sparse.kron(S, sp.sparse.identity(dim))
        SeB = Se @ B
        BSSB = SeB.T @ SeB
        
        l0 = sim.l0.copy()
        a_prev = sim.l0.copy()
        a_min = a_prev.copy()*0.5
        a_max = a_prev.copy()*1.5
        
        def energy(a):
            sim.l0[control_edge_indices] = a
            z_next = sim.step(z, z_dot)
            reg_loss = 0.5 * np.linalg.norm(a - l0)**2
            matching_loss =0.5 * np.linalg.norm( (SeB @ z_next)  - target_pos.reshape(-1, 1))**2
            energy_loss = sim.energy_gradient(z_next)
            l = matching_loss + gamma_spring_reg * reg_loss + gamma_energy * energy_loss
            return l

        def gradient(a):
            nonlocal a_prev
            sim.l0[control_edge_indices] = a
            z_next = sim.step(z, z_dot)
            sim_dloss_dz =   BSSB @ z - SeB.T @ target_pos.reshape(-1, 1)
            if gamma_energy > 0:
                sim_dloss_dz += sim.energy_gradient(z_next) * gamma_energy
            
            # print("Sim grad norm : ", np.linalg.norm(sim.energy_gradient(z_next)))
            sim_dloss_da =  sim.backward(z_next,sim_dloss_dz)
                    
            reg_dloss_da = gamma_spring_reg * (a - l0)
            reg_dot_dloss_da = gamma_spring_dot_reg * (a - a_prev)
            a_prev = a.copy()
            dloss_da =  gamma_spring_reg * reg_dloss_da + sim_dloss_da + reg_dot_dloss_da 
            # print("Final grad norm : ", np.linalg.norm(dloss_da))
            return dloss_da
                
        params = LevenbergMarquardtParams(max_iter=lm_iters, tolerance=lm_tol, do_line_search=lm_do_line_search, lam=lm_lam, alpha0=lm_alpha0)
        optimizer = LevenbergMarquardtSolver(energy, gradient, params)
        Xs, Es, II, JJ= igl.remove_unreferenced(sim.X, E_actuated)
        z, z_dot = sim.rest_state()
            
        Z = np.zeros((z.shape[0], num_timesteps))
        actuations = np.zeros((sim.l0.shape[0], num_timesteps))
        Ps = np.zeros((dim, num_timesteps))
        pIs = np.zeros((1, num_timesteps))
        for i in range(num_timesteps):
            target_pos = target_pos_func(i)
            a = optimizer.solve(sim.l0[control_edge_indices].copy(), x_min=a_min, x_max=a_max)
            sim.l0[control_edge_indices] = a
            
            z_next = sim.step(z, z_dot)
            z_dot = (z_next - z) / sim.params.h
            z = z_next.copy()
            Z[:, i] = z.flatten()
            actuations[:, i] = a.flatten()
            Ps[:, i] = target_pos.flatten()
            pIs[0, i] = target_indices
        return Z, actuations, Ps, pIs
        
    

directory = os.path.dirname(os.path.abspath(__file__))
name = "baby_dragon"
method = "force_dual_skinning_eigenmodes"
data_dir = directory + "/../../../data/3d/" + name + "/"
result_dir = directory + "/results/" + name + "/"
texture_path = data_dir + "/" + name + "_tex.png"

[X_tex, uv_tex, _, F_tex, F_uv_tex, F_n_tex] = igl.readOBJ(data_dir + name + "_tex.obj")

tex_uv = uv_tex[F_uv_tex.flatten()]


os.makedirs(result_dir, exist_ok=True)

[X, T, F] = igl.readMESH(data_dir + name + ".mesh")



map =closest_point_map(X_tex, X)
F, _, _= igl.boundary_facets(T)
E = igl.edges(T)

lm_iters = 1
lm_lam = 0.5


lm_tol = 1e-9
lm_do_line_search = False

num_timesteps = 420

gamma_pin = 1e9          
gamma_spring_reg = 0
gamma_energy = 0
gamma_spring_dot_reg = 0

ym_edges = 1e8
ym_vol = 1e6
rho = 1e0
m_disp = 30
m_skin = 5
m_skin_sub = 3
m_disp_sub = 3
read_subspace_cache = False
read_result_cache = False

compute_times = False


muscle_names = ["left_back_leg", "right_back_leg", "left_front_leg", "right_front_leg", 
                "tail", "head", "back", "right_wing", "left_wing"]
muscle_names = ["left_front_leg", "right_front_leg", "head"]
muscle_names = ["head"]
muscle_names = ["left_wing"]

muscle_names = [ "body", "head",   "left_front_leg",    "tail", ]
    

    


dim = X.shape[1]

if name == "baby_dragon":
    is_bottom = X[:, 1] < np.min(X[:, 1]) + 0.2
    is_mid_leg = (X[:, 2] < np.mean(X[:, 2]) + 0.25) * (X[:, 2] > np.mean(X[:, 2]) - 0.05)
    pinned_vertices = np.where(is_bottom * is_mid_leg)[0]
    m_disp = 10
    # ym_edges = 1e7
    # ym_vol = 1e5
    local_global_max_iter = 10

H_pin, b_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices, :], X.shape[0], gamma_pin)
H_elastic = linear_elasticity_hessian(X=X, T=T)
H = H_elastic + H_pin

methods = [ "force_dual_skinning_eigenmodes"] #, "eigs"]

# video_from_image_dir(result_dir + "/../baby_dragon_still/force_dual_eigs_body_final_render", result_dir + "/../baby_dragon_still/force_dual_eigs_body_final_render.mp4")
# def target_indices_head_func(i):
#     return np.array([3843])

# def target_pos_head_func(i):
    
#     radius = 0.75
#     if i < int(num_timesteps * 0.8):

#         start_pos = X[target_indices_head_func(0)[0]]
#         # move it in a circle 
#         angle = i * 2 * np.pi / ( num_timesteps * 0.8 / 2.0)
        
#         pos = start_pos + radius *np.array([np.sin(angle), 1-np.cos(angle) , 0])
#     else:
#         pos = X[target_indices_head_func(0)[0]]
#     # starting at the bottom most pole
#     return pos
    


def target_indices_left_leg_func(i):
    return np.array([1815])

def target_pos_left_leg_func(i):

    radius = 0.75
    if i < int(num_timesteps * 0.8):

        start_pos = X[target_indices_left_leg_func(0)[0]]
        # move it in a circle 
        angle = -i * 2 * np.pi / ( num_timesteps * 0.8 / 2.0)
        
        pos = start_pos + radius *np.array([ 0, 1-np.cos(angle) , np.sin(angle)])
    else:
        pos = X[target_indices_left_leg_func(0)[0]]
    # starting at the bottom most pole
    return pos
    

def target_indices_head_func(i):
    return np.array([3843])

def target_pos_head_func(i):
    
    radius = 0.75
    if i < int(num_timesteps * 0.8):

        start_pos = X[target_indices_head_func(0)[0]]
        # move it in a circle 
        angle = i * 2 * np.pi / ( num_timesteps * 0.8 / 2.0)
        
        pos = start_pos + radius *np.array([np.sin(angle), 1-np.cos(angle) , 0])
    else:
        pos = X[target_indices_head_func(0)[0]]
    # starting at the bottom most pole
    return pos

def target_indices_tail_func(i):
    return np.array([15379])

def target_pos_tail_func(i):
    radius = 1.0
    if i < int(num_timesteps * 0.8):

        start_pos = X[target_indices_tail_func(0)[0]]
        # move it in a circle 
        angle = i * 2 * np.pi / ( num_timesteps * 0.8 )
        
        pos = start_pos + radius *np.array([(np.sin(angle)), 0, -(1-np.cos(angle))])
    else:
        pos = X[target_indices_tail_func(0)[0]]
    return pos


def target_indices_body_func(i):
    return np.array([3843])

def target_pos_body_func(i):
    
    radius = 2
    if i < int(num_timesteps * 0.8):

        start_pos = X[target_indices_body_func(0)[0]]
        # move it in a circle 
        angle = i * 2 * np.pi / ( num_timesteps * 0.8 // 0.5)
        
        pos = start_pos + radius *np.array([0, - 0.05 *( np.sin(angle) ), np.sin(angle)])
    else:
        pos = X[target_indices_body_func(0)[0]]
    # starting at the bottom most pole
    return pos


for muscle_name in muscle_names:
    for method in methods:
        
        if method == "force_dual_eigs":
            lm_alpha0 = 1e-1
        else:
            lm_alpha0 = 1e-2
            
        lm_alpha0 = 1e0
        # E_actuated_list = []
        # for i in range(len(muscle_names)):
        #     X_boxes, _, _, F_boxes, _, _ = igl.readOBJ(data_dir + name + "_spring_muscles_" + muscle_names[i] + ".obj")
        #     spring_Vs_i, E_actuated_i = boxes_to_edges(X_boxes, F_boxes)
            
        #     [sqrdD, bI, cp] = igl.point_mesh_squared_distance(spring_Vs_i, X, np.arange(X.shape[0])[:, None])
        #     spring_Vs_on_mesh_i = X[bI]
        #     E_actuated_on_mesh_i = bI[E_actuated_i]
        #     E_actuated_list.append(E_actuated_on_mesh_i)
        # E_actuated = np.vstack(E_actuated_list)
        
        X_boxes, _, _, F_boxes, _, _ = igl.readOBJ(data_dir + name + "_spring_muscles_" + muscle_name + ".obj")
        spring_Vs_i, E_actuated_i = boxes_to_edges(X_boxes, F_boxes)
        [sqrdD, bI, cp] = igl.point_mesh_squared_distance(spring_Vs_i, X, np.arange(X.shape[0])[:, None])
        spring_Vs_on_mesh_i = X[bI]
        E_actuated = bI[E_actuated_i]
    
        control_edge_indices = np.arange(E_actuated.shape[0]) 


        cache_path = result_dir + "/" + method + "_" + muscle_name + "_subspace.npz"
        compute_subspace_func = lambda : compute_subspace()
        [B, W, cI, cW, labels] = compute_with_cache_check(compute_subspace_func, cache_path, read_cache=read_subspace_cache)
        params = ARAPSpringActuatorPDSimParams(rho=rho, h=1e-2, ym=ym_vol, ym_edge=ym_edges, Q0=B.T @ H_pin @ B, b0=B.T @ b_pin)
        params.solver_p.max_iter = local_global_max_iter
        sim = ARAPSpringActuatorPDSim(control_edge_indices,E_actuated,  X, T, B, labels=labels, params=params)
        interactive_app(sim)
        # from scipy.io import savemat
        # savemat(directory + "/" + muscle_name + ".mat", {"B": B, "X": X, "T": T})

        

        
        if compute_times:
            params = ARAPSpringActuatorPDSimParams(rho=rho, h=1e-2, ym=ym_vol, ym_edge=ym_edges, Q0=B.T @ H_pin @ B, b0=B.T @ b_pin)
            params.solver_p.max_iter = local_global_max_iter
            sim = ARAPSpringActuatorPDSim(control_edge_indices,E_actuated,  X, T, B, labels=labels, params=params)
            
            Z = None 
            def compute_result_func():
                global Z
                Z, actuations, Ps, pIs = simulate(sim, num_timesteps, target_indices_body_func, target_pos_body_func)
                return Z, actuations, Ps, pIs
            
            num_runs = 3
            if method == "full":
                num_timesteps = 10
            
            timings = timeit.repeat(compute_result_func, number=1, repeat=num_runs)
            print("Dragon " + method + " sim time: " + str(np.array(timings).mean()/num_timesteps))    
            
            def reconstruct():
                Us = (B @ Z).reshape(X.shape[0], dim, num_timesteps)
                return Us
            
            
            if method is not "full":
                timings_reconstruct = timeit.repeat(reconstruct, number=1, repeat=num_runs)
                print("Dragon " + method + " reconstruct time: " + str(np.array(timings_reconstruct).mean()/num_timesteps))    
            else:
                timings_reconstruct = np.array([np.inf])
                
            # save all the above metrics into a human readable text file
            metrics = {
                "Dragon " + method + " vertices: " :  str(X.shape[0]),
                "Dragon " + method + " subspace dim: " :  str(sim.B.shape[1]),
                "Dragon " + method + " cubature points: " :  str(cI.shape[0]),
                "Dragon " + method + " timesteps: " :  str(num_timesteps),
                "Dragon " + method + " sim time: " :  str(np.array(timings).mean()/num_timesteps),
                "Dragon " + method + " reconstruct time: " :  str(np.array(timings_reconstruct).mean()/num_timesteps),
            }
            with open(result_dir + "/" + method + "_metrics.txt", "w") as f:
                for key, value in metrics.items():
                    f.write(key + ": " + value + "\n")
                

        else:
            def compute_result_func():
                params = ARAPSpringActuatorPDSimParams(rho=rho, h=1e-2, ym=ym_vol, ym_edge=ym_edges, Q0=B.T @ H_pin @ B, b0=B.T @ b_pin)
                params.solver_p.max_iter = local_global_max_iter
                sim = ARAPSpringActuatorPDSim(control_edge_indices,E_actuated,  X, T, B, labels=labels, params=params)
                
                # recording_path = result_dir + "/" +  muscle_name + "_recording.npz"
                # interactive_app(recording_path)
                
            
                
                if muscle_name == "head":
                    Z, actuations, Ps, pIs = simulate(sim, num_timesteps, target_indices_head_func, target_pos_head_func)
                elif muscle_name == "left_front_leg":
                    Z, actuations, Ps, pIs = simulate(sim, num_timesteps, target_indices_left_leg_func, target_pos_left_leg_func)
                elif muscle_name == "tail":
                    Z, actuations, Ps, pIs = simulate(sim, num_timesteps, target_indices_tail_func, target_pos_tail_func)
                elif muscle_name == "body":
                    Z, actuations, Ps, pIs = simulate(sim, num_timesteps, target_indices_body_func, target_pos_body_func)
                return Z, actuations, Ps, pIs
            
            result_cache_path = result_dir + "/" + method + "_" + muscle_name + "_result.npz"
            Z, actuations, Ps, pIs = compute_with_cache_check(compute_result_func, result_cache_path, read_cache=read_result_cache)
            
            U = (B @ Z).reshape(X.shape[0], dim, num_timesteps)
            animation_viewer(U, F, P=Ps.reshape(1, dim, num_timesteps))
            
            
            # kwargs = {
            #     "camLocation": np.array([12, -9, 1.145]),
            #     "lookAtLocation": np.array([0, -2, 0]),
            #     "numSamples": 200,
            #     "imgRes_x": 2048,
            #     "imgRes_y": 2048,
            #     "exposure": 2,
            #     "location": np.array([0, 0, 0]),
            #     "rotation": np.array([90, 0, 0]),
            #     "scale": np.array([1, 1, 1]),
            #     "lightAngle": np.array([-50, 5, -127.90]),
            #     "lightStrength": 1,
            #     "lightAngle2": np.array([-86.7904, 0.25, 172]),            
            #     "tex_png": None,
            #     "tex_uv": None,
            #     "fps": 60,
            #     "shade_smooth": True,
            #     "save_blend_file": True,
            #     "mesh_color": [255/255, 255/255, 255/255, 255/255],
            #     "edge_radius": 0.05,
            #     "edge_color": [49/255, 130/255, 189/255, 255/255],
            #     "alpha": 0.2,
            #     "transmission": 0.6,
            #     "edge_colormap_path": data_dir + "/../../colormaps/Blues_11.png",
            #     "source_color": [20/255, 255/255, 20/255, 255/255], # nice green
            #     "target_color": [49/255, 130/255, 189/255, 255/255], # nice red
            #     "source_radius": 0.1,
            #     "target_radius": 0.1,
            #     "tex_png": texture_path,
            #     "tex_uv": tex_uv,
            #     "uv_type" : "per_corner"
                
            # }
            
            # render_path = result_dir + "/" + method + "_" + muscle_name + "_edge_vis.mp4"
            # actuations_0 = actuations[:, [0]]
            # actuations = np.abs(actuations - actuations_0)
            # Y =U.reshape(X.shape[0], dim, num_timesteps)
            # pIs = pIs.astype(int).flatten()
            # time = np.arange(num_timesteps)
            # P_source = Y[pIs, :, time].T
            # P_target = Ps
            # frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # render_edges_on_transparent_mesh_animation(Y[:, :, frames],  
            #                                         F, E_actuated, 
            #                                         render_path, P_target=P_target[:, frames], P_source=P_source[:, frames],
            #                                         edge_scalars=actuations[:, frames], frames=frames,**kwargs)
            
            

            # render_path = result_dir + "/" + method + "_" + muscle_name + "_final_render.mp4"
            # Y_tex = (map @ Y.reshape(X.shape[0], -1)).reshape(X_tex.shape[0], dim, num_timesteps)
            # render_spring_ik_animation(Y_tex[:, :, frames], F_tex, 
            #                         Y[:, :, frames], F, E_actuated, render_path, 
            #                         P_target=P_target[:, frames], P_source=P_source[:, frames], edge_scalars=actuations[:, frames], **kwargs, 
            #                         frames=frames)
        
    