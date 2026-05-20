
import os
import timeit
import igl
import numpy as np
import scipy as sp
import time

import sys
from PIL import Image




sys.path.append(os.path.dirname(__file__) + "/../../../")

import simkit as sk
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_scalar_field import view_scalar_field
from simkit.spectral_clustering import spectral_clustering
from simkit.blender.render_vertex_scalars import render_vertex_scalars
from simkit.average_onto_simplex import average_onto_simplex
from simkit.blender.render_scene_animation import render_scene_animation
from simkit.closest_point_map import closest_point_map
from simkit.diffuse_scalar import diffuse_scalar
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.filesystem.compute_with_cache_check import compute_with_cache_check
from simkit.fold_in_vector_subspace import fold_in_vector_subspace
# from simkit.force_dual_modes import force_dual_modes_diagonal
from simkit.lbs_jacobian import lbs_jacobian
from simkit.log_likelihood import conditional_likelihoods_diagonal, log_likelihoods_diagonal
from simkit.polyscope.view_animation import view_animation
from simkit.polyscope.view_scalar_fields import view_scalar_fields
# from simkit.sims.elastic.AdaptiveSphereContactFEMSim import AdaptiveSphereContactFEMSim
from simkit.sims.elastic.ElasticFEMSim import ElasticFEMSimParams
from simkit.spectral_cubature import spectral_cubature
from simkit.eigs import eigs
from simkit.fold_vector_hessian import fold_vector_hessian
from simkit.normalize_and_center import normalize_and_center
from simkit.ympr_to_lame import ympr_to_lame
# from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.massmatrix import massmatrix

from gmm_force_expansion import gmm_force_expansion
from AdaptiveClickFEMSim import AdaptiveClickFEMSim

def interactive_mouse_app(mode, sim, Bs, cIs, cWs, diagonal_distributions, uv=None, tex=None, M_tex=None, F_tex=None, marginalize_ind=None):
    Sigma_Fs = [sp.sparse.diags(diagonal_distributions[i].flatten()).tocsc() for i in range(len(diagonal_distributions))]
   
   
    var = fold_in_vector_subspace(Sigma_Fs[mode].diagonal()[:, None], 3).sum(axis=1)
    
    import polyscope as ps
    import polyscope.imgui as psim
    ps.init()
   
    do_texture = False
    if uv is not None and tex is not None and M_tex is not None and F_tex is not None:
        do_texture = True
        
        mesh = ps.register_surface_mesh("mesh2", M_tex @ X_norm, F_tex)
        mesh.add_parameterization_quantity("test_param",  tex_uv,
                                defined_on='corners')
        mesh.add_color_quantity("test_vals", tex_png[:, :, 0:3]/255,
                                defined_on='texture', param_name="test_param",
                                        enabled=True)
    else:
        mesh = ps.register_volume_mesh("couch", sim.X, sim.T)
        mesh.add_scalar_quantity("variance", var.flatten(), enabled=True)
        M_tex = sp.sparse.identity(sim.X.shape[0])

    pc = None
    # pc  = ps.register_point_cloud("sphere", sim.sphere_center, radius=sim.sphere_radius/2)
    # ps.show()
    z, z_dot = sim.rest_state()
    # pc2 = ps.register_point_cloud("contact", (sim.SB @ z + sim.Sex0).reshape(-1, 3), radius=0.01)
    
    x_mean = sim.X.mean(axis=0)[None, :]
    camera_pos2 = x_mean + np.array([[0, 0, 5]])
    ps.look_at(camera_pos2.flatten(), x_mean.flatten())
    
    M = massmatrix(sim.X, sim.T)
    Me = sp.sparse.kron(M, sp.sparse.eye(dim)).tocsc()
    z_value = x_mean[:, -1]
    speed = 0.01
    d = 0
    def callback():
        nonlocal  z, z_dot, mode, sim, z_value, pc, d

        win_pos = psim.GetMousePos()    
        view = ps.get_camera_view_matrix()
        camera_pos = np.linalg.inv(view)[:3, 3]  # get camera position

        # if right mouse button is clicked, place a point on the mesh.
        if psim.IsMouseClicked(1):     
            if sim.num_handles > 0:
                sim.remove_handle(0)
                
            pos = ps.screen_coords_to_world_position(win_pos).reshape(-1, 3)    # use polyscope to find intersection into scene
            d = np.linalg.norm(pos - camera_pos)    # remember depth for future dragging
            pc = ps.register_point_cloud("clicked", pos, radius=0.01)   #vis
            
            U = sim.X + (sim.B @ z).reshape(-1, 3)
            [_sqrD, vI, _cp] = igl.point_mesh_squared_distance(pos,U, np.arange(sim.X.shape[0])[:, None])
            
            sim.add_handle(vI,U[vI, :])
            
            # if point being moved exists, and space is being held down, move the point by dragging mouse around
        if pc is not None and psim.IsKeyDown(psim.GetKeyIndex(psim.ImGuiKey_Space)): 
            ray = ps.screen_coords_to_world_ray(win_pos)   # shoot ray from camera into scene
            P = (camera_pos + d * ray).reshape(-1, 3)   # get final position
            pc.update_point_positions(P)

            sim.update_handle_position(0, P)
    
        if sim.num_handles > 0:
            force = sim.query_handle_force(z)
            # marginalize_ind = np.where(np.abs(force) > 1e-6)[0]
            cl, cll = conditional_likelihoods_diagonal(force, Sigma_Fs, marginalize=marginalize_ind)               
                
            new_force_mixture_index = np.argmax(cl)
            if new_force_mixture_index != mode:
                mode = new_force_mixture_index
                B_old = sim.B.copy()  
                
                BMB = Bs[mode].T @ Me @ Bs[mode]
                BMB_old = Bs[mode].T @ Me @ B_old
                z = sp.sparse.linalg.spsolve(BMB, BMB_old @ z).reshape(-1, 1)
                z_dot = sp.sparse.linalg.spsolve(BMB, BMB_old @ z_dot).reshape(-1, 1)
                # sim.update_subspace(B, cI, cW)
                sim.update_subspace(Bs[mode], cIs[mode], cWs[mode])
                # var = fold_in_vector_subspace(Sigma_Fs[mode].diagonal()[:, None], 3).sum(axis=1)
                if not do_texture:
                    mesh.add_scalar_quantity("variance", var.flatten(), enabled=True)
        
        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z) / sim.p.h
        z = z_next.copy()
        
        # pc.update_point_positions(sim.sphere_center)
        U =sim.X + (sim.B @ z).reshape(-1, 3)
        mesh.update_vertex_positions(M_tex @ U)
    
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(callback)
    ps.show()
    
    
def simulate(mode, bI_func, bc_pos_func, sim : AdaptiveClickFEMSim, Bs, cIs, cWs, diagonal_distributions, num_timesteps, Sigma_Fs_inv=None, sumlogcovs=None, marginalize_ind=None):
    # Sigma_Fs = [sp.sparse.diags(diagonal_distributions[i].flatten()).tocsc() for i in range(len(diagonal_distributions))]
   
    z, z_dot = sim.rest_state()
    x_mean = sim.X.mean(axis=0)[None, :]
    camera_pos2 = x_mean + np.array([[0, 0, 5]])
    
    M = massmatrix(sim.X, sim.T)
    Me = sp.sparse.kron(M, sp.sparse.eye(dim)).tocsc()
    
    max_dim = 0
    for i in range(len(Bs)):
        max_dim = max(max_dim, Bs[i].shape[1])
        
    Z = np.zeros((max_dim, num_timesteps))
    mode_indices = np.zeros(num_timesteps, dtype=int)
    vI = bI_func(0)
    bc_handle = bc_pos_func(0)
    for i,vi in enumerate(vI):
        sim.add_handle(np.array([vi]), bc_handle[i]) 
    
    i = 0
    # import polyscope as ps
    # ps.init()
    # ps.set_ground_plane_mode("none")
    # mesh_new = ps.register_volume_mesh("mesh", X, T)
    for i in range(num_timesteps):
        # if point being moved exists, and space is being held down, move the point by dragging mouse around
        # vI_new = bI_func(i)
        # if vI_new != vI:
        #     sim.remove_handle(0)
        #     sim.add_handle(vI_new, bc_pos_func(i))
        #     vI = vI_new#
        
        bc_handle = bc_pos_func(i)
        for c,vi in enumerate(vI):
            sim.update_handle_position(c, bc_handle[c])
            
        if sim.num_handles > 0:
            force = sim.query_handle_force(z)
            marginalize_ind = np.where(np.abs(force) > 1e-12)[0]
            log_likelihood = log_likelihoods_diagonal(force, Sigma_Fs, covs_inv=Sigma_Fs_inv, sumlogcovs=sumlogcovs, marginalize=marginalize_ind)
            cl, cll = conditional_likelihoods_diagonal(force, Sigma_Fs, covs_inv=Sigma_Fs_inv, sumlogcovs=sumlogcovs, marginalize=marginalize_ind)#, marginalize=marginalize_ind)               
            new_force_mixture_index = np.argmax(cl)
            if new_force_mixture_index != mode:
                print("########################################################")
                print("Mode Changed at time: " + str(i))
                print("Marginalized indices: " + str(marginalize_ind))
                print("old mode: " + str(mode))
                print("new mode: " + str(new_force_mixture_index))
                print("force: " + str(force[marginalize_ind]))
                print("cl: " + str(cl))
                print("cll: " + str(cll))
                print("ll : " + str(log_likelihood))
                mode = new_force_mixture_index
                B_old = sim.B.copy()  
                B_new = Bs[mode]
                BMB = B_new.T @ (Me @ B_new)
                BMB_old = B_new.T @ (Me @ B_old)
                z_new = np.linalg.solve(BMB, BMB_old @ z).reshape(-1, 1)
                z_dot_new = np.linalg.solve(BMB, BMB_old @ z_dot).reshape(-1, 1)
                
                # u_new = B_new @ z_new
                # u_old = B_old @ z
                # import polyscope as ps
                # ps.init()
                # ps.set_ground_plane_mode("none")
                # ps.remove_all_structures()
                # # pc = ps.register_point_cloud("pc", X[bI_handle, :])
                # mesh_new = ps.register_volume_mesh("mesh", X + u_new.reshape(-1, 3), T)
                # mesh_old = ps.register_volume_mesh("mesh_old", X + u_old.reshape(-1, 3), T)
                # pc_end = ps.register_point_cloud("pc_end", bc_handle_end)
                # pc_start = ps.register_point_cloud("pc_start", bc_handle_start)
                # u_handle = sim.SB @ z
                # pc_handle = ps.register_point_cloud("pc_handle", X0[bI_handle, :] + u_handle.reshape(-1, 3))
                # # pc_handle.add_vector_quantity("force", force[marginalize_ind].reshape(-1, 3))
                # # mesh.add_scalar_quantity("u_new", X + u_new.reshape(-1, 3), enabled=True)
                # # pc.add_scalar_quantity("u_old", X + u_old.reshape(-1, 3), enabled=True)
                # ps.show()
                # z = np.zeros((z.shape[0], 1)) #sp.sparse.linalg.spsolve(BMB, BMB_old @ z).reshape(-1, 1)
                # z_dot = np.zeros((z_dot.shape[0], 1)) #sp.sparse.linalg.spsolve(BMB, BMB_old @ z_dot).reshape(-1, 1)
                # sim.update_subspace(B, cI, cW)
                sim.update_subspace(B_new, cIs[mode], cWs[mode])
                z = z_new.copy()
                z_dot = z_dot_new.copy()
                
                # u_new = Bs[mode] @ z
                # mesh_new.update_vertex_positions(X + u_new.reshape(-1, 3))
                # ps.frame_tick()
                
   
        z_next = sim.step(z, z_dot)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        
        u_new = Bs[mode] @ z
        # mesh_new.update_vertex_positions(X + u_new.reshape(-1, 3))
        # ps.frame_tick()
        Z[:Bs[mode].shape[1], i] = z.flatten()
        
        mode_indices[i] = mode
    sim.remove_handle(0)
    return Z, mode_indices
        
  


    
def compute_subspace(X, T, H, M,  subspace_dim, num_cubature_points, method):
    
    F = igl.boundary_facets(T)[0]
    bI = np.unique(F)
    L = fold_vector_hessian(H + H_pin, 3 )
    Ms = fold_vector_hessian(M, 3 )
    
    if method == "full":
        B = sp.sparse.identity(X.shape[0]*X.shape[1])
        W = sp.sparse.identity(X.shape[0]*X.shape[1])
        cI = np.arange(T.shape[0])
        cW = sk.volume(X, T)
        return [B], [W], cI[None, :], cW[None, :], [np.ones((1, X.shape[0]*X.shape[1]))]
    
    if method == "skinning_eigenmodes":
        [Eval, W] = eigs(L, k=subspace_dim, M=Ms) 
        B = lbs_jacobian(X, W)
        cI, cW = spectral_cubature(X, T, W, num_cubature_points )
        Ma = massmatrix(X, F)
        m = Ma.diagonal()
        m = m[bI]
        p = m / m.sum()
        
        return B[None, :], W[None, :], cI[None, :], cW[None, :], [np.ones((1, X.shape[0]*X.shape[1]))]

    elif method == "force_dual_skinning_eigenmodes" or method == "force_dual_skinning_eigenmodes_expanded":
        # build distribution 
        # initialize empty numpy arrays instead of lists
        Bs = np.zeros((distributions.shape[1], X.shape[0]*X.shape[1], subspace_dim*12))
        Ws = np.zeros((distributions.shape[1], X.shape[0], subspace_dim))
        cIs = np.zeros((distributions.shape[1], num_cubature_points), dtype=int)
        cWs = np.zeros((distributions.shape[1], num_cubature_points))
        # Sigma_Fs = np.zeros((distributions.shape[1], X.shape[0], X.shape[0]))
        diagonal_distributions = np.zeros((distributions.shape[1], X.shape[0]*X.shape[1]))

        for i in range(distributions.shape[1]):
            diagonal_distributions[i] = (np.kron(distributions[:, [i]], np.ones((X.shape[1], 1)))).flatten()
            Sigma_F = sp.sparse.diags((distributions[:, i]))
            Sigma_F = sp.sparse.kron(Sigma_F, sp.sparse.eye(dim))
            Sigma_F_inv = sp.sparse.diags(1.0 / Sigma_F.diagonal())
            HiH = H @ Sigma_F_inv @ H.T
            L = fold_vector_hessian(HiH, dim)
            Msi = sp.sparse.diags(1.0/ Ms.diagonal())
            Di, Wi = eigs(L, subspace_dim, M=Ms)
            Ws[i, :, :] = Wi
            B = lbs_jacobian(X, Wi)
            Bs[i, :, :] = B
            cI, cW = spectral_cubature(X, T, Wi, num_cubature_points)  
            cWs[i, :] = cW
            cIs[i, :] = cI
            Ma = massmatrix(X, F)
            m   = Ma.diagonal()

    if method == "force_dual_skinning_eigenmodes_expanded":
        diagonal_distributions_expanded, force_mixture_subspaces = gmm_force_expansion(diagonal_distributions, 2)
        Bs_expanded = [] 
        Ws_expanded = []
        cIs_expanded = []
        cWs_expanded = []
        for i in range(len(diagonal_distributions_expanded)):
            
            W = np.concatenate(Ws[force_mixture_subspaces[i], :, :].transpose(0, 2, 1), axis=0).T
            B = lbs_jacobian(X, W)
            cI, cW = spectral_cubature(X, T, W, num_cubature_points)  
            
            Ws_expanded.append(W)
            Bs_expanded.append(B)
            cIs_expanded.append(cI)
            cWs_expanded.append(cW)
            
            # B = np.concatenate(Bs[force_mixture_subspaces[i], :, :].transpose(0, 2, 1), axis=0).T
            # Bs_expanded.append(B)
            # # sk.polyscope.view_displacement_modes(X, T, B)
            # Ws_expanded[i, :, :] = Ws[force_mixture_subspaces[i], :, :]
            # cI, cW = spectral_cubature(X, T, Ws_expanded[i, :, :], num_cubature_points)  
            # cIs_expanded[i, :] = cI
            # cWs_expanded[i, :] = cW
            # diagonal_distributions_expanded[i] = diagonal_distributions_expanded[i]
        
        Bs = Bs_expanded
        Ws = Ws_expanded
        cIs = cIs_expanded
        cWs = cWs_expanded
        diagonal_distributions = np.array(diagonal_distributions_expanded)
    return Bs, Ws, cIs, cWs, diagonal_distributions


start_pos_wing = np.array([[0.962, 0.57, 0.15]])
end_pos_wing = np.array([[0.253, -0.2, -0.0]])

start_pos_horn = np.array([[0.012, 0.53, -0.79]])
end_pos_horn = np.array([[0.02, 0.06, -0.90]]) 

num_timesteps_wing = 100
num_timesteps_horn = 100
num_timesteps_return = 100
num_timesteps = num_timesteps_wing + num_timesteps_horn + num_timesteps_return
def bI_func(i):
    return bI_handle
    # if i < num_timesteps_wing:
    #     return igl.point_mesh_squared_distance(start_pos_wing, X, np.arange(X.shape[0])[:, None])[1]
    # else:
    #     return igl.point_mesh_squared_distance(start_pos_horn, X, np.arange(X.shape[0])[:, None])[1]
    
    
# def bI_func
def bc_pos_func(i): 
    
    bc_handle = np.zeros((2, 3), dtype=float)
    
    # if i <  num_timesteps:
    #     bc_handle = bc_handle_start + (bc_handle_end - bc_handle_start) * i / (num_timesteps_return)
    if i < num_timesteps_wing:
        bc_handle[1] = bc_handle_start[1] + (bc_handle_end[1] - bc_handle_start[1]) * i / (num_timesteps_wing)
        bc_handle[0] = bc_handle_start[0]
    elif i < num_timesteps_wing + num_timesteps_horn:
        bc_handle[0] = bc_handle_start[0] + (bc_handle_end[0] - bc_handle_start[0]) * (i - num_timesteps_wing) / (num_timesteps_horn)
        bc_handle[1] = bc_handle_end[1]
    
    # if i < num_timesteps_horn:
    #     bc_handle[0] = bc_handle_start[0] + (bc_handle_end[0] - bc_handle_start[0]) * i / (num_timesteps_horn)
    #     bc_handle[0] = bc_handle_start[0]
    # elif i < num_timesteps_wing + num_timesteps_horn:
    #     bc_handle[0] = bc_handle_start[0] + (bc_handle_end[0] - bc_handle_start[0]) * (i - num_timesteps_horn) / (num_timesteps_wing)
    #     bc_handle[1] = bc_handle_end[1]
    elif i < num_timesteps_wing + num_timesteps_horn + num_timesteps_return:
        bc_handle = bc_handle_end + (bc_handle_start - bc_handle_end) * (i - num_timesteps_wing - num_timesteps_horn) / (num_timesteps_return)
    return bc_handle
name = "pegasus"
methods = ["skinning_eigenmodes"] #  "force_dual_skinning_eigenmodes_expanded"] #,  "skinning_eigenmodes", ]

# method = "skinning_eigenmodes"#
directory = os.path.dirname(__file__)
data_dir = directory + "/../../data/3d/" + name
load_colormap = directory + "/../../data/colormaps/Purples_11.png"
result_dir = directory + "/results/" + name + "/"


distributions = np.load(data_dir + "/pegasus_distribution_global_sharp.npy")

distributions = np.clip(distributions, 1e-6, 1)[:, [0, 2]]

k_handle = 1e6
distributions = distributions * k_handle
gamma_pin = 1e8
subspace_dim = 4
num_cubature_points = 200
read_subspace_cache = False
read_result_cache =  False
os.makedirs(result_dir, exist_ok=True)
[X0, T, F] = igl.readMESH(data_dir + "/" + name + ".mesh")
F = igl.boundary_facets(T)[0]
bI = np.unique(F)

timing = False


# Xs =X0[bI, :]
# Xs_norm, to, so = normalize_and_center(Xs, return_params=True)
# X_norm = X0.copy()
# X_norm = X_norm + to
# X_norm *= so

[X_tex, UV_tex, N_tex, F_tex, FUV_tex, FN_tex] = igl.readOBJ(data_dir + "/unicorn_tex.obj")
# X_norm += X_tex.mean(axis=0)

map =closest_point_map(X_tex, X0)

# import polyscope as ps
# ps.init()
# ps.set_ground_plane_mode("none")
# mesh = ps.register_surface_mesh("unicorn", X_tex, F_tex)
# mesh2 = ps.register_volume_mesh("unicorn", X0, T)
# ps.show()
M_tex = sp.sparse.kron(map, sp.sparse.identity(3))
tex_uv = UV_tex[FUV_tex.flatten()]
tex_path = data_dir + "/unicorn_tex.png"
tex_png = np.array(Image.open(tex_path) )

X, to, so =normalize_and_center(X0, return_params=True)
dim = X.shape[1]
materials= np.load(data_dir + "/" + name + "_materials.npy")
ym = materials[:, 0]
pr = materials[:, 1]
mu, lam = sk.ympr_to_lame(ym, 0.0)
M = sk.massmatrix(X, T)
distributions = np.sqrt(M) @ distributions
M_inv_sqrt = sp.sparse.diags(1.0 / np.sqrt(  M.diagonal()))
H = sk.energies.linear_elasticity_hessian(X=X, T=T, mu=mu, lam=lam).tocsc()

# [_sqrD, vI, _cP] = igl.point_mesh_squared_distance(start_pos, X, np.arange(X.shape[0])[:, None])
# view_scalar_fields(X, T, distributions)
pinned_vertices = np.where(np.linalg.norm(X - X.mean(axis=0)[None, :] - np.array([[0, -0.15, 0.0]]), axis=1) < 2e-1)[0]
# pinned_vertices = np.where(X[:, 1] < X[:, 1].min() + 1e-1)[0]

bc = X[pinned_vertices, :]
H_pin, b_pin = dirichlet_penalty(pinned_vertices, 0*bc, X.shape[0], gamma_pin)


wing_index = igl.point_mesh_squared_distance(start_pos_wing, X, np.arange(X.shape[0])[:, None])[1]
horn_index = igl.point_mesh_squared_distance(start_pos_horn, X, np.arange(X.shape[0])[:, None])[1]
bI_handle = np.concatenate([horn_index, wing_index])
bc_handle = X[bI_handle, :].copy()
bc_handle_end = np.concatenate([end_pos_horn, end_pos_wing])
bc_handle_start = bc_handle.copy()

marginalize_ind = (np.repeat(bI_handle[:, None], 3, axis=1) * 3 + np.arange(3)[None, :]).flatten()
L = fold_vector_hessian(H + H_pin, 3)

Me = sp.sparse.kron(M, sp.sparse.eye(dim)).tocsc()

ball_radius = 0.05

[X_tex_ball, UV_tex_ball, N_tex_ball, F_tex_ball, FUV_tex_ball, FN_tex_ball] = igl.readOBJ(data_dir + "/../ball/ball_tex.obj")
tex_ball_png = (data_dir + "/../ball/ball_red_tex.png")
tex_ball_uv = UV_tex_ball[FUV_tex_ball.flatten()]


X_tex_ball *= ball_radius
X_tex_ball0 = X_tex_ball.copy()



for method in methods:
    subspace_cache_dir = result_dir + "/" + method + "_subspace.npz"
    compute_subspace_func = lambda :  compute_subspace(X, T, H + H_pin, Me, 
                                                    subspace_dim,
                                                    num_cubature_points,
                                                        method)

    [Bs, Ws, cIs, cWs, diagonal_distributions] = compute_subspace_func()
    # [Bs, Ws, cIs, cWs, diagonal_distributions] = compute_with_cache_check(compute_subspace_func, 
    #                                                         subspace_cache_dir,
    #                                                         read_cache=read_subspace_cache)
    
    Sigma_Fs = [sp.sparse.diags(diagonal_distributions[i].flatten()).tocsc() for i in range(len(diagonal_distributions))]
    Sigma_Fs_inv = [sp.sparse.diags(1.0 / Sigma_Fs[i].diagonal()).tocsc() for i in range(len(Sigma_Fs))]
    sumlogcovs =[np.sum(np.log(Sigma_Fs[i].diagonal())) for i in range(len(Sigma_Fs))
                ]
    
    
    if timing:
        sim_params = ElasticFEMSimParams(ym=ym, pr=0.45, h=0.01, Q0=H_pin, b0=b_pin)
        sim_params.solver_p.max_iter = 1
        sim_params.solver_p.tolerance = 0
        

        mode = 0
        sim = AdaptiveClickFEMSim(k_handle, X, T,Bs[mode], cIs[mode], cWs[mode], sim_params=sim_params, q=X.reshape(-1, 1))
        Zs = None
        mode_indices = None
        def compute_result():
            global Zs, mode_indices
            [Zs, mode_indices] = simulate(mode, bI_func, bc_pos_func,  sim, Bs, cIs, cWs, Sigma_Fs, num_timesteps, Sigma_Fs_inv=Sigma_Fs_inv, sumlogcovs=sumlogcovs)   
            return Zs, mode_indices
     
        # Zs, mode_indices = compute_result()
        
        #print number of vertices
        print("Pegasus " + method + " vertices: " + str(X.shape[0]))
        # print subspace dimension  
        print("Pegasus " + method + " subspace dim: " + str(sim.B.shape[1]))
        # print number of contact points
        
        # print number of clusters/cubature
        print("Pegasus " + method + " cubature points: " + str(cIs[0].shape[0]))
        
        # print number of timesteps
        print("Pegasus " + method + " timesteps: " + str(num_timesteps))
        
        #print max iter
        print("Pegasus " + method + " max iter: " + str(sim_params.solver_p.max_iter))
        
        
        num_runs = 3
        if method == "full":
            num_timesteps = 10
            
        timings = timeit.repeat(compute_result, number=1, repeat=num_runs)
        print(" Pegasus " + method + " sim time: " + str(np.array(timings).mean()/num_timesteps))    
                
        def reconstruct():
            Us = (Bs[ mode_indices, :] @ Zs.T[:, :, None]).reshape(mode_indices.shape[0], -1, 3)
            
        if method is not "full":
            timings_reconstruct = timeit.repeat(reconstruct, number=1, repeat=num_runs)
            print(" Pegasus " + method + " reconstruct time: " + str(np.array(timings_reconstruct).mean()/num_timesteps))    
        else:
            timings_reconstruct = np.array([np.inf])
        # save all the above metrics into a human readable text file
        metrics = {
            "Pegasus " + method + " vertices: " :  str(X.shape[0]),
            "Pegasus " + method + " subspace dim: " :  str(sim.B.shape[1]),
            "Pegasus " + method + " cubature points: " :  str(cIs[0].shape[0]),
            "Pegasus " + method + " timesteps: " :  str(num_timesteps),
            "Pegasus " + method + " sim time: " :  str(np.array(timings).mean()/num_timesteps),
            "Pegasus " + method + " reconstruct time: " :  str(np.array(timings_reconstruct).mean()/num_timesteps),
        }
        # save into a human readoble file
        with open(result_dir + "/" + method + "_metrics.txt", "w") as f:
            for key, value in metrics.items():
                f.write(key + ": " + value + "\n")
        
    elif not timing:
        
        def compute_result():
            sim_params = ElasticFEMSimParams(ym=ym, pr=0.45, h=0.01, Q0=H_pin, b0=b_pin)
            sim_params.solver_p.max_iter = 1
            sim_params.solver_p.tolerance = 0
            
            mode = 0
            sim = AdaptiveClickFEMSim(k_handle, X, T,Bs[mode], cIs[mode], cWs[mode], sim_params=sim_params, q=X.reshape(-1, 1))
        
            [Zs, mode_indices] = simulate(mode, bI_func, bc_pos_func,  sim, Bs, cIs, cWs,  Sigma_Fs, num_timesteps, Sigma_Fs_inv=Sigma_Fs_inv, sumlogcovs=sumlogcovs, marginalize_ind=marginalize_ind)   
            return Zs, mode_indices
        
        result_cache_dir = result_dir + "/" + method + "_result.npz"
        [Zs, mode_indices] = compute_result()
        # [Zs, mode_indices] = compute_with_cache_check(compute_result, result_cache_dir, read_cache=read_result_cache)    
        
        Us = np.zeros((X.shape[0], X.shape[1], num_timesteps))
        for t in range(num_timesteps):
            mode_indices[t] = mode_indices[t]
            B = Bs[ mode_indices[t]]
            dim_B = B.shape[1]
            Us[:, :, t] = ( B@ Zs[:dim_B, t ]).reshape(-1, 3)
        # Us = (Bs[ mode_indices, :] @ Zs.T[:, :, None]).reshape(mode_indices.shape[0], -1, 3)
        view_animation(X, T, Us.reshape(X.shape[0]*X.shape[1], -1))
        
        # sk.polyscope.view_scalar_fields(X, T, distributions)
        # sk.polyscope.view_displacement_modes(X, T, Bs[1])
        # mode = 0
        # sim_params = ElasticFEMSimParams(ym=ym, pr=0.45, h=0.01, Q0=H_pin, b0=b_pin)
        # sim_params.solver_p.max_iter = 1
        # sim_params.solver_p.tolerance = 0    
        # sim = AdaptiveClickFEMSim(k_handle, X, T,Bs[mode], cIs[mode], cWs[mode], sim_params=sim_params, q=X.reshape(-1, 1))
        # # interactive_mouse_app(mode, sim, Bs, cIs, cWs, diagonal_distributions, uv=tex_uv, tex=tex_png, M_tex=map, F_tex=F_tex)
        
        # import polyscope as ps
        # ps.init()
        # ps.set_ground_plane_mode("none")
        # mesh = ps.register_surface_mesh("mesh", X, T)
        # pc = ps.register_point_cloud("pc", X[bI_handle, :])
        # ps.show()
        
        # X_tex_ball += X[vI, :]
        Us_ball_1 = np.zeros(( num_timesteps, X_tex_ball.shape[0], dim,))
        Us_ball_2 = np.zeros(( num_timesteps, X_tex_ball.shape[0], dim,))
        for i in range(num_timesteps):
            Us_ball_1[i, :,:] = bc_pos_func(i)[0]# - X[bI_func(i), :]
            Us_ball_2[i, :,:] = bc_pos_func(i)[1]# - X[bI_func(i), :]

        # # render_scalar_fields(X, T, diagonal_distributions)
        
        # render_path = result_dir + "/" + method + "_render.mp4"
        imgRes_x = 2038
        imgRes_y = 2048
        numSamples = 200    
        camLocation = [2.38077, 2.425, 1.1097]
        lookAtLocation = [0, 0, 0]
        lightAngle = [50.476, 3.6586, -122.22]
        lightStrength = 1
        lightAngle2 = [73.4, -15,125]
        lightStrength2 = 1
        frames =np.arange(num_timesteps)
        animation_kwargs = [
            {
                'X': (( M_tex @ X.reshape(-1, 1))).reshape(-1, dim),
                'F': F_tex,
                'U': (M_tex @ Us.reshape(-1, num_timesteps)).reshape(X_tex.shape[0], dim, -1)[:, :, frames],#.reshape(X.shape[0], dim, -1),
                'tex_png': tex_path,
                'tex_uv': tex_uv
            },
            {
                'X': X_tex_ball,
                'F': F_tex_ball,
                'U': (Us_ball_1).transpose(1, 2, 0).reshape(-1, num_timesteps).reshape(X_tex_ball.shape[0], dim, -1)[:, :, frames],  # Static collider
                'tex_png': tex_ball_png,
                'tex_uv': tex_ball_uv
            },
            {
                'X': X_tex_ball,
                'F': F_tex_ball,
                'U': (Us_ball_2).transpose(1, 2, 0).reshape(-1, num_timesteps).reshape(X_tex_ball.shape[0], dim, -1)[:, :, frames],  # Static collider
                'tex_png': tex_ball_png,
                'tex_uv': tex_ball_uv
            }
        ]

        # val = diagonal_distributions.reshape( -1, X.shape[0], X.shape[1])
        # vals = np.mean(val, axis=2).T
        
        # view_scalar_fields(X, T, val[:, :, 2].T)
        # render_path = result_dir + "/" + method + "_load_render/"
        # M_inv_sqrt_e = sp.sparse.kron(M_inv_sqrt, sp.sparse.eye(dim))
        # render_vertex_scalars(X, F, M_inv_sqrt_e @ diagonal_distributions.T, render_path,load_colormap,
        #                     imgRes_x=imgRes_x, imgRes_y=imgRes_y, numSamples=numSamples, 
        #                     camLocation = camLocation, lookAtLocation=lookAtLocation, lightAngle=lightAngle, 
        #                     lightStrength=lightStrength, lightAngle2=lightAngle2, lightStrength2=lightStrength2,
        #                     shade_smooth=True
        #                     )
        render_path = result_dir + "/" + method + "_sim_render.mp4"
        render_scene_animation(render_path, imgRes_x, imgRes_y, numSamples, 
                            camLocation = camLocation, lookAtLocation=lookAtLocation, lightAngle=lightAngle, 
                            lightStrength=lightStrength, lightAngle2=lightAngle2, lightStrength2=lightStrength2, 
                            animation_kwargs=animation_kwargs, shade_smooth=True, save_blend_file=True)
        
        
        