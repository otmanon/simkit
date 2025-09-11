# from simkit.worlds import ForceWorlds

import scipy as sp
from scipy.sparse import hstack, vstack
import numpy as np
import igl
import igl.triangle
import os
import polyscope as ps
import polyscope.imgui as psim
import sys

import blendertoolbox as bt

sys.path.append(os.path.dirname(__file__) + "/../../../")
from simkit import common_selections
from simkit.average_onto_simplex import average_onto_simplex
from simkit.blender.vertexScalarToUV_unnormalized import vertexScalarToUV_unnormalized
from simkit.diffuse_scalar import diffuse_scalar
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
from simkit.gravity_force import gravity_force
from simkit.harmonic_coordinates import harmonic_coordinates
from simkit.linear_elasticity_hessian import linear_elasticity_hessian
from simkit.massmatrix import massmatrix
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_fields import view_scalar_fields
from simkit.random_impulse_vibes import random_impulse_vibes
from simkit.rotate import rotate
from simkit.selection_matrix import selection_matrix
from simkit.shape_outlines import rectangle_outline, star_sign_outline
from simkit.sims.elastic.ElasticROMPDSim import ElasticROMPDSim, ElasticROMPDSimParams
from simkit.spectral_clustering import spectral_clustering
from simkit.umfpack_lu_solve import umfpack_lu_solve
from simkit.ympr_to_lame import ympr_to_lame


directory = os.path.dirname(__file__)
class ElephantConfig():
    def __init__(self):
        self.character_name = "elephant"
        self.pre_rotate = np.array([0,0 , 0])
        self.data_dir = directory + "/../../../data/3d/" + str(self.character_name) + "/"
        self.mesh_path = self.data_dir + str(self.character_name) + ".mesh"
        self.keypoints_path = self.data_dir + "key_vertices.npy"
        self.interaction_indices = [1] # indices into keypoints

        self.pinned_selection = "near"
        # self.pinned_selection_kwargs = {"thresh_top": 1, "thresh_bot": 0.1}

        self.pinned_selection_kwargs = {"t": 0.2, 'c' : np.array([[0, -0.5, 0]])}
        self.force =  1e3 * np.array([0, 1, 0])
        self.eye_pos = np.array([-2, 0.5, 3])
        self.look_at = np.array([0, 0, 0])
        self.m = 8


def sinusoidal_handle_updown(bI, X, num_timesteps, period, a=1):
    n = X.shape[0]
    dim = X.shape[1]
    y = X[bI, :].reshape(-1, 1)
    Y = np.repeat(y, num_timesteps, axis=1)
    direction = np.array([[0, 1, 0]])
    dd = np.ones((bI.shape[0], dim)) * direction
    d = dd.reshape(-1, 1)
    D= np.repeat(d, num_timesteps, axis=1)
    A = a*np.sin(2 * np.pi * np.arange(num_timesteps) / period)
    U = A * D + Y
    return U

def distributed_load_modes(H, m, d, pi, M=None):
    pass

def load_modes(H, m, D, M=None):
    D2 = umfpack_lu_solve(H, D)       
    B_interaction = np.linalg.svd(D2, full_matrices=False)[0]


    if m < B_interaction.shape[1]:
        B_la = B_interaction[:, :m]
    else:
        remainder = m - B_interaction.shape[1]
        # and do the rest as orthogonal to B_interaction
        J = B_interaction.copy()
        Z = sp.sparse.csc_matrix((J.shape[1], J.shape[1]))
        H_eq = vstack([hstack([H, J]), \
                    hstack([J.T, Z])]).tocsc()
        M_eq = sp.sparse.block_diag([M, Z]).tocsc()
        B_ortho_tmp = eigs(H_eq, remainder, M=M_eq)[1]
        B_ortho = B_ortho_tmp[:-J.shape[1], :]
        B_la = np.concatenate((B_interaction, B_ortho), axis=1)
    return B_la


class DistributedLinearForceWorld():
    def __init__(self, X, T, pinned_points, simulation_params):
        self.X = X
        self.T = T
        self.dim = X.shape[1]
        self.pinned_points = pinned_points
        self.simulation_params = simulation_params
        return

    def build_sim(self, B=None, labels=None, Q0=None):
        self.simulation_params.Q0 = Q0
        sim = ElasticROMPDSim(self.X, self.T, B=B, labels=labels, params=self.simulation_params, displacement=True)
        return sim
    
    
    def interact(self, B, labels, gamma=1e7):
        
        dim = self.X.shape[1]
        n = self.X.shape[0]
        ss = self.T.shape[1]

        # f = gravity_force(X, T, a=-9.8, rho=1e3).reshape(-1, 1)
        # self.simulation_params.Q0 *= 1e-5
        # self.simulation_params.b0 *= 1e-5
        # self.simulation_params.b0 -= f
        

        sim = ElasticROMPDSim(self.X, self.T, B=B, labels=labels, params=self.simulation_params, displacement=True)
        z , z_dot = sim.rest_state()
        z*= 0
        z_dot *= 0

        import polyscope as ps
        ps.init()
        ps.remove_all_structures()
        # ps.register_surface_mesh("mesh", self.X, self.T)
        mesh = ps.register_volume_mesh("mesh", self.X, self.T)
        pc = ps.register_point_cloud("keypoints", self.X[self.pinned_points, :])
        ps.set_ground_plane_mode("none")

        
        class Handle:
            def __init__(self, bI, bc):
                self.bI = np.array(bI).reshape(-1)
                self.bc = np.array(bc).reshape(-1, dim)

        handle = None
        Q_picked = None
        b_picked = None
        BSGamma = None
        speed = 1e-2

        Q0 = sim.params.Q0
        b0 = sim.params.b0
        handle_mesh = None
        handle_vertex_mesh = None
        def callback():
            nonlocal z, z_dot, handle, Q_picked, b_picked, BSGamma, sim, handle_mesh, handle_vertex_mesh

            if psim.IsMouseClicked(0):
                name, i= ps.get_selection()
                print(name)
                P = (B @ z).reshape(-1,dim) + self.X
                if name == 'mesh':
                    print("Clicked object : ", name)
                    is_vertex = i < n
                    is_face = i > n 
                    clicked_mesh = is_face or is_vertex
                    if is_vertex:
                        print("Clicked vertex index : ", i)
                        handle = Handle(i, P[i, :])

                    elif is_face:
                        print("Clicked face index : ", i - n)
                        # Faces = igl.boundary_facets(self.T)
                        bi = self.T[i - n, 0]
                        handle = Handle(bi, P[bi, :])

                    if clicked_mesh:
                        Q_picked, b_picked, SGamma = dirichlet_penalty(handle.bI, handle.bc,
                                                                                n,  gamma=gamma, only_b=False, return_SGamma=True)
                        BSGamma = B.T @ SGamma
                        sim.set_quadratic_penalty(Q_picked + sim.params.Q0, sim.params.b0)

                        handle_mesh = ps.register_point_cloud("handle", P[handle.bI, :], radius=0.02)
                        handle_vertex_mesh = ps.register_point_cloud("handle_vertex", P[handle.bI, :], radius=0.02)

            if handle is not None:
                if psim.IsKeyPressed(psim.ImGuiKey_J):
                    handle.bc += speed * np.array([[-1, 0., 0]])
                if psim.IsKeyPressed(psim.ImGuiKey_L):
                    handle.bc += speed * np.array([[1, 0., 0]])
                if psim.IsKeyPressed(psim.ImGuiKey_I):
                    handle.bc += speed * np.array([[0, 1., 0]])
                if psim.IsKeyPressed(psim.ImGuiKey_K):
                    handle.bc += speed * np.array([[0, -1., 0]])
                if psim.IsKeyPressed(psim.ImGuiKey_U):
                    handle.bc += speed * np.array([[0, 0., 1]])
                if psim.IsKeyPressed(psim.ImGuiKey_O): 
                    handle.bc += speed * np.array([[0, 0., -1]])

                handle_mesh.update_point_positions(handle.bc)
                b_picked =  -BSGamma @ handle.bc.reshape(-1, 1)

            z_next = sim.step(z, z_dot, b_ext=b_picked)
            z_dot = (z_next - z)/sim.params.h
            z = z_next.copy()

            u = B @ z
            U = u.reshape(-1, self.dim) + self.X
            mesh.update_vertex_positions(U)
            pc.update_point_positions(U[self.pinned_points, :])

            if handle_vertex_mesh is not None:
                handle_vertex_mesh.update_point_positions(U[handle.bI, :])

        ps.set_user_callback(callback)
        ps.show()

    def simulate(self, sim, num_timesteps, F=None):
        z , z_dot = sim.rest_state()
        z*= 0
        z_dot *= 0
        Z = np.zeros((sim.B.shape[1], num_timesteps))
        if F is None:
            F = np.zeros((z.shape[0], num_timesteps))
        for i in range(num_timesteps):
            f = F[:, i].reshape(-1, 1)
            z_next = sim.step(z, z_dot, b_ext=-f)
            z_dot = (z_next - z)/sim.params.h
            z = z_next.copy()
            Z[:, i] = z.flatten()

      
        return Z

    def render(self, B, Z, path=None, bI=None, eye_pos=None, look_at=None):
    
        if path is not None:
            import pathlib
            stem = pathlib.Path(path).stem
            result_dir = os.path.dirname(path) + "/" + stem + "/"
            os.makedirs(result_dir, exist_ok=True)
        ps.init()
        ps.remove_all_structures()
        mesh = ps.register_volume_mesh("mesh", self.X, self.T)
        pc = ps.register_point_cloud("keypoints", self.X[self.pinned_points, :])

        if bI is not None:
            pbI = ps.register_point_cloud("bI", self.X[bI, :], enabled=True, radius=0.01)
        ps.set_ground_plane_mode("none")
        
        if eye_pos is not None and look_at is not None:
            ps.look_at(eye_pos, look_at)

        for i in range(num_timesteps):
            if i < Z.shape[1]:
                z = Z[:, i]
                u = B @ z
                U = u.reshape(-1, self.dim) + self.X
                mesh.update_vertex_positions(U)
                pc.update_point_positions(U[self.pinned_points, :])

                if bI is not None:
                    pbI.update_point_positions(U[bI, :])

                if path is not None:
                    ps.screenshot(result_dir + str(i).zfill(4) + ".png", transparent_bg=False)

                ps.frame_tick()
            
        if path is not None:
            video_from_image_dir(result_dir, path, fps=30)
            gif_path = path[:-4] + ".gif"
            mp4_to_gif(path,  gif_path)
        return 
    
    def blender_render(self, B, Z, path, bI=None, eye_pos=[0, -5, 0], look_at=[0, 0,0 ], numSamples=2, exposure=2, imgres_x=1920, imgres_y=1080, frames=[0, 75], location=[0, 0, 0], rotation_euler=[90, 0, 0], scale=[1, 1, 1], color=bt.derekBlue):
    
        import pathlib
        stem = pathlib.Path(path).stem
        result_dir = os.path.dirname(path) + "/" + stem + "/"
        os.makedirs(result_dir, exist_ok=True)

        import blendertoolbox as bt
        import bpy


        if frames is None:
            frames = np.arange(Z.shape[1])

        for frame in frames:
            z = Z[:, frame]
            u = B @ z
            U = u.reshape(-1, self.dim) + self.X
            bt.blenderInit(imgres_x, imgres_y, numSamples=numSamples, exposure=exposure)
            mesh=  bt.readNumpyMesh( U, igl.boundary_facets(self.T)[0], location=location, rotation_euler=rotation_euler, scale=scale)  
            bpy.context.view_layer.objects.active = mesh
            mesh.select_set(True)
            bpy.ops.object.shade_smooth() 
            meshColor = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 2.0)
            bt.setMat_balloon(mesh, meshColor, 1)
            bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 
            # bt.createMesh("elephant", self.X, self.T)
            # bt.createMesh("deformed", U, self.T)
            # bt.createMesh("keypoints", self.X[self.pinned_points, :], None)
            if bI is not None:
                points = bt.readNumpyPoints( U[bI, :], location=location, rotation_euler=rotation_euler, scale=scale)
                pointColor = bt.colorObj(bt.cb_purple, 0.5, 1.0, 1.0, 0.0, 2.0)
                bt.setMat_pointCloud(points, pointColor, 0.06)

            ## set light
            lightAngle = (-20, -70, 78) 
            strength = 1
            shadowSoftness = 0.3
            sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
            
            lightAngle2 = (54, -66, 30) 
            strength2 = 0.6
            shadowSoftness2 = 0.3
            sun2 = bt.setLight_sun(lightAngle2, strength2, shadowSoftness2)


            cam= bt.setCamera(eye_pos, look_at, focalLength=35)
            bt.invisibleGround(shadowBrightness=0.9, location=[0, 0, self.X[:, 1].min()-0.01])
            bpy.ops.wm.save_mainfile(filepath=result_dir + '/test.blend')
            bt.renderImage(result_dir + str(frame).zfill(4) + ".png", camera=cam)
        video_from_image_dir(result_dir, path, fps=30)
        gif_path = path[:-4] + ".gif"
        mp4_to_gif(path,  gif_path)

    def blender_render_weight(self, w, path, bI=None, eye_pos=[0, -5, 0], look_at=[0, 0,0 ], numSamples=2, exposure=2, imgres_x=1920, imgres_y=1080, frames=[0, 75], location=[0, 0, 0], rotation_euler=[90, 0, 0], scale=[1, 1, 1]):
        import pathlib
        result_dir = os.path.dirname(path) 
        os.makedirs(result_dir, exist_ok=True)

        import blendertoolbox as bt
        import bpy
        bt.blenderInit(imgres_x, imgres_y, numSamples=numSamples, exposure=exposure)
        mesh=  bt.readNumpyMesh( self.X, igl.boundary_facets(self.T)[0], location=location, rotation_euler=rotation_euler, scale=scale)  
        bpy.context.view_layer.objects.active = mesh
        mesh.select_set(True)
        bpy.ops.object.shade_smooth() 

        # maxim = np.abs(w).max()
        # Cdata =  w +  maxim
        # Cdata = Cdata / (2.0*maxim)
        # Cdata += 1e-3
        # Cdata  = np.minimum(Cdata, 0.99)
        
        # normalize from 0 to 1 Cdata
        Cdata = w
        # add a threshold so it doesn't go over 1 or under 0
        
        Cdata = (Cdata - Cdata.min()) / (Cdata.max() - Cdata.min())
        Cdata = np.clip(Cdata, 1e-2, 0.99)
        # vertex_scalars = Cdata  # vertex color list
        # color_type = 'vertex'
        # color_map = 'default'
        vertexScalarToUV_unnormalized(mesh, Cdata)
        # bt.vertexScalarToUV_unnormalized(mesh, Cdata)

        useless = (0,0,0,1)
        meshColor = bt.colorObj(useless, 0.5,1, 1, 0.0, 0.0)
        bt.setMat_texture(mesh, directory + "/../../../data/colormaps/Purples_11.png", meshColor)

        bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

        if bI is not None:
            points = bt.readNumpyPoints( self.X[bI, :], location=location, rotation_euler=rotation_euler, scale=scale)
            pointColor = bt.colorObj(bt.cb_purple, 0.5, 1.0, 1.0, 0.0, 2.0)
            bt.setMat_pointCloud(points, pointColor, 0.06)

        ## set light
        lightAngle = (-20, -70, 78) 
        strength = 1
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
        
        lightAngle2 = (54, -66, 30) 
        strength2 = 0.6
        shadowSoftness2 = 0.3
        sun2 = bt.setLight_sun(lightAngle2, strength2, shadowSoftness2)

        cam= bt.setCamera(eye_pos, look_at, focalLength=35)
        bt.invisibleGround(shadowBrightness=0.9, location=[0, 0, self.X[:, 1].min()-0.01])

        bt.renderImage(path, camera=cam)





cs = [ ElephantConfig()]
for c in cs:
    [X, T, F] = igl.readMESH(c.mesh_path)
    X = normalize_and_center(X)
    dim = X.shape[1]
    n = X.shape[0]
    X = rotate(X, c.pre_rotate)

    keypoint_vertices = np.load(c.keypoints_path)
    # pin these vertices
    

    _p, pinned_vertices = common_selections.create_selection(c.pinned_selection, X, kwargs=c.pinned_selection_kwargs)

    # simulation parameters
    dim = X.shape[1]
    ym = 1e5
    pr = 0.0
    mu, lam = ympr_to_lame(ym, pr)
    gamma = 1e5
    rho=1e-6
    h=1e-2
    Q, b = dirichlet_penalty(bI=pinned_vertices, y=X[pinned_vertices, :], nv = X.shape[0], gamma=gamma)
    simulation_params = ElasticROMPDSimParams(rho=rho, h=h, ym=ym, pr=lam, Q0=Q, b0=b)

    #experiment parameters 
    num_timesteps = 300
    force = c.force 
    period = 300
    m = c.m
    k = 30

    read_cache = True
    cache_dir = directory + "/cache/" 


    # build subspaces
    H = linear_elasticity_hessian(X=X, T=T, mu=mu, lam=lam) + Q
    M = sp.sparse.kron(massmatrix(X, T), sp.sparse.eye(dim))


    y = [np.array([[1.0]]) for i in range(keypoint_vertices.shape[0])]
    bI = [ np.array(i) for i in keypoint_vertices[:keypoint_vertices.shape[0]]]
    pi0 = []
    for i in range(len(bI)):
        pi_i = diffuse_scalar(X, T, bI=bI[i], y=y[i], h=1, ord=1 ) # normalize these to sum to 1
        pi0.append(pi_i)
    pi0=np.hstack(pi0)
    ord = 5
    i = 1
    pi = pi0**ord / np.sum(pi0**ord, axis=1)[:, None]
    try:
        if read_cache:
            B_dla = np.load(cache_dir + f"{c.character_name}_{ord}_dla.npy")
        else:
            print("no cache")
            raise Exception("No cache")
    except:
        print("computing from scratch")
        # for i in range(1):
        cc = pi[:, i]**(0.5) * 1e5
        C = sp.sparse.diags(cc.flatten())
        Ce = sp.sparse.kron(C, sp.sparse.eye(dim))
        Ci = sp.sparse.diags(1 / cc.flatten())
        H2 = H + Ce
        Cie = sp.sparse.kron(Ci, sp.sparse.eye(dim))
        D, B_dla = eigs( Cie @ H2, m)
        os.makedirs(cache_dir, exist_ok=True)
        
        np.save(cache_dir + f"{c.character_name}_{ord}_dla.npy", B_dla)

    
    # view_scalar_modes(X, T, pi, dir = dir + f"/{c.character_name}_{ord}_pi/")
    # view_displacement_modes(X, T, B_dla, a=10, path = dir + f"/{c.character_name}_{ord}_dla_modes.mp4", fps=30)
    try:
        if read_cache:
            B_lma = np.load(cache_dir + f"{c.character_name}_{ord}_lma.npy")
        else:
            print("no cache")
            raise Exception("No cache")
    except:
        B_lma = eigs(H, m)[1]
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_dir + f"{c.character_name}_{ord}_lma.npy", B_lma)

    # # build labels independent of subspace
    W_m = eigs(dirichlet_laplacian(X, T), k=10, M=massmatrix(X, T))[1]
    W_t = average_onto_simplex(W_m, T)
    l = spectral_clustering(W_t, k)[0]
    bI = np.array([keypoint_vertices[i]])
    U = sinusoidal_handle_updown(bI, X, num_timesteps, period, a=0.3)
    Q0, b0 , SGamma = dirichlet_penalty(bI, U[:, 0].reshape(1, -1), n, gamma, return_SGamma=True)      

    subspaces = { "dla":B_dla , "full": sp.sparse.identity(B_lma.shape[0]), "lma":B_lma}
    # world.interact(B=B_lma, labels=l)     

    for key, B in subspaces.items():
        world = DistributedLinearForceWorld(X, T, pinned_vertices, simulation_params)
        
        if key == "dla":
            world.blender_render_weight(pi[:, i], path=directory + f"/{c.character_name}_{key}_weights.png", bI=bI, numSamples=200)

        sim = world.build_sim(B=B, labels=l, Q0=Q0 + Q)
        BF = B.T @ SGamma @ U
        Z = world.simulate(sim, num_timesteps=num_timesteps, F=BF)

        color = bt.derekBlue
        if key == "lma":
            color = bt.coralRed
        elif key == "full":
            color =  np.array([120,198,121, 0])/255#green
        # world.render(B, Z, bI=bI, eye_pos=c.eye_pos, look_at=c.look_at)#, path=dir + f"/{c.character_name}_{ord}_{key}_sim.mp4") 
        world.blender_render(B, Z, path=directory +f"/{c.character_name}_{key}_blender.mp4",  bI=bI, numSamples=200, color=color, frames= np.arange(num_timesteps))
        # view_scalar_modes(X, T, pi, dir = dir + f"/{c.character_name}_{ord}_pi/")
        # view_displacement_modes(X, T, B_dla, a=10)#, path = dir + f"/{c.character_name}_{ord}_{key}_modes.mp4", fps=30)