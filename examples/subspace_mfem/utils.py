
import numpy as np
import igl
import os
from pathlib import Path
import simkit as sk


ag = -9.81
k_pin = 1e8

def load_mesh(geometry_path):            
    file_extension = geometry_path.split(".")[-1]
    if file_extension == "obj":
        [X, _, _, T, _, _] = igl.readOBJ(geometry_path)
        X = X[:, 0:2]
    elif file_extension == "mesh":
        [X, T, _] = igl.readMESH(geometry_path)
    else:
        raise ValueError("Geometry path must be a .obj or .mesh file")
    return X, T

def normalize_mesh(X):
    X = X - X.mean(axis=0)
    X = X / max(X.max(axis=0) - X.min(axis=0))
    return X

def compute_subspace(X, T, m, k, mu=None, bI=None):
    # Compute skinning modes and cubature points
    
    if mu is not None:
        if isinstance(mu, str):
            mu = np.load(mu).reshape(-1, 1)
        elif isinstance(mu, float) or isinstance(mu, int):
            mu = mu * np.ones((T.shape[0], 1))
        elif isinstance(mu, np.ndarray):
            assert(mu.shape[0] == T.shape[0])
            assert(mu.shape[1] == 1)
        
    assert(mu.shape[1] == 1)
    assert(mu.shape[0] == T.shape[0])
    
    if isinstance(bI, str):
        bI = np.load(bI).astype(int)
        
    if m is None:    
        W = None
        E = None
        B = None
    else:
        [W, E,  B] = sk.skinning_eigenmodes(X, T, m, mu=mu, bI=bI)
    
    if k is None:
        cI = None
        cW = None
        labels = None
    else:
        [cI, cW, labels] = sk.spectral_cubature(X, T, W, k, return_labels=True)
    return W, E, B, cI, cW, labels

def create_mfem_sim(X, T, ym, rho, h,
                    max_iter, do_line_search,
                    B=None, cI=None, cW=None):
    
        
    if ym is not None:
        if isinstance(ym, str):
            ym = np.load(ym).reshape(-1, 1)
        elif isinstance(ym, float) or isinstance(ym, int):
            ym = ym * np.ones((T.shape[0], 1))
            
    assert(ym.shape[0] == T.shape[0])
    
    sim_params = sk.sims.elastic.ElasticMFEMSimParams()  
    sim_params.ym = ym  # Young's modulus (Pa)
    sim_params.h = h   # time step (s)
    sim_params.rho = rho # density kg/m^3
    sim_params.gamma = np.linalg.norm(ym) # set constraint weight to same as ym
    sim_params.solver_p.max_iter= max_iter
    sim_params.solver_p.do_line_search = do_line_search
    q = X.reshape(-1, 1) # rest geometry.
    sim = sk.sims.elastic.ElasticMFEMSim(X, T, sim_params=sim_params, q=q, B=B, cI=cI, cW=cW)
    return sim


def create_fem_sim(X, T, ym, rho, h,  max_iter, do_line_search, B=None, cI=None, cW=None):
    
    if ym is not None:
        if isinstance(ym, str):
            ym = np.load(ym).reshape(-1, 1)
        elif isinstance(ym, float) or isinstance(ym, int):
            ym = ym * np.ones((T.shape[0], 1))
        elif isinstance(ym, np.ndarray):
            assert(ym.shape[0] == T.shape[0])
            assert(ym.shape[1] == 1)
        
    assert(ym.shape[1] == 1)
    assert(ym.shape[0] == T.shape[0])
            
    sim_params = sk.sims.elastic.ElasticFEMSimParams()
    sim_params.ym = ym  # Young's modulus (Pa)
    sim_params.h = h   # time step (s)
    sim_params.rho = rho # density kg/m^3
    
    sim_params.solver_p.max_iter= max_iter
    sim_params.solver_p.do_line_search = do_line_search
    q = X.reshape(-1, 1) # rest geometry.
    sim = sk.sims.elastic.ElasticFEMSim(X, T, sim_params=sim_params, q=q, B=B, cI=cI, cW=cW)
    return sim


def view_animation(X, T, U, path=None, fps=60, eye_pos=None, look_at=None):
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    if eye_pos is not None and look_at is not None:
        ps.look_at(eye_pos, look_at)
    else:
        ps.look_at(np.array([0, 0, 3]), np.array([0, 0, 0]))
    dim = X.shape[1]
    
    if path is not None:
        stem = Path(path).stem
        dir = Path(path).parent
        dirstem = os.path.join(dir, stem)
        os.makedirs(dirstem, exist_ok=True)

    
    if T.shape[1] == 3:
        mesh = ps.register_surface_mesh("mesh", X, T, edge_width=0.01)
    elif T.shape[1] == 4:
        mesh = ps.register_volume_mesh("mesh", X, T, edge_width=0.01)
    else:
        raise ValueError("T must be 3 or 4")
    
    for i in range(U.shape[1]):
        x = X.reshape(-1, 1) + U[:, [i]]
        mesh.update_vertex_positions(x.reshape(-1, dim))
        ps.frame_tick()
        
        if path is not None:
            ps.screenshot(dirstem + "/" + str(i + 1).zfill(4) + ".png", transparent_bg=True)

    if path is not None:
        sk.filesystem.video_from_image_dir(dirstem, path, fps=fps)
        sk.filesystem.mp4_to_gif(path, path.replace(".mp4", ".gif"))

    ps.remove_all_structures()
    
    return
