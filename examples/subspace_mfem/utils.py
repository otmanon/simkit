
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



def simulate_drop_mfem(sim : sk.sims.elastic.ElasticMFEMSim, bI,
                       num_timesteps, return_info=False):
    
    if isinstance(bI, str):
        bI = np.load(bI).astype(int)

    assert(isinstance(bI, np.ndarray))
    assert(bI.dtype == int)
    
    dim = sim.X.shape[1]
    bg =  -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    bc0 = (sim.X - sim.q.reshape(-1, dim))[bI, :]
    [Q_ext, b_ext] = sk.dirichlet_penalty(bI, bc0, sim.X.shape[0],  k_pin)
    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ (b_ext + bg)
    
    z, s, z_dot = sim.rest_state()
    Zs = np.zeros((z.shape[0], num_timesteps + 1))
    As = np.zeros((s.shape[0], num_timesteps + 1))

    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
        
    for i in range(num_timesteps):
        if return_info:
            z_next, s_next, info = sim.step(z, s, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        else:
            z_next, s_next = sim.step(z, s, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
        z_dot = (z_next - z) / sim.sim_params.h    
        z = z_next.copy()
        s = s_next.copy()
        
        Zs[:, i+1] = z.flatten()
        As[:, i+1] = s.flatten()

        if return_info:
            info_history[i] = info
            
    if return_info:
        return Zs, As, info_history
    else:
        return Zs, As


def simulate_slingshot_mfem(
    sim : sk.sims.elastic.ElasticMFEMSim, 
    bI, pullI, 
    pull_disp, 
    pull_timesteps, free_timesteps, 
    return_info=False, return_history=False):
    
    if isinstance(bI, str):
        bI = np.load(bI).astype(int)
    if isinstance(pullI, str):
        pullI = np.load(pullI).astype(int)
    
    dim = sim.X.shape[1]
    assert(isinstance(bI, np.ndarray))
    assert(isinstance(pullI, np.ndarray))
    assert(pullI.dtype == int)
    assert(pull_disp.shape[0] == 1)
    assert(pull_disp.shape[1] == dim)
    
    bg =  -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    pull_disp = pull_disp / np.linalg.norm(pull_disp)
    pull_bc0 = (sim.X - sim.q.reshape(-1, dim))[pullI, :]
    bc0 = (sim.X - sim.q.reshape(-1, dim))[bI, :]


    Q_ext_pin, b_ext_pin = sk.dirichlet_penalty(bI, bc0, sim.X.shape[0],  k_pin)
    BQB_ext_pin = sim.B.T @ Q_ext_pin @ sim.B
    Bb_ext_pin = sim.B.T @ b_ext_pin
    
    [Q_ext_pull, b_ext_pull, SGamma] = sk.dirichlet_penalty(pullI, pull_bc0, sim.X.shape[0],  k_pin,
                                                    return_SGamma=True)
    BSGamma = sim.B.T @ SGamma
    BQB_ext_pull = sim.B.T @ Q_ext_pull @ sim.B
    Bb_ext_pull = sim.B.T @ b_ext_pull
    
    Bb_gravity = sim.B.T @ bg

    z, s, z_dot = sim.rest_state()
    Zs = np.zeros((z.shape[0], pull_timesteps + free_timesteps + 1))
    As = np.zeros((s.shape[0], pull_timesteps + free_timesteps + 1))

        
    num_timesteps = pull_timesteps + free_timesteps
    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
    BQB_ext = BQB_ext_pull + BQB_ext_pin 
    for i in range(num_timesteps):
        
        if i < pull_timesteps:
            pull_bc = (pull_bc0.reshape(-1, dim) + (i / pull_timesteps) * pull_disp).reshape(-1, 1)
            Bb_ext_pull = BSGamma @pull_bc 
            Bb_ext = Bb_ext_pull + Bb_ext_pin + Bb_gravity
        else:
            pull_bc0 = pull_bc0
            BQB_ext = BQB_ext_pin
            Bb_ext = Bb_ext_pin + Bb_gravity
            
        if return_info:
            z_next, s_next, info = sim.step(z, s, z_dot, Q_ext=BQB_ext,
                                            b_ext=Bb_ext, return_info=return_info)
            info_history[i] = info

        else:
            z_next, s_next = sim.step(z, s, z_dot, Q_ext=BQB_ext,
                                        b_ext=Bb_ext)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        s = s_next.copy()

        Zs[:, i+1] = z.flatten()
        As[:, i+1] = s.flatten()

    if return_info:
        return Zs, As, info_history
    else:
        return Zs, As
          


def simulate_drop_fem(sim : sk.sims.elastic.ElasticFEMSim, bI,
                      num_timesteps, return_info=False):
    
    if isinstance(bI, str):
        bI = np.load(bI).astype(int)

    assert(isinstance(bI, np.ndarray))
    assert(bI.dtype == int)
    
    dim = sim.X.shape[1]
    bg =  -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    bc0 = (sim.X - sim.q.reshape(-1, dim))[bI, :]
    [Q_ext, b_ext] = sk.dirichlet_penalty(bI, bc0, sim.X.shape[0],   k_pin)
    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ (b_ext + bg)

    z, z_dot = sim.rest_state()
    Zs = np.zeros((z.shape[0], num_timesteps + 1))

    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
    for i in range(num_timesteps):
        
        if return_info:
            z_next, info = sim.step(z, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        else:
            z_next = sim.step(z, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        
        Zs[:, i+1] = z.flatten()
        
        if return_info:
            info_history[i] = info
        
    if return_info:
        return Zs, info_history
    else:
        return Zs


def simulate_slingshot_fem(sim : sk.sims.elastic.ElasticFEMSim, 
                           bI, pullI, 
                           pull_disp, pull_timesteps, 
                           free_timesteps, 
                           return_info=False):
        
    if isinstance(bI, str):
        bI = np.load(bI).astype(int)
    if isinstance(pullI, str):
        pullI = np.load(pullI).astype(int)
    
    assert(isinstance(bI, np.ndarray))
    assert(bI.dtype == int)

    dim = sim.X.shape[1]
    bg = -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    Bb_gravity = sim.B.T @ bg
    
    pull_disp = pull_disp / np.linalg.norm(pull_disp)
    pull_bc0 = (sim.X - sim.q.reshape(-1, dim))[pullI, :]
    
    # pull_disp_bc0 = pull_bc0.reshape(-1, dim)
    [Q_pull, b_pull, SGamma] = sk.dirichlet_penalty(pullI, 
                                                    pull_bc0, sim.X.shape[0], 
                                                    k_pin, 
                                                    return_SGamma=True)
    BSGamma = sim.B.T @ SGamma
    BQB_pull = sim.B.T @ Q_pull @ sim.B
    
    # Pin boundary vertices in place
    bc0 = (sim.X - sim.q.reshape(-1, dim))[bI, :]
    [Q_pin, b_pin] = sk.dirichlet_penalty(bI, bc0, sim.X.shape[0], k_pin)
    BQB_pin = sim.B.T @ Q_pin @ sim.B
    Bb_pin = sim.B.T @ (b_pin + bg)

    # Initialize simulation state
    z, z_dot = sim.rest_state()
    total_timesteps = pull_timesteps + free_timesteps
    Zs = np.zeros((z.shape[0], total_timesteps + 1))
    Zs[:, 0] = z.flatten()

    BQB_ext = BQB_pull + BQB_pin 
    
    num_timesteps = pull_timesteps + free_timesteps
    
    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
    # Pull phase - gradually displace pullI vertices
    for i in range(num_timesteps):
        # Linearly interpolate displacement
        if i < pull_timesteps:
            pull_bc = (pull_bc0.reshape(-1, dim) + (i / pull_timesteps) * pull_disp).reshape(-1, 1)            
            Bb_ext_pull = BSGamma @ pull_bc 
            Bb_ext = Bb_ext_pull + Bb_pin + Bb_gravity
        else:
            pull_bc = pull_bc0
            BQB_ext = BQB_pin
            Bb_ext = Bb_pin + Bb_gravity
    

        if return_info:
            z_next, info = sim.step(z, z_dot, Q_ext=BQB_ext,
                                    b_ext=Bb_ext, return_info=True)
            info_history[i] = info
            
        else:
            z_next = sim.step(z, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
            
            
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        Zs[:, i+1] = z.flatten()

    if return_info:
        return Zs, info_history
    else:
        return Zs


    
    

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
