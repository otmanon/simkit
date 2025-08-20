
import numpy as np
import simkit as sk

ag = -9.81
k_pin = 1e8
def compute_subspace(X, T, m, k):
    # Compute skinning modes and cubature points
    m = 10
    k = 100
    [W, E,  B] = sk.skinning_eigenmodes(X, T, m)
    [cI, cW, labels] = sk.spectral_cubature(X, T, W, k, return_labels=True)
    return W, E, B, cI, cW, labels

def create_mfem_sim(X, T, ym, rho, h, max_iter, do_line_search):
    sim_params = sk.sims.elastic.ElasticMFEMSimParams()  
    sim_params.ym = ym  # Young's modulus (Pa)
    sim_params.h = h   # time step (s)
    sim_params.rho = rho # density kg/m^3
    sim_params.gamma = ym # set constraint weight to same as ym
    sim_params.solver_p.max_iter= max_iter
    sim_params.solver_p.do_line_search = do_line_search
    q = X.reshape(-1, 1) # rest geometry.
    sim = sk.sims.elastic.ElasticMFEMSim(X, T, sim_params=sim_params, q=q)
    return sim


def create_fem_sim(X, T, ym, rho, h,  max_iter, do_line_search):
    sim_params = sk.sims.elastic.ElasticFEMSimParams()
    sim_params.ym = ym  # Young's modulus (Pa)
    sim_params.h = h   # time step (s)
    sim_params.rho = rho # density kg/m^3
    sim_params.solver_p.max_iter= max_iter
    sim_params.solver_p.do_line_search = do_line_search
    q = X.reshape(-1, 1) # rest geometry.
    sim = sk.sims.elastic.ElasticFEMSim(X, T, sim_params=sim_params, q=q)
    return sim


def simulate_mfem(sim : sk.sims.elastic.ElasticMFEMSim, num_timesteps, return_info=False):
    dim = sim.X.shape[1]
    bg =  -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    # Pinning DOF
    bI =  np.where(sim.X[:, 0] < 0.001 + sim.X[:, 0].min())[0]
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
        z_next, s_next, info = sim.step(z, s, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        z_dot = (z_next - z) / sim.sim_params.h    
        z = z_next.copy()
        s = s_next.copy()
        
        Zs[:, i+1] = z.flatten()
        As[:, i+1] = s.flatten()

        if return_info:
            info_history[i] = info
            
    return Zs, As, info_history

def simulate_fem(sim : sk.sims.elastic.ElasticFEMSim, num_timesteps, return_info=False):
    dim = sim.X.shape[1]
    bg =  -sk.gravity_force(sim.X, sim.T, a=ag, rho=1e3).reshape(-1, 1)

    # Pinning DOF
    bI =  np.where(sim.X[:, 0] < 0.001 + sim.X[:, 0].min())[0]
    bc0 = (sim.X - sim.q.reshape(-1, dim))[bI, :]
    [Q_ext, b_ext] = sk.dirichlet_penalty(bI, bc0, sim.X.shape[0],   k_pin)
    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ (b_ext + bg)


    z, z_dot = sim.rest_state()
    Zs = np.zeros((z.shape[0], num_timesteps + 1))

    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
    for i in range(num_timesteps):
        z_next, info = sim.step(z, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        
        Zs[:, i+1] = z.flatten()
        
        if return_info:
            info_history[i] = info
            
    return Zs, info_history



def view_animation(X, T, U):
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.look_at(np.array([0, 0, 3]), np.array([0, 0, 0]))
    mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
    for i in range(U.shape[1]):
        x = X.reshape(-1, 1) + U[:, [i]]
        mesh.update_vertex_positions(x.reshape(-1, 2))
        ps.frame_tick()
    
    return
