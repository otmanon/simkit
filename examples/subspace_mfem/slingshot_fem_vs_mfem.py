import numpy as np
import sys
sys.path.append("../../")

import igl
import polyscope as ps
import scipy as sp
import simkit as sk
import os

from utils import *
from config import *

def simulate_slingshot_mfem(
    sim : sk.sims.elastic.ElasticMFEMSim, 
    bI, pullI, 
    pull_disp, 
    pull_timesteps, free_timesteps, 
    return_info=False):
    
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

    z, s, la, z_dot = sim.rest_state()
    Zs = np.zeros((z.shape[0], pull_timesteps + free_timesteps + 1))
    As = np.zeros((s.shape[0], pull_timesteps + free_timesteps + 1))
    las = np.zeros((la.shape[0], pull_timesteps + free_timesteps + 1))
        
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
            z_next, s_next, la_next, info = sim.step(z, s, la, z_dot, Q_ext=BQB_ext,
                                            b_ext=Bb_ext, return_info=return_info)
            info_history[i] = info

        else:
            z_next, s_next , la_next= sim.step(z, s, la,  z_dot, Q_ext=BQB_ext,
                                        b_ext=Bb_ext)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
        s = s_next.copy()
        la = la_next.copy()

        Zs[:, i+1] = z.flatten()
        As[:, i+1] = s.flatten()
        las[:, i+1] = la.flatten()

    if return_info:
        return Zs, As, las, info_history
    else:
        return Zs, As, las
          

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



dirname =  os.path.dirname(__file__)

configs = [ gatormanConfig()]
pull_timesteps = 100
free_timesteps = 300

for c in configs:
    print(c.name)
    [X, T] = load_mesh(c.geometry_path)
    dim = X.shape[1]
    X = normalize_mesh(X)


    W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym, bI=c.bI)

    # sk.polyscope.view_scalar_modes(X, T, W)
    
    video_path_mfem = dirname + "/results/slingshot/" + c.name + "_mfem.mp4"
    mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                            c.h, c.max_iter, 
                            c.do_line_search,
                            B=B, cI=cI, cW=cW)
    
    [Zs, As, Las] = simulate_slingshot_mfem(mfem_sim, c.bI, 
                                       c.pullI,
                                       c.pull_disp,
                                       pull_timesteps,
                                       free_timesteps, 
                                       return_info=False)
    view_animation(X, T, (mfem_sim.B @ Zs), 
                path=video_path_mfem,
                eye_pos=c.eye_pos,
                look_at=c.look_at)


    video_path_fem = dirname + "/results/slingshot/" + c.name + "_fem.mp4"
    fem_sim = create_fem_sim(X, T, c.ym, 
                            c.rho, c.h, 
                            c.max_iter,
                            c.do_line_search,
                            B=B, cI=cI, cW=cW)
    Zs = simulate_slingshot_fem(fem_sim, c.bI, 
                                c.pullI,
                                c.pull_disp,
                                pull_timesteps,
                                free_timesteps, 
                                return_info=False)
    view_animation(X, T, (fem_sim.B @ Zs),
                path=video_path_fem,
                eye_pos=c.eye_pos,
                look_at=c.look_at)

