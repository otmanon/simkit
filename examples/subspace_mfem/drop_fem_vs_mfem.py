import os
import sys

import igl
import numpy as np
import polyscope as ps
import scipy as sp

import simkit as sk

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import *  # noqa: E402,F401,F403
from config import *  # noqa: E402,F401,F403


def simulate_drop_mfem(sim, bI,
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
    
    z, s, la, z_dot = sim.rest_state()
    z_curr = z.copy()
    z_prev = z.copy()
    Zs = np.zeros((z.shape[0], num_timesteps + 1))
    As = np.zeros((s.shape[0], num_timesteps + 1))
    Ls = np.zeros((la.shape[0], num_timesteps + 1))
    Zs[:, 0] = z.flatten()
    As[:, 0] = s.flatten()

    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
        
    for i in range(num_timesteps):
        if return_info:
            z_next, s_next, la, info = sim.step(z_curr, z_prev, s, la,  Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        else:
            z_next, s_next, la = sim.step(z_curr, z_prev, s, la, Q_ext=BQB_ext, b_ext=Bb_ext)
        z_prev = z_curr.copy()
        z_curr = z_next.copy()
        z = z_next.copy()
        s = s_next.copy()
        
        Zs[:, i+1] = z.flatten()
        As[:, i+1] = s.flatten()
        Ls[:, i+1] = la.flatten()
        
        if return_info:
            info_history[i] = info
            
    if return_info:
        return Zs, As, Ls, info_history
    else:
        return Zs, As, Ls

def simulate_drop_fem(sim, bI,
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
    z_curr = z.copy()
    z_prev = z_curr.copy()
    Zs = np.zeros((z.shape[0], num_timesteps + 1))

    if return_info:
        info_history = np.empty(num_timesteps, dtype=object)
    for i in range(num_timesteps):
        
        if return_info:
            z_next, info = sim.step(z_curr, z_prev, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext, return_info=return_info)
        else:
            z_next = sim.step(z_curr, z_prev, z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
        z_dot = (z_next - z) / sim.sim_params.h
        z_prev = z_curr.copy()
        z_curr = z_next.copy()
        z = z_next.copy()
        
        Zs[:, i+1] = z.flatten()
        
        if return_info:
            info_history[i] = info
        
    if return_info:
        return Zs, info_history
    else:
        return Zs



if __name__ == "__main__":  
    dirname =  os.path.dirname(__file__)


    configs = [crabConfig()]
    num_timesteps = 400
    for c in configs:
        print(c.name)

        [X, T] = load_mesh(c.geometry_path)
        dim = X.shape[1]
        X = normalize_mesh(X)
        
        W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym)

        video_path_mfem = os.path.join(dirname, "results", "drop", c.name + "_mfem.mp4")
        mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                                c.h, c.max_iter, 
                                c.do_line_search,
                                B=B, cI=cI, cW=cW)
        
        [Zs, As, Ls] = simulate_drop_mfem(mfem_sim, c.bI,
                                    num_timesteps, 
                                    return_info=False)
        view_animation(X, T, (mfem_sim.B @ Zs), 
                    path=video_path_mfem, eye_pos=c.eye_pos,
                    look_at=c.look_at)


        video_path_fem = os.path.join(dirname, "results", "drop", c.name + "_fem.mp4")
        fem_sim = create_fem_sim(X, T, c.ym, 
                                c.rho, c.h, 
                                c.max_iter,
                                c.do_line_search,
                                B=B, cI=cI, cW=cW)
        
        
        Zs = simulate_drop_fem(fem_sim, c.bI, num_timesteps,
                            return_info=False)
        view_animation(X, T, (fem_sim.B @ Zs),
                    path=video_path_fem, eye_pos=c.eye_pos,
                    look_at=c.look_at)

