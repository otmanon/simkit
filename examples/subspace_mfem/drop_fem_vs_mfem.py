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

dirname =  os.path.dirname(__file__)


configs = [crabConfig(), cthuluConfig()]
num_timesteps = 400
for c in configs:
    print(c.name)

    [X, T] = load_mesh(c.geometry_path)
    dim = X.shape[1]
    X = normalize_mesh(X)

    
    W, E, B, cI, cW, labels = compute_subspace(X, T, c.m, c.k, mu=c.ym)

    video_path_mfem = dirname + "/results/drop/" + c.name + "_mfem.mp4"
    mfem_sim = create_mfem_sim(X, T, c.ym, c.rho, 
                            c.h, c.max_iter, 
                            c.do_line_search,
                            B=B, cI=cI, cW=cW)
    
    [Zs, As] = simulate_drop_mfem(mfem_sim, c.bI,
                                num_timesteps, 
                                return_info=False)
    view_animation(X, T, (mfem_sim.B @ Zs), 
                path=video_path_mfem, eye_pos=c.eye_pos,
                look_at=c.look_at)


    video_path_fem = dirname + "/results/drop/" + c.name + "_fem.mp4"
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





    # # dp_hist = np.array(info_history[0]['dx'])[:, : , 0]
    # # search_directions = np.array(dp_hist)
    # # step_sizes = np.array(info_history[0]['alphas'])
    # # du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
    # view_animation(X, T, (fem_sim.B @ Zs))

    # mfem_sim = create_mfem_sim(X, T, ym, rho, h, max_iter, do_line_search)
    # [Zs, As,  info_history] = simulate_mfem(mfem_sim, num_timesteps, return_info=True)
    # step_sizes = np.array(info_history[0]['alphas'])
    # dp_hist = np.array(info_history[0]['dp'])[:, :, 0]
# alphas = np.array([info['alphas'] for info in info_history])
# search_directions = np.array([mfem_sim.z_a_from_p(dp)[0] for dp in dp_hist])
# du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
# grads_s = np.array([mfem_sim.z_a_from_p(info['g'][0])[1] for info in info_history])
# print("alphas", alphas)
# print("gs", np.array([np.linalg.norm(info['g'][-1]) for info in info_history]))
# view_animation(X, T, (mfem_sim.B @ du.T))
# view_animation(X, T, (mfem_sim.B @ Zs))

# view_animation(X, T, (mfem_sim.B @ Zs))
# g_norms_mfem = [np.linalg.norm(mfem_sim.z_a_from_p(info['g'][-1])[0] )for info in info_history]


# view_animation(X, T, (fem_sim.B @ Zs))

# view_animation(X, T, Zs, As, Z_dots, path=result_dir + "/mfem_animation.mp4")
# plot_grad_norms(info_history, path=result_dir + "/mfem_grad_norms.png")

# fem_sim = create_fem_sim(X, T, ym, rho, h, max_iter, do_line_search)
# [Zs, info_history] = simulate_fem(fem_sim, num_timesteps, return_info=True)
# view_animation(X, T, Zs, path=result_dir + "/fem_animation.mp4")
# plot_grad_norms(info_history, path=result_dir + "/fem_grad_norms.png")

