import numpy as np
import sys
sys.path.append("../../")


import igl
import polyscope as ps
import scipy as sp
import simkit as sk

from utils import *
# from s.filesystem import get_data_directory
# from sk.sims.elastic import ElasticROMMFEMSim, ElasticROMMFEMSimParams, ElasticROMFEMSim, ElasticROMFEMSimParams

# Simulation type


ym = 1e8
h = 1e-2
rho = 1e-3
m = 10
k = 100
max_iter = 2

do_line_search = True

# Load mesh
# [X, _, _, T, _, _] = igl.readOBJ(sk.filesystem.get_data_directory() + "/2d/cthulu/cthulu.obj")
X = np.array([[0, 0.],
              [1, 0],
              [0, 1]])
T = np.array([[0, 1, 2]])
X = X[:, 0:2]
X = X / max(X.max(axis=0) - X.min(axis=0))
dim = X.shape[1]

# W, E, B, cI, cW, labels = compute_subspace(X, T, m, k)

num_timesteps = 1


# fem_sim = create_fem_sim(X, T, ym, rho, h, max_iter, do_line_search)
# [Zs,  info_history] = simulate_fem(fem_sim, num_timesteps, return_info=True)
# dp_hist = np.array(info_history[0]['dx'])[:, : , 0]
# search_directions = np.array(dp_hist)
# step_sizes = np.array(info_history[0]['alphas'])
# du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
# view_animation(X, T, (fem_sim.B @ du.T))

mfem_sim = create_mfem_sim(X, T, ym, rho, h, max_iter, do_line_search)
[Zs, As,  info_history] = simulate_mfem(mfem_sim, num_timesteps, return_info=True)
step_sizes = np.array(info_history[0]['alphas'])
dp_hist = np.array(info_history[0]['dp'])[:, :, 0]
alphas = np.array([info['alphas'] for info in info_history])
search_directions = np.array([mfem_sim.z_a_from_p(dp)[0] for dp in dp_hist])
du = np.cumsum(search_directions * step_sizes[:, None], axis=0)
grads_s = np.array([mfem_sim.z_a_from_p(info['g'][0])[1] for info in info_history])
print("alphas", alphas)
# print("gs", np.array([np.linalg.norm(info['g'][-1]) for info in info_history]))
view_animation(X, T, (mfem_sim.B @ du.T))


# g_norms_mfem = [np.linalg.norm(mfem_sim.z_a_from_p(info['g'][-1])[0] )for info in info_history]


# view_animation(X, T, (fem_sim.B @ Zs))





# view_animation(X, T, Zs, As, Z_dots, path=result_dir + "/mfem_animation.mp4")
# plot_grad_norms(info_history, path=result_dir + "/mfem_grad_norms.png")

# fem_sim = create_fem_sim(X, T, ym, rho, h, max_iter, do_line_search)
# [Zs, info_history] = simulate_fem(fem_sim, num_timesteps, return_info=True)
# view_animation(X, T, Zs, path=result_dir + "/fem_animation.mp4")
# plot_grad_norms(info_history, path=result_dir + "/fem_grad_norms.png")

