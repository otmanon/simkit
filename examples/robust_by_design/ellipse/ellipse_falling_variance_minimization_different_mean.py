import os
import scipy as sp
import numpy as np
import simkit as sk

from examples.robust_by_design.ellipse.main import *
from simkit.solvers import CMAESSolver
from simkit.solvers.CMAESSolver import CMAESSolverParams


read_cache = True
p_mean = np.array([1.35, 0.65])
p_min = np.array([0.5, 0.5])
p_max = np.array([2.0, 2.0])

p_scale = p_mean / 10

def transform_dp(dp):
    p = p_mean + dp * p_scale
    p = np.clip(p, p_min, p_max)
    return p
def variance_min_objective(dp):
    p = transform_dp(dp)
    com_ys_all = evaluate_com_height_across_thetas(thetas, p)
    # variance through time
    var = np.var(com_ys_all, axis=0)
    all_var = np.sum(var)
    return all_var

def compute_optimal_p():
    solver_params = CMAESSolverParams(maxiter = 20, popsize=8, num_processes=1)
    solver = CMAESSolver(variance_min_objective, solver_params)
    p_opt, history = solver.solve(np.zeros(2), return_history=True)
    return history



# get two endpoints  on curve with normal n that goes through contact_p
current_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = current_dir + '/results/optimization_different_mean/'
os.makedirs(result_dir, exist_ok=True)


num_samples = 3
thetas = np.linspace(0, np.pi/2, num_samples)

cache = result_dir 
[history, ] = sk.filesystem.compute_with_cache_check(compute_optimal_p,
                                                     result_dir + "/cache.npz", 
                                                     read_cache=read_cache )

dp_best = history[-1]['xbest']
p_best = transform_dp(dp_best)

print("p_best : ", p_best)
fs = [hist['fbest'] for hist in history]

plt.plot(fs)
plt.ylabel("objective")
plt.xlabel("iteration")
plt.savefig(result_dir + "/convergence.png", dpi=300)
plt.show()
plt.clf()


com_ys_all, Zs_list, X_list = evaluate_com_height_across_thetas(thetas,p_best, return_Zs=True)
# print shape in polyscope
com_ys_all_mean, Zs_list_mean, X_mean_list = evaluate_com_height_across_thetas(thetas, p_mean, return_Zs=True)

save_shape(result_dir + "/best_shape", p_best)
save_shape(result_dir + "/first_shape", p_mean)
    

os.makedirs(result_dir + "/plot_best/", exist_ok=True)
os.makedirs(result_dir + "/plot_mean/", exist_ok=True)
coms_vars_plots(com_ys_all, thetas, result_dir=result_dir + "/plot_best/", legend=True)
coms_vars_plots(com_ys_all_mean, thetas, result_dir=result_dir + "/plot_mean/", legend=True)

for i, Zs in enumerate(Zs_list):
    print("Zs_best_theta_" + str(np.round(thetas[i], 3)))
    view_animation(X_list[i], T, Zs - X_list[i].reshape(-1, 1) , V_ground, E_ground, 
                   path=result_dir + "/Zs_best_theta_" + str(np.round(thetas[i], 3)) + "/")
for i, Zs in enumerate(Zs_list_mean):
    view_animation(X_mean_list[i], T, Zs - X_mean_list[i].reshape(-1, 1), V_ground, E_ground, 
                   path=result_dir + "/Zs_mean_theta_" + str(np.round(thetas[i], 3)) + "/")