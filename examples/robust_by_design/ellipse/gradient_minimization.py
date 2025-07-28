import os
import scipy as sp
import numpy as np
import simkit as sk

from examples.robust_by_design.ellipse.main import *
from simkit.solvers import CMAESSolver
from simkit.solvers.CMAESSolver import CMAESSolverParams


read_cache = True
p_mean = np.array([1.35, 0.65])
p_min = np.array([0.10, 0.10])
p_max = np.array([4.0, 4.0])

p_scale = p_mean / 20

def transform_dp(dp):
    p = p_mean + dp * p_scale
    p = np.clip(p, p_min, p_max)
    return p

def dcom_dtheta(thetas, p):
    epsilon = 1e-5
    q_pos = evaluate_com_height_across_thetas(thetas + epsilon, p)
    q_neg = evaluate_com_height_across_thetas(thetas - epsilon, p)
    dq_dtheta = (q_pos - q_neg)/ (2 * epsilon)
    return dq_dtheta
    
    
def gradient_min_objective(dp):
    p = transform_dp(dp)

    dq_dtheta = dcom_dtheta(thetas, p)    
    
    dq_dtheta_squared = dq_dtheta**2
    energy = dq_dtheta_squared.sum()
    return energy

def compute_optimal_p():
    solver_params = CMAESSolverParams(maxiter = 20, popsize=16, num_processes=8)
    solver = CMAESSolver(gradient_min_objective, solver_params)
    p_opt, history = solver.solve(np.zeros(2), return_history=True)
    return history


def plot_dcom_dtheta(dq_dtheta, thetas, path):
    for i in range(dq_dtheta.shape[0]):
        plt.plot(dq_dtheta[i],   label = '$\\theta$ =' + str(np.round(thetas[i], 2)))
    plt.ylabel('$\\frac{\\partial  q}{\\partial \\theta }$')
    plt.xlabel("time")
    plt.legend()
    # plt.semilogy()
    # plt.ylim(10**(-8), 10**3)
    plt.savefig(path )
    plt.show()
    plt.clf()
    
    
    

num_samples = 3
thetas = np.linspace(0, np.pi/2, num_samples)

if __name__ == "__main__":
    # get two endpoints  on curve with normal n that goes through contact_p
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = current_dir + '/results/grad_optimization/'
    os.makedirs(result_dir, exist_ok=True)

    dq_dtheta = dcom_dtheta(thetas, p_mean)

    plot_dcom_dtheta(dq_dtheta, thetas, path=result_dir + "/grad_first_shape.png")
    com_ys_all_mean, Zs_list_mean, X_mean_list = evaluate_com_height_across_thetas(thetas, p_mean, return_Zs=True)
    save_shape(result_dir + "/first_shape", p_mean)
    # coms_vars_plots(com_ys_all_mean, thetas, result_dir=result_dir + "/plot_mean/", legend=True)

    # plot gradient info 

    # os.makedirs(result_dir + "/plot_mean/", exist_ok=True)
    # for i, Zs in enumerate(Zs_list_mean):
    #     view_animation(X_mean_list[i], T, Zs - X_mean_list[i].reshape(-1, 1), V_ground, E_ground, 
    #                    path=result_dir + "/Zs_mean_theta_" + str(np.round(thetas[i], 3)) + "/")
                
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


    dq_dtheta = dcom_dtheta(thetas, p_best)
    plot_dcom_dtheta(dq_dtheta, thetas, path=result_dir + "/grad_best_optimized.png")
    save_shape(result_dir + "/starting_shape", p_mean)
    save_shape(result_dir + "/sbest_shape", p_best)
    # os.makedirs(result_dir + "/plot_best/", exist_ok=True)
