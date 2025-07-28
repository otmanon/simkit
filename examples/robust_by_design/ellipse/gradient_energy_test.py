import os
from unittest import result
import scipy as sp
import numpy as np
import simkit as sk

from examples.robust_by_design.ellipse.main import *
from simkit.solvers import CMAESSolver
from simkit.solvers.CMAESSolver import CMAESSolverParams


read_cache = False
p_mean = np.array([1.35, 0.65])
p_min = np.array([0.10, 0.10])
p_max = np.array([4.0, 4.0])

p_scale = p_mean / 20

def dcom_dtheta(thetas, p):
    epsilon = 1e-5
    q_pos = evaluate_com_height_across_thetas(thetas + epsilon, p)
    q_neg = evaluate_com_height_across_thetas(thetas - epsilon, p)
    dq_dtheta = (q_pos - q_neg)/ (2 * epsilon)
    return dq_dtheta
        



def plot_dcom_dtheta(dq_dtheta, thetas, path):
    plt.figure()
    for i in range(thetas.shape[0]):
        plt.plot(np.arange(num_timesteps)*1e-2, dq_dtheta[i, :], label="$\\theta=" + str(np.round(thetas[i], 3)) + "$")
    plt.ylabel("dq/dtheta")
    plt.xlabel("Simulation time (s)")
    plt.legend()
    plt.savefig(path, dpi=300)
    plt.show()
    
    
p_real = np.array([1.0, 1.0])

num_samples = 3
thetas = np.linspace(0, np.pi/2, num_samples)

if __name__ == "__main__":
    # get two endpoints  on curve with normal n that goes through contact_p
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = current_dir + '/results/grad_optimization/'
    os.makedirs(result_dir, exist_ok=True)


    
    # dq_dtheta = dcom_dtheta(thetas, p_mean)
    # save_shape(result_dir + "/shape_start.png", p_mean)
    # plot_dcom_dtheta(dq_dtheta, thetas, path=result_dir + "/grad_start.png")
    
    
    dq_dtheta_real = dcom_dtheta(thetas, p_real)
    save_shape(result_dir + "/shape_circle_200.png", p_real)
    plot_dcom_dtheta(dq_dtheta_real, thetas, path=result_dir + "/grad_circle_200.png")
    
    