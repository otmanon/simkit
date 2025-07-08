import os
import igl
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import simkit as sk

from simkit.polyscope.view_displacement_modes import view_displacement_modes

from animation_viewers import animation_viewer_2D

def objective(Zs, B, X, T, forward_dir=np.array([1, 0]), SB=None, GAJB=None):
    z0 = Zs[:, 0]
    z1 = Zs[:, -1]
    z_diff = z1 - z0 # distance traveled throughout the simulation
    d_com = sk.subspace_com(z_diff, B, X, T, SB=SB)
    
    
    v_hat = forward_dir
    
    
    J_disp = d_com @ v_hat
    
    d_com_hist = sk.subspace_com(Zs - Zs[:, [0]], B, X, T, SB=SB)
    J_com_y = np.max(np.abs(d_com_hist[:, 1]))
    


    R = sk.subspace_rotation(Zs, B, X, T, GAJB=GAJB)
    forward_dirs = (R @ forward_dir).T
    J_alignment = np.min(forward_dirs.T @ forward_dir[:, None])
    J_alignment = max(J_alignment, 0)
    
    J = (J_disp)* J_alignment#* np.exp(1 - J_alignment)
    
    # print("J_disp: ", J_disp) 
    # print("J_alignment: ", J_alignment)
    return J


def simulate(sim : sk.sims.elastic.ModalMuscleSim, p : np.ndarray, num_timesteps=1000):
    """
    Simulates the modal muscle system for a given set of parameters.
    
    Parameters
    ----------
    sim : sk.sims.ModalMuscleSim
        The modal muscle simulation object.
    p (m, 3): np.ndarray
        The sinusoidal actuation parameters of the modal muscle simulation.
        First column is the amplitude, second column is the period, third column is the phase.
    num_timesteps : int, optional
        The number of timesteps to simulate.
        
    Returns
    -------
    Zs (n, num_timesteps): np.ndarray
        The modal muscle activations.
    """
    Zs = np.zeros((sim.B.shape[1], num_timesteps ))
    z, z_dot, a = sim.rest_state()
    for i in range(num_timesteps):   
        Zs[:, i] = z.flatten()
        t = i 
        y = p[:, 0] * (np.sin(2 * np.pi * (t + p[:,2])/ p[:, 1]))
        a[:-1] = y.sum(axis=0)
        f = np.zeros((sim.B.shape[1], 1))
        z_next = sim.step(z, z_dot, a, b_ext= f)
        z_dot = (z_next - z) / sim.sim_params.h
        z = z_next.copy()
    return Zs


class SolutionTransform():
    def __init__(self, P):
        self.p_mean = P.flatten()
        
        P_scale = P.copy()
        P_scale[:, 0] = 0.3 * P[:, 0]
        P_scale[:, 1] = 0.3 * P[:, 1]
        P_scale[:, 2] = np.pi/2
        self.p_scale = P_scale.flatten()

        P_max = P.copy()
        P_max[:, 0] = 2 * P[:, 0]
        P_max[:, 1] = 3 * P[:, 1]
        P_max[:, 2] = np.pi/2
        self.p_max = P_max.flatten()

        P_min = P.copy()
        P_min[:, 0] = 0
        P_min[:, 1] = 0.3* P[:, 1]
        P_min[:, 2] = 0
        self.p_min = P_min.flatten()
        
    def transform(self, dp ):
        p = self.p_mean + dp*self.p_scale
        p = np.clip(p, self.p_min, self.p_max)
        return p



dir = os.path.join(os.path.dirname(__file__))
name = "horse"
m = 6
nc = 30
k = 5
max_s = 1
period = 50
fps = 120

num_threads = 1
phase = 0
modeset = [3, 4]
dim_a = len(modeset)
num_timesteps = 500
vis_data = False
read_result = True
data_dir = os.path.join(dir, '..\\..\\data\\2d', name)
result_dir = os.path.join(dir, 'results\\2d\\',name)

[X, _, _, T, _, _] = igl.readOBJ(os.path.join( data_dir,name +'.obj'))
X = X[:, :2]
X = sk.normalize_and_center(X)
X[:, 1] = X[:, 1] - np.min(X[:, 1]) + 1e-6

dim = X.shape[1]
[W, _E, B] = sk.skinning_eigenmodes(X, T, m)
B = sk.orthonormalize(B, M=sp.sparse.kron(sk.massmatrix(X, T), sp.sparse.identity(dim)))
[_E, D] = sk.linear_modal_analysis(X, T, max(modeset) + 1)    
D_modeset = D[:, modeset]
D_modeset_one = np.hstack([D_modeset, X.reshape(-1, 1)])
limit_a = sk.limit_actuation_dirichlet_energy(X, T, D, max_s=max_s)

[_cI, _cW, l] = sk.spectral_cubature(X, T, W, k, return_labels=True)
d = np.zeros(T.shape[0])
g = B.T @ sk.gravity_force(X, T, a=-9.8, rho=1e3).reshape(-1, 1)

fI = np.unique(igl.boundary_facets(T)[0])
cfI = sk.farthest_point_sampling(X[fI, :], nc)
cI = fI[cfI]

if vis_data:
    LA = np.diag(limit_a)
    sk.polyscope.view_clusters(X, T, l, path=os.path.join(result_dir, 'clusters.png'))

    sk.polyscope.view_sample_points(X, T, X[cI], path=os.path.join(result_dir, 'contact_points.png'))
    
    sk.polyscope.view_displacement_modes(X, T, D @ LA, a=1, period=fps, fps=fps, path=os.path.join(result_dir, 'modes.mp4'))
    sk.polyscope.view_displacement_modes(X, T, D_modeset @ np.diag(limit_a[modeset]), a=1, period=fps, fps=fps, path=os.path.join(result_dir, 'actuated_modes.mp4'))


sim_params = sk.sims.elastic.ModalMuscleSimParams(mu=1e6,gamma=1e6, rho=1e3, 
                                                alpha=1.0, contact=True, b0=-g)
sim_params.solver_p.max_iter = 10

ground_height = X.min(axis=0)[1]
sim = sk.sims.elastic.ModalMuscleSim(X, T, B, D_modeset_one, l, d, cI=cI, sim_params=sim_params, plane_pos=np.array([[0,ground_height]]).T)
[z, z_dot, a] = sim.rest_state()

P = np.vstack([limit_a[modeset], period * np.ones(dim_a), phase * np.ones(dim_a)]).T
p_mean = P.flatten()
tr = SolutionTransform(P)
dp0 = np.zeros(P.shape[0]*P.shape[1])
d_com, SB = sk.subspace_com(np.zeros(B.shape[1]), B, X, T, return_SB=True)
R, GAJB = sk.subspace_rotation(np.zeros(B.shape[1]), B, X, T, return_GAJB=True)


def objective_func(dp):
    p = tr.transform(dp)
    Zs = simulate(sim, p.reshape(-1, 3), num_timesteps=num_timesteps)
    r = objective(Zs, B, X, T, SB=SB, GAJB=GAJB)
    return r

if __name__ == "__main__":
    cmaes_params = sk.solvers.CMAESSolverParams(maxiter=200, popsize=16, seed=0, num_processes=8)
    cmaes_solver = sk.solvers.CMAESSolver(objective_func, cmaes_params)
    
    # Zs = simulate(sim, p_mean.reshape(-1, 3), num_timesteps=num_timesteps)
    # r = objective(Zs, B, X, T, SB=SB, GAJB=GAJB)
    # print("reward: ", r)
    # animation_viewer_2D(Zs, B, X, T, cI, ground_height)
    if not read_result:
        dp_opt, result, running_history = cmaes_solver.solve(dp0, return_result=True, return_history=True)
 
        p = tr.transform(dp_opt)
        np.save( os.path.join(result_dir, "p.npy"), p)
        np.save( os.path.join(result_dir, "result.npy"), np.array(result, dtype=object))
        np.save( os.path.join(result_dir, "running_history.npy"), np.array(running_history, dtype=object))
    
    p = np.load( os.path.join(result_dir, "p.npy"))
    result = np.load( os.path.join(result_dir, "result.npy"), allow_pickle=True)
    running_history = np.load( os.path.join(result_dir, "running_history.npy"), allow_pickle=True)

        
    fbest_history = np.array([r['fbest'] for r in running_history])
    xbest_history = np.array([r['xbest'] for r in running_history])
    
    # plot fbest_history
    plt.plot(fbest_history)
    plt.title("Best Objective")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.savefig(os.path.join(result_dir, "fbest_history.png"))
    plt.show()

    
    Zs = simulate(sim, p.reshape(-1, 3), num_timesteps=num_timesteps)
    r = objective(Zs, B, X, T, SB=SB, GAJB=GAJB)
    print("final reward: ", r)
    animation_viewer_2D(Zs, B, X, T, cI, ground_height)

