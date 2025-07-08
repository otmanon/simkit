import numpy as np
import scipy as sp
import os 
import igl
import gpytoolbox as gpy


from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.harmonic_coordinates import harmonic_coordinates
from simkit.uniform_line import uniform_line
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.random_impulse_vibes  import random_impulse_vibes
from simkit.sims.elastic.ElasticROMMFEMSim import ElasticROMMFEMSim, ElasticROMMFEMSimParams, SQPMFEMSolverParams
from simkit.sims.elastic.ElasticROMPDSim import ElasticROMPDSim, ElasticROMPDSimParams, BlockCoordSolverParams
from simkit.spectral_basis_localization import spectral_basis_localization
from simkit.normalize_and_center import normalize_and_center
from simkit.orthonormalize import orthonormalize
import matplotlib.pyplot as plt
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.colors import *
from simkit.lbs_jacobian import lbs_jacobian
from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir
    
from simkit.modal_analysis import modal_analysis

def one_step_heat_response_vary_constraint(X, T, bI, bc, B, sigma=1, dt=1e-2):
    n = X.shape[0]
    labels = np.arange(T.shape[0])
    us = []
    for i in bI:
        bc2 = bc + X[bI[i, None]]
        Q_ext, b_ext = dirichlet_penalty(bI=bI[i, None], y=bc2, nv = n, gamma=1e10)
        Q_ext = B.T @ Q_ext @ B
        b_ext = B.T @ b_ext
        sim_params = ElasticROMPDSimParams(rho=1e3, h=1e-2, ym=1e8, pr=0, solver_params=solver_params, Q0=Q_ext, b0=b_ext)
        sim = ElasticROMPDSim(X, T, B=B, labels=labels, params=sim_params)
        u, u_dot = sim.rest_state()
        u = sim.step(u,u_dot)
        us.append( sim.B @ u)
    return us
        

# def one_step_heat_response_vary_impulse(X, T, bI, bc, B, sigma=1, dt=1e-2):
#     n = X.shape[0]
#     sim_params = ElasticROMPDSimParams(rho=1e3, h=1e-2, ym=1e8, pr=0, solver_params=solver_params)
#     labels = np.arange(T.shape[0])
#     sim = ElasticROMPDSim(X, T, B=B, labels=labels, params=sim_params)
#     us = []
#     fs = []
#     z = np.zeros((n, 1))
#     M = massmatrix(X, T)
#     for i in bI:
#         u = sim.zero_state()
#         f = z.copy()
#         f[i] = bc
#         b_ext = - M @ f
#         u =  sim.step(u,  b_ext=b_ext)
#         us.append( sim.B @ u)
#         fs.append(f)
#     return us, fs


plt.ion()
fig, ax = plt.subplots()

dir = os.path.dirname(__file__)

num_timesteps = 400


[X, T] = gpy.regular_square_mesh(40, 4)
X[:, 1] = X[:, 1] * 0.1
X = X[:, :2]

X = normalize_and_center(X)
X[:, 0] *= 1.5
dim = X.shape[1]
# import polyscope as ps
# ps.init()
# ps_mesh = ps.register_surface_mesh("mesh", X, T)
# ps.show()

n = X.shape[0]
h = 1e-1
bI = np.where(X[:, 0] == X[:, 0].min())[0]
bc = np.array([0, 0.5])[None, :] + X[bI]
Q_ext, b_ext = dirichlet_penalty(bI=bI, y=bc, nv = n, gamma=1e9)

W_eigs = eigs(dirichlet_laplacian(X, T), k=10, M=massmatrix(X, T))[1].real
B_eigs = lbs_jacobian(X, W_eigs)
W_riv = random_impulse_vibes(X, T, 10, h=1e-1)[0].real
B_riv = lbs_jacobian(X, W_riv)
W_spectral = spectral_basis_localization(X, T, 10, order=1)[0]
B_spectral = lbs_jacobian(X, W_spectral)
B_full = sp.sparse.identity(n*dim)

bI = farthest_point_sampling(X, 10)
W_harm = harmonic_coordinates(X, T, bI)
B_harm = lbs_jacobian(X, W_harm)

bI = farthest_point_sampling(X, 19)
W_harm = harmonic_coordinates(X, T, bI)
B_harm_hat = np.kron( W_harm, np.eye(dim))
B_harm_hat = np.hstack([B_harm_hat, X.reshape(-1, 1)])

# B_full = modal_analysis(X, T, 10)[0]
names =  ['harmonic_coordinates', 'harmonic_coordinates_hat']
colors = [light_magenta, light_yellow]
Bs =  [B_harm, B_harm_hat]

solver_params = BlockCoordSolverParams(tolerance=1e-6, max_iter=2) #SQPMFEMSolverParams(max_iter=1, tol=1e-3, do_line_search=False)

from simkit.matplotlib.TriangleMesh import TriangleMesh
title = 'Elastic Equation Response to Impulse'
for i, B in enumerate(Bs):
    name = names[i]
    plt.title(title + ' : ' + name + ' subspace')
    results = os.path.join(dir, 'results', 'impulse_response', name)
    os.makedirs(results, exist_ok=True)

    bI = np.arange(n)
    bc = np.array([ 0, 0.5])[: None] 
    # Us_ref, _Fs = one_step_heat_response_vary_constraint(X, T, bI, bc, B_full)

    Us = one_step_heat_response_vary_constraint(X, T, bI, bc, B)


    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.xticks([])
    plt.yticks([])
    X3 = np.hstack([Us[0].reshape(-1, 2)])

    
    mesh = TriangleMesh(X3, T, facecolors=colors[i])
    os.makedirs(results, exist_ok=True)
    for j, u in enumerate(Us):
        X3 = np.hstack([u.reshape(-1, 2)])
        # plt.semilogy()
        # update ax.viewLim using the new dataLim
        # ax.autoscale_view()
        mesh.update_vertex_positions(X3)
        plt.pause(0.01)
        plt.savefig(os.path.join(results, '{:04d}.png'.format(j)), dpi=150)
    mesh.remove()
    
    video_from_image_dir(results, os.path.join(results, name + '.mp4'), fps=30)
    mp4_to_gif(os.path.join(results, name + '.mp4'), os.path.join(results, '..',  name + '.gif'))

