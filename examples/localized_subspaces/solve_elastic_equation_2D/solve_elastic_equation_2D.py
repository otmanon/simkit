import numpy as np
import scipy as sp
import os 
import igl
import gpytoolbox as gpy


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
W_riv = random_impulse_vibes(X, T, 10, h=1)[0].real
B_riv = lbs_jacobian(X, W_riv)
W_spectral = spectral_basis_localization(X, T, 10, order=1)[0]
B_spectral = lbs_jacobian(X, W_spectral)
B_full = sp.sparse.identity(n*dim)
# B_full = modal_analysis(X, T, 10)[0]
names =  ['eigs', 'riv', 'spectral', 'full']
colors = [light_red, light_green, light_blue, light_orange]
Bs =  [B_eigs, B_riv, B_spectral, B_full]

solver_params = BlockCoordSolverParams(tolerance=1e-6, max_iter=2) #SQPMFEMSolverParams(max_iter=1, tol=1e-3, do_line_search=False)
sims = []
us = []
u_dots = []
ss = []

from simkit.matplotlib.TriangleMesh import TriangleMesh

for i, B in enumerate(Bs):
    if isinstance(B, np.ndarray):
        Be = orthonormalize(B, M=sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))
    else:
        Be = B
    sim_params = ElasticROMPDSimParams(rho=1e3, h=1e-2, ym=1e8, pr=0, Q0=Be.T @ Q_ext @ Be, b0= Be.T @ b_ext, solver_params=solver_params)
    # sim = ElasticROMMFEMSim(X, T, B=B, p=sim_params)
    labels = np.arange(T.shape[0])
    sim = ElasticROMPDSim(X, T, B=Be, labels=labels, params=sim_params)
    sims.append(sim)

    u,  u_dot = sim.rest_state()
    us.append(u)
    # ss.append(s)
    u_dots.append(u_dot)

plt.axis('equal')
plt.xlim(-2, 2)
plt.xticks([])
plt.yticks([])
# plt.legend(handles=[curve.proxy_line for curve in curves], loc='upper left')

for j, sim in enumerate(sims):
    
    u = us[j]
    u_dot = u_dots[j]
    mesh = TriangleMesh(X, T, facecolors=colors[j], label=names[j])
    results = os.path.join(dir, 'results', 'elastic_sims',  names[j])
    os.makedirs(results, exist_ok=True)
    for i in range(num_timesteps):

        plt.title('Elastic Equation with ' + str(names[j]) + ' Subspace : t=' + '{:.2f}'.format(i*sim.params.h) + 's')
        # s = ss[j]
        u_prev = u.copy()
        u_next =  sim.step(u, u_dot)
        u_dot = (u_next - u_prev)/h
        u = u_next.copy()
        # us[j] = u
        # ss[j] = s
        # u_dots[j] = u_dot

        X3 = (sim.B @ u).reshape(-1, dim)
        mesh.update_vertex_positions(X3)
        plt.pause(0.01)
        plt.savefig(os.path.join(results, '{:04d}.png'.format(i)), dpi=150)
    mesh.remove()

    video_from_image_dir(results, os.path.join(results,  names[j] +  '.mp4'), fps=24)
    mp4_to_gif(os.path.join(results, names[j] + '.mp4'), os.path.join(results,  names[j] +  '.gif'))