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
from simkit.sims.WaveSim import WaveSim, WaveSimParams
from simkit.spectral_basis_localization import spectral_basis_localization

import matplotlib.pyplot as plt
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.colors import *

from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir


def one_step_heat_response_vary_constraint(X, T, bI, bc, B, sigma=1, dt=1e-2):
    n = X.shape[0]
    sim_params = WaveSimParams(sigma=sigma, dt=h)
    sim = WaveSim(X, T, sim_params, B=B)
    us = []
    for i in bI:
        u = sim.zero_state()
        Q_ext, b_ext = dirichlet_penalty(bI=bI[i, None], y=bc, nv = n, gamma=1e10)
        u =  sim.step(u, Q_ext, b_ext)
        us.append( sim.B @ u)
    return us
        

def one_step_heat_response_vary_impulse(X, T, bI, bc, B, sigma=1, dt=1e-2):
    n = X.shape[0]
    sim_params = WaveSimParams(sigma=sigma, dt=h)
    sim = WaveSim(X, T, sim_params, B=B)
    us = []
    fs = []
    z = np.zeros((n, 1))
    M = massmatrix(X, T)
    for i in bI:
        u = sim.zero_state()
        f = z.copy()
        f[i] = bc
        b_ext = - M @ f
        u =  sim.step(u,  b_ext=b_ext)
        us.append( sim.B @ u)
        fs.append(f)
    return us, fs

plt.ion()
fig, ax = plt.subplots()


dir = os.path.dirname(__file__)

num_timesteps = 200

N = 100
[X, T] = uniform_line(N, return_simplex=True)


n = X.shape[0]
h = 1e-3

B_eigs = eigs(dirichlet_laplacian(X, T), k=10, M=massmatrix(X, T))[1].real
B_riv = random_impulse_vibes(X, T, 10, h=1)[0].real
B_riv_1e1 = random_impulse_vibes(X, T, 10, h=1e-1)[0].real
B_riv_1e2 = random_impulse_vibes(X, T, 10, h=1e-2)[0].real
B_spectral = spectral_basis_localization(X, T, 10, order=2)[0]
B_full = None
names =  [ 'eigs', 'riv', 'sl', 'riv_1e1', 'riv_1e2']
colors =  [red, orange, green, orange, orange]
Bs =  [ B_eigs, B_riv, B_spectral, B_riv_1e1, B_riv_1e2]

X2 = np.hstack((X, np.zeros((n, 1))))
curve0 = Curve(X2, T)

plt.xlim(-0.25, 1.25)
plt.ylim(-0.025, 0.125)
plt.xticks([])
plt.yticks([0, 0.1])

plt.title('Heat Equation Response to Dirichlet Constraint')
results = os.path.join(dir, 'results', 'dirichlet_constraint')

# for i, B in enumerate(Bs):
#     name = names[i]
#     results = os.path.join(dir, 'results', 'dirichlet_constraint', name)
#     os.makedirs(results, exist_ok=True)

#     bI = np.arange(N)
#     bc = np.array([ 0.1])[:, None]
#     Us = one_step_heat_response_vary_constraint(X, T, bI, bc, B)

#     X3 = np.hstack([X, Us[0]])
#     curve = Curve(X3, T, color=colors[i], label=names[i])
#     for j, u in enumerate(Us):
#         X3 = np.hstack([X, u])
#         curve.update_vertex_positions(X3)
#         plt.pause(0.01)
#         plt.savefig(os.path.join(results, '{:04d}.png'.format(j)), dpi=150)
#     curve.remove()   

#     video_from_image_dir(results, os.path.join(results, name + '.mp4'), fps=30)
#     mp4_to_gif(os.path.join(results, name + '.mp4'), os.path.join(results, name + '.gif'))



# title = 'Heat Equation Response to Impulse'
# for i, B in enumerate(Bs):
#     name = names[i]
    
#     plt.title(title + ' : ' + name + ' subspace')
#     results = os.path.join(dir, 'results', 'impulse_response', name)
#     os.makedirs(results, exist_ok=True)

#     bI = np.arange(N)
#     bc = np.array([ 500])[:, None]
#     Us, Fs = one_step_heat_response_vary_impulse(X, T, bI, bc, B)

#     X3 = np.hstack([X, Us[0]])
#     curve = Curve(X3, T, color=colors[i], label='response')
#     X4 = np.hstack([X, Fs[0]])
#     curve_force = Curve(X4, T, color=colors[i], linestyle='dashed', linewidth=1, label='impulse')
#     for j, u in enumerate(Us):
#         X3 = np.hstack([X, u])
#         X4 = np.hstack([X, 0.0001* Fs[j]])
#         curve.update_vertex_positions(X3)
#         curve_force.update_vertex_positions(X4)
#         plt.legend(handles=[curve.proxy_line, curve_force.proxy_line], loc='upper right')
#         plt.pause(0.01)
#         plt.savefig(os.path.join(results, '{:04d}.png'.format(j)), dpi=150)
#     curve.remove()   
#     curve_force.remove()
    
#     video_from_image_dir(results, os.path.join(results, name + '.mp4'), fps=30)
#     mp4_to_gif(os.path.join(results, name + '.mp4'), os.path.join(results, '..',  name + '.gif'))



title = 'Heat Equation Error Response to Impulse'
for i, B in enumerate(Bs):
    name = names[i]
    plt.title(title + ' : ' + name + ' subspace')
    results = os.path.join(dir, 'results', 'impulse_response_error', name)
    os.makedirs(results, exist_ok=True)

    bI = np.arange(N)
    bc = np.array([ 500])[:, None]

    Us_ref, _Fs = one_step_heat_response_vary_impulse(X, T, bI, bc, None)

    Us, Fs = one_step_heat_response_vary_impulse(X, T, bI, bc, B)
    plt.xlim(-0.25, 1.25)
    plt.ylim(-0.0001,  0.1)
    plt.yticks([0,  0.025, 0.05, 0.1])
    X3 = np.hstack([X, Us[0]])
    curve = Curve(X3, T, color=colors[i], label='response')
    for j, u in enumerate(Us):
        u_ref = Us_ref[j]
        X3 = np.hstack([X, np.sqrt((u - u_ref)**2)])
        # plt.semilogy()


        # update ax.viewLim using the new dataLim
        # ax.autoscale_view()
        curve.update_vertex_positions(X3)
        plt.pause(0.01)
        plt.savefig(os.path.join(results, '{:04d}.png'.format(j)), dpi=150)
    curve.remove()   
    
    video_from_image_dir(results, os.path.join(results, name + '.mp4'), fps=30)
    mp4_to_gif(os.path.join(results, name + '.mp4'), os.path.join(results, '..',  name + '.gif'))

# for j, sim in enumerate(sims):
#     curve = Curve(X2, T, color=colors[j], label=names[j])
#     u = us[j]
#     for i in range(N):
#         u = sim.zero_state()
#         bI = np.array([i])
#         bc = np.array([ 0.1])[:, None]        
#         Q_ext, b_ext = dirichlet_penalty(bI=bI, y=bc, nv = n, gamma=1e4)
#         u =  sim.step(u, Q_ext, b_ext)
#         X3 = np.hstack([X, sim.B @ u])
#         curve.update_vertex_positions(X3)

#         plt.pause(0.01)
#     curve.remove()
#     # plt.savefig(os.path.join(results, '{:04d}.png'.format(i)), dpi=150)

