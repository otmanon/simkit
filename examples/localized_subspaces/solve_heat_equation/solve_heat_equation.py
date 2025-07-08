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
from simkit.sims.HeatSim import HeatSim, HeatSimParams
from simkit.spectral_basis_localization import spectral_basis_localization

import matplotlib.pyplot as plt
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.colors import *

from simkit.filesystem.mp4_to_gif import mp4_to_gif
from simkit.filesystem.video_from_image_dir import video_from_image_dir


        
plt.ion()
fig, ax = plt.subplots()


dir = os.path.dirname(__file__)
results = os.path.join(dir, 'results')
os.makedirs(results, exist_ok=True)
num_timesteps = 200

N = 100
[X, T] = uniform_line(N, return_simplex=True)


n = X.shape[0]
h = 1e-3
bI = np.array([ N-1])
bc = np.array([ 0.1])[:, None]
Q_ext, b_ext = dirichlet_penalty(bI=bI, y=bc, nv = n, gamma=1e4)

B_eigs = eigs(dirichlet_laplacian(X, T), k=10, M=massmatrix(X, T))[1].real
B_riv = random_impulse_vibes(X, T, 10, h=1)[0].real
B_spectral = spectral_basis_localization(X, T, 10, order=1)[0]
B_full = None

names = ['eigs', 'riv', 'spectral', 'full']
colors = [red, green, blue, orange]
Bs = [B_eigs, B_riv, B_spectral, B_full]
sim_params = HeatSimParams(sigma=1, dt=h, Q0=Q_ext, b0=b_ext)
sims = []
us = []

X2 = np.hstack((X, np.zeros((n, 1))))
curve0 = Curve(X2, T)
curves = []
for i, B in enumerate(Bs):
    sim = HeatSim(X, T, sim_params, B=B)
    sims.append(sim)

    u = sim.zero_state()
    us.append(u)

        
    curve = Curve(X2, T, color=colors[i], label=names[i])
    curves.append(curve)

plt.xlim(-0.25, 1.25)
plt.ylim(-0.025, 0.125)
plt.xticks([])
plt.yticks([])
plt.title('Heat Equation with Different Subspaces : t=0s')
plt.legend(handles=[curve.proxy_line for curve in curves], loc='upper left')

for i in range(num_timesteps):

    for j, sim in enumerate(sims):
        u = us[j]
        u =  sim.step(u)
        X3 = np.hstack([X, sim.B @ u])
        curves[j].update_vertex_positions(X3)
    
    
    plt.title('Heat Equation with Different Subspaces : t=' + "{:.3f}".format(i*h) + 's')   
    plt.pause(0.01)

    plt.savefig(os.path.join(results, '{:04d}.png'.format(i)), dpi=150)

video_from_image_dir(results, os.path.join(dir, 'heat_equation.mp4'), fps=24)
mp4_to_gif(os.path.join(dir, 'heat_equation.mp4'), os.path.join(dir, 'heat_equation.gif'))