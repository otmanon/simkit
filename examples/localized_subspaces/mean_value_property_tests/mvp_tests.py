import numpy as np
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.harmonic_coordinates import harmonic_coordinates
from simkit.biharmonic_coordinates import biharmonic_coordinates
from simkit.massmatrix import massmatrix
from simkit.modal_analysis import modal_analysis
from simkit.sims.HeatSim import HeatSim, HeatSimParams
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.uniform_line import uniform_line
import matplotlib.pyplot as plt

from simkit.matplotlib.colors import *



N = 100
[X, T]  = uniform_line(N, return_simplex=True)
X /= N

bI = np.array([0, N-1])
y = np.array([0, 1.0])[:, None]
[Q0, b0] = dirichlet_penalty(bI, y, N, 1e8)

M = massmatrix(X, T)
f = np.zeros((X.shape[0], 1))
f[N//2] = 0
f = M @ f


[B, E, _D] = skinning_eigenmodes(X, T, 20)

bI = np.linspace(0, N-1, 20).astype(int)
B = biharmonic_coordinates(X, T, bI )
B = harmonic_coordinates(X, T, bI )

params = HeatSimParams(sigma=1, rho=1, dt=1e-7, Q0=Q0, b0=b0)
sim = HeatSim(X, T, params, B=B)

z = sim.zero_state()
plt.figure()
# plt.ylim(-0.1, 2)
# plt.ion()

plt.plot(X, np.zeros((X.shape[0], 1)), color=red)


z_next = sim.step(z)
z = z_next.copy()




u = sim.B @ z

# mvp = np.zeros_like(u)
# mvp[1:-1] = (u[:-2] + u[2:] ) / 2 
# mvp_error = np.abs(mvp - u)
# mvp_error[0] = 0
# mvp_error[-1] = 0

# plt.plot(X, B)
plt.plot( X, u, color=blue)
# plt.plot(X, mvp_error, color=green)
# plt.ylim(-0.01, (u).max())
plt.show()