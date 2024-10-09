import os
import igl
import gpytoolbox as gpy
import matplotlib.pyplot as plt

from simkit.farthest_point_sampling import farthest_point_sampling
from simkit.gravity_force import gravity_force
from simkit.matplotlib.Curve import Curve
from simkit.matplotlib.PointCloud import PointCloud
from simkit.matplotlib.TriangleMesh import TriangleMesh
from simkit.matplotlib.colors import *
from simkit.orthonormalize import orthonormalize
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from simkit.polyscope.view_scalar_modes import view_scalar_modes
from simkit.skinning_eigenmodes import skinning_eigenmodes
from simkit.subspace_com import subspace_com
from simkit.modal_analysis import modal_analysis
from simkit.spectral_cubature import spectral_cubature
from simkit.normalize_and_center import normalize_and_center
from simkit.sims.elastic.ModalMuscleSim import *

from ModalMuscleTestSim import ModalMuscleTestSim, ModalMuscleTestSimParams
dir = os.path.join(os.path.dirname(__file__))

def simulate(sim, p, num_timesteps=1000):
    Zs = np.zeros((B.shape[1], num_timesteps + 1))
    z, z_dot, a = sim.rest_state()
    Zs[:, 0] = z.flatten()
    for i in range(num_timesteps):   
        t = i * sim_params.h
        y = p[:, 0]  * (np.sin(2 * np.pi * (t/ p[:, 1] + p[:, 2])))
        a[:-1] = y.sum(axis=0)
        f = np.zeros((B.shape[1], 1))
        z_next = sim.step(z, z_dot, a, b_ext= f - g)
        z_dot = (z_next - z) / sim_params.h
        z = z_next.copy()
        Zs[:, i+1] = z_next.flatten()
    return Zs


m = 10
nc = 30
k = 10
modeset = [3, 4, 5]
num_timesteps = 1000

[X, T] = gpy.regular_square_mesh(20, 20)
# [X, _, _, T, _, _] = igl.read_obj(dir + "/../../data/2d/beam/beam.obj")
X = X[:, :2]
X[:, 1] *= 0.2
X = normalize_and_center(X)
dim = X.shape[1]


[W, _E, B] = skinning_eigenmodes(X, T, m)
B = orthonormalize(B, M=sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim)))

[_E, D] = modal_analysis(X, T, max(modeset) + 1)    
D = D[:, modeset]
D = np.hstack([D, X.reshape(-1, 1)])

[_cI, _cW, l] = spectral_cubature(X, T, W, k, return_labels=True)
d = np.zeros(T.shape[0])

g = B.T @ gravity_force(X, T, a=-9.8, rho=1e3).reshape(-1, 1)


fI = np.unique(igl.boundary_facets(T))
cfI = farthest_point_sampling(X[fI, :], nc)
cI = fI[cfI]

sim_params = ModalMuscleSimParams(mu=1e5,gamma=1e5, rho=1e3, alpha=1.0, contact=True)
sim_params.solver_p.max_iter = 10

sim = ModalMuscleSim(X, T, B, D, l, d, cI=cI, params=sim_params, plane_pos=np.array([[0, -2]]).T)
[z, z_dot, a] = sim.rest_state()

# Mv = sp.sparse.kron(massmatrix(X, T), sp.sparse.identity(dim))
# BMD = B.T @ Mv @ D 

amp = 0.1
period = 0.3
phase = 0

p = np.hstack([amp, period, phase])[None, :]
plt.ion()
[fig, ax] = plt.subplots(dpi=100, figsize=(10, 10))
plt.axis('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([-5, -1.5, 2])
ax.set_yticks([-2, 0, 2])
mesh = TriangleMesh(X, T, edgecolors=gray, linewidths=0.1,  outlinewidth=2)
pc = PointCloud(X[cI, :], size=10)

com, SB = subspace_com(z, B, X, T, return_SB=True)
pc2 = PointCloud(com.reshape(-1, 2), size=100, color='red')
plt.axline((0, -2), slope=0, color='black', linewidth=4, zorder=-1)


Zs = simulate(sim, p, num_timesteps=num_timesteps)
Us = B @ Zs
for i in range(num_timesteps):    
    U = Us[:, i].reshape(-1, dim)    
    mesh.update_vertex_positions(U)
    pc.update_vertex_positions(U[cI, :])
    com = subspace_com(Zs[:, i], B, X, T, SB=SB)
    pc2.update_vertex_positions(com.reshape(-1, 2))
    plt.pause(0.0001)

    