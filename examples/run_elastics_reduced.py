import numpy as np
import igl
import polyscope as ps
import scipy as sp
# import gpytoolbox as gpy

from simkit import (
    deformation_jacobian,
    volume,
    selection_matrix,
    dirichlet_penalty,
    gravity_force,
    stretch,
    symmetric_stretch_map,
    skinning_eigenmodes,
    spectral_cubature,
    project_into_subspace,
)
from simkit.polyscope import (
    view_clusters,
    view_cubature,
    view_displacement_modes,
    view_scalar_modes)

from simkit.filesystem import get_data_directory
from simkit.sims.elastic import ElasticROMMFEMSim, ElasticROMMFEMSimParams, ElasticROMFEMSim, ElasticROMFEMSimParams
from simkit.spectral_moment_fitting_cubature import spectral_moment_fitting_cubature

[X, _, _, T, _, _] = igl.readOBJ(get_data_directory() + "2d/cthulu/cthulu.obj")
# [X, T] = gpy.regular_square_mesh(30, 30)
X = X[:, 0:2]

X = X / max(X.max(axis=0) - X.min(axis=0))

dim = X.shape[1]

m = 10
[W, E,  B] = skinning_eigenmodes(X, T, m)

k = 100
[cI, cW, labels] = spectral_cubature(X, T, W, k, return_labels=True)

# p = 1 will give you m * k total cubature points (roughly)
# p = 2 will give you about m choose 2 per cluster.
# For m=k=10, you end up with up to 650 samples.
k = 10
p = 2
[cI, cW, labels] = spectral_moment_fitting_cubature(X, T, W, k, return_labels=True, p=p)

# B =  np.identity(X.shape[0]*dim)
# cI = np.arange(T.shape[0])
# cW = volume(X, T) 
# view_scalar_modes(X, T, W)
# view_displacement_modes(X, T, B, a=1)
view_cubature(X, T, cI, cW, labels)

A = np.zeros(((W.shape[1], dim, dim+1)))
A[:, :dim, :dim] = np.identity(dim)
z = project_into_subspace(X.reshape(-1, 1), B)


# s = np.ones((k, 3))
# s[:, -1] = 0
# l = np.zeros(s.shape)
z_dot = np.zeros(z.shape)

G = selection_matrix(cI, T.shape[0])
Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim))
J = deformation_jacobian(X, T)
GJB = Ge @ J @ B

F = (GJB @ z).reshape(-1, dim, dim)
C , Ci = symmetric_stretch_map(cI.shape[0], dim)
a = (Ci @ stretch(F).reshape(-1, 1)).reshape(-1, 1)
# l = np.zeros(a.shape)
z_dot = np.zeros(z.shape)

rho = 1e3
bg =  - gravity_force(X, T, rho=rho).reshape(-1, 1) # negative sign of force we desire bc the b's are gonna be in rhs of newton's method
bI =  np.where(X[:, 0] < 0.001 + X[:, 0].min())[0]
bc0 = (X[bI, :])

# sim_params = ElasticROMFEMSimParams()
sim_params = ElasticROMMFEMSimParams()


sim_params.ym = 5e5
sim_params.h = 1e-2
sim_params.rho = rho
sim_params.solver_p.max_iter= 1
sim_params.solver_p.do_line_search = True #True

# sim = ElasticROMFEMSim(X, T,B, cI=cI, cW=cW, p=sim_params)
sim = ElasticROMMFEMSim(X, T, B,  cI, cW, sim_params)

period = 100
ps.init()
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
for i in range(1000):
    
    bc = bc0 + np.sin( 2.0 * np.pi * i / (period)) * np.array([[1, 0]])
    [Q_ext, b_ext] = dirichlet_penalty(bI, bc, X.shape[0],  1e8)

    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ (b_ext + bg)
    # z_next = sim.step(z,  z_dot, BQB_ext, Bb_ext)
    # z = z_next.copy()

    z_next, a_next = sim.step(z, a,  z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
    z_dot = (z_next - z) / sim_params.h    
    z = z_next.copy()
    a = a_next.copy()
    
    x = B @ z
    
    mesh.update_vertex_positions(x.reshape(-1, 2))
    ps.frame_tick()




# What Ty needs :
# Energy per-stencil
