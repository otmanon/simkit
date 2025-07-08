import numpy as np
import igl
import polyscope as ps
import scipy as sp
from simkit import deformation_jacobian, massmatrix
from simkit.sims.elastic import *
from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.grad import grad
from simkit.gravity_force import gravity_force
from simkit.sims.elastic import ElasticMFEMSim
from simkit.stretch import stretch
from simkit.symmetric_stretch_map import symmetric_stretch_map
from simkit.filesystem import get_data_directory

[X, _, _, T, _, _] = igl.readOBJ(get_data_directory() + "2d/beam/beam.obj")

# [X, _, _, T, _, _] = igl.readOBJ("../data/2d/bingby/bingby.obj")

# [X, _, _, T, _, _] = igl.readOBJ("../data/2d/T/T.obj")

# [X, _, _, T, _, _] = igl.readOBJ("./data/2d/cthulu/cthulu.obj")

X = X[:, 0:2]
X = X / max(X.max(axis=0) - X.min(axis=0))

x = X.reshape(-1, 1)
J = deformation_jacobian(X, T)

dim = X.shape[1]
F = (J @ x).reshape(-1, dim, dim)

C , Ci = symmetric_stretch_map(T.shape[0], dim)
s = (Ci @ stretch(F).reshape(-1, 1)).reshape(-1, 1)
x_dot = np.zeros(x.shape)

rho = 1e3
bg =  - gravity_force(X, T, rho=rho).reshape(-1, 1)
bI =  np.where(X[:, 0] < 0.001 + X[:, 0].min())[0]
bc0 = (X[bI, :])
[Qp, bp] = dirichlet_penalty(bI, bc0, X.shape[0],  1e7)

sim_params = ElasticFEMSimParams()
# sim_params = ElasticMFEMSimParams()

# sim_params.b = bg
# sim_params.Q = Qp
sim_params.ym = 1e7
sim_params.h = 1e-2
sim_params.rho = rho
sim_params.solver_p.max_iter= 3
sim_params.solver_p.do_line_search = True #True


sim = ElasticFEMSim(X, T, sim_params)
# sim = ElasticMFEMSim(X, T, sim_params)

period = 100
ps.init()
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
for i in range(1000):
    bc = bc0 + np.sin( 2 * np.pi * i / (period)) * np.array([[1, 0]])
    [Q_ext, b_ext] = dirichlet_penalty(bI, bc, X.shape[0],  1e8)
    
    x_next = sim.step(x,  x_dot, Q_ext, b_ext + bg)
    x_dot = (x_next - x) / sim_params.h    
    x = x_next.copy()

    # NOTE: sim.step() fails with MFEM solver because step() expects 'l' (lagrange multiplier) param
    # x_next, s_next = sim.step(x, s,  x_dot, Q_ext, b_ext + bg)
    # x_dot = (x_next - x) / sim_params.h
   
    # x = x_next.copy()
    # s = s_next.copy()

    mesh.update_vertex_positions(x.reshape(-1, 2))
    ps.frame_tick()




# What Ty needs :
# Energy per-stencil
