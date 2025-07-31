import numpy as np
import igl
import polyscope as ps
import scipy as sp
import simkit
from simkit.filesystem import get_data_directory
from simkit.sims.elastic import ElasticROMMFEMSim, ElasticROMMFEMSimParams, ElasticROMFEMSim, ElasticROMFEMSimParams

# Simulation type
sim_type = "mfem" # "mfem" or "fem"

# Load mesh
[X, _, _, T, _, _] = igl.readOBJ(get_data_directory() + "2d/cthulu/cthulu.obj")
X = X[:, 0:2]
X = X / max(X.max(axis=0) - X.min(axis=0))
dim = X.shape[1]

# Compute skinning modes and cubature points
m = 10
k = 100
[W, E,  B] = simkit.skinning_eigenmodes(X, T, m)
[cI, cW, labels] = simkit.spectral_cubature(X, T, W, k, return_labels=True)

# Uncomment to view modes, cubature, etc.
# from simkit.polyscope import (
#     view_clusters,
#     view_cubature,
#     view_displacement_modes,
#     view_scalar_modes)
# view_scalar_modes(X, T, W)
# view_displacement_modes(X, T, B, a=1)
# view_cubature(X, T, cI, cW, labels)

# Initialize subspace DOF
z = simkit.project_into_subspace(X.reshape(-1, 1), B)

# Cubature deformation Jacobian
G = simkit.selection_matrix(cI, T.shape[0])
Ge = sp.sparse.kron(G, sp.sparse.identity(dim*dim)) # maps cubature points to full mesh
J = simkit.deformation_jacobian(X, T)               # full-space deformation Jacobian
GJB = Ge @ J @ B                                    # deformation Jacobian * modes at cubature points

# Initializing stretch DOF
F = (GJB @ z).reshape(-1, dim, dim) # initial deformation gradients
C , Ci = simkit.symmetric_stretch_map(cI.shape[0], dim)
a = (Ci @ simkit.stretch(F).reshape(-1, 1)).reshape(-1, 1)

# Initial velocity
z_dot = np.zeros(z.shape)

# Creating simulator
if sim_type == "mfem":
    sim_params = ElasticROMMFEMSimParams()
elif sim_type == "fem":
    sim_params = ElasticROMFEMSimParams()
else:
    raise ValueError(f"Invalid sim type: {sim_type}")

sim_params.ym = 5e5  # Young's modulus (Pa)
sim_params.h = 1e-2  # time step (s)
sim_params.rho = 1e3 # density kg/m^3
sim_params.solver_p.max_iter= 1
sim_params.solver_p.do_line_search = True #True

if sim_type == "mfem":
    sim = ElasticROMMFEMSim(X, T, B,  cI, cW, sim_params)
elif sim_type == "fem":
    sim = ElasticROMFEMSim(X, T,B, cI=cI, cW=cW, p=sim_params)
else:
    raise ValueError(f"Invalid sim type: {sim_type}")

# Gravity force (negated because b's are in rhs of newton's method)
bg =  -simkit.gravity_force(X, T, rho=sim_params.rho).reshape(-1, 1)

# Pinning DOF
bI =  np.where(X[:, 0] < 0.001 + X[:, 0].min())[0]
bc0 = (X[bI, :])

period = 100
ps.init()
ps.set_ground_plane_mode("none")
mesh = ps.register_surface_mesh("mesh", X, T, edge_width=1)
for i in range(1000):
    bc = bc0 + np.sin( 2.0 * np.pi * i / (period)) * np.array([[1, 0]])
    [Q_ext, b_ext] = simkit.dirichlet_penalty(bI, bc, X.shape[0],  1e8)

    BQB_ext = sim.B.T @ Q_ext @ sim.B
    Bb_ext = sim.B.T @ (b_ext + bg)

    if sim_type == "mfem":
        z_next, a_next = sim.step(z, a,  z_dot, Q_ext=BQB_ext, b_ext=Bb_ext)
        z_dot = (z_next - z) / sim_params.h    
        z = z_next.copy()
        a = a_next.copy()
    else:
        z_next = sim.step(z,  z_dot, BQB_ext, Bb_ext)
        z_dot = (z_next - z) / sim_params.h
        z = z_next.copy()
    
    x = B @ z
    
    mesh.update_vertex_positions(x.reshape(-1, 2))
    ps.frame_tick()