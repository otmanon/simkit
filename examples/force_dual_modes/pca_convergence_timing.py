import os
import igl
import sys
import numpy as np
import scipy as sp
import timeit

from simkit.dirichlet_penalty import dirichlet_penalty
from simkit.eigs import eigs
from simkit.massmatrix import massmatrix
from simkit.polyscope.view_displacement_modes import view_displacement_modes
from sklearn.decomposition import PCA



directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory + "/../../../")

from simkit.linear_elasticity_hessian import linear_elasticity_hessian


# 2D character creation
# name = "jester"
# data_dir = directory + "/../../../data/2d/" + name + "/"
# mesh_file = data_dir +  name + ".obj"
# distribution_file = data_dir +  name + "_distribution.npy"
# [X,_, _, T, _, _] = igl.readOBJ(mesh_file)
# X = X[:, :2]
# F = T
# pinned_vertices = np.where(X[:, 1] < X[:, 1].min() + 100)[0]

# 3D character creation
name = "seal"
data_dir = directory + "/../../../data/3d/" + name + "/"
mesh_file = data_dir +  name + ".mesh"
distribution_file = data_dir +  name + "_distribution.npy"
[X, T, F] = igl.readMESH(mesh_file)
F = igl.boundary_facets(T)[0]
dim = X.shape[1]
pinned_vertices = np.where(X[:, 2] < X[:, 2].min() + 1e-1)[0]


result_dir = directory + "/results/" + name + "/"
os.makedirs(result_dir, exist_ok=True)

M = massmatrix(X, T)
Me = sp.sparse.kron(M, sp.sparse.identity(dim))
H_elastic = linear_elasticity_hessian(X=X, T=T)


H_pin = dirichlet_penalty(pinned_vertices, X[pinned_vertices], X.shape[0], 1e4)[0]
H = H_elastic + H_pin

# import polyscope as ps
# ps.init()
# mesh = ps.register_surface_mesh("mesh", X, F)
# pc = ps.register_point_cloud("pinned_vertices", X[pinned_vertices, :], radius=0.01)
# ps.show()

variances = np.load(distribution_file)
std_devs = np.sqrt(variances)

std_dev = std_devs[:, 0]
variance = variances[:, 0]
Sigma_F_sqrt = sp.sparse.diags(std_dev)
Sigma_F_sqrt_e = sp.sparse.kron(Sigma_F_sqrt, sp.sparse.identity(dim))
k = 10

# assert(k <= num_trials)
# force dual modes
Sigma_F_inv = sp.sparse.diags(1/variance)
Sigma_F_inv_e = sp.sparse.kron(Sigma_F_inv, sp.sparse.identity(dim))

def compute_fdm(k, Sigma_F_inv_e, H):
    Q = H @ Sigma_F_inv_e @ H
    [e_val, u_fdm] = eigs(Q, k=k)
    return u_fdm

read_cache = True
# view_displacement_modes(X, F, u_fdm, a= 10, period=10)


powers = np.arange(4, 14)
num_trials_list = 2** powers
num_runs = 5
times = np.zeros((num_trials_list.shape[0], num_runs))

def compute_pca(k,  sigma_F_sqrt_e,num_trials):
    chol_H = sp.sparse.linalg.factorized(H )
    Y = np.random.randn(X.shape[0]*dim, num_trials)
    # compute the force sample
    f = Sigma_F_sqrt_e @ Y
    # compute the displacement
    U = chol_H(f)

    U_mean = U.mean(axis=1).reshape(-1, 1)
    U_centered = U - U_mean
    [u_pca, s_pca, v_pca] = np.linalg.svd(U_centered, full_matrices=False)

    pca = PCA(n_components=k, svd_solver='randomized')
    pca.fit(U_centered.T)
    u_pca = pca.components_[:k].T
    
    return u_pca

if not read_cache:
    

    times_fdm = timeit.repeat(lambda: compute_fdm(k, Sigma_F_sqrt_e, H), number=1, repeat=num_runs)
    
    np.save(result_dir + "times_fdm.npy", times_fdm)
    for i, num_trials in enumerate(num_trials_list):
        
        times[i] = timeit.repeat(lambda: compute_pca(k, Sigma_F_sqrt_e, num_trials), number=1, repeat=num_runs)
    
        np.save(result_dir + "times_pca.npy", times)        
        print("num_trials :", num_trials)
        print("time : ", times[i])
    # print("e_val_error :", np.abs(e_val_pca[:k] - e_val).mean())
else:
    times_pca = np.load(result_dir + "times_pca.npy")
    times_fdm = np.load(result_dir + "times_fdm.npy")
# write matplotlib code to plot the cosine similarity
import matplotlib.pyplot as plt
plt.plot(num_trials_list,times_pca.mean(axis=1))
plt.xlabel("#Samples")
plt.ylabel("Time(s)")
plt.semilogx()
plt.semilogy()
plt.axhline(y=times_fdm.mean(), color='r')
# create new x axis at times_fdm
plt.plot()
# make sure there are 4 x ticks
plt.xticks([ 10, 100, 1000, 10000])
plt.yticks([1e0, 1e1, 1e2, 1e3])
plt.savefig(result_dir + "_timing_computation.png")
plt.savefig(result_dir + "_timing_computation.svg")
plt.show()

# repeat = 5
# for i, num_trials in enumerate(num_trials_list):
#     times[i] = timeit.timeit(lambda: compute_pca(k, chol_H, Sigma_F_sqrt_e, num_trials), number=1, repeat=repeat)

# np.save(result_dir + "times.npy", times)

# # import polyscope as ps
# # ps.init()
# # mesh = ps.register_surface_mesh("mesh", X, F)
# # for i in range(variances.shape[1]):
# #     mesh.add_scalar_quantity("variance " + str(i), variances[:, i].flatten(), enabled=True)
# # ps.show()
# import numpy as np
# import timeit
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import os

# def setup_timing():
#     # Parameters
#     dim = 3  # 3D problem
#     k = 10   # number of PCA components
#     num_trials = 1000  # number of samples
    
#     # Assuming these are your input matrices
#     # You'll need to replace these with your actual matrices
#     X = np.random.randn(1000, dim)  # Example mesh vertices
#     T = np.random.randint(0, 1000, (2000, 3))  # Example triangles
#     H = np.random.randn(X.shape[0]*dim, X.shape[0]*dim)  # Example Hessian
#     H = H.T @ H  # Make it positive definite
    
#     # Compute mass matrix
#     M = np.eye(X.shape[0]*dim)  # Simplified mass matrix
    
#     # Compute Cholesky factorization
#     chol_H = np.linalg.cholesky(H)
    
#     # Compute force sampling matrix
#     Sigma_F_sqrt_e = np.eye(X.shape[0]*dim)  # Simplified force sampling matrix
    
#     return X, T, H, M, chol_H, Sigma_F_sqrt_e, dim, k, num_trials

# def compute_pca(X, T, H, M, chol_H, Sigma_F_sqrt_e, dim, k, num_trials):
#     # Sample random forces
#     Y = np.random.randn(X.shape[0]*dim, num_trials)
    
#     # Compute force samples
#     f = Sigma_F_sqrt_e @ Y
    
#     # Solve for displacements
#     U = np.linalg.solve(chol_H, f)
    
#     # Center the data
#     U_mean = U.mean(axis=1).reshape(-1, 1)
#     U_centered = U - U_mean
    
#     # Compute PCA
#     pca = PCA(n_components=k, svd_solver='randomized')
#     pca.fit(U_centered.T)
#     u_pca = pca.components_[:k].T
    
#     return u_pca, U_centered

# def main():
#     # Create results directory if it doesn't exist
#     results_dir = "results"
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Setup
#     X, T, H, M, chol_H, Sigma_F_sqrt_e, dim, k, num_trials = setup_timing()
    
#     # Store results for each run
#     all_times = []
#     all_u_pcas = []
#     all_u_fdms = []
    
#     # Define the function to time
#     def time_pca():
#         u_pca, U_centered = compute_pca(X, T, H, M, chol_H, Sigma_F_sqrt_e, dim, k, num_trials)
#         all_u_pcas.append(u_pca)
#         all_u_fdms.append(U_centered)
#         return u_pca
    
#     # Time the computation
#     num_runs = 5
#     times = timeit.repeat(time_pca, number=1, repeat=num_runs)
    
#     # Print results
#     print(f"Timing results for computing {k} PCA components:")
#     print(f"Number of trials: {num_trials}")
#     print(f"Number of runs: {num_runs}")
#     print(f"Average time: {np.mean(times):.4f} seconds")
#     print(f"Min time: {np.min(times):.4f} seconds")
#     print(f"Max time: {np.max(times):.4f} seconds")
#     print(f"Standard deviation: {np.std(times):.4f} seconds")
    
#     # Save results
#     np.save(os.path.join(results_dir, "timings.npy"), np.array(times))
#     np.save(os.path.join(results_dir, "u_pcas.npy"), np.array(all_u_pcas))
#     np.save(os.path.join(results_dir, "u_fdms.npy"), np.array(all_u_fdms))
    
#     # Save parameters for reference
#     params = {
#         'dim': dim,
#         'k': k,
#         'num_trials': num_trials,
#         'num_runs': num_runs,
#         'X_shape': X.shape,
#         'T_shape': T.shape
#     }
#     np.save(os.path.join(results_dir, "params.npy"), params)

# if __name__ == "__main__":
#     main() 