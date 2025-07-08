import numpy as np
import scipy as sp
import os 
import igl


from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.farthest_point_sampling import farthest_point_sampling

from simkit.polyscope.view_scalar_modes import view_scalar_modes

dir = os.path.dirname(__file__)

pre = dir + "/../../../"

result_dir = dir + "/results/"
os.makedirs(result_dir, exist_ok=True)
[V, T, F] = igl.read_mesh(os.path.join(pre, 'data/3d/octopus/octopus.mesh'))



# F = np.identity(H.shape[0])[:, cI]
# G = sp.sparse.linalg.spsolve(H, F )

# G = G / G.sum(axis=1)[:, None]


# view_scalar_modes(V, T, G, period=0.25)


ms = np.arange(1, 200, 5)
hs = [5e-1, 1e-1, 5e-2, 1e-2 ]
threshold = 1e-3

import matplotlib.pyplot as plt

for h in hs:
    L = dirichlet_laplacian(V, T)
    M = massmatrix(V,  T)
    H = L + M / h**2

    density_ratios = []
    cI_full = farthest_point_sampling(V, ms.max())
    F = np.identity(H.shape[0])[:, cI_full]
    G_full = sp.sparse.linalg.spsolve(H, F ).reshape(F.shape)
    for m in ms:

        G = G_full[:, :m]
        G[G < 1e-10] = 1e-10
        G = G / G.sum(axis=1)[:, None]

        Gt = G.copy()
        Gt[G < threshold] = 0
        G_sparse = sp.sparse.csc_matrix(Gt)

        dr = density_ratio(G_sparse)
        density_ratios.append(dr)

        print("m: ", m, "Density Ratio: ", dr)

    density_ratios = np.array(density_ratios)

    np.save(result_dir + "density_ratios_h_" + str(h) + ".npy", density_ratios)

    plt.plot(ms, density_ratios, label='h = ' + str(h))

plt.legend()
plt.xlabel('m')
plt.ylabel('Density Ratio')
plt.title('Density Ratio Falloff with h')
plt.savefig(result_dir + "density_ratio_vs_m.png")