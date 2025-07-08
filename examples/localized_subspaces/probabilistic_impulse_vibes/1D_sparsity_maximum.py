import numpy as np
import scipy as sp
import os 

from localized_subspaces.uniform_line import uniform_line
from simkit.gaussian_rbf import gaussian_rbf
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix


# This script will highlight how our probabilistic impulse vibes becomes sparser or denser.
pre = os.path.dirname(__file__) + '/'
N = 1000
[X, T] = uniform_line(N, return_simplex=True)
L = dirichlet_laplacian(X, T)
M = massmatrix(X,  T)
Mi = sp.sparse.diags(1 / M.diagonal())
I = np.identity(M.shape[0])


def probabilistic_impulse_vibes_sparsity_max( h, ns, gamma, smax = 2):
    sampleI = (np.linspace(0, N-1, ns)).astype(int)
    H = L + M / h**2

    P = np.hstack([X[sampleI], np.ones((ns, 1)) * gamma])
    F = gaussian_rbf(X, P)
    F = F / F.max()
    
    Prob = F
    j = np.argsort(Prob, axis=1)[:,-smax:]

    i = np.repeat(np.arange(j.shape[0])[:, None], ( j.shape[1]), axis=1)
    
    # get indices i, j  of Prob
    top_prob = Prob[i, j]

    Prob = sp.sparse.csc_matrix((top_prob.flatten(), (i.flatten(), j.flatten())), shape=Prob.shape).toarray()


    Probcond = Prob / (Prob.sum(axis=1)[:, None])
    E_ri_cond = sp.sparse.linalg.spsolve(H, M @ Probcond)


    return E_ri_cond, Probcond, Prob


sparsity_maximum = 2


ns = 10
h = 1e-1
gamma = 10


G, Probcond, Prob = probabilistic_impulse_vibes_sparsity_max(h, ns, gamma, sparsity_maximum)

import matplotlib.pyplot as plt

# make plot higher res, high dpi
plt.rcParams['figure.dpi'] = 300
fig, axs = plt.subplots(3, 1)

axs[0].plot(X, Prob)
axs[0].set_title('E[x in Gi]')
axs[0].set_xlabel('x')
axs[1].plot(X, Probcond)
axs[1].set_title('E[x in Gi | x in G]')
axs[1].set_xlabel('x')

axs[2].plot(X,   G)
axs[2].set_title('Subspace (E[ri | x in G])')
axs[2].set_xlabel('x')

plt.tight_layout()
plt.savefig(pre + 'sparsity_maximum_ns' + str(ns) + "_h" + "{:.2E}".format(h) + "_g" + "{:.2E}".format(gamma) + ".png")
plt.show()

