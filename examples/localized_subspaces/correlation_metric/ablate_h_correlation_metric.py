


import numpy as np
import scipy as sp
import os 

from simkit.uniform_line import uniform_line
from simkit.gaussian_rbf import gaussian_rbf
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix
from simkit.random_impulse_vibes import random_impulse_vibes
from simkit.subspace_corrolation import subspace_corrolation
from simkit.eigs import eigs
"""
This script will evalutate the correlation metric detailed by Philip
"""

dir = os.path.dirname(__file__)

N = 1000
[X, T] = uniform_line(N, return_simplex=True)
m = 10

L = dirichlet_laplacian(X, T)
M = massmatrix(X,  T)


# import matplotlib.pyplot as plt
# plt.plot(X, G)
# plt.title('Green\'s Functions for each Impulse h=1e-2')
# plt.xlabel('x')
# plt.ylabel('Impulse Response')
# plt.show()
# sI = np.zeros(X.shape[0]).astype(int)
# tI = np.arange(X.shape[0])

import matplotlib.pyplot as plt



# plt.show()



ms = np.arange(1, 100, 5)
dt = 1e-2
hs = [1e-3, 1e-2, 1e-1, 1]

H = L + M / dt**2
Hi = sp.sparse.linalg.spsolve(H, np.identity(H.shape[0]))
errors_eig = []
errors_rivs = [[], [], [], []]
cor = subspace_corrolation(Hi,   B=None, M=M)
for m in ms:
    [E, G] = eigs(L, M=M, k = m)
    sub_cor_eig  = subspace_corrolation(Hi,  B=G, M=M)
    errors_eig.append((np.abs(np.abs(sub_cor_eig) - np.abs(cor))).sum())
    

    for c, h in enumerate(hs):
        [G, cI, G_full] = random_impulse_vibes(X, T, m, h=h)
        sub_cor_riv = subspace_corrolation(Hi,  B=G, M=M)

        errors_rivs[c].append((np.abs(np.abs(sub_cor_riv) - np.abs(cor))).sum())


plt.plot(ms, np.log(errors_eig), label='eigs')
for i, h in enumerate(hs):
    plt.plot(ms, np.log(errors_rivs[i]), label='riv h={:.0E}'.format(h))

plt.title('Correlation Error Across Subspaces')
plt.legend()
plt.xlabel('m')
plt.ylabel('Log Excess Correlation Error')
plt.savefig(os.path.join(dir, 'correlation_error_ablate_h.png'))
plt.show()
# plt.show()