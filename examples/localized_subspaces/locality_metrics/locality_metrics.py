
import numpy as np
import scipy as sp
import os 

from localized_subspaces.uniform_line import uniform_line
from localized_subspaces.eigs import eigs

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
sampleI = N // 2
h = 1e-2

H = L + M / h**2


G = sp.sparse.linalg.spsolve(H, I[:, sampleI])

m = 20
[Ev, B] = eigs(L, M=M, k = m)
B = B.real
G2 = B @ (np.linalg.solve(B.T @ H @ B,  B.T[:, sampleI]))


E = (G2 - G)**2
import matplotlib.pyplot as plt

# make plot higher res, high dpi
plt.rcParams['figure.dpi'] = 300
fig, axs = plt.subplots(3, 1)

axs[0].plot(X, G)
axs[0].set_title('Green\'s Functions for each Impulse h=1e-2')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Impulse Response')

axs[1].plot(X, G2)
axs[1].set_title('BBHBB Green\'s Function response using Modal Analysis')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Impulse Response')


axs[2].plot(X, E)
axs[2].set_title('Squared Error')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Impulse Response')
# put more space between both plots
plt.tight_layout()
plt.savefig(os.path.join(pre, 'compare_eigs_full_' + 'm' +str(m) + '.png'))

plt.show()


def probabilistic_impulse_vibes( h, ns, gamma):
    sampleI = (np.linspace(0, N-1, ns)).astype(int)
    H = L + M / h**2

    P = np.hstack([X[sampleI], np.ones((ns, 1)) * gamma])
    F = gaussian_rbf(X, P)
    F = F / F.max()
    P = np.hstack([X[sampleI], np.ones((ns, 1)) * gamma])
    F = gaussian_rbf(X, P)
    F = F / F.max()
    Prob = F
    Probcond = Prob / (Prob.sum(axis=1)[:, None])
    E_ri_cond = sp.sparse.linalg.spsolve(H, M @ Probcond)


    return E_ri_cond, Probcond, Prob

ns = m 
gammas = [10, 20, 30, 50]

for gamma in gammas:

    D, Probcond, Prob = probabilistic_impulse_vibes(h, ns, gamma)

    Gg = D @ (np.linalg.solve(D.T @ H @ D,  D.T[:, sampleI]))

    E = (Gg - G)**2

    import matplotlib.pyplot as plt

    # make plot higher res, high dpi
    plt.rcParams['figure.dpi'] = 300
    fig, axs = plt.subplots(3, 1)


    axs[0].plot(X, G)
    axs[0].set_title('Green\'s Functions for each Impulse h=1e-2')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Impulse Response')

    axs[1].plot(X, Gg)
    axs[1].set_title('BBHBB Green\'s Function response using SIV')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Impulse Response')


    axs[2].plot(X, E)
    axs[2].set_title('Squared Error')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Impulse Response')

    plt.tight_layout()
    plt.savefig(os.path.join(pre, 'compare_eigs_siv_' + 'm' +str(m) + '_gamma' + str(gamma) + '.png'))
    plt.show()


    # also plot the impulse vibes
    import matplotlib.pyplot as plt

    plt.plot(X, D)
    plt.title('Impulse Vibes for gamma = ' + str(gamma))
    plt.xlabel('x')
    plt.ylabel('Impulse Vibe')

    plt.tight_layout()
    plt.savefig(os.path.join(pre, 'impulse_vibes_' + 'm' +str(m) + '_gamma' + str(gamma) + '.png'))

    plt.show()