import numpy as np
import scipy as sp
import os 

from simkit.uniform_line import uniform_line
from simkit.gaussian_rbf import gaussian_rbf
from simkit.dirichlet_laplacian import dirichlet_laplacian
from simkit.massmatrix import massmatrix

"""
This script will highlight how our probabilistic impulse vibes works.

We will start with a simple 1D line mesh, and visualize random individual impulses, as narrow hat funciton impulses.

Then for each one we will plot the resulting green's function, for a propagation time h=1e-2.

"""

dir = os.path.dirname(__file__)

N = 100
[X, T] = uniform_line(N, return_simplex=True)
ns = 3
sampleI = (np.linspace(0, N-1, ns)).astype(int)

L = dirichlet_laplacian(X, T)

M = massmatrix(X,  T)


h = 1e-2
H = L + M / h**2

Mi = sp.sparse.diags(1 / M.diagonal())

I = np.identity(H.shape[0])
F = Mi @ I[:, sampleI]
G = sp.sparse.linalg.spsolve(H, F)


# make two subplots. the first, on top, shows the impulses M @ I on the unit line. The second, below, shows the resulting Green's function G.
import matplotlib.pyplot as plt

# make plot higher res, high dpi
plt.rcParams['figure.dpi'] = 300
fig, axs = plt.subplots(2, 1)

axs[0].plot(X, F)
axs[0].set_title('Impulses')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Impulse')

axs[1].plot(X, G)
axs[1].set_title('Green\'s Functions for each Impulse')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Impulse Response')
# put more space between both plots
plt.tight_layout()
plt.savefig(os.path.join(dir, 'impulses_and_responses_1D.png'))
plt.clf()
# now instead of impulse hat function responses, we will use gaussians

P = np.hstack([X[sampleI], np.ones((ns, 1)) * 100.0])
F = gaussian_rbf(X, P)

F = F / F.max()


# Next, let's say that Gi denotes the i-th impulse of mangitude a.

# With each Gi we associate a set of probability distributions over all vertices.
# This probability distribution is denoted P(x in Gi), and has an expected value of E[x in Gi]

# It's also associated with a distribution of responses r, whose expected value is E[r] = H^-1 E[x in Gi] a

E_ri =  sp.sparse.linalg.spsolve(H,  M @ F)



# We can then define the Expected response to each impulse at x as H^-1 E[ x in Gi] a
Prob = F
Probcond = Prob / (Prob.sum(axis=1)[:, None])

fig, axs = plt.subplots(2, 1)
axs[0].plot(X, M @ Prob)
axs[0].set_title('E[x in Gi]')
axs[0].set_xlabel('x')
axs[0].set_ylabel('E[x in Gi]')

axs[1].plot(X, E_ri)
axs[1].set_title('E[ri]')
axs[1].set_xlabel('x')
axs[1].set_ylabel('E_ri')
plt.tight_layout()
plt.savefig(os.path.join(dir, 'probability_distribution.png'))
plt.clf()



fig, axs = plt.subplots(2, 1)
axs[0].plot(X, M @ Probcond)
axs[0].set_title('P[x in Gi | x in G]')
axs[0].set_xlabel('x')
axs[0].set_ylabel('P[x in Gi | x in G]')


E_ri_cond =sp.sparse.linalg.spsolve(H, M @ Probcond)
axs[1].plot(X, E_ri_cond)
axs[1].set_title('E[ri | x in G]')
axs[1].set_xlabel('x')
axs[1].set_ylabel('E[ri | x in G]')

plt.tight_layout()
plt.savefig(os.path.join(dir, 'conditional_probability.png'))


# Finally we can define the conditional probability P(x in Gi | x in G) as the probability that x is in Gi given that x is affected by an impulse.


# With this distribution we can define the Expected response at x given that an impulse has been applied on x as sum_i H^-1  P(x in Gi | x in G) a_i

