"""mesh4PDE coarse mesh (1,200 v) scored through the full-space closing's PCA components
(weighted by sigma, x RHO) side by side with the rest Hessian's eigenmodes (1/lambda, through
the 3rd mode of every pad). -> data/sdm_hand/hand_coarse_1200.npz. Needs MESH4PDE on the path."""
import inspect
import os
import sys

import numpy as np
import scipy as sp

import simkit
from simkit.energies import stable_neo_hookean_hessian_x
from hand import DATA, load, build_hand_system, close_hand

sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import coarsen  # noqa: E402

TARGET, RHO = 1200, 3.0
fine, rig = load("hand_tets.npz"), load("hand_rig.npz")
X, T, E, nu = fine["X"], fine["T"], fine["E"], fine["nu"]
sysd = build_hand_system(fine, rig)
m = np.repeat(simkit.massmatrix(X, T, rho=fine["rho"][:, None]).diagonal(), 3)
snaps = close_hand(sysd)["x"]
lam, Phi = simkit.eigs(sysd["system"](sysd["x0"], 0.0)[2].tocsc(), k=120, M=sp.sparse.diags(m))
o = np.argsort(lam)
lam, Phi = lam[o], Phi[:, o]
k = 60
for i, n in enumerate(fine["part_names"]):
    if n.endswith("_pad"):
        sel = fine["part"] == i
        Jp = simkit.deformation_jacobian(X, T[sel]).tocsc()
        Jp.resize((Jp.shape[0], 3 * len(X)))
        Hp = stable_neo_hookean_hessian_x(X, Jp, (E / (2 * (1 + nu)))[sel, None],
                                          (E * nu / ((1 + nu) * (1 - 2 * nu)))[sel, None], simkit.volume(X, T[sel]))
        k = max(k, np.nonzero(np.einsum("ij,ij->j", Phi, Hp @ Phi) / lam >= 0.5)[0][2] + 1)
W, sig, _ = np.linalg.svd(np.sqrt(m)[:, None] * (snaps - X.ravel()).T, full_matrices=False)
npc = int(np.searchsorted(np.cumsum(sig ** 2) / (sig ** 2).sum(), 1 - 1e-6) + 1)
c = RHO * np.linalg.norm(1 / lam[:k]) / np.linalg.norm(sig[:npc])
extra = {"min_quality": 0.3} if "min_quality" in inspect.signature(coarsen).parameters else {}
Xc, Tc, P = coarsen(X=X, T=T, B=np.hstack([W[:, :npc] / np.sqrt(m)[:, None], Phi[:, :k]]),
                    eigenvalues=np.concatenate([1 / (c * sig[:npc]), lam[:k]]), target_vertices=TARGET,
                    max_extrapolation=1.0, **extra)
Tc = np.asarray(Tc, np.int64)
cand = sp.spatial.cKDTree(X[T].mean(1)).query(Xc[Tc].mean(1), k=48)[1]      # fine tet holding each centroid
a, b, cc, d = (X[T[cand][..., i]] for i in range(4))
bc = np.linalg.solve(np.stack([b - a, cc - a, d - a], -1), (Xc[Tc].mean(1)[:, None] - a)[..., None])[..., 0]
t = cand[np.arange(len(Tc)), np.concatenate([1 - bc.sum(-1, keepdims=True), bc], -1).min(-1).argmax(1)]
P = sp.sparse.csc_matrix(P)
np.savez_compressed(os.path.join(DATA, f"hand_coarse_{TARGET}.npz"), X=Xc, T=Tc, part=fine["part"][t], E=E[t],
                    nu=nu[t], rho=fine["rho"][t], P_data=P.data, P_indices=P.indices, P_indptr=P.indptr,
                    P_shape=np.array(P.shape))
print(f"{npc} PCA + {k} eigenmodes -> {len(Xc)} vertices, {len(Tc)} tets")
