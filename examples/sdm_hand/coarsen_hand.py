"""Coarsen the hand with mesh4PDE, scoring collapses through PCA + elastic modes.

The coarse mesh has to reproduce two things: the closing motion (joints bending
while the blocks move rigidly) and the soft tissue (pad deformation). So the
basis mesh4PDE scores collapses through has two halves, side by side:

* the lowest Hessian eigenmodes of the hand at rest, up to the 3rd deformation mode
  of every pad, weighted ``1/lambda`` as usual;
* the principal components of the full-space closing (mass-weighted PCA of its
  quasi-static snapshots), weighted by their singular values ``sigma`` and scaled
  so this half carries ``RHO`` times the elastic half's weight.

mesh4PDE refuses collapses that would leave a fine vertex more than ``MAX_EXTRAP``
coarse tets outside its coarse tet, and (``min_quality``, when the build supports
it) collapses that would leave a tet with mean-ratio quality below ``MIN_QUALITY``.
Each coarse tet takes the material of the fine tet its centroid lies in.

Needs mesh4PDE (``MESH4PDE=/path/to/mesh4PDE``). Writes
``data/sdm_hand/hand_coarse_1200.npz`` (``X, T, part, E, nu, rho`` and ``P``)::

    python examples/sdm_hand/coarsen_hand.py
"""
import inspect
import os
import sys

import numpy as np
import scipy as sp

import simkit
from simkit.energies import stable_neo_hookean_hessian_x

from hand import DATA, load_fine, build_hand_system, close_hand

sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
from lib import coarsen  # noqa: E402  (mesh4PDE)

TARGET, RHO, MAX_EXTRAP, MIN_QUALITY = 1200, 3.0, 1.0, 0.3
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def elastic_modes(fine, sysd, M, k_max=120, per_pad=3, k_min=60):
    """Lowest eigenmodes of the rest Hessian, through the ``per_pad``-th mode that puts
    at least half its energy into each pad."""
    X, T = fine["X"], fine["T"]
    H = sysd["system"](sysd["x0"], 0.0)[2].tocsc()
    lam, Phi = simkit.eigs(H, k=k_max, M=M)
    o = np.argsort(lam)
    lam, Phi = lam[o], Phi[:, o]
    E, nu = fine["E"], fine["nu"]
    names = list(fine["part_names"])
    need = k_min
    for n in names:
        if n.endswith("_pad"):
            sel = fine["part"] == names.index(n)
            J = simkit.deformation_jacobian(X, T[sel]).tocsc()
            J.resize((J.shape[0], 3 * len(X)))
            Hp = stable_neo_hookean_hessian_x(X, J, (E / (2 * (1 + nu)))[sel, None],
                                              (E * nu / ((1 + nu) * (1 - 2 * nu)))[sel, None],
                                              simkit.volume(X, T[sel]), psd=True)
            share = np.einsum("ij,ij->j", Phi, Hp @ Phi) / lam
            idx = np.nonzero(share >= 0.5)[0]
            need = max(need, idx[per_pad - 1] + 1 if len(idx) >= per_pad else k_max)
    return Phi[:, :need], lam[:need]


def pca_modes(snapshots, X, m, energy=1 - 1e-6):
    """Mass-weighted PCA of displacement snapshots: M-orthonormal components and
    their singular values."""
    sq = np.sqrt(m)
    W, sig, _ = np.linalg.svd(sq[:, None] * (snapshots - X.reshape(-1)).T, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(sig ** 2) / (sig ** 2).sum(), energy) + 1)
    return W[:, :k] / sq[:, None], sig[:k]


def locate(X, T, Q, k=48):
    """Index of the tet of (X, T) containing each point of Q (or the one it is least
    outside of, among the k with the nearest centroids)."""
    cand = sp.spatial.cKDTree(X[T].mean(1)).query(Q, k=k)[1]
    a, b, c, d = (X[T[cand][..., i]] for i in range(4))
    lam = np.linalg.solve(np.stack([b - a, c - a, d - a], -1), (Q[:, None, :] - a)[..., None])[..., 0]
    bary = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], -1).min(-1)
    return cand[np.arange(len(Q)), bary.argmax(1)]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fine, rig = load_fine()
    X, T = fine["X"], fine["T"]
    sysd = build_hand_system(fine, rig)
    m = np.repeat(simkit.massmatrix(X, T, rho=fine["rho"][:, None]).diagonal(), 3)
    snap = os.path.join(RESULTS, "full_closing.npz")
    if not os.path.exists(snap):
        run = close_hand(sysd)
        np.savez_compressed(snap, x=run["x"], a=run["a"])
        print(f"full-space closing: {run['seconds']:.0f} s")
    Phi_e, lam_e = elastic_modes(fine, sysd, sp.sparse.diags(m))
    Phi_p, sig = pca_modes(np.load(snap)["x"], X, m)
    c = RHO * np.linalg.norm(1 / lam_e) / np.linalg.norm(sig)
    print(f"basis: {Phi_p.shape[1]} PCA components + {Phi_e.shape[1]} eigenmodes (PCA weight x{RHO})")
    extra = {"min_quality": MIN_QUALITY} if "min_quality" in inspect.signature(coarsen).parameters else {}
    Xc, Tc, P = coarsen(X=X, T=T, B=np.hstack([Phi_p, Phi_e]), eigenvalues=np.concatenate([1 / (c * sig), lam_e]),
                        target_vertices=TARGET, max_extrapolation=MAX_EXTRAP, **extra)
    Tc = np.asarray(Tc, np.int64)
    part = fine["part"][locate(X, T, Xc[Tc].mean(1))]
    lut = {p: i for i, p in enumerate(fine["part"])}          # first fine tet of each part: its material
    E, nu, rho = (fine[k][[lut[p] for p in part]] for k in ("E", "nu", "rho"))
    P = sp.sparse.csc_matrix(P)
    np.savez_compressed(os.path.join(DATA, f"hand_coarse_{TARGET}.npz"), X=Xc, T=Tc, part=part, E=E, nu=nu, rho=rho,
                        P_data=P.data, P_indices=P.indices, P_indptr=P.indptr, P_shape=np.array(P.shape))
    print(f"coarse: {len(Xc)} vertices, {len(Tc)} tets, min P weight {P.data.min():.2f}")


if __name__ == "__main__":
    main()
