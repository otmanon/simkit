"""Eigenfunctions of the compliant hand's operator.

``A = H_elastic + H_tendons`` at the open pose (``a = 0``, tendons at rest
length), restricted to the free DOFs (wrist base pinned), generalized with the
lumped mass matrix: ``A phi = lambda M phi``. Each mode's energy
``phi^T A phi`` is split into stiff blocks, flexure joints, pads and tendons.

    python sdm_modes.py            (after sdm_geometry.py, sdm_tets.py)

Writes ``output/sdm_modes.npz`` and ``output/renders/10_eigenfunctions.png``.
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simkit
import simkit.energies as energies

from sdm_sim import Hand, OUT
from sdm_render import tet_grid, shot

pv.OFF_SCREEN = True
REN = os.path.join(OUT, "renders")
N_MODES = 20


def main():
    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = Hand(scene)
    X, T = hand.X, hand.T
    x0 = X.reshape(-1)
    fr = hand.free
    A = hand.hessian(x0, 0.0)[fr][:, fr].tocsc()
    M = sp_diag(hand.m[fr])
    lam, Bf = simkit.eigs(A, k=N_MODES, M=M)
    order = np.argsort(lam)
    lam, Bf = lam[order], Bf[:, order]
    B = np.zeros((3 * hand.n, N_MODES))
    B[fr] = Bf

    # energy breakdown per mode
    kind_t = np.array(hand.kinds)[hand.part]
    parts = {}
    for k in ("palm", "link", "flexure", "pad"):
        sel = kind_t == k
        J = simkit.deformation_jacobian(X, T[sel]).tocsc()
        J.resize((J.shape[0], 3 * hand.n))
        parts[k] = energies.stable_neo_hookean_hessian_x(X, J, hand.mu[sel], hand.lam[sel],
                                                          hand.vol[sel], psd=True)
    parts["tendons"] = energies.mass_springs_hessian_x(X, hand.E_t, hand.ym, hand.svol,
                                                       hand.l0(0.0), psd=True)
    label = {"palm": "palm", "link": "blocks", "flexure": "flexures", "pad": "pads",
             "tendons": "tendons"}
    rows = []
    for i in range(N_MODES):
        f = B[:, i]
        e = {label[k]: float(f @ (H @ f)) for k, H in parts.items()}
        tot = sum(e.values())
        rows.append({k: v / tot for k, v in e.items()})
    np.savez_compressed(os.path.join(OUT, "sdm_modes.npz"), eigenvalues=lam, B=B)
    for i, r in enumerate(rows):
        top = sorted(r.items(), key=lambda kv: -kv[1])[:3]
        print(f"mode {i+1:2d}  f = {np.sqrt(lam[i])/(2*np.pi):7.2f} Hz   "
              + ", ".join(f"{k} {v:.0%}" for k, v in top))

    # render: hand upright, palm side three-quarter; colour = |phi|
    surf0 = tet_grid(X, T).extract_surface(algorithm="dataset_surface")
    ext = np.ptp(X, 0).max()
    view = ((0.9, -1.4, 0.45), (0, 0, 1))
    imgs = []
    for i in range(N_MODES):
        u = B[:, i].reshape(-1, 3)
        mag = np.linalg.norm(u, axis=1)
        s = 0.12 * ext / mag.max()
        g = tet_grid(X + s * u, T)
        g.point_data["phi"] = mag / mag.max()
        surf = g.extract_surface(algorithm="dataset_surface")

        def add(pl, surf=surf):
            pl.add_mesh(surf0, color="#d9d9d4", opacity=0.18)
            pl.add_mesh(surf, scalars="phi", cmap="magma", clim=(0, 1), show_scalar_bar=False,
                        smooth_shading=False)
        imgs.append(shot(add, view, size=(700, 800), zoom=1.25, bounds_mesh=surf0))
    ncol = 5
    fig, axes = plt.subplots(N_MODES // ncol, ncol, figsize=(3.6 * ncol, 4.3 * (N_MODES // ncol)))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(imgs[i])
        ax.axis("off")
        top = sorted(rows[i].items(), key=lambda kv: -kv[1])[:2]
        ax.set_title(f"mode {i+1}: {np.sqrt(lam[i])/(2*np.pi):.1f} Hz\n"
                     + ", ".join(f"{k} {v:.0%}" for k, v in top), fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, label="|displacement| (normalised)")
    fig.suptitle("Eigenfunctions of the compliant hand: A = H_elastic + H_tendons, wrist pinned "
                 "(A phi = lambda M phi; grey = rest shape, deformation exaggerated)", fontsize=13)
    path = os.path.join(REN, "10_eigenfunctions.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("wrote", os.path.relpath(path))


def sp_diag(v):
    import scipy.sparse as sp
    return sp.diags(v).tocsc()


if __name__ == "__main__":
    main()
