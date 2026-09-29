"""Eigenfunctions of the compliant hand's operator.

``A = H_elastic + H_tendons`` at the open pose (``a = 0``, tendons at rest
length), restricted to the free DOFs (wrist base pinned), generalized with the
lumped mass matrix: ``A phi = lambda M phi``. Each mode's energy
``phi^T A phi`` is split into stiff blocks, flexure joints, pads and tendons.

    python sdm_modes.py            (after sdm_geometry.py, sdm_tets.py)
    python sdm_modes.py --until-soft   # keep going until soft deformation takes over
    python sdm_modes.py --fingertips   # the fingertip-pad modes, one row per fingertip

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
SOFT_SWITCH = 0.5      # a mode is "soft" when >= 50% of its kinetic energy is in soft material


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
    parts["tendons"] = hand.tendon_hessian(X.reshape(-1), 0.0)
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


def until_soft(want=6, k0=40, k_max=600):
    """Keep computing modes until soft deformation takes over.

    Energy share cannot separate articulation (blocks move rigidly, flexures
    bend: ~98% flexure energy) from genuinely soft deformation, so the test is
    kinematic: the share of a mode's kinetic energy (M-weighted |phi|^2) that
    lives in soft material (flexures + pads). A mode is *soft* when that share
    is >= SOFT_SWITCH. Modes are computed in growing batches until ``want``
    soft modes have appeared; the pads' share of the energy is tracked too.
    """
    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = Hand(scene)
    X, T = hand.X, hand.T
    fr = hand.free
    A = hand.hessian(X.reshape(-1), 0.0)[fr][:, fr].tocsc()
    M = sp_diag(hand.m[fr])
    kind_t = np.array(hand.kinds)[hand.part]
    # vertex -> soft if every incident tet is soft (flexure/pad)
    stiff_v = np.zeros(hand.n, bool)
    stiff_v[T[np.isin(kind_t, ["palm", "link"])].ravel()] = True
    soft_dof = np.repeat(~stiff_v, 3)
    sel = kind_t == "pad"
    J = simkit.deformation_jacobian(X, T[sel]).tocsc()
    J.resize((J.shape[0], 3 * hand.n))
    H_pad = energies.stable_neo_hookean_hessian_x(X, J, hand.mu[sel], hand.lam[sel], hand.vol[sel], psd=True)
    k = k0
    while True:
        lam, Bf = simkit.eigs(A, k=k, M=M)
        o = np.argsort(lam)
        lam, Bf = lam[o], Bf[:, o]
        B = np.zeros((3 * hand.n, k))
        B[fr] = Bf
        kin = hand.m[:, None] * B ** 2
        soft_kin = kin[soft_dof].sum(0) / kin.sum(0)
        pad_E = np.einsum("ij,ij->j", B, H_pad @ B) / lam / np.einsum("ij,ij->j", B, hand.m[:, None] * B)
        soft = np.where(soft_kin >= SOFT_SWITCH)[0]
        print(f"k={k}: soft modes so far {list(soft + 1)[:10]}", flush=True)
        if len(soft) >= want or k >= k_max:
            break
        k = min(2 * k, k_max)
    np.savez_compressed(os.path.join(OUT, "sdm_modes_long.npz"), eigenvalues=lam, B=B,
                        soft_kin=soft_kin, pad_E=pad_E)
    render_until_soft(X, T, B, lam, soft_kin, pad_E)


def takeover_index(soft_kin, window=10, need=8):
    """First mode from which soft modes dominate: >= need of the next window are soft
    (isolated local flexure modes earlier in the spectrum do not count)."""
    soft = soft_kin >= SOFT_SWITCH
    for i in range(len(soft) - window + 1):
        if soft[i] and soft[i:i + window].sum() >= need:
            return i
    return int(np.argmax(soft)) if soft.any() else len(soft) - 1


def render_until_soft(X, T, B, lam, soft_kin, pad_E):
    k = len(lam)
    first = takeover_index(soft_kin)
    spikes = [i for i in np.where(soft_kin >= SOFT_SWITCH)[0] if i < first]
    f_hz = lambda i: np.sqrt(lam[i]) / (2 * np.pi)
    print(f"soft deformation takes over at mode {first + 1} ({f_hz(first):.0f} Hz); "
          f"isolated local soft modes before that: {[i + 1 for i in spikes]}")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    idx = np.arange(1, k + 1)
    ax.plot(idx, soft_kin, "-", color="#2a78d6", lw=2, label="kinetic energy in soft material")
    ax.plot(idx, pad_E, "-", color="#eb6834", lw=2, label="elastic energy in pads")
    ax.axhline(SOFT_SWITCH, color="#8c8c86", lw=1, ls="--")
    ax.axvline(first + 1, color="#3d3d3a", lw=1)
    ax.annotate(f"soft takes over: mode {first + 1} ({f_hz(first):.0f} Hz)", (first + 1, 0.5),
                xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=10,
                bbox=dict(fc="white", ec="none"))
    ax.set_xlabel("mode number")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, k)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.grid(True, color="#e5e4df", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Articulation (stiff blocks moving on flexures) vs soft deformation, by mode",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(REN, "11_soft_takeover.png"), dpi=130)
    plt.close(fig)

    # sample of the articulation regime, the isolated local modes, then the takeover
    before = list(np.unique(np.linspace(0, first - 1, 7).astype(int)))
    show = before + spikes[:3] + list(range(first, min(first + 10, k)))
    show = list(dict.fromkeys(show))
    surf0 = tet_grid(X, T).extract_surface(algorithm="dataset_surface")
    ext = np.ptp(X, 0).max()
    view = ((0.9, -1.4, 0.45), (0, 0, 1))
    imgs = []
    for i in show:
        u = B[:, i].reshape(-1, 3)
        mag = np.linalg.norm(u, axis=1)
        soft_i = soft_kin[i] >= SOFT_SWITCH
        g = tet_grid(X + ((0.03 if soft_i else 0.12) * ext / mag.max()) * u, T)
        g.point_data["phi"] = mag / mag.max()
        surf = g.extract_surface(algorithm="dataset_surface")
        focus, bounds = None, surf0
        if soft_i:                       # close-up on where the mode lives
            hot = X[mag >= 0.3 * mag.max()]
            focus = hot.mean(0)
            r = max(np.ptp(hot, 0).max(), 0.03)
            bounds = pv.Box(bounds=(focus[0] - r, focus[0] + r, focus[1] - r, focus[1] + r,
                                    focus[2] - r, focus[2] + r))

        def add(pl, surf=surf):
            pl.add_mesh(surf0, color="#d9d9d4", opacity=0.18)
            pl.add_mesh(surf, scalars="phi", cmap="magma", clim=(0, 1), show_scalar_bar=False,
                        smooth_shading=False)
        imgs.append(shot(add, view, size=(700, 800), zoom=1.25 if not soft_i else 1.1,
                         focus=focus, bounds_mesh=bounds))
    ncol = 5
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 4.3 * nrow))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i, img in zip(axes.ravel(), show, imgs):
        ax.imshow(img)
        soft_i = soft_kin[i] >= SOFT_SWITCH
        tag = ("  SOFT (close-up)" if i >= first else "  local (close-up)") if soft_i else ""
        ax.set_title(f"mode {i+1}: {f_hz(i):.0f} Hz{tag}\n"
                     f"soft kinetic {soft_kin[i]:.0%}, pad energy {pad_E[i]:.0%}",
                     fontsize=10, color="#b2182b" if soft_i else "black")
    fig.suptitle(f"Compliant-hand modes: articulation up to mode {first}, soft deformation from "
                 f"mode {first + 1} ({f_hz(first):.0f} Hz). Grey = rest shape; deformation exaggerated",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(REN, "12_modes_until_soft.png"), dpi=110)
    print("wrote renders 11_soft_takeover.png, 12_modes_until_soft.png")


def fingertip_modes(per_tip=3, share=0.5):
    """For each fingertip pad, the first modes with > ``share`` of their elastic
    energy in that pad, rendered as close-ups (from sdm_modes_long.npz)."""
    scene = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
    hand = Hand(scene)
    X, T = hand.X, hand.T
    d = np.load(os.path.join(OUT, "sdm_modes_long.npz"))
    B, lam = d["B"], d["eigenvalues"]
    tot = np.einsum("ij,ij->j", B, hand.hessian(X.reshape(-1), 0.0) @ B)
    tips = ["index", "middle", "ring", "little", "thumb"]
    picks, first = {}, {}
    for f in tips:
        i = hand.names.index(f"{f}_pad")
        sel = hand.part == i
        J = simkit.deformation_jacobian(X, T[sel]).tocsc()
        J.resize((J.shape[0], 3 * hand.n))
        H = energies.stable_neo_hookean_hessian_x(X, J, hand.mu[sel], hand.lam[sel], hand.vol[sel], psd=True)
        sh = np.einsum("ij,ij->j", B, H @ B) / tot
        m = np.where(sh > share)[0][:per_tip]
        picks[f] = [(int(k), float(sh[k])) for k in m]
        first[f] = int(m[0]) if len(m) else None
        pad_c = X[np.unique(T[sel])].mean(0)
        picks[f] = [(k, s_, pad_c) for k, s_ in picks[f]]
    fh = lambda k: np.sqrt(lam[k]) / (2 * np.pi)
    for f in tips:
        print(f"{f:7s} pad: modes {[k + 1 for k, _, _ in picks[f]]}  "
              f"(first at {fh(first[f]):.0f} Hz)" if first[f] is not None else f"{f}: none")

    surf0 = tet_grid(X, T).extract_surface(algorithm="dataset_surface")
    view = ((0.55, -1.4, 0.55), (0, 0, 1))          # palm side, looking at the pads
    fig, axes = plt.subplots(len(tips), per_tip, figsize=(4.0 * per_tip, 4.1 * len(tips)))
    for r, f in enumerate(tips):
        for c in range(per_tip):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks[f]):
                continue
            k, s_, pc = picks[f][c]
            u = B[:, k].reshape(-1, 3)
            mag = np.linalg.norm(u, axis=1)
            g = tet_grid(X + (0.004 / mag.max()) * u, T)
            g.point_data["phi"] = mag / mag.max()
            surf = g.extract_surface(algorithm="dataset_surface")
            rr = 0.036
            pc = pc + np.array([0.0, 0.0, 0.006])
            box = pv.Box(bounds=(pc[0] - rr, pc[0] + rr, pc[1] - rr, pc[1] + rr, pc[2] - rr, pc[2] + rr))

            def add(pl, surf=surf):
                pl.add_mesh(surf0, color="#d9d9d4", opacity=0.15)
                pl.add_mesh(surf, scalars="phi", cmap="magma", clim=(0, 1), show_scalar_bar=False,
                            smooth_shading=False)
            ax.imshow(shot(add, view, size=(700, 700), zoom=1.0, focus=pc, bounds_mesh=box))
            ax.set_title(f"{f} fingertip: mode {k + 1}, {fh(k):.0f} Hz\n"
                         f"{s_:.0%} of the energy in the {f} pad", fontsize=10)
    fig.suptitle("Fingertip pad modes of the compliant hand (close-ups from the palm side; "
                 "deformation exaggerated to 4 mm, grey = rest shape)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(REN, "13_fingertip_pad_modes.png"), dpi=110)
    print("wrote renders/13_fingertip_pad_modes.png")


if __name__ == "__main__":
    import sys
    if "--until-soft" in sys.argv and "--render-only" in sys.argv:
        d = np.load(os.path.join(OUT, "sdm_modes_long.npz"))
        sc = np.load(os.path.join(OUT, "sdm_tets.npz"))
        render_until_soft(sc["X"].astype(float), sc["T"].astype(np.int64), d["B"],
                          d["eigenvalues"], d["soft_kin"], d["pad_E"])
    elif "--fingertips" in sys.argv:
        fingertip_modes()
    elif "--until-soft" in sys.argv:
        until_soft()
    else:
        main()
