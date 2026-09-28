"""Eigenmodes of the spring-actuated elastic hand, until the fingertip modes appear.

Operator: ``A = H_elastic + H_springs + Q_pin`` (``hand_springs.hand_operator``),
generalized problem ``A phi = lambda M phi``. The palm is pinned and the springs
tie every link to it, so there is no zero mode: mode 1 is already the first
non-zero eigenfunction.

For each mode the energy ``phi^T A phi = lambda`` is split into actuator springs,
hinge springs and the elastic energy of each material. A mode is a *fingertip
mode* when at least 30% of its total energy is elastic energy of the fingertip
rubber (a *pad mode* likewise for the silicone pads).
Modes are computed in growing batches until at least ``--want`` fingertip modes
have appeared.

    python eigen_modes.py [--want 4] [--start 32]
    python eigen_modes.py --render-only       # re-render the saved modes

Writes ``output/hand_modes.npz`` and ``output/renders/hand_modes.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simkit
import simkit.energies as energies

from hand_springs import HandSprings, hand_operator
from materials import part_material
from render_hand import WORLD, tet_grid, shot

pv.OFF_SCREEN = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
SHORT = {"silicone_shore_20A": "pads", "polyurethane_40A": "tip rubber",
         "aluminium_6061_T6": "Al6061", "aluminium_7075_T6": "Al7075",
         "steel_AISI_4140": "steel"}


def material_hessians(X, T, part, names, aux):
    mats = np.array([part_material(names[p]) for p in part])
    out = {}
    for m in np.unique(mats):
        sel = mats == m
        J = simkit.deformation_jacobian(X, T[sel]).tocsc()
        J.resize((J.shape[0], 3 * len(X)))
        out[m] = energies.macklin_mueller_neo_hookean_hessian_x(
            X, J, aux["mu"][sel].reshape(-1, 1), aux["lam"][sel].reshape(-1, 1),
            aux["vol"][sel], psd=False)
    return out


def classify(B, lam, Hm, H_hinge, H_act, Q):
    rows = []
    for i in range(B.shape[1]):
        f = B[:, i]
        e = {SHORT[m]: float(f @ (H @ f)) for m, H in Hm.items()}
        e["hinge springs"] = float(f @ (H_hinge @ f))
        e["actuators"] = float(f @ (H_act @ f))
        e["pin"] = float(f @ (Q @ f))
        tot = sum(e.values())
        el = sum(e[SHORT[m]] for m in Hm)
        rows.append(dict(lam=float(lam[i]), freq=float(np.sqrt(max(lam[i], 0)) / (2 * np.pi)),
                         share={k: v / tot for k, v in e.items()},
                         tip_share=e["tip rubber"] / tot, pad_share=e["pads"] / tot,
                         elastic_share=el / tot))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=4, help="stop after this many fingertip modes")
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--render-only", action="store_true",
                    help="re-render from output/hand_modes.npz without recomputing")
    args = ap.parse_args()
    if args.render_only:
        return render_only(args.want)

    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    meta = json.load(open(os.path.join(OUT, "hand_meta.json")))
    names = [str(n) for n in scene["part_names"]]
    springs = HandSprings(scene, meta)
    t0 = time.time()
    A, M, X, T, part, aux = hand_operator(scene, meta, springs)
    Hm = material_hessians(X, T, part, names, aux)
    keep = np.unique(scene["T"][scene["part"] != names.index("cup")])
    remap = -np.ones(len(scene["X"]), np.int64)
    remap[keep] = np.arange(len(keep))
    E = remap[springs.E]
    nh = len(springs.E_hinge)

    def spring_H(sel):
        return energies.mass_springs_hessian_x(X, E[sel], springs.ym[sel], springs.vol[sel],
                                               springs.l0_rest[sel].reshape(-1, 1), psd=True)
    H_hinge, H_act = spring_H(np.arange(nh)), spring_H(np.arange(nh, len(E)))
    print(f"operator: {A.shape[0]} DOFs [{time.time()-t0:.0f}s]", flush=True)

    k = args.start
    while True:
        t0 = time.time()
        lam, B = simkit.eigs(A, k=k, M=M)
        order = np.argsort(lam)
        lam, B = lam[order], B[:, order]
        rows = classify(B, lam, Hm, H_hinge, H_act, aux["Q"])
        tips = [i for i, r in enumerate(rows) if r["tip_share"] >= 0.3]
        print(f"k={k}: {len(tips)} fingertip modes {[i + 1 for i in tips]} [{time.time()-t0:.0f}s]",
              flush=True)
        if len(tips) >= args.want or k >= args.max:
            break
        k = min(2 * k, args.max)

    print(f"{'mode':>4} {'lambda':>10} {'f [Hz]':>8}  dominant energy")
    for i, r in enumerate(rows):
        top = sorted(r["share"].items(), key=lambda kv: -kv[1])[:3]
        tag = "  <-- FINGERTIP" if i in tips else ""
        print(f"{i+1:4d} {r['lam']:10.3e} {r['freq']:8.1f}  "
              + ", ".join(f"{n} {v:.0%}" for n, v in top) + tag)
    np.savez_compressed(os.path.join(OUT, "hand_modes.npz"), eigenvalues=lam, B=B,
                        keep=keep, tips=np.array(tips))
    json.dump(dict(rows=rows, tips=tips), open(os.path.join(OUT, "hand_modes.json"), "w"), indent=1)

    render_modes(X, T, B, rows, tips, args.want)


def render_modes(X, T, B, rows, tips, want):
    """Modes 1..first fingertip mode (whole hand), then each fingertip mode as a
    whole-hand view plus a close-up of the fingertip it lives in."""
    last = tips[0] if tips else len(rows) - 1
    show = list(range(min(last, 15)))
    Xw = X @ WORLD.T
    c = Xw.mean(0)
    cam = (c + [0.30, 0.30, -0.50], c + [0.0, 0.03, 0.0])
    panels = [(i, cam, False) for i in show]
    for i in tips[:want]:
        u = np.linalg.norm(B[:, i].reshape(-1, 3), axis=1)
        focus = Xw[u >= 0.5 * u.max()].mean(0)
        panels += [(i, cam, False), (i, (focus + [0.05, 0.05, -0.07], focus), True)]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (i, cm, zoom) in zip(axes.ravel(), panels):
        u = (B[:, i].reshape(-1, 3)) @ WORLD.T
        mag = np.linalg.norm(u, axis=1)
        u *= (0.004 if zoom else 0.012) / mag.max()   # drawn max displacement
        g = tet_grid(Xw + u, T)
        g.point_data["|phi|"] = mag / mag.max()
        surf = g.extract_surface(algorithm="dataset_surface")
        ax.imshow(shot(lambda pl: pl.add_mesh(surf, scalars="|phi|", cmap="magma",
                                              clim=(0, 1), show_scalar_bar=False), cm,
                       size=(760, 700)))
        r = rows[i]
        top = sorted(r["share"].items(), key=lambda kv: -kv[1])[:2]
        ax.set_title(f"mode {i+1}{'  FINGERTIP' if i in tips else ''}{' (close-up)' if zoom else ''}\n"
                     f"f = {r['freq']:.0f} Hz; " + ", ".join(f"{n} {v:.0%}" for n, v in top),
                     fontsize=10, color="#b2182b" if i in tips else "black")
    fig.suptitle("Eigenmodes of the spring-actuated elastic Allegro hand: modes 1-"
                 f"{show[-1] + 1 if show else 0}, then the first fingertip modes "
                 "(colour: |displacement|, exaggerated)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(OUT, "renders", "hand_modes.png")
    fig.savefig(path, dpi=110)
    print("wrote", os.path.relpath(path))


def render_only(want):
    scene = dict(np.load(os.path.join(OUT, "scene_tets.npz")))
    d = np.load(os.path.join(OUT, "hand_modes.npz"))
    info = json.load(open(os.path.join(OUT, "hand_modes.json")))
    names = [str(n) for n in scene["part_names"]]
    keep = d["keep"]
    remap = -np.ones(len(scene["X"]), np.int64)
    remap[keep] = np.arange(len(keep))
    T = remap[scene["T"][scene["part"] != names.index("cup")]]
    render_modes(scene["X"][keep], T, d["B"], info["rows"], info["tips"], want)


if __name__ == "__main__":
    main()
