"""Hyper-reduced simulation of the compliant hand in mesh4PDE's coarse subspace.

The unknowns are the coarse vertex positions ``x`` (3 per coarse vertex). The fine
hand is never simulated; its vertices follow the coarse ones through mesh4PDE's
prolongation, ``x_fine = B x`` with ``B = kron(P, I_3)``
(``lib.prolongation_to_subspace``). Every term of the energy is evaluated at the
coarse size:

* elastic   -- stable Neo-Hookean integrated on the COARSE mesh (its tets are the
               cubature), with each coarse tet's material from winding numbers;
* tendons   -- the fine hand's 15 embedded tendon springs, ``d = (G_t B) x``;
* hinges    -- the fine hand's 30 hinge pins, ``(G_p B) x`` vs their rest value;
* wrist     -- the fine base vertices, ``(S B) x = X_base``, as a quadratic penalty;
* inertia   -- ``M_r = B^T M B`` (dynamics only).

``build_hand_system(level)`` returns ``system(x, a) -> (E, g, H)``; the actuation
``a`` only changes the tendons' rest lengths ``l0 = (1 - c a) l_rest``, so sweeping
``a`` closes the hand. ``newton(system, x0, a)`` minimises it (projected Newton,
backtracking line search, stops on the Newton decrement). ``level=None`` builds
the same system with ``P = I`` on the fine mesh: the full-space reference.

    python sdm_reduced.py [--levels 300 600 1200 2500 5000] [--fine]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy as sp

import simkit
import simkit.energies as energies
from simkit.backtracking_line_search import backtracking_line_search

sys.path.insert(0, os.environ.get("MESH4PDE", os.path.expanduser("~/mesh4pde")))
if not os.path.isdir(sys.path[0]):
    sys.path[0] = "/home/user/mesh4pde"
from lib import prolongation_to_subspace  # noqa: E402  (mesh4PDE)

from sdm_sim import Hand, OUT, K_PIN, solve_spd, smoothstep  # noqa: E402

CO = os.path.join(OUT, "coarse")
K_BASE = 1e8          # N/m, wrist-base penalty per fine base vertex
_FINE = {}


def fine_hand():
    """The fine hand: the source of the tendons, hinge pins, base and mass."""
    if "hand" not in _FINE:
        _FINE["scene"] = dict(np.load(os.path.join(OUT, "sdm_tets.npz")))
        _FINE["hand"] = Hand(_FINE["scene"])
    return _FINE["hand"]


def load_level(level):
    d = dict(np.load(os.path.join(CO, f"sdm_tets_{level}.npz")))
    P = sp.sparse.csc_matrix((d["P_data"], d["P_indices"], d["P_indptr"]), shape=tuple(d["P_shape"]))
    return d, P


def build_hand_system(level=None, k_pin=None):
    """Return ``system(x, a) -> (E, g, H)`` plus what the solver and renderer need.
    ``k_pin`` overrides the hinge-pin stiffness (0 drops the pins: in a coarse
    subspace that cannot reproduce an exact hinge rotation they act as a locking
    constraint)."""
    K_PIN_ = K_PIN if k_pin is None else k_pin
    hf = fine_hand()
    if level is None:
        sc, P = _FINE["scene"], sp.sparse.identity(hf.n, format="csc")
    else:
        sc, P = load_level(level)
    Xc = np.asarray(sc["X"], float)
    Tc = np.asarray(sc["T"], np.int64)
    B = prolongation_to_subspace(P=P, dof=3)                      # kron(P, I3)

    # elastic, integrated on the coarse mesh
    E, nu = np.asarray(sc["E"], float), np.asarray(sc["nu"], float)
    mu = (E / (2 * (1 + nu))).reshape(-1, 1)
    lam = (E * nu / ((1 + nu) * (1 - 2 * nu))).reshape(-1, 1)
    J = simkit.deformation_jacobian(Xc, Tc)
    vol = simkit.volume(Xc, Tc)
    x0 = Xc.reshape(-1).copy()
    E_rest = energies.stable_neo_hookean_energy_x(Xc, J, mu, lam, vol)

    # fine-side linear maps, pulled back into the subspace
    Gt = (hf.Gt @ B).tocsr()                                      # tendons
    Gp = (hf.G @ B).tocsr()                                       # hinge pins
    gp0 = hf.g0
    base = hf.pinned
    S = sp.sparse.csr_matrix((np.ones(3 * len(base)), (np.arange(3 * len(base)),
                              (3 * base[:, None] + np.arange(3)).ravel())), shape=(3 * len(base), 3 * hf.n))
    Sb = (S @ B).tocsr()
    xb0 = hf.X[base].reshape(-1)
    H_quad = (K_PIN_ * (Gp.T @ Gp) + K_BASE * (Sb.T @ Sb)).tocsr()  # constant Hessian part
    M_r = (B.T @ sp.sparse.diags(hf.m) @ B).tocsc()
    k_t = hf.ym.ravel()
    l_rest, c = hf.l_rest, hf.c

    def tendons(x, a, energy_only=False):
        d = (Gt @ x).reshape(-1, 3)
        l = np.linalg.norm(d, axis=1)
        l0 = (1 - c * a) * l_rest
        e = 0.5 * float((k_t * (l - l0) ** 2).sum())
        if energy_only:
            return e, None, None
        u = d / l[:, None]
        g = Gt.T @ ((k_t * (l - l0))[:, None] * u).ravel()
        r = l0 / l
        blk = (k_t * np.maximum(1 - r, 0))[:, None, None] * np.eye(3) + \
            (k_t * r)[:, None, None] * u[:, :, None] * u[:, None, :]
        H = Gt.T @ sp.sparse.block_diag(list(blk), format="csr") @ Gt
        return e, g, H

    def system(x, a, energy_only=False):
        """Energy, gradient, Hessian of the hand at coarse state x and actuation a
        (``energy_only``: just E, for the line search)."""
        Xm = x.reshape(-1, 3)
        e = energies.stable_neo_hookean_energy_x(Xm, J, mu, lam, vol) - E_rest
        if energy_only:
            rp = Gp @ x - gp0
            rb = Sb @ x - xb0
            return float(e + tendons(x, a, True)[0] + 0.5 * K_PIN_ * (rp @ rp) + 0.5 * K_BASE * (rb @ rb)), None, None
        g = energies.stable_neo_hookean_gradient_x(Xm, J, mu, lam, vol).ravel()
        H = energies.stable_neo_hookean_hessian_x(Xm, J, mu, lam, vol, psd=True)
        et, gt, Ht = tendons(x, a)
        rp = Gp @ x - gp0
        rb = Sb @ x - xb0
        e += et + 0.5 * K_PIN_ * (rp @ rp) + 0.5 * K_BASE * (rb @ rb)
        g = g + gt + K_PIN_ * (Gp.T @ rp) + K_BASE * (Sb.T @ rb)
        H = (H + Ht + H_quad).tocsr()
        return float(e), g, H

    tips = [hf.tip_vertices(f) for f in hf.fingers]
    return dict(system=system, x0=x0, M=M_r, P=P, B=B, X=Xc, T=Tc, scene=sc, level=level,
                n_dof=len(x0), n_tets=len(Tc), tips=tips, fingers=hf.fingers,
                Gt=Gt, Gp=Gp, Sb=Sb)


def newton(system, x0, a, x_tilde=None, M=None, h=None, tol=1e-9, max_iters=60):
    """Projected Newton on system(., a) [+ inertia 1/(2h^2) |x - x_tilde|_M^2];
    stops when half the Newton decrement, -g.dx / 2, drops below ``tol`` (J)."""
    inert = x_tilde is not None
    Mh = M / h ** 2 if inert else None

    def total(x, need_grad=True):
        e, g, H = system(x, a, energy_only=not need_grad)
        if inert:
            d = x - x_tilde
            e += 0.5 * d @ (Mh @ d)
            if need_grad:
                g = g + Mh @ d
                H = H + Mh
        return e, g, H

    x = x0.copy()
    it = 0
    for it in range(1, max_iters + 1):
        e, g, H = total(x)
        dx = solve_spd(H.tocsc(), -g)
        dec = -g @ dx
        if 0.5 * dec < tol:
            break
        alpha, x, _ = backtracking_line_search(lambda y: total(y, False)[0], x, g, dx)
        if alpha == 0:
            break
    return x, it


def simulate(sysd, n_static=12, h=1 / 60, t_ramp=1.2, t_end=2.0, log=print, dynamics=True):
    system, x0, M = sysd["system"], sysd["x0"], sysd["M"]
    P = sysd["P"]
    X_f = fine_hand().X

    def tips(x):
        xf = P @ x.reshape(-1, 3)
        return np.array([np.linalg.norm(xf[t].mean(0) - X_f[t].mean(0)) for t in sysd["tips"]])

    t0 = time.time()
    a_vals = np.linspace(0, 1, n_static + 1)
    xs, its = [], []
    x = x0.copy()
    for a in a_vals:
        start = 2 * xs[-1] - xs[-2] if len(xs) > 1 else x
        x, it = newton(system, start, a)
        xs.append(x.copy())
        its.append(it)
    t_static = time.time() - t0
    log(f"  statics: {len(a_vals)} steps, {sum(its)} Newton its, {t_static:.2f} s; tips at a=1 (mm) "
        + " ".join(f"{f} {v*1e3:.1f}" for f, v in zip(sysd["fingers"], tips(xs[-1]))))

    if not dynamics:
        return dict(static_a=a_vals, static_x=np.array(xs), static_its=np.array(its), t_static=t_static,
                    static_tip=np.array([tips(xx) for xx in xs]))
    t0 = time.time()
    n_steps = int(round(t_end / h))
    x, v = x0.copy(), np.zeros_like(x0)
    frames, ts, a_t, dits = [x.copy()], [0.0], [0.0], []
    for k in range(1, n_steps + 1):
        t = k * h
        a = float(smoothstep(t / t_ramp))
        x_new, it = newton(system, x + h * v, a, x_tilde=x + h * v, M=M, h=h)
        v = (x_new - x) / h
        x = x_new
        frames.append(x.copy())
        ts.append(t)
        a_t.append(a)
        dits.append(it)
    t_dyn = time.time() - t0
    tipd = np.array([tips(f) for f in frames])
    log(f"  dynamics: {n_steps} steps, {sum(dits)} Newton its, {t_dyn:.2f} s; mean tip at t=2 s "
        f"{tipd[-1].mean()*1e3:.1f} mm")
    return dict(static_a=a_vals, static_x=np.array(xs), static_its=np.array(its), t_static=t_static,
                dyn_t=np.array(ts), dyn_a=np.array(a_t), dyn_x=np.array(frames), dyn_its=np.array(dits),
                t_dynamic=t_dyn, dyn_tip=tipd, static_tip=np.array([tips(xx) for xx in xs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="*")
    ap.add_argument("--fine", action="store_true", help="also run the full-space reference (P = I)")
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--n-static", type=int, default=12)
    ap.add_argument("--k-pin", type=float, default=None, help="hinge-pin stiffness in the subspace")
    ap.add_argument("--names", nargs="*", help="level file tags instead of --levels, e.g. 1200_q30")
    args = ap.parse_args()
    summ = json.load(open(os.path.join(CO, "coarse_summary.json")))
    levels = args.names if args.names else (args.levels if args.levels is not None
                                            else [r["target"] for r in summ["levels"]])
    pin_tag = "" if args.k_pin is None else f"_kpin{args.k_pin:g}"
    runs = [(lv, f"_{lv}{pin_tag}") for lv in levels] + ([(None, "_fine")] if args.fine else [])
    report = {}
    for lv, tag in runs:
        t0 = time.time()
        sysd = build_hand_system(lv, k_pin=args.k_pin)
        print(f"== {'fine (P = I)' if lv is None else f'level {lv}'}: {sysd['n_dof']} DOFs, "
              f"{sysd['n_tets']} integration tets [build {time.time() - t0:.1f} s]", flush=True)
        res = simulate(sysd, n_static=args.n_static, log=lambda s: print(s, flush=True),
                       dynamics=not args.no_dynamics)
        np.savez_compressed(os.path.join(OUT, f"reduced_sim{tag}.npz"),
                            **{k: (v.astype(np.float32) if k in ("static_x", "dyn_x") else v)
                               for k, v in res.items()})
        report[tag] = dict(level=lv, dofs=int(sysd["n_dof"]), tets=int(sysd["n_tets"]),
                           t_static=res["t_static"], static_newton=int(res["static_its"].sum()),
                           static_tip_mm=(res["static_tip"][-1] * 1e3).round(2).tolist())
        if "dyn_x" in res:
            report[tag].update(t_dynamic=res["t_dynamic"], dyn_newton=int(res["dyn_its"].sum()),
                               dyn_tip_mm=(res["dyn_tip"][-1] * 1e3).round(2).tolist())
        prev = os.path.join(OUT, "reduced_report.json")
        old = json.load(open(prev)) if os.path.exists(prev) else {}
        old.update(report)
        json.dump(old, open(prev, "w"), indent=1)


if __name__ == "__main__":
    main()
