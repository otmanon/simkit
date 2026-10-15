"""Time the C++ backend against the NumPy implementations.

Builds a tetrahedral grid, then times the deformation Jacobian, the polar
decomposition, PSD projection, and the energy / gradient / Hessian of every
dispatched material under both backends.

    python scripts/benchmark_cpp.py            # default 14^3 grid, 16k tets
    python scripts/benchmark_cpp.py --n 20     # bigger
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import simkit  # noqa: E402
from simkit.energies import elastic as E  # noqa: E402
from simkit.psd_project import psd_project  # noqa: E402
from simkit.polar_svd import polar_svd  # noqa: E402


def grid_tets(n: int):
    """Cube [0,1]^3 split into n^3 cells of 6 tetrahedra each."""
    xs = np.linspace(0.0, 1.0, n + 1)
    X = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    c = idx[:-1, :-1, :-1].reshape(-1)
    dx, dy, dz = (n + 1) ** 2, (n + 1), 1
    v = [c, c + dx, c + dy, c + dx + dy, c + dz, c + dx + dz, c + dy + dz, c + dx + dy + dz]
    tets = [[0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7]]
    T = np.concatenate([np.stack([v[i] for i in tet], axis=1) for tet in tets], axis=0)
    return X, T


def timeit(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14, help="grid resolution per axis")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    X, T = grid_tets(args.n)
    rng = np.random.default_rng(0)
    U = X + 0.005 * rng.standard_normal(X.shape)
    t = T.shape[0]
    mu = np.full((t, 1), 1.0)
    lam = np.full((t, 1), 3.0)
    print(f"mesh: {X.shape[0]} vertices, {t} tets; cpp available: {simkit.cpp_available()}")

    backends = ["python"] + (["cpp"] if simkit.cpp_available() else [])
    rows = []

    def record(name, fns):
        times = {}
        for b in backends:
            simkit.set_backend(b)
            times[b] = timeit(fns, args.repeat)
        rows.append((name, times))

    record("deformation_jacobian", lambda: simkit.deformation_jacobian(X, T))
    simkit.set_backend("python")
    J = simkit.deformation_jacobian(X, T)
    vol = simkit.volume(X, T)
    F = (J @ U.reshape(-1, 1)).reshape(-1, 3, 3)
    record("polar_svd", lambda: polar_svd(F))
    H = rng.standard_normal((t, 9, 9))
    H = H + H.transpose(0, 2, 1)
    record("psd_project (9x9)", lambda: psd_project(H))
    record("block_diag_sandwich", lambda: simkit.block_diag_sandwich(J, H))
    P = rng.standard_normal((t, 3, 3))
    record("assembled_gradient", lambda: simkit.assembled_gradient(J, P))
    record("assembled_hessian", lambda: simkit.assembled_hessian(J, H))
    B, _ = np.linalg.qr(rng.standard_normal((J.shape[1], 60)))
    JB = J @ B
    record("assembled_subspace_gradient (r=60)", lambda: simkit.assembled_subspace_gradient(JB, P))
    record("assembled_subspace_hessian (r=60)", lambda: simkit.assembled_subspace_hessian(JB, H))
    for mat in E._MATERIALS:
        record(f"{mat} energy_x", lambda: E.elastic_energy_x(U, J, mu, lam, vol, mat))
        record(f"{mat} gradient_x", lambda: E.elastic_gradient_x(U, J, mu, lam, vol, mat))
        record(f"{mat} hessian_x", lambda: E.elastic_hessian_x(U, J, mu, lam, vol, mat))

    header = f"{'function':40s}" + "".join(f"{b:>12s}" for b in backends) + ("     speedup" if len(backends) > 1 else "")
    print(header)
    print("-" * len(header))
    for name, times in rows:
        line = f"{name:40s}" + "".join(f"{times[b] * 1e3:10.1f}ms" for b in backends)
        if len(backends) > 1:
            line += f"{times['python'] / times['cpp']:11.1f}x"
        print(line)


if __name__ == "__main__":
    main()
