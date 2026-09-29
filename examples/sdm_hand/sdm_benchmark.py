"""Wall-clock benchmark: full space vs hyper-reduced (1200_palmar_mixw3), run one after
another on an otherwise idle machine. Statics, 25 actuation steps, same Newton solver.

    python sdm_benchmark.py      # -> output/benchmark.json
"""
import json
import os
import time

import numpy as np

import sdm_reduced as R
from sdm_cup import Cup

CUP = dict(center=(-0.015, -0.044, 0.120), R=0.035, yaw=-15.0)
CASES = [("air", None, 0.0), ("cup + friction", "cup", 1e10)]
MODELS = [("reduced (1200 v, 3,600 DOFs)", "1200_palmar_mixw3", 1e5), ("full space (34,008 DOFs)", None, None)]


def main():
    out = []
    R.fine_hand()                                   # load the fine hand once, outside the timings
    for case, obj_kind, fr in CASES:
        for model, level, kp in MODELS:
            obj = Cup(center=CUP["center"], R=CUP["R"], yaw=CUP["yaw"]) if obj_kind else None
            t0 = time.perf_counter()
            sysd = R.build_hand_system(level, k_pin=kp, obj=obj, friction=fr)
            t_build = time.perf_counter() - t0
            res = R.simulate(sysd, n_static=24, dynamics=False, log=lambda s: print(s, flush=True))
            row = dict(case=case, model=model, dofs=int(sysd["n_dof"]), build_s=round(t_build, 2),
                       statics_s=round(float(res["t_static"]), 2), newton_its=int(res["static_its"].sum()),
                       per_newton_ms=round(1e3 * float(res["t_static"]) / int(res["static_its"].sum()), 1),
                       tips_mm=(res["static_tip"][-1] * 1e3).round(1).tolist())
            out.append(row)
            print(row, flush=True)
            json.dump(out, open(os.path.join(R.OUT, "benchmark.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
