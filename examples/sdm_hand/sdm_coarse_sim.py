"""Run the full statics + dynamics (``sdm_sim.run``) on every mesh4PDE level.

    python sdm_coarse_sim.py [--targets 300 600 ...]   (after sdm_coarsen.py)

Writes ``output/sdm_sim_<n>.npz`` / ``sim_report_<n>.json`` per level.
"""
from __future__ import annotations

import argparse
import json
import os

from sdm_sim import run, OUT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="*")
    args = ap.parse_args()
    s = json.load(open(os.path.join(OUT, "coarse", "coarse_summary.json")))
    for t in args.targets or [r["target"] for r in s["levels"]]:
        print(f"===== mesh4PDE level {t} =====", flush=True)
        run(scene_file=os.path.join("coarse", f"sdm_tets_{t}.npz"), tag=f"_{t}")
