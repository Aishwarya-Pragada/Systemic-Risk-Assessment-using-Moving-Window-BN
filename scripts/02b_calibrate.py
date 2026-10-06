#!/usr/bin/env python
"""Calibration helper: how do the BDeu equivalent sample size (ess) and the candidate count C change the learned
networks?  The paper's Fig. 8 (4th panel) shows modified network density of roughly 0.6-0.8, so we look for
settings whose networks are in that range.  Runs a few days in this process (fast with numba).

  python scripts/02b_calibrate.py
  python scripts/02b_calibrate.py --ess 1 5 20 --C 10 14 --days 30 1000 2000 3403
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                    # noqa: E402
from bnsr.data.dataset import Dataset                                  # noqa: E402
from bnsr.networks.learner import LearnParams, learn_network          # noqa: E402
from bnsr.networks.stats import modified_network_density               # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="light",
                    help="iteration counts are taken from this profile")
    ap.add_argument("--ess", type=float, nargs="+",
                    default=[0.5, 1, 2, 5, 10, 20, 50])
    ap.add_argument("--C", type=int, nargs="+", default=[10, 14])
    ap.add_argument("--days", type=int, nargs="+",
                    default=[30, 1000, 2000, 3403])
    ap.add_argument("--iters", type=int, default=None,
                    help="override all iteration counts (testing only)")
    a = ap.parse_args()

    cfg = load_config()
    ds = Dataset.load(cfg)
    w, M = cfg["bayesian_network"]["w"], cfg["bayesian_network"]["M"]
    base = LearnParams.from_config(cfg, a.profile)
    if a.iters:
        base = replace(base, order_iter=a.iters,
                       order_burn=a.iters // 5, struct_iter=a.iters)
    data = {t: ds.window(t, w).to_numpy(dtype="int64") for t in a.days}

    print(f"profile={a.profile} M={M} days={a.days}")
    print(f"{'ess':>6} {'C':>3} {'mean edges':>11} {'mean MND':>9} {'avg parents':>12} "
          f"{'nodes at C cap':>15} {'s/day':>7}")
    for C in a.C:
        for ess in a.ess:
            p = replace(base, C=C, ess=ess)
            e, d, par, cap, secs = [], [], [], [], []
            for t in a.days:
                t0 = time.time()
                net = learn_network(
                    data[t], p, seed=cfg["learning"]["seed"] + t)
                secs.append(time.time() - t0)
                v = net.info["v"]
                indeg = net.adj.sum(axis=0)
                e.append(net.info["edges"])
                d.append(modified_network_density(net.info["edges"], v, M))
                par.append(indeg.mean())
                cap.append(float((indeg >= net.info["C"]).mean()))
            print(f"{ess:6g} {C:3d} {np.mean(e):11.1f} {np.mean(d):9.3f} {np.mean(par):12.2f} "
                  f"{np.mean(cap):15.2%} {np.mean(secs):7.2f}")


if __name__ == "__main__":
    main()
