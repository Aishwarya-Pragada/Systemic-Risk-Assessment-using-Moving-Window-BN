#!/usr/bin/env python
"""Supporting information S5: how accurate is the normalised topological order for different K?
For a sample of days, the median-order estimate NT is recomputed `reps` times with K random initial orders
(K = 10, 20, ..., 90, 100, 200, 300, 500, 1000); the spread of those replicates is the RMSE of S5 Eq. (15):
    sqrt( mean_i || NT_i - mean(NT) ||^2 ).
Reads the learned networks; writes results/figures/fig13_k_sensitivity.png and data/processed/k_sensitivity_<profile>.csv"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                         # noqa: E402
from bnsr.indicators.topological import sample_topological_positions         # noqa: E402
from bnsr.networks.runner import out_dir_for                                # noqa: E402
from bnsr.networks.storage import list_days, load_network                   # noqa: E402

KS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 1000]


def nt_vector(pos):
    M = np.median(pos, axis=0)
    w = pos.shape[1] - M
    return w / w.sum()


def rmse_for_K(adj, K, reps, seed):
    nts = np.array([nt_vector(sample_topological_positions(
        adj, K, seed + r)) for r in range(reps)])
    return float(np.sqrt(((nts - nts.mean(axis=0)) ** 2).sum(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--every", type=int, default=50,
                    help="use every n-th day (paper: all days)")
    ap.add_argument("--reps", type=int, default=100,
                    help="replicates per K (paper: 100)")
    a = ap.parse_args()
    cfg = load_config()
    d = out_dir_for(cfg, a.profile)
    days = list_days(d)[:: a.every]
    print(f"{len(days)} days, {a.reps} replicates, K in {KS}")
    sample_topological_positions(
        np.zeros((3, 3), dtype=bool), 2, 0)          # compile
    res = {K: [] for K in KS}
    secs = {K: [] for K in KS}
    for n, t in enumerate(days):
        adj = load_network(d, t)["adj"]
        for K in KS:
            t0 = time.time()
            res[K].append(rmse_for_K(adj, K, a.reps, 1000 * t))
            secs[K].append((time.time() - t0) / a.reps)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(days)}")
    tab = pd.DataFrame({"K": KS, "median_RMSE": [np.median(res[K]) for K in KS],
                        "mean_seconds_per_estimate": [np.mean(secs[K]) for K in KS]})
    tab.to_csv(Path(cfg["paths"]["processed_dir"]) /
               f"k_sensitivity_{a.profile}.csv", index=False)
    print(tab.round(5).to_string(index=False))
    print("\nPaper (S5): RMSE about 0.001-0.0025 at K=100 (typically ~0.0017) and ~0.0013 at K=1000; "
          "K=1000 takes about 11x longer than K=100.")
    out = Path(cfg["paths"]["results_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([res[K] for K in KS], showfliers=False)
    ax.set_xticklabels([str(K) for K in KS])
    ax.set_xlabel("K")
    ax.set_ylabel("Estimated RMSE of the normalised order")
    ax.set_title(
        "Fig 13 (S5). Accuracy of the median topological order versus K")
    ax.grid(alpha=0.3)
    fig.savefig(out / "fig13_k_sensitivity.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'fig13_k_sensitivity.png'}")


if __name__ == "__main__":
    main()
