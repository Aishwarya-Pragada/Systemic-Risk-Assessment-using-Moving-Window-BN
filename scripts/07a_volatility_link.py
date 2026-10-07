#!/usr/bin/env python
"""Supporting information S3 (Figs 11 and 12): the autocorrelation of the order distance and how often the
order-distance coefficient is positive at different volatility levels.
Needs data/processed/hsi.csv and indicators_<profile>.csv.  Writes results/figures/fig11_*.png, fig12_*.png and
data/processed/volatility_link_<profile>.csv."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                   # noqa: E402
from bnsr.evaluation.series import load_series                        # noqa: E402
from bnsr.evaluation.volatility_link import (UPPERS, log_abs, rolling_beta1,   # noqa: E402
                                             share_positive_by_level)
from bnsr.indicators.compute import acf                               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    a = ap.parse_args()
    cfg = load_config()
    w, m = cfg["bayesian_network"]["w"], cfg["lasso"]["m"]
    y, mnd, od, dates, T = load_series(cfg, a.profile)
    ar = y["abs_ret"]
    out = Path(cfg["paths"]["results_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    beta = rolling_beta1(ar, mnd, od, w + m, T, m)
    logy = log_abs(ar[w + m: T + 1])
    counts, share = share_positive_by_level(beta, logy)
    labels = [f"(-inf,{UPPERS[0]:g}]"] + [f"({UPPERS[i-1]:g},{UPPERS[i]:g}]" for i in range(1, len(UPPERS))] \
        + [f"({UPPERS[-1]:g},inf)"]
    tab = pd.DataFrame(
        {"interval": labels, "days": counts, "share_positive": share})
    tab.to_csv(Path(cfg["paths"]["processed_dir"]) /
               f"volatility_link_{a.profile}.csv", index=False)
    print(f"{int((~np.isnan(beta)).sum())} rolling regressions; overall share of positive b_t: "
          f"{np.nanmean(beta > 0):.3f}")
    print(tab.round(3).to_string(index=False))
    # skip the sparse end intervals
    mid = slice(1, len(share) - 1)
    rho = pd.Series(np.arange(len(share))[mid]).corr(
        pd.Series(share[mid]), method="spearman")
    print(f"\nSpearman correlation between volatility level and share positive (interior intervals): {rho:+.2f}"
          "   (paper: share rises with volatility, about 0.41 -> 0.55)")

    # ---- Fig 12 ----
    x = np.append(UPPERS, UPPERS[-1] + 0.4)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax2 = ax.twinx()
    ax2.bar(x, counts, width=0.3, color="0.35")
    ax2.set_ylabel("Number of trading days in the interval")
    ax2.set_ylim(0, counts.max() * 3)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(x, share, "ko-", ms=3)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Log absolute return")
    ax.set_ylabel("Share of b_t greater than 0")
    ax.set_title(
        "Fig 12 (S3). Share of positive order-distance coefficients by volatility level")
    ax.grid(alpha=0.3)
    fig.savefig(out / "fig12_volatility_link.png",
                dpi=150, bbox_inches="tight")

    # ---- Fig 11 ----
    odv = od[~np.isnan(od)]
    r = acf(odv, 35)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.vlines(range(0, 36), 0, [1.0] + r, color="black", lw=1)
    ci = 1.96 / np.sqrt(len(odv))
    ax.axhline(ci, ls="--", color="royalblue", lw=0.8)
    ax.axhline(-ci, ls="--", color="royalblue", lw=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title("Fig 11a (S3). Autocorrelation of the order distance")
    fig.savefig(out / "fig11_od_acf.png", dpi=150, bbox_inches="tight")
    print(f"\nOD autocorrelation lag 1/5/10/20/35: {[round(r[k-1], 2) for k in (1, 5, 10, 20, 35)]} "
          "(paper, Fig 11a: about 0.35 at lag 1, slowly decaying)")
    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()
