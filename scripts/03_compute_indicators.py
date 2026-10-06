#!/usr/bin/env python
"""Stage 3: modified network density, topological orders (Kahn, K=100) and order distance for every day.
Reads  data/interim/networks/<profile>/   Writes  data/processed/indicators_<profile>.csv,
relative_order_<profile>.csv, median_order_<profile>.csv"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                          # noqa: E402
from bnsr.indicators.compute import (acf, compute_indicators, save_indicators,  # noqa: E402
                                     sector_means)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--K", type=int, default=None, help="random initial orders per network (default 100)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    ind, rm, md = compute_indicators(cfg, a.profile, a.K, a.start, a.end)
    out = save_indicators(cfg, a.profile, ind, rm, md)
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    od = ind["OD"].dropna()
    print(f"\nOrder distance: n={len(od)} mean={od.mean():.3f} sd={od.std():.3f} min={od.min():.3f} max={od.max():.3f}")
    print("OD autocorrelation lags 1-5:", [round(x, 3) for x in acf(od, 5)])
    print(f"MND: mean={ind.MND.mean():.3f} min={ind.MND.min():.3f} max={ind.MND.max():.3f}")
    print("\nMean relative topological order by sector (1 = head of order):")
    print(sector_means(rm).round(3).to_string())


if __name__ == "__main__":
    main()
