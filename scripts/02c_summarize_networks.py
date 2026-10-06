#!/usr/bin/env python
"""Summarise the learned networks: edges, modified network density (Eq. 4) per day, and yearly means.
Writes data/processed/network_summary_<profile>.csv  (columns: t, date, v, edges, MND, max_parents)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                    # noqa: E402
from bnsr.networks.runner import out_dir_for                           # noqa: E402
from bnsr.networks.stats import modified_network_density               # noqa: E402
from bnsr.networks.storage import list_days, load_network              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    a = ap.parse_args()
    cfg = load_config()
    M, w, T = cfg["bayesian_network"]["M"], cfg["bayesian_network"]["w"], cfg["data"]["expected_returns"]
    d = out_dir_for(cfg, a.profile)
    days = list_days(d)
    expected = set(range(w, T + 1))
    print(f"{len(days)} networks in {d}; missing: {len(expected - set(days))}")
    rows = []
    for t in days:
        z = load_network(d, t)
        v, e = len(z["nodes"]), int(z["adj"].sum())
        rows.append((t, z["date"], v, e, modified_network_density(
            e, v, M), int(z["adj"].sum(axis=0).max())))
    df = pd.DataFrame(
        rows, columns=["t", "date", "v", "edges", "MND", "max_parents"])
    out = Path(cfg["paths"]["processed_dir"]) / \
        f"network_summary_{a.profile}.csv"
    df.to_csv(out, index=False)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    print(df.groupby("year").agg(days=("t", "size"), nodes=("v", "mean"), edges=("edges", "mean"),
                                 MND=("MND", "mean")).round(3).to_string())
    print(f"overall: MND mean {df.MND.mean():.3f}, min {df.MND.min():.3f}, max {df.MND.max():.3f}; "
          f"max parents seen {df.max_parents.max()} (limit {M})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
