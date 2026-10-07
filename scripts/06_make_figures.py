#!/usr/bin/env python
"""Stage 6: Figures 6-9 -> results/figures/*.png
Needs the outputs of Stages 1-4 (hsi.csv, indicators_<profile>.csv, relative_order_<profile>.csv,
granger_<profile>.csv) and the learned networks for Fig 6."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                  # noqa: E402
from bnsr.data.universe import sector_map                            # noqa: E402
from bnsr.networks.runner import out_dir_for                         # noqa: E402
from bnsr.networks.storage import load_network, net_path             # noqa: E402
from bnsr.viz import figures as F                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    cfg = load_config()
    p = Path(cfg["paths"]["processed_dir"])
    out = Path(cfg["paths"]["results_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    sector_of = sector_map()

    hsi = pd.read_csv(p / "hsi.csv", index_col="date", parse_dates=["date"])
    ind = pd.read_csv(p / f"indicators_{a.profile}.csv")
    rm = pd.read_csv(p / f"relative_order_{a.profile}.csv", index_col="date")

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=a.dpi, bbox_inches="tight")
        print("saved", path)

    save(F.fig7_sector_order(rm, hsi, ind, sector_of), "fig7_sector_order.png")
    save(F.fig8_series(hsi, ind), "fig8_series.png")

    gpath = p / f"granger_{a.profile}.csv"
    if gpath.exists():
        save(F.fig9_granger(hsi, pd.read_csv(gpath), cfg["granger"]["alpha"]), "fig9_granger.png")
    else:
        print("skip Fig 9: run scripts/04_granger_tests.py first")

    nets, rows, titles = [], [], []
    ma = F.moving_average_order(rm)
    ndir = out_dir_for(cfg, a.profile)
    for date, title in F.FIG6_DATES:
        hit = ind.loc[ind["date"] == date, "t"]
        if hit.empty or not net_path(ndir, int(hit.iloc[0])).exists():
            print(f"skip {date}: no network")
            continue
        nets.append(load_network(ndir, int(hit.iloc[0])))
        rows.append(ma.loc[date])
        titles.append(title)
    if nets:
        save(F.fig6_networks(nets, rows, sector_of, titles), "fig6_networks.png")


if __name__ == "__main__":
    main()
