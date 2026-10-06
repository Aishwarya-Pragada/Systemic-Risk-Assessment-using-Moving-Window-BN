#!/usr/bin/env python
"""Stage 4: rolling Granger-causality tests (Table 1 of the paper).
Needs data/processed/hsi.csv (Stage 1) and data/processed/indicators_<profile>.csv (Stage 3).
Writes data/processed/granger_<profile>.csv (p-values for every day, response, hypothesis and lag)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                       # noqa: E402
from bnsr.evaluation.granger import HYPS, rolling_granger, significant_days  # noqa: E402

PAPER = {"loss": (1108, 890, 1100), "abs_ret": (942, 682, 953)}          # Table 1 (H1a, H1b, H2)


def load_series(cfg, profile):
    p = Path(cfg["paths"]["processed_dir"])
    hsi = pd.read_csv(p / "hsi.csv", index_col="date", parse_dates=["date"])
    ind = pd.read_csv(p / f"indicators_{profile}.csv", parse_dates=["date"])
    T = int(hsi["t"].max())
    mnd, od = np.full(T + 1, np.nan), np.full(T + 1, np.nan)
    mnd[ind["t"].to_numpy()] = ind["MND"].to_numpy()
    od[ind["t"].to_numpy()] = ind["OD"].to_numpy()
    y = {"loss": hsi["loss"].to_numpy(), "abs_ret": hsi["abs_ret"].to_numpy()}
    dates = hsi.index.to_numpy()
    return y, mnd, od, dates, T


def shifted(mnd, od, w, T, shift):
    """Placebo: circularly shift MND and OD together inside their valid range (keeps their autocorrelation
    and joint behaviour, breaks the link to the market data)."""
    idx = np.arange(w + 1, T + 1)
    m2, o2 = np.full_like(mnd, np.nan), np.full_like(od, np.nan)
    m2[idx], o2[idx] = np.roll(mnd[idx], shift), np.roll(od[idx], shift)
    return m2, o2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--placebo-shift", type=int, default=1500, help="days to shift indicators for the placebo")
    ap.add_argument("--no-placebo", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    w, g = cfg["bayesian_network"]["w"], cfg["granger"]
    y, mnd, od, dates, T = load_series(cfg, a.profile)
    t_first, t_last = w + g["w_gc"], T
    days = np.arange(t_first, t_last + 1)
    print(f"{len(days)} test days (t={t_first}..{t_last}), w_GC={g['w_gc']}, lags={g['lags']}, alpha={g['alpha']}")

    def run(m, o, tag):
        t0 = time.time()
        P = {k: rolling_granger(v, m, o, t_first, t_last, tuple(g["lags"]), g["w_gc"]) for k, v in y.items()}
        sig = {k: significant_days(p, g["alpha"]) for k, p in P.items()}
        print(f"[{tag}] done in {time.time() - t0:.0f}s")
        return P, sig

    P, sig = run(mnd, od, "real")
    # ---- save p-values ----
    cols = {"t": days, "date": pd.to_datetime(dates[days]).strftime("%Y-%m-%d")}
    for k in P:
        for ai, h in enumerate(HYPS):
            for bi, L in enumerate(g["lags"]):
                cols[f"{k}_{h}_L{L}"] = P[k][:, ai, bi]
    pdir = Path(cfg["paths"]["processed_dir"])
    pd.DataFrame(cols).to_csv(pdir / f"granger_{a.profile}.csv", index=False)

    def table(sg, title):
        print(f"\n{title}  (days with >= 1 significant lag at {g['alpha']:.0%}, out of {len(days)})")
        print(f"{'':10s}" + "".join(f"{h:>12s}" for h in HYPS))
        for k in sg:
            print(f"{k:10s}" + "".join(f"{int(sg[k][:, i].sum()):>8d} ({sg[k][:, i].mean():4.0%})" for i in range(3)))

    table(sig, "YOUR RESULT")
    print("\nPAPER Table 1" + "\n" + f"{'':10s}" + "".join(f"{h:>12s}" for h in HYPS))
    for k, v in PAPER.items():
        print(f"{k:10s}" + "".join(f"{n:>8d} ({n / 3334:4.0%})" for n in v))

    if not a.no_placebo:
        m2, o2 = shifted(mnd, od, w, T, a.placebo_shift)
        _, sig0 = run(m2, o2, "placebo")
        table(sig0, f"PLACEBO (indicators shifted by {a.placebo_shift} days: what chance alone gives)")

    split = np.searchsorted(pd.to_datetime(dates[days]), pd.Timestamp("2015-01-01"))
    print("\nShare of significant days, 2008-2014 vs 2015-2021 (paper: more significant results after 2015):")
    for k in sig:
        print(f"  {k:8s} " + "  ".join(f"{h}: {sig[k][:split, i].mean():.0%} -> {sig[k][split:, i].mean():.0%}"
                                      for i, h in enumerate(HYPS)))
    print(f"\nsaved {pdir / f'granger_{a.profile}.csv'}")


if __name__ == "__main__":
    main()
