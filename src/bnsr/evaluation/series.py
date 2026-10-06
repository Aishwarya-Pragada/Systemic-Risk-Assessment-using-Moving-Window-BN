"""Daily series used by Stages 4-5, all indexed by day t = 0..T (t = 0 is 2 Jan 2008)."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(cfg, profile):
    p = Path(cfg["paths"]["processed_dir"])
    hsi = pd.read_csv(p / "hsi.csv", index_col="date", parse_dates=["date"])
    ind = pd.read_csv(p / f"indicators_{profile}.csv", parse_dates=["date"])
    T = int(hsi["t"].max())
    mnd, od = np.full(T + 1, np.nan), np.full(T + 1, np.nan)
    mnd[ind["t"].to_numpy()] = ind["MND"].to_numpy()
    od[ind["t"].to_numpy()] = ind["OD"].to_numpy()
    y = {"loss": hsi["loss"].to_numpy(), "abs_ret": hsi["abs_ret"].to_numpy()}
    return y, mnd, od, hsi.index.to_numpy(), T


def shifted(mnd, od, w, T, shift):
    """Placebo: circularly shift MND and OD *together* inside their valid range (keeps their autocorrelation and
    their joint behaviour, breaks the link to the market data)."""
    idx = np.arange(w + 1, T + 1)
    m2, o2 = np.full_like(mnd, np.nan), np.full_like(od, np.nan)
    m2[idx], o2[idx] = np.roll(mnd[idx], shift), np.roll(od[idx], shift)
    return m2, o2
