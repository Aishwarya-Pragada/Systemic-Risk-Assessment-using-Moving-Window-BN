"""Access layer over the processed data, used by every later stage.

Time convention follows the paper: t = 1 is 3 Jan 2008 and t = T = 3403 is 4 Nov 2021
(day 0 = 2 Jan 2008 is only used to compute the first return).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame          # T x 60 binary indicators (Int8, <NA> before a stock is listed)
    returns: pd.DataFrame    # T x 60 log-returns
    prices: pd.DataFrame     # (T+1) x 60 cleaned prices, row 0 = day 0
    hsi: pd.DataFrame        # (T+1) rows: t, close, ret, abs_ret, loss
    universe: pd.DataFrame   # name, symbol, sector, first_price_date, first_return_date

    @classmethod
    def load(cls, cfg: dict) -> "Dataset":
        p = Path(cfg["paths"]["processed_dir"])
        rd = lambda f: pd.read_csv(p / f, index_col="date", parse_dates=["date"])
        X = rd("indicators.csv").astype("Int8")
        return cls(X=X, returns=rd("returns.csv"), prices=rd("prices.csv"),
                   hsi=rd("hsi.csv"), universe=pd.read_csv(p / "universe.csv"))

    @property
    def T(self) -> int:
        return len(self.X)

    @property
    def symbols(self) -> list:
        return list(self.X.columns)

    def date_of(self, t: int) -> pd.Timestamp:
        """Calendar date of day t (t = 0 .. T)."""
        return self.hsi.index[t]

    def window(self, t: int, w: int) -> pd.DataFrame:
        """D_{(t-w+1):t}: indicator data of days t-w+1..t restricted to the variables that are
        available on *every* day of the window (the paper: G_t only includes variables that are
        common in all data sets in the window). Returns a w x v_t integer DataFrame."""
        if not (w <= t <= self.T):
            raise ValueError(f"need w <= t <= T, got t={t}, w={w}, T={self.T}")
        win = self.X.iloc[t - w:t]
        cols = win.columns[win.notna().all(axis=0).values]
        return win[cols].astype(int)

    def node_counts(self, w: int) -> pd.Series:
        """v_t for t = w..T."""
        return self.X.notna().astype(int).rolling(w).min().sum(axis=1).iloc[w - 1:].astype(int)
