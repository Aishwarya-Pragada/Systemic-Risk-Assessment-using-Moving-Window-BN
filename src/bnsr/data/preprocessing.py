"""Preprocessing pipeline: raw CSVs -> prices, log-returns, binary return indicators X_it, HSI series.

Paper (Data section):
  * closing prices of the HSI constituents and of the HSI, 2 Jan 2008 - 4 Nov 2021 (3,404 trading days)
  * R_it = log P_it - log P_i(t-1), day 0 = 2 Jan 2008  ->  T = 3403 returns, t = 1 is 3 Jan 2008
  * X_it = 1 if R_it >= 0 else 0                                  (Eq. 8)
  * Loss_t = -min{R_t, 0};  |R_t| is the volatility proxy         (Eq. 9, Eq. 10)
  * number of nodes v_t grows from 46 to 60 as constituents are listed
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import load_constituent_prices, load_index
from .universe import symbols, universe_frame, sector_map, SECTOR_COUNTS


# --------------------------------------------------------------------------------------
# 1. trading calendar
# --------------------------------------------------------------------------------------
def select_calendar(prices, index_df, start, end, price_col="Close"):
    """Trading days used by the study.

    A date is kept if (a) it is in [start, end], (b) it appears in both files, (c) the HSI close is
    not missing and (d) at least one of the 60 stocks has a price (this removes half-day sessions
    such as 24/31 Dec, and typhoon days, on which the stock file is empty).
    This rule yields exactly the 3,404 trading days reported in the paper.
    Returns (calendar, dict of dropped dates by reason).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    p_dates = prices.index[(prices.index >= start) & (prices.index <= end)]
    i_sub = index_df.loc[(index_df.index >= start) & (index_df.index <= end)]

    only_stock_file = p_dates.difference(i_sub.index)
    only_index_file = i_sub.index.difference(p_dates)
    common = p_dates.intersection(i_sub.index)

    idx_nan = common[i_sub.loc[common, price_col].isna().values]
    common = common.difference(idx_nan)

    syms = [s for s in symbols() if s in prices.columns]
    no_stock_px = common[prices.loc[common, syms].notna().sum(axis=1).values == 0]
    calendar = common.difference(no_stock_px)

    dropped = {
        "only_in_constituents_file": [d.strftime("%Y-%m-%d") for d in only_stock_file],
        "only_in_index_file": [d.strftime("%Y-%m-%d") for d in only_index_file],
        "index_close_missing": [d.strftime("%Y-%m-%d") for d in idx_nan],
        "no_constituent_prices": [d.strftime("%Y-%m-%d") for d in no_stock_px],
    }
    return calendar, dropped


# --------------------------------------------------------------------------------------
# 2. price cleaning
# --------------------------------------------------------------------------------------
def clean_prices(prices, calendar, forward_fill=True, max_ffill=None):
    """Restrict to the 60-stock universe and the study calendar; fill suspension gaps.

    Forward-filling never creates prices before a stock's first listing day (leading NaNs stay NaN),
    so late-listed stocks only enter the networks once they have a full window of returns.
    A filled gap gives R = 0 and therefore X = 1 (since X = 1 if R >= 0).
    """
    syms = symbols()
    missing = [s for s in syms if s not in prices.columns]
    if missing:
        raise KeyError(f"Universe symbols missing from constituents file: {missing}")

    px = prices.loc[calendar, syms].copy()
    if (px <= 0).any().any():
        raise ValueError("Non-positive prices found in the 60-stock universe")

    first_valid = px.apply(pd.Series.first_valid_index)
    na_before = px.isna()
    if forward_fill:
        px = px.ffill(limit=max_ffill)
    filled = na_before & px.notna()
    stats = {
        "cells_forward_filled": int(filled.sum().sum()),
        "forward_filled_per_stock": {k: int(v) for k, v in filled.sum().items() if v > 0},
        "remaining_interior_gaps": int(
            sum(px[s].loc[first_valid[s]:].isna().sum() for s in syms)
        ),
    }
    return px, first_valid, stats


# --------------------------------------------------------------------------------------
# 3. returns / indicators / HSI series
# --------------------------------------------------------------------------------------
def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    """R_it = log P_it - log P_i(t-1). Row 0 (day 0 = 2 Jan 2008) is dropped -> T rows."""
    return np.log(px).diff().iloc[1:]


def binary_indicators(ret: pd.DataFrame) -> pd.DataFrame:
    """Eq. (8): X_it = 1 if R_it >= 0 else 0. Missing returns (not yet listed) stay missing."""
    return (ret >= 0).astype("Int8").where(ret.notna())


def hsi_series(index_df, calendar, price_col="Close") -> pd.DataFrame:
    """HSI close, log-return R_t, |R_t| (volatility proxy) and Loss_t = -min{R_t, 0}."""
    close = index_df.loc[calendar, price_col].astype(float)
    r = np.log(close).diff()
    out = pd.DataFrame({"close": close, "ret": r})
    out["abs_ret"] = out["ret"].abs()
    out["loss"] = np.maximum(-out["ret"], 0.0)
    out.insert(0, "t", np.arange(len(out)))  # t = 0 is 2 Jan 2008 (day 0), t = 1 is 3 Jan 2008
    return out


# --------------------------------------------------------------------------------------
# 4. driver
# --------------------------------------------------------------------------------------
def run_preprocessing(cfg: dict, strict: bool = True) -> dict:
    d = cfg["data"]
    raw_prices = load_constituent_prices(cfg)
    raw_index = load_index(cfg)

    calendar, dropped = select_calendar(
        raw_prices, raw_index, d["start_date"], d["end_date"], d["price_column"]
    )
    px, first_valid, fill_stats = clean_prices(
        raw_prices, calendar, d["forward_fill_gaps"], d["max_forward_fill"]
    )
    ret = log_returns(px)
    X = binary_indicators(ret)
    hsi = hsi_series(raw_index, calendar, d["price_column"])

    # ---- node count v_t over time (stocks with a valid X for the whole 30-day window) ----
    w = cfg["bayesian_network"]["w"]
    full_window = X.notna().astype(int).rolling(w).min()
    node_counts = full_window.sum(axis=1).iloc[w - 1:].astype(int)
    steps = node_counts[node_counts.diff().fillna(node_counts.iloc[0]) != 0]

    first_ret = X.apply(pd.Series.first_valid_index)
    uni = universe_frame()
    uni["first_price_date"] = uni["symbol"].map(first_valid).dt.strftime("%Y-%m-%d")
    uni["first_return_date"] = uni["symbol"].map(first_ret).dt.strftime("%Y-%m-%d")

    report = {
        "n_trading_days": int(len(calendar)),
        "n_returns_T": int(len(ret)),
        "first_day": calendar[0].strftime("%Y-%m-%d"),
        "first_return_day": ret.index[0].strftime("%Y-%m-%d"),
        "last_day": calendar[-1].strftime("%Y-%m-%d"),
        "n_stocks_total": int(px.shape[1]),
        "sector_counts": uni["sector"].value_counts().to_dict(),
        "raw_file_tickers": int(raw_prices.shape[1]),
        "tickers_not_in_paper_universe": int(raw_prices.shape[1] - px.shape[1]),
        "dropped_dates": dropped,
        "price_cleaning": fill_stats,
        "nodes_at_first_window": int(node_counts.iloc[0]),
        "nodes_at_last_window": int(node_counts.iloc[-1]),
        "node_count_changes": {k.strftime("%Y-%m-%d"): int(v) for k, v in steps.items()},
        "fraction_up_days_all_stocks": float(X.stack().astype(float).mean()),
        "hsi_missing_returns": int(hsi["ret"].iloc[1:].isna().sum()),
    }

    if strict:
        assert len(calendar) == d["expected_trading_days"], report["n_trading_days"]
        assert len(ret) == d["expected_returns"], report["n_returns_T"]
        assert report["sector_counts"] == SECTOR_COUNTS, report["sector_counts"]
        assert node_counts.min() == d["min_nodes"] and node_counts.max() == d["max_nodes"], (
            node_counts.min(), node_counts.max())
        assert (node_counts.diff().dropna() >= 0).all(), "node count must be non-decreasing"
        assert report["hsi_missing_returns"] == 0
        assert report["price_cleaning"]["remaining_interior_gaps"] == 0 or not d["forward_fill_gaps"]

    return {"prices": px, "returns": ret, "X": X, "hsi": hsi, "universe": uni,
            "node_counts": node_counts, "report": report}


def save_processed(out: dict, cfg: dict) -> Path:
    pdir = Path(cfg["paths"]["processed_dir"])
    pdir.mkdir(parents=True, exist_ok=True)
    out["prices"].to_csv(pdir / "prices.csv", index_label="date")
    out["returns"].to_csv(pdir / "returns.csv", index_label="date")
    out["X"].to_csv(pdir / "indicators.csv", index_label="date")
    out["hsi"].to_csv(pdir / "hsi.csv", index_label="date")
    out["universe"].to_csv(pdir / "universe.csv", index=False)
    out["node_counts"].rename("v_t").to_csv(pdir / "node_counts.csv", index_label="date")
    with open(pdir / "preprocessing_report.json", "w") as f:
        json.dump(out["report"], f, indent=2, default=str)
    return pdir
