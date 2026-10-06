"""Raw-data loaders for the two provided CSV files."""
from pathlib import Path

import pandas as pd


def _read(path, date_col, date_format) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    if df.index.duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    return df


def load_constituent_prices(cfg: dict) -> pd.DataFrame:
    """Daily closing prices of all 103 tickers in the raw file (dates x symbols)."""
    path = Path(cfg["paths"]["raw_dir"]) / cfg["data"]["constituents_file"]
    return _read(path, "date", cfg["data"]["date_format"])


def load_index(cfg: dict) -> pd.DataFrame:
    """HSI index file (Open/High/Low/Close/Adj Close/Volume), indexed by date."""
    path = Path(cfg["paths"]["raw_dir"]) / cfg["data"]["index_file"]
    return _read(path, "Date", cfg["data"]["date_format"])
