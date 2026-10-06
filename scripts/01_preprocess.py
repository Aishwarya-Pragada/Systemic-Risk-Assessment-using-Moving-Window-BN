#!/usr/bin/env python
"""Stage 1: raw CSVs -> data/processed/*  (prices, returns, binary indicators, HSI series, report)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                        # noqa: E402
from bnsr.data.preprocessing import run_preprocessing, save_processed      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config YAML (default config/config.yaml)")
    ap.add_argument("--no-strict", action="store_true", help="do not assert paper-reported counts")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = run_preprocessing(cfg, strict=not a.no_strict)
    pdir = save_processed(out, cfg)

    r = out["report"]
    print(f"Trading days : {r['n_trading_days']}  (first {r['first_day']}, last {r['last_day']})")
    print(f"Returns T    : {r['n_returns_T']}  (t=1 is {r['first_return_day']})")
    print(f"Stocks       : {r['n_stocks_total']} of {r['raw_file_tickers']} raw tickers  {r['sector_counts']}")
    print(f"Nodes v_t    : {r['nodes_at_first_window']} -> {r['nodes_at_last_window']}")
    print(f"Gap filling  : {r['price_cleaning']['cells_forward_filled']} cells forward-filled")
    print("Node-count changes (date: v_t):")
    print(json.dumps(r["node_count_changes"], indent=2))
    print(f"Saved to {pdir}")


if __name__ == "__main__":
    main()
