#!/usr/bin/env python
"""Stage 5: rolling one-day-ahead LASSO predictions and the RMSE tables (Tables 2 and 3 of the paper).
Needs data/processed/hsi.csv and data/processed/indicators_<profile>.csv.
Writes data/processed/lasso_predictions_<profile>.csv and lasso_table2_<profile>.csv.

  python scripts/05_lasso_prediction.py --profile faithful
  python scripts/05_lasso_prediction.py --profile faithful --placebo     # also refit with shifted indicators
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                      # noqa: E402
from bnsr.evaluation.lasso import MODELS, rolling_predictions            # noqa: E402
from bnsr.evaluation.metrics import rmse, tail_mask                      # noqa: E402
from bnsr.evaluation.series import load_series, shifted                  # noqa: E402

# Paper Table 2: (response, case) -> [(H0, alt, pct) for H1a, H1b, H2]
PAPER_T2 = {
    ("loss", "0.1%"): [(.0392, .0404, 3.2), (.0392, .0387, -1.1), (.0377, .0375, -0.5)],
    ("abs_ret", "0.1%"): [(.0392, .0396, 1.0), (.0392, .0365, -7.0), (.0394, .0372, -5.6)],
    ("loss", "1%"): [(.0349, .0359, 2.8), (.0349, .0349, -0.1), (.0342, .0340, -0.6)],
    ("abs_ret", "1%"): [(.0338, .0342, 1.2), (.0338, .0320, -5.4), (.0342, .0326, -4.6)],
    ("loss", "2%"): [(.0310, .0319, 2.7), (.0310, .0310, -0.1), (.0306, .0307, 0.3)],
    ("abs_ret", "2%"): [(.0305, .0309, 1.4), (.0305, .0298, -2.2), (.0305, .0299, -1.9)],
    ("loss", "all"): [(.0094, .0098, 4.5), (.0094, .0098, 4.8), (.0091, .0096, 5.2)],
    ("abs_ret", "all"): [(.0100, .0106, 6.0), (.0100, .0104, 4.9), (.0097, .0104, 6.7)],
}
PAPER_T3 = {"loss": (43, 73, 105), "abs_ret": (37, 69, 116)}
PCT = {"0.1%": 99.9, "1%": 99.0, "2%": 98.0}
PAIRS = [("H1a", "H0_L5"), ("H1b", "H0_L5"), ("H2", "H0_L3")]      # (alternative, its null model)


def evaluate(y, preds, t_first, t_last, m_prime):
    """Table-2 style results: {(resp, case): [(rmse_H0, rmse_alt, pct) x3]} and Table-3 counts."""
    out, counts = {}, {}
    for resp in preds:
        truth = y[resp][t_first + 1: t_last + 2]
        cases = {c: tail_mask(y[resp], t_first, t_last, p, m_prime) for c, p in PCT.items()}
        cases["all"] = None
        counts[resp] = tuple(int(cases[c].sum()) for c in PCT)
        for case, mask in cases.items():
            row = []
            for alt, null in PAIRS:
                r0, r1 = rmse(preds[resp][null], truth, mask), rmse(preds[resp][alt], truth, mask)
                row.append((r0, r1, 100.0 * (r1 / r0 - 1.0)))
            out[(resp, case)] = row
    return out, counts


def show(res, counts, title):
    print(f"\n{title}")
    print(f"{'case':>6} {'response':>8} | " + " | ".join(f"{h:^30s}" for h in ("H1a", "H1b", "H2")))
    print(f"{'':>6} {'':>8} | " + " | ".join(f"{'H0 / alt  (pct)  [paper pct]':^30s}" for _ in range(3)))
    for case in ("0.1%", "1%", "2%", "all"):
        for resp in ("loss", "abs_ret"):
            cells = []
            for i, (r0, r1, pc) in enumerate(res[(resp, case)]):
                pp = PAPER_T2[(resp, case)][i][2]
                cells.append(f"{r0:.4f}/{r1:.4f} ({pc:+5.1f}%) [{pp:+5.1f}%]")
            print(f"{case:>6} {resp:>8} | " + " | ".join(f"{c:^30s}" for c in cells))
    print("\nTable 3 (days accepted)   yours -> paper")
    for resp in counts:
        print(f"  {resp:8s} 0.1%/1%/2%: {counts[resp]} -> {PAPER_T3[resp]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--placebo", action="store_true", help="also run with the indicators shifted in time")
    ap.add_argument("--placebo-shift", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None, help="only the first N origins (quick test)")
    a = ap.parse_args()

    cfg = load_config()
    w, lc = cfg["bayesian_network"]["w"], cfg["lasso"]
    workers = a.workers or cfg["learning"]["workers"]
    y, mnd, od, dates, T = load_series(cfg, a.profile)
    t_first, t_last = w + lc["m"], T - 1
    if a.limit:
        t_last = min(t_last, t_first + a.limit - 1)
    print(f"{t_last - t_first + 1} one-day-ahead predictions (origins t={t_first}..{t_last}), m={lc['m']}, "
          f"L=5/3, m'={lc['m_prime']}, workers={workers}")

    def run(m_, o_, tag):
        t0 = time.time()
        preds = rolling_predictions(y, m_, o_, t_first, t_last, m=lc["m"], cv_splits=lc.get("cv_splits", 5),
                                    n_alphas=lc.get("n_alphas", 50), n_jobs=workers)
        print(f"[{tag}] fitted in {(time.time() - t0) / 60:.1f} min")
        return preds

    preds = run(mnd, od, "real")
    res, counts = evaluate(y, preds, t_first, t_last, lc["m_prime"])
    show(res, counts, "YOUR TABLE 2  (RMSE of H0 / alternative, % change; paper's % change in brackets)")

    pdir = Path(cfg["paths"]["processed_dir"])
    cols = {"t": np.arange(t_first, t_last + 1)}
    for resp in preds:
        cols[f"{resp}_next"] = y[resp][t_first + 1: t_last + 2]
        for mn, v in preds[resp].items():
            cols[f"{resp}_{mn}"] = v
    pd.DataFrame(cols).to_csv(pdir / f"lasso_predictions_{a.profile}.csv", index=False)
    rows = [(resp, case, alt, r0, r1, pc) for (resp, case), row in res.items()
            for (alt, _), (r0, r1, pc) in zip(PAIRS, row)]
    pd.DataFrame(rows, columns=["response", "case", "model", "rmse_H0", "rmse_alt", "pct_change"]).to_csv(
        pdir / f"lasso_table2_{a.profile}.csv", index=False)

    if a.placebo:
        m2, o2 = shifted(mnd, od, w, T, a.placebo_shift)
        res0, _ = evaluate(y, run(m2, o2, "placebo"), t_first, t_last, lc["m_prime"])
        show(res0, counts, f"PLACEBO TABLE 2 (indicators shifted by {a.placebo_shift} days = what chance gives)")
    print(f"\nsaved to {pdir}")


if __name__ == "__main__":
    main()
