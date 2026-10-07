"""Supporting-information S3: is the order distance more often positively associated with volatility when
volatility is high?  (Eq. 13, Figs 11-12 of the supporting information.)

On every day t = w + m .. T an OLS regression on the last m = 40 days
    |R_s| = b0 + b1 OD_{s-1} + b2 MND(G_{s-1}; M) + b3 |R_{s-1}| + e_s
gives b_t = b1.  Days are grouped by log|R_t| into (-inf, -8], (-8, -7.6], ..., (-3.6, -3.2], (-3.2, inf) and the
share of days with b_t > 0 is computed per group.
"""
import numpy as np

# 13 upper bounds -> 14 intervals
UPPERS = np.round(np.arange(-8.0, -3.2 + 1e-9, 0.4), 10)


def rolling_beta1(y, mnd, od, t_first, t_last, m=40):
    out = np.full(t_last - t_first + 1, np.nan)
    for r, t in enumerate(range(t_first, t_last + 1)):
        # m - 1 rows; the first day only gives lags
        s = np.arange(t - m + 2, t + 1)
        X = np.column_stack([np.ones(len(s)), od[s - 1], mnd[s - 1], y[s - 1]])
        if np.isnan(X).any():
            continue
        out[r] = np.linalg.lstsq(X, y[s], rcond=None)[0][1]
    return out


def log_abs(y):
    with np.errstate(divide="ignore"):
        return np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), -np.inf)


def share_positive_by_level(beta, logy):
    """Returns (counts, share_positive) for the 14 intervals."""
    ok = ~np.isnan(beta)
    # (a, b] intervals, last one open
    idx = np.searchsorted(UPPERS, logy[ok], side="left")
    b = beta[ok]
    counts = np.bincount(idx, minlength=len(UPPERS) + 1)
    pos = np.bincount(idx, weights=(b > 0).astype(
        float), minlength=len(UPPERS) + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(counts > 0, pos / counts, np.nan)
    return counts, share
