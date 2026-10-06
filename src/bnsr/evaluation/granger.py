"""Rolling Granger-causality tests (paper, Eq. 9 and Eq. 10, Fig. 3c / Fig. 4, Table 1).

On every test day t (t = w + w_GC .. T) the last w_GC = 40 days t-w_GC+1..t are used.  For a lag L the first L
observations only provide lagged values, so each regression has w_GC - L rows:

  H0 : y_s = a0 + sum_i a_i y_{s-i}
  H1a: H0 + sum_j b_j MND_{s-j}
  H1b: H0 + sum_j b_j OD_{s-j}
  H2 : H0 + sum_j b_j MND_{s-j} + sum_k c_k OD_{s-k} + sum_l d_l OD_{s-l} * MND_{s-l}

y is Loss (Eq. 9) or |R| (Eq. 10).  Each alternative is tested against H0 with the usual F-test for the
added regressors; a day counts as "significant" if at least one lag L = 1..5 is significant at the 10% level.
"""
import numpy as np
from scipy import stats

HYPS = ("H1a", "H1b", "H2")


def _fit(Y, X):
    beta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    r = Y - X @ beta
    return float(r @ r), int(rank)


def f_test_pvalue(Y, X0, X1):
    """p-value of the F-test that the extra columns of X1 (over X0) are jointly zero."""
    rss0, k0 = _fit(Y, X0)
    rss1, k1 = _fit(Y, X1)
    q, df = k1 - k0, len(Y) - k1
    if q <= 0 or df <= 0:
        return np.nan
    if rss1 <= 1e-300:
        return 0.0 if rss0 > rss1 else np.nan
    F = max((rss0 - rss1) / q, 0.0) / (rss1 / df)
    return float(stats.f.sf(F, q, df))


def design_matrices(y, mnd, od, t, L, w_gc):
    """Response vector and the four design matrices for one test day t and lag L (series indexed by day)."""
    sl = slice(t - w_gc + 1, t + 1)
    yw, mw, ow = y[sl], mnd[sl], od[sl]
    n = w_gc - L
    Y = yw[L:]
    lag = lambda a, i: a[L - i: w_gc - i]
    ones = np.ones((n, 1))
    ylags = np.column_stack([lag(yw, i) for i in range(1, L + 1)])
    mlags = np.column_stack([lag(mw, j) for j in range(1, L + 1)])
    olags = np.column_stack([lag(ow, j) for j in range(1, L + 1)])
    inter = olags * mlags
    X0 = np.hstack([ones, ylags])
    return Y, {"H0": X0, "H1a": np.hstack([X0, mlags]), "H1b": np.hstack([X0, olags]),
               "H2": np.hstack([X0, mlags, olags, inter])}


def day_pvalues(y, mnd, od, t, lags=(1, 2, 3, 4, 5), w_gc=40):
    """{(hyp, L): p-value} for one test day."""
    out = {}
    for L in lags:
        Y, X = design_matrices(y, mnd, od, t, L, w_gc)
        for h in HYPS:
            out[(h, L)] = f_test_pvalue(Y, X["H0"], X[h])
    return out


def rolling_granger(y, mnd, od, t_first, t_last, lags=(1, 2, 3, 4, 5), w_gc=40):
    """(n_days, len(HYPS), len(lags)) array of p-values for t = t_first..t_last."""
    P = np.full((t_last - t_first + 1, len(HYPS), len(lags)), np.nan)
    for r, t in enumerate(range(t_first, t_last + 1)):
        d = day_pvalues(y, mnd, od, t, lags, w_gc)
        for a, h in enumerate(HYPS):
            for b, L in enumerate(lags):
                P[r, a, b] = d[(h, L)]
    return P


def significant_days(P, alpha=0.1):
    """Boolean (n_days, n_hyps): at least one lag significant at level alpha."""
    with np.errstate(invalid="ignore"):
        return np.nanmin(np.where(np.isnan(P), np.inf, P), axis=2) < alpha
