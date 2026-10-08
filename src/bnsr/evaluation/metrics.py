"""RMSE (Eq. 11), tail-event RMSE (Eq. 12) and the tail-event day selection.

Tail-event rule (paper, Eq. 12 and Table 3): the prediction made on day t (for day t+1) is evaluated only if the
value it predicts, y_{t+1}, exceeds q_t(p), the p-th percentile (linear interpolation) of the last m' = 100
observed values {y_t, y_{t-1}, ..., y_{t-m'+1}}.  I.e. the tail-event RMSE measures how well the model predicts
the days on which the series jumps to an extreme level.
Evidence that this is the paper's rule (not "y_t is extreme"): the day counts equal the paper's Table 3
(43/73/105 for Loss, 37/69/116 for |R|), and a naive predictor evaluated on these target days gives RMSEs of
about 0.038/0.035/0.031, the same size as the paper's H0 columns (0.039/0.035/0.031), whereas conditioning on the
extreme *origin* day gives about 0.02.
"""
import numpy as np


def tail_mask(y, t_first, t_last, p, m_prime):
    """Boolean array over origins t = t_first..t_last: y_{t+1} > p-th percentile of y_{t-m'+1}..y_t.
    Day 0 has no return and is never part of the window."""
    out = np.zeros(t_last - t_first + 1, dtype=bool)
    for r, t in enumerate(range(t_first, t_last + 1)):
        window = y[max(1, t - m_prime + 1): t + 1]
        out[r] = y[t + 1] > np.percentile(window, p)
    return out


def rmse(pred, truth, mask=None):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def paired_tail_test(pred0, pred1, truth, mask, n_boot=5000, seed=0):
    """Is the alternative (pred1) really better than the null model (pred0) on the accepted days?
    Uses d_t = e0_t^2 - e1_t^2 on the days in `mask`.  Returns
    (pct_change, t_stat, p_one_sided, ci_low, ci_high): the % change in RMSE, the t statistic of mean(d)
    (positive = alternative better), the one-sided p-value of that t statistic, and a 95% bootstrap
    interval for the % change in RMSE (resampling days).  Tail days cluster in crisis episodes, so the
    t-test is optimistic; read it as a rough guide, not a proof."""
    from scipy import stats
    e0 = (np.asarray(pred0, float) - np.asarray(truth, float))[mask]
    e1 = (np.asarray(pred1, float) - np.asarray(truth, float))[mask]
    n = len(e0)
    d = e0 ** 2 - e1 ** 2
    se = d.std(ddof=1) / np.sqrt(n)
    t = float(d.mean() / se) if se > 0 else np.nan
    p = float(stats.t.sf(t, n - 1)) if np.isfinite(t) else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ratio = np.sqrt((e1[idx] ** 2).mean(axis=1) / (e0[idx] ** 2).mean(axis=1))
    lo, hi = np.percentile(100 * (ratio - 1), [2.5, 97.5])
    return 100.0 * (np.sqrt((e1 ** 2).mean() / (e0 ** 2).mean()) - 1.0), t, p, float(lo), float(hi)
