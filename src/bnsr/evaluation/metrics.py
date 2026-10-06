"""RMSE (Eq. 11), tail-event RMSE (Eq. 12) and the tail-event day selection.

Tail-event rule (paper, Eq. 12 and Table 3): the prediction made on day t is evaluated only if the observed
y_t exceeds q_t(p), the p-th percentile of the previous m' = 100 observations.  The paper's text writes the window
as {y_t, ..., y_{t-m'+1}}, but only the window {y_{t-1}, ..., y_{t-m'}} (current day excluded, linear
interpolation) reproduces the day counts of the paper's Table 3 exactly (43/73/105 for Loss, 37/69/116 for |R|),
so that is what is implemented.
"""
import numpy as np


def tail_mask(y, t_first, t_last, p, m_prime):
    """Boolean array over origins t = t_first..t_last: y_t > p-th percentile of y_{t-m'}..y_{t-1}.
    Day 0 has no return and is never part of the window."""
    out = np.zeros(t_last - t_first + 1, dtype=bool)
    for r, t in enumerate(range(t_first, t_last + 1)):
        window = y[max(1, t - m_prime): t]
        out[r] = y[t] > np.percentile(window, p)
    return out


def rmse(pred, truth, mask=None):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return float(np.sqrt(np.mean((pred - truth) ** 2)))
