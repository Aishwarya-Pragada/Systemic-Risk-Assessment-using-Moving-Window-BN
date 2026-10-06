"""Rolling one-day-ahead LASSO prediction (paper, Eq. 9-10 models, Fig. 3d / Fig. 5, Eq. 11).

On every origin day t = w + m .. T-1 the last m = 40 days (t-m+1..t) are used to fit a LASSO model, giving
m - L training rows (the first L observations only supply lags).  The fitted model predicts y_{t+1}.
Models (y = Loss or |R|):  H0 (lags of y only), H1a (+ MND lags), H1b (+ OD lags),
H2 (+ MND lags + OD lags + OD*MND lags).  L = 5 for H0/H1a/H1b and L = 3 for H2 (and its H0).
The LASSO penalty is chosen by time-series cross-validation (expanding window, scikit-learn TimeSeriesSplit)
on standardised regressors; the exact folds/penalty grid are not given in the paper.
"""
import warnings

import numpy as np
from joblib import Parallel, delayed

MODELS = {            # name -> (kind, L)
    "H0_L5": ("H0", 5), "H1a": ("H1a", 5), "H1b": ("H1b", 5),
    "H0_L3": ("H0", 3), "H2": ("H2", 3),
}


def features(kind, L, y, mnd, od, s_idx):
    """Regressor matrix for target days s_idx: lags 1..L of y, and (depending on kind) of MND, OD, OD*MND."""
    s_idx = np.asarray(s_idx)
    def lag(a): return np.column_stack([a[s_idx - i] for i in range(1, L + 1)])
    cols = [lag(y)]
    if kind in ("H1a", "H2"):
        cols.append(lag(mnd))
    if kind in ("H1b", "H2"):
        cols.append(lag(od))
    if kind == "H2":
        cols.append(lag(od) * lag(mnd))
    return np.hstack(cols)


def _predict_one(kind, L, y, mnd, od, t, m, cv_splits, n_alphas):
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    s_train = np.arange(t - m + 1 + L, t + 1)
    Xtr, ytr = features(kind, L, y, mnd, od, s_train), y[s_train]
    xte = features(kind, L, y, mnd, od, np.array([t + 1]))
    try:
        scaler = StandardScaler().fit(Xtr)
        Xs = scaler.transform(Xtr)
        # explicit penalty grid (works in every scikit-learn version): from alpha_max, where all
        # coefficients are zero, down to alpha_max / 1000, log-spaced
        alpha_max = float(np.max(np.abs(Xs.T @ (ytr - ytr.mean()))) / len(ytr))
        if not alpha_max > 0:
            return float(ytr.mean())
        grid = np.geomspace(alpha_max, alpha_max * 1e-3, n_alphas)
        lasso = LassoCV(alphas=grid, cv=TimeSeriesSplit(
            n_splits=cv_splits), max_iter=20000, tol=1e-4)
        lasso.fit(Xs, ytr)
        return float(lasso.predict(scaler.transform(xte))[0])
    # degenerate window (e.g. constant response)
    except Exception:
        return float(ytr.mean())


def _day(t, ys, mnd, od, m, cv_splits, n_alphas):
    warnings.filterwarnings("ignore")
    return [[_predict_one(kind, L, y, mnd, od, t, m, cv_splits, n_alphas) for kind, L in MODELS.values()]
            for y in ys]


def rolling_predictions(y_by_resp, mnd, od, t_first, t_last, m=40, cv_splits=5, n_alphas=50, n_jobs=1):
    """Predictions made on origins t_first..t_last for the day after.
    Returns {response: {model: array(len = t_last - t_first + 1)}}."""
    names = list(y_by_resp)
    ys = [y_by_resp[k] for k in names]
    days = list(range(t_first, t_last + 1))
    res = Parallel(n_jobs=n_jobs, batch_size=20)(
        delayed(_day)(t, ys, mnd, od, m, cv_splits, n_alphas) for t in days)
    # (days, responses, models)
    arr = np.array(res)
    return {k: {mn: arr[:, i, j] for j, mn in enumerate(MODELS)} for i, k in enumerate(names)}
