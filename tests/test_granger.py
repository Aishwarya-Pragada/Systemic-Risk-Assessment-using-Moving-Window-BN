import numpy as np

from bnsr.evaluation.granger import (day_pvalues, design_matrices, f_test_pvalue, rolling_granger,
                                     significant_days)


def test_design_alignment_bruteforce():
    rng = np.random.default_rng(0)
    T = 120
    y, m, o = rng.random(T), rng.random(T), rng.random(T)
    t, L, W = 100, 3, 40
    Y, X = design_matrices(y, m, o, t, L, W)
    rows = list(range(t - W + 1 + L, t + 1))
    assert Y.shape == (W - L,) and len(rows) == W - L
    assert np.allclose(Y, y[rows])
    for r, s in enumerate(rows):
        exp0 = [1.0] + [y[s - i] for i in range(1, L + 1)]
        assert np.allclose(X["H0"][r], exp0)
        assert np.allclose(X["H1a"][r, len(exp0):], [m[s - j] for j in range(1, L + 1)])
        assert np.allclose(X["H1b"][r, len(exp0):], [o[s - j] for j in range(1, L + 1)])
        h2 = X["H2"][r, len(exp0):]
        assert np.allclose(h2, [m[s - j] for j in range(1, L + 1)] + [o[s - j] for j in range(1, L + 1)]
                           + [o[s - j] * m[s - j] for j in range(1, L + 1)])
    assert X["H2"].shape[1] == 1 + L + 3 * L


def test_f_test_matches_textbook():
    rng = np.random.default_rng(1)
    n = 60
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    Y = 1 + 2 * x1 + 0.0 * x2 + rng.normal(size=n)
    X0 = np.column_stack([np.ones(n), x1])
    X1 = np.column_stack([np.ones(n), x1, x2])
    # compare with the t-test of a single added regressor (F = t^2)
    beta = np.linalg.lstsq(X1, Y, rcond=None)[0]
    res = Y - X1 @ beta
    s2 = res @ res / (n - 3)
    se = np.sqrt(s2 * np.linalg.inv(X1.T @ X1)[2, 2])
    from scipy import stats
    p_t = 2 * stats.t.sf(abs(beta[2] / se), n - 3)
    assert abs(f_test_pvalue(Y, X0, X1) - p_t) < 1e-10


def test_detects_true_lagged_effect_and_not_noise():
    rng = np.random.default_rng(2)
    T = 400
    x = rng.normal(size=T)
    y = np.zeros(T)
    for s in range(1, T):
        y[s] = 0.9 * x[s - 1] + 0.3 * rng.normal()
    noise = rng.normal(size=T)
    P_true = rolling_granger(y, x, rng.normal(size=T), 100, 140, lags=(1, 2), w_gc=40)
    P_null = rolling_granger(rng.normal(size=T), noise, rng.normal(size=T), 100, 300, lags=(1, 2), w_gc=40)
    assert significant_days(P_true)[:, 0].mean() > 0.95            # H1a picks up the MND effect
    assert significant_days(P_null)[:, 0].mean() < 0.35            # union over lags at 10%: well below power
    assert np.isnan(P_true).sum() == 0


def test_constant_regressor_does_not_crash():
    T = 100
    y = np.random.default_rng(3).random(T)
    d = day_pvalues(y, np.ones(T), np.ones(T), 80, lags=(1, 2), w_gc=40)
    assert all(np.isnan(v) or 0 <= v <= 1 for v in d.values())
