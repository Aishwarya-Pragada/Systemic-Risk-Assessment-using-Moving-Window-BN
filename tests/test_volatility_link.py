import numpy as np

from bnsr.evaluation.volatility_link import (UPPERS, log_abs, rolling_beta1,
                                             share_positive_by_level)


def test_rolling_beta1_recovers_lagged_coefficient():
    rng = np.random.default_rng(0)
    T = 300
    od, mnd = rng.random(T), rng.random(T)
    y = np.zeros(T)
    for s in range(1, T):
        y[s] = 0.1 + 2.0 * od[s - 1] - 0.5 * mnd[s - 1] + \
            0.3 * y[s - 1] + 1e-4 * rng.normal()
    b = rolling_beta1(y, mnd, od, 100, 200, m=40)
    assert len(b) == 101 and np.allclose(b, 2.0, atol=0.05)


def test_bins_and_log_abs():
    assert len(UPPERS) == 13 and UPPERS[0] == - \
        8.0 and abs(UPPERS[-1] + 3.2) < 1e-9
    y = np.array([0.0, np.exp(-8.0), np.exp(-7.9), np.exp(-3.0)])
    ly = log_abs(y)
    assert np.isinf(ly[0]) and ly[0] < 0
    beta = np.array([1.0, -1.0, 1.0, 1.0])
    counts, share = share_positive_by_level(beta, ly)
    assert counts.sum() == 4 and len(counts) == 14
    # (-inf, -8] holds the zero return and exp(-8)
    assert counts[0] == 2 and share[0] == 0.5
    assert counts[1] == 1 and share[1] == 1.0          # (-8, -7.6]
    assert counts[13] == 1 and share[13] == 1.0        # (-3.2, inf)
