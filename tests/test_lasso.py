from pathlib import Path

import numpy as np

from bnsr.evaluation.lasso import MODELS, features, rolling_predictions
from bnsr.evaluation.metrics import rmse, tail_mask


def test_feature_alignment_bruteforce():
    rng = np.random.default_rng(0)
    y, m, o = rng.random(80), rng.random(80), rng.random(80)
    s = np.array([60, 61])
    X = features("H2", 3, y, m, o, s)
    assert X.shape == (2, 12)
    for r, t in enumerate(s):
        exp = [y[t - i] for i in (1, 2, 3)] + [m[t - i] for i in (1, 2, 3)] + [o[t - i] for i in (1, 2, 3)] \
            + [o[t - i] * m[t - i] for i in (1, 2, 3)]
        assert np.allclose(X[r], exp)
    assert features("H0", 5, y, m, o, s).shape == (2, 5)
    assert features("H1a", 5, y, m, o, s).shape == (2, 10)
    assert features("H1b", 5, y, m, o, s).shape == (2, 10)


def test_rmse_and_tail_mask():
    assert rmse([1, 2, 3], [1, 2, 5]) == np.sqrt(4 / 3)
    assert rmse([1, 2, 3], [1, 2, 5], np.array([False, False, True])) == 2.0
    # strictly increasing: always a new maximum
    y = np.concatenate([[np.nan], np.arange(1.0, 301.0)])
    assert tail_mask(y, 150, 200, 99.9, 100).all()
    # strictly decreasing: never
    z = np.concatenate([[np.nan], np.arange(300.0, 0.0, -1.0)])
    assert not tail_mask(z, 150, 200, 98, 100).any()


def test_tail_mask_looks_at_the_target_day():
    rng = np.random.default_rng(5)
    # steadily falling: no new highs
    y = np.concatenate([[np.nan], 0.05 - 0.0001 * np.arange(300)])
    y[201] = 0.5                                       # spike on day 201
    m = tail_mask(y, 150, 250, 99.0, 100)
    # origin t = 200 predicts the spike day
    assert m[200 - 150] and m.sum() == 1


def test_table3_counts_reproduced_if_data_present():
    p = Path(__file__).resolve().parents[1] / "data" / "processed" / "hsi.csv"
    if not p.exists():
        return
    import pandas as pd
    h = pd.read_csv(p)
    for k, exp in (("loss", (43, 73, 105)), ("abs_ret", (37, 69, 116))):
        got = tuple(int(tail_mask(h[k].to_numpy(), 70, 3402, q, 100).sum()) for q in (
            99.9, 99.0, 98.0))
        assert got == exp


def test_alternative_beats_null_when_indicator_is_informative():
    rng = np.random.default_rng(1)
    T = 260
    od = rng.normal(size=T)
    mnd = rng.normal(size=T)
    y = np.zeros(T)
    for s in range(1, T):
        y[s] = 0.8 * od[s - 1] + 0.2 * rng.normal()
    P = rolling_predictions({"r": y}, mnd, od, 60, 120,
                            m=40, cv_splits=4, n_alphas=20, n_jobs=1)["r"]
    truth = y[61:122]
    assert set(P) == set(MODELS) and all(len(v) == 61 for v in P.values())
    assert rmse(P["H1b"], truth) < 0.7 * rmse(P["H0_L5"], truth)
    assert rmse(P["H2"], truth) < 0.8 * rmse(P["H0_L3"], truth)
