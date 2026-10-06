import numpy as np

from bnsr.indicators.compute import acf
from bnsr.indicators.order_distance import normalized_order, order_distance, relative_order
from bnsr.indicators.topological import median_orders, sample_topological_positions


def _fig2():
    """Paper Fig. 2: X1 -> X3, X2 -> X3, X1 -> X4, X3 -> X4  (indices 0..3)."""
    adj = np.zeros((4, 4), dtype=bool)
    adj[0, 2] = adj[1, 2] = adj[0, 3] = adj[2, 3] = True
    return adj


def test_samples_are_valid_topological_orders():
    rng = np.random.default_rng(0)
    v = 12
    adj = np.triu(rng.random((v, v)) < 0.3, k=1)            # DAG: edges only from lower to higher index
    perm = rng.permutation(v)
    adj = adj[np.ix_(perm, perm)]                              # relabel nodes
    pos = sample_topological_positions(adj, 50, 1)
    for k in range(50):
        assert sorted(pos[k]) == list(range(1, v + 1))
        for j, i in zip(*np.nonzero(adj)):
            assert pos[k, j] < pos[k, i]                       # parent precedes child


def test_fig2_example():
    pos = sample_topological_positions(_fig2(), 100, 3)
    assert (pos[:, 2] == 3).all() and (pos[:, 3] == 4).all()   # X3, X4 fixed
    assert ((pos[:, 0] + pos[:, 1]) == 3).all()                # X1, X2 take positions {1, 2}
    M = median_orders(_fig2(), 100, 3)
    assert M[2] == 3 and M[3] == 4 and abs(M[0] + M[1] - 3) < 1e-9


def test_relative_and_normalized_order():
    M = np.array([1.0, 2.0, 3.0, 4.0])
    rm = relative_order(M, 4)
    assert np.allclose(rm, [1, 2 / 3, 1 / 3, 0])
    nt = normalized_order(M, 4, [0, 1, 2, 3])
    assert abs(nt.sum() - 1) < 1e-12 and np.allclose(nt, [3 / 6, 2 / 6, 1 / 6, 0])


def test_order_distance():
    nodes = ["a", "b", "c", "d"]
    M = np.array([1.0, 2.0, 3.0, 4.0])
    assert order_distance(M, nodes, M, nodes) == 0.0
    M2 = np.array([4.0, 3.0, 2.0, 1.0])
    od = order_distance(M, nodes, M2, nodes)
    assert 0 < od <= 2
    # a new node entering on day t: distance only uses the common nodes a, b, c
    od2 = order_distance(M[:3], nodes[:3], np.array([1.0, 2.0, 3.0, 4.0]), nodes)
    assert od2 >= 0 and np.isfinite(od2)


def test_acf():
    x = np.tile([1.0, -1.0], 50)
    assert acf(x, 2)[0] < -0.9 and acf(x, 2)[1] > 0.9
