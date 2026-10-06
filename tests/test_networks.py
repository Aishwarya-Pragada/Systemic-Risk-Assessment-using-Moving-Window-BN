import itertools
import math
import tempfile

import numpy as np

from bnsr.networks.bdeu import (build_tables, local_score_reference, node_score_table,
                                subset_transforms)
from bnsr.networks.candidates import select_candidates
from bnsr.networks.learner import LearnParams, is_acyclic, learn_network
from bnsr.networks.storage import list_days, load_network, save_network
from bnsr.networks.stats import modified_network_density


def _data(N=30, v=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(N, v))
    X[:, 1] = np.where(rng.random(N) < 0.85, X[:, 0], 1 - X[:, 0])      # node 1 depends on node 0
    X[:, 2] = np.where(rng.random(N) < 0.85, X[:, 1], 1 - X[:, 1])
    return X.astype(np.int64)


def test_bdeu_table_matches_reference():
    X = _data()
    cand_i = np.array([0, 2, 3, 5], dtype=np.int64)
    sc = node_score_table(X, 1, cand_i, 3, 1.0)
    for s in range(1 << 4):
        pars = [int(cand_i[c]) for c in range(4) if (s >> c) & 1]
        if len(pars) > 3:
            assert sc[s] == -np.inf
        else:
            assert abs(sc[s] - local_score_reference(X, 1, pars, 1.0)) < 1e-9


def test_subset_transforms_bruteforce():
    X = _data()
    cand_i = np.array([0, 2, 3, 5], dtype=np.int64)
    sc = node_score_table(X, 1, cand_i, 4, 0.5)
    F, G, A = subset_transforms(sc, 4)
    for mask in range(16):
        subs = [s for s in range(16) if (s & mask) == s]
        vals = np.array([sc[s] for s in subs])
        assert abs(F[mask] - (vals.max() + math.log(np.exp(vals - vals.max()).sum()))) < 1e-9
        assert abs(G[mask] - vals.max()) < 1e-12
        assert sc[A[mask]] == G[mask] and (A[mask] & mask) == A[mask]


def test_candidates_pick_dependent_node():
    X = _data()
    cand = select_candidates(X, 2, 1.0)
    assert cand.shape == (6, 2)
    assert 2 in cand[1] and 1 in cand[2]      # nodes 1 and 2 are strongly dependent in _data()
    assert all(i not in cand[i] for i in range(6))


def _all_dags(v):
    pairs = [(a, b) for a in range(v) for b in range(v) if a != b]
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        adj = np.zeros((v, v), dtype=bool)
        for (a, b), x in zip(pairs, bits):
            adj[a, b] = bool(x)
        if is_acyclic(adj):
            yield adj


def test_learner_finds_exhaustive_optimum_4_nodes():
    X = _data(N=30, v=4, seed=3)
    cache = {}

    def ls(i, pars):
        key = (i, tuple(pars))
        if key not in cache:
            cache[key] = local_score_reference(X, i, list(pars), 1.0)
        return cache[key]

    best, n = -np.inf, 0
    for adj in _all_dags(4):
        n += 1
        tot = sum(ls(i, np.nonzero(adj[:, i])[0]) for i in range(4))
        best = max(best, tot)
    assert n == 543
    p = LearnParams(M=3, C=3, ess=1.0, n_chains=1, order_iter=1500, order_burn=100, thin=1,
                    struct_iter=500)
    net = learn_network(X, p, seed=11)
    assert abs(net.score - best) < 1e-8
    tot = sum(ls(i, np.nonzero(net.adj[:, i])[0]) for i in range(4))
    assert abs(tot - net.score) < 1e-8


def test_acyclic_parent_limit_and_determinism():
    X = _data(N=30, v=9, seed=5)
    p = LearnParams(M=2, C=5, n_chains=1, order_iter=800, order_burn=100, thin=5, struct_iter=800)
    n1 = learn_network(X, p, seed=1)
    n2 = learn_network(X, p, seed=1)
    assert is_acyclic(n1.adj) and n1.adj.sum(axis=0).max() <= 2
    assert np.array_equal(n1.adj, n2.adj) and n1.score == n2.score


def test_structure_phase_never_worse_than_order_phase():
    X = _data(N=30, v=7, seed=8)
    a = learn_network(X, LearnParams(M=3, C=4, order_iter=300, order_burn=50, thin=5, struct_iter=0), 2)
    b = learn_network(X, LearnParams(M=3, C=4, order_iter=300, order_burn=50, thin=5, struct_iter=600), 2)
    assert b.score >= a.score - 1e-9


def test_storage_roundtrip_and_density():
    X = _data(v=5)
    net = learn_network(X, LearnParams(M=2, C=3, order_iter=100, order_burn=10, thin=5, struct_iter=100), 1)
    with tempfile.TemporaryDirectory() as d:
        save_network(d, 31, "2008-02-18", [f"s{i}" for i in range(5)], net)
        assert list_days(d) == [31]
        z = load_network(d, 31)
        assert np.array_equal(z["adj"], net.adj) and z["nodes"][0] == "s0" and z["date"] == "2008-02-18"
    assert modified_network_density(9, 5, 3) == 9 / (5 * 3 - 6)
