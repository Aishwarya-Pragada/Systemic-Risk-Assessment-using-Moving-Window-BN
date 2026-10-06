"""Learn one Bayesian network G_t from a window of binary data.

Pipeline (per window):
  1. candidate parents per node (top-C pairwise BDeu gain)               -> candidates.py
  2. BDeu tables for every parent subset of the candidates, M-limited    -> bdeu.py
  3. Order MCMC (n_chains independent chains); keep the best DAG found    -> order_mcmc.py
  4. Structure MCMC (add/delete/reverse) started from that DAG            -> structure_mcmc.py
  5. G_t = the DAG with the highest BDeu score seen in steps 3-4
"""
from dataclasses import dataclass

import numpy as np

from .bdeu import build_tables
from .candidates import select_candidates
from .order_mcmc import order_mcmc
from .structure_mcmc import structure_mcmc


@dataclass
class LearnParams:
    M: int = 13
    C: int = 14
    ess: float = 1.0
    n_chains: int = 1
    order_iter: int = 20000
    order_burn: int = 5000
    thin: int = 10
    struct_iter: int = 20000

    @classmethod
    def from_config(cls, cfg: dict, profile: str) -> "LearnParams":
        lc = cfg["learning"]
        if profile not in lc["profiles"]:
            raise KeyError(f"unknown profile {profile!r}; available: {list(lc['profiles'])}")
        return cls(M=cfg["bayesian_network"]["M"], ess=lc["ess"], **lc["profiles"][profile])


@dataclass
class LearnedNetwork:
    adj: np.ndarray        # (v, v) bool; adj[j, i] = True  <=>  edge j -> i
    score: float           # total BDeu log-score of the DAG
    info: dict


def is_acyclic(adj: np.ndarray) -> bool:
    indeg = adj.sum(axis=0).astype(int)
    stack = [i for i in range(adj.shape[0]) if indeg[i] == 0]
    seen = 0
    while stack:
        n = stack.pop()
        seen += 1
        for c in np.nonzero(adj[n])[0]:
            indeg[c] -= 1
            if indeg[c] == 0:
                stack.append(c)
    return seen == adj.shape[0]


def subsets_to_adjacency(sub, cand) -> np.ndarray:
    v, C = cand.shape
    adj = np.zeros((v, v), dtype=bool)
    for i in range(v):
        for c in range(C):
            if (int(sub[i]) >> c) & 1:
                adj[cand[i, c], i] = True
    return adj


def learn_network(data: np.ndarray, params: LearnParams, seed: int) -> LearnedNetwork:
    """data: (N, v) array of 0/1 values (one row per day of the window)."""
    X = np.ascontiguousarray(data, dtype=np.int64)
    v = X.shape[1]
    cand = select_candidates(X, params.C, params.ess)
    scores, F, G, A = build_tables(X, cand, params.M, params.ess)

    best_score, best_sub = -np.inf, None
    acc_order = []
    for ch in range(params.n_chains):
        sc, sub, acc = order_mcmc(F, G, A, cand, params.order_iter, params.order_burn,
                                  params.thin, seed + 7919 * ch)
        acc_order.append(acc)
        if sc > best_score:
            best_score, best_sub = sc, sub

    s_score, s_sub, acc_struct = structure_mcmc(scores, cand, best_sub, params.struct_iter,
                                                seed + 104729)
    if s_score > best_score:
        best_score, best_sub = s_score, s_sub

    adj = subsets_to_adjacency(best_sub, cand)
    assert is_acyclic(adj), "learned graph contains a cycle"
    assert adj.sum(axis=0).max() <= params.M, "parent limit violated"
    info = {"v": v, "edges": int(adj.sum()), "order_acc": float(np.mean(acc_order)),
            "struct_acc": float(acc_struct), "C": int(cand.shape[1])}
    return LearnedNetwork(adj=adj, score=float(best_score), info=info)


def warmup():
    """Compile (and cache) all numba functions on a tiny problem before workers start."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(30, 8))
    learn_network(X, LearnParams(M=3, C=4, n_chains=1, order_iter=50, order_burn=10, thin=5,
                                 struct_iter=50), seed=1)
