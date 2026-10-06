"""Sampling topological orders of a DAG with Kahn's algorithm (paper, "Topological orders").

Convention (paper): if X_i precedes X_j in the order then X_j cannot be a parent of X_i, i.e. parents
(sources of disturbance) come first.  adj[j, i] = True means the edge j -> i, so j precedes i.

Kahn's algorithm is run with a random *initial* order: among all nodes whose parents are already placed it
always picks the one ranked first in the initial order.  Different random initial orders give different
topological orders compatible with the DAG; K = 100 of them are drawn per network (paper, K = 100).
"""
import numpy as np

from .._jit import njit


@njit(cache=True)
def sample_topological_positions(adj, K, seed):
    """Returns pos of shape (K, v): pos[k, i] in 1..v is the position T(X_i) in the k-th sampled order."""
    np.random.seed(seed)
    v = adj.shape[0]
    pos = np.empty((K, v), dtype=np.int64)
    indeg0 = np.zeros(v, dtype=np.int64)
    for i in range(v):
        for j in range(v):
            if adj[j, i]:
                indeg0[i] += 1
    for k in range(K):
        rank = np.random.permutation(v)            # random initial order: rank[node]
        indeg = indeg0.copy()
        placed = np.zeros(v, dtype=np.bool_)
        for step in range(v):
            best = -1
            best_rank = v + 1
            for n in range(v):
                if (not placed[n]) and indeg[n] == 0 and rank[n] < best_rank:
                    best = n
                    best_rank = rank[n]
            if best < 0:
                raise ValueError("graph has a cycle")
            placed[best] = True
            pos[k, best] = step + 1
            for c in range(v):
                if adj[best, c]:
                    indeg[c] -= 1
    return pos


def median_orders(adj, K, seed):
    """M_t(X_i): median position over the K sampled topological orders (float array of length v)."""
    pos = sample_topological_positions(np.ascontiguousarray(adj, dtype=np.bool_), K, seed)
    return np.median(pos, axis=0)
