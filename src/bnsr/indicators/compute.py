"""Stage 3: risk indicators for every day t = w..T from the learned networks.

  MND_t  modified network density (Eq. 4)
  OD_t   order distance between G_{t-1} and G_t (Eq. 7), defined for t = w+1..T
  RM_t   relative topological order of every stock (Eq. 5), used for Figs 6-7
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.universe import sector_map, symbols
from ..networks.runner import out_dir_for
from ..networks.stats import modified_network_density
from ..networks.storage import list_days, load_network
from .order_distance import order_distance, relative_order
from .topological import median_orders


def compute_indicators(cfg: dict, profile: str, K=None, t_first=None, t_last=None, verbose=True):
    w, M = cfg["bayesian_network"]["w"], cfg["bayesian_network"]["M"]
    K = K or cfg["bayesian_network"]["K"]
    seed0 = cfg["learning"]["seed"]
    d = out_dir_for(cfg, profile)
    days = [t for t in list_days(d) if (t_first is None or t >= t_first) and (t_last is None or t <= t_last)]
    if not days:
        raise FileNotFoundError(f"no networks found in {d}")
    if verbose:
        print(f"{len(days)} networks from {d}, K={K}")

    syms = symbols()
    rows, rm_rows, md_rows = [], {}, {}
    prev = None
    for n, t in enumerate(days):
        z = load_network(d, t)
        nodes, adj = z["nodes"], z["adj"]
        v, e = len(nodes), int(adj.sum())
        Mt = median_orders(adj, K, seed0 + t)
        od = np.nan
        if prev is not None and prev["t"] == t - 1:
            od = order_distance(prev["M"], prev["nodes"], Mt, nodes)
        rows.append((t, z["date"], v, e, modified_network_density(e, v, M), od))
        rm = dict(zip(nodes, relative_order(Mt, v)))
        rm_rows[z["date"]] = rm
        md_rows[z["date"]] = dict(zip(nodes, Mt))
        prev = {"t": t, "M": Mt, "nodes": nodes}
        if verbose and (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(days)}")
    ind = pd.DataFrame(rows, columns=["t", "date", "v", "edges", "MND", "OD"])
    rm = pd.DataFrame.from_dict(rm_rows, orient="index").reindex(columns=syms)
    md = pd.DataFrame.from_dict(md_rows, orient="index").reindex(columns=syms)
    rm.index.name = md.index.name = "date"
    return ind, rm, md


def save_indicators(cfg, profile, ind, rm, md) -> Path:
    p = Path(cfg["paths"]["processed_dir"])
    p.mkdir(parents=True, exist_ok=True)
    ind.to_csv(p / f"indicators_{profile}.csv", index=False)
    rm.to_csv(p / f"relative_order_{profile}.csv")
    md.to_csv(p / f"median_order_{profile}.csv")
    return p


def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    den = (x * x).sum()
    return [float((x[:-k] * x[k:]).sum() / den) for k in range(1, nlags + 1)]


def sector_means(rm: pd.DataFrame) -> pd.Series:
    sm = sector_map()
    return rm.mean(axis=0).groupby(lambda s: sm[s]).mean().sort_values(ascending=False)
