"""One compressed .npz file per day: data/interim/networks/<profile>/net_<t>.npz"""
from pathlib import Path

import numpy as np


def net_path(out_dir, t: int) -> Path:
    return Path(out_dir) / f"net_{t:05d}.npz"


def save_network(out_dir, t, date, nodes, net):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"net_{t:05d}.tmp.npz"
    np.savez_compressed(
        tmp, t=np.int64(t), date=str(date), nodes=np.array(nodes, dtype="U10"),
        adj=net.adj.astype(bool), score=np.float64(net.score),
        order_acc=np.float64(net.info["order_acc"]), struct_acc=np.float64(net.info["struct_acc"]),
    )
    tmp.replace(net_path(out_dir, t))          # atomic: a half-written file is never picked up on resume


def load_network(out_dir, t):
    z = np.load(net_path(out_dir, t), allow_pickle=False)
    return {"t": int(z["t"]), "date": str(z["date"]), "nodes": [str(s) for s in z["nodes"]],
            "adj": z["adj"], "score": float(z["score"])}


def list_days(out_dir):
    return sorted(int(p.stem.split("_")[1]) for p in Path(out_dir).glob("net_?????.npz"))
