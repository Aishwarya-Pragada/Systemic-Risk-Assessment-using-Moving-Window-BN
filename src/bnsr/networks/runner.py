"""Worker-side code for scripts/02_learn_networks.py (must live in an importable module for Windows spawn)."""
import time
from pathlib import Path

from ..data.dataset import Dataset
from .learner import LearnParams, learn_network
from .storage import save_network

_S = {}


def init_worker(cfg: dict, profile: str):
    from .learner import warmup
    warmup()                                   # loads the numba cache (or compiles once)
    _S["cfg"] = cfg
    _S["ds"] = Dataset.load(cfg)
    _S["params"] = LearnParams.from_config(cfg, profile)
    _S["w"] = cfg["bayesian_network"]["w"]
    _S["seed"] = cfg["learning"]["seed"]
    _S["out_dir"] = Path(cfg["paths"]["interim_dir"]) / "networks" / profile


def out_dir_for(cfg: dict, profile: str) -> Path:
    return Path(cfg["paths"]["interim_dir"]) / "networks" / profile


def run_day(t: int, save: bool = True):
    ds, w = _S["ds"], _S["w"]
    win = ds.window(t, w)
    t0 = time.time()
    net = learn_network(win.to_numpy(dtype="int64"), _S["params"], seed=_S["seed"] + t)
    secs = time.time() - t0
    if save:
        save_network(_S["out_dir"], t, ds.date_of(t).strftime("%Y-%m-%d"), list(win.columns), net)
    return {"t": t, "secs": secs, "v": net.info["v"], "edges": net.info["edges"],
            "score": net.score, "order_acc": net.info["order_acc"], "struct_acc": net.info["struct_acc"]}
