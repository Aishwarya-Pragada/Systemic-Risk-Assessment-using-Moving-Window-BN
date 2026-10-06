#!/usr/bin/env python
"""Stage 2: learn the rolling-window Bayesian networks G_w .. G_T (3,374 networks).

Examples
  python scripts/02_learn_networks.py --profile light --benchmark          # time 3 days, no files written
  python scripts/02_learn_networks.py --profile light                      # all days, resumable
  python scripts/02_learn_networks.py --profile faithful --workers 10
Re-running skips days that are already saved, so you can stop (Ctrl+C) and continue later.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnsr.config import load_config                                    # noqa: E402
from bnsr.networks import runner                                       # noqa: E402
from bnsr.networks.learner import warmup                               # noqa: E402
from bnsr.networks.stats import modified_network_density               # noqa: E402
from bnsr.networks.storage import net_path                             # noqa: E402
from bnsr._jit import HAVE_NUMBA                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="light", help="light | faithful (see config/config.yaml)")
    ap.add_argument("--workers", type=int, default=None, help="default: learning.workers in the config")
    ap.add_argument("--start", type=int, default=None, help="first t (default w = 30)")
    ap.add_argument("--end", type=int, default=None, help="last t (default T = 3403)")
    ap.add_argument("--benchmark", action="store_true", help="time 3 days in this process, save nothing")
    ap.add_argument("--force", action="store_true", help="recompute days that already exist")
    a = ap.parse_args()

    if not HAVE_NUMBA:
        sys.exit("numba is not installed - run `pip install numba` first (plain Python would take months).")

    cfg = load_config(a.config)
    w, M = cfg["bayesian_network"]["w"], cfg["bayesian_network"]["M"]
    T = cfg["data"]["expected_returns"]
    t_first, t_last = a.start or w, a.end or T
    workers = a.workers or cfg["learning"]["workers"]
    out_dir = runner.out_dir_for(cfg, a.profile)

    print(f"profile={a.profile}  M={M}  w={w}  days {t_first}..{t_last}  numba=on")
    t0 = time.time()
    warmup()
    print(f"numba warm-up (compile/cache load): {time.time() - t0:.1f}s")

    if a.benchmark:
        runner.init_worker(cfg, a.profile)
        days = sorted({t_first, (t_first + t_last) // 2, t_last})
        res = []
        for t in days:
            r = runner.run_day(t, save=False)
            res.append(r)
            print(f"  t={t:5d} v={r['v']:2d} edges={r['edges']:4d} "
                  f"MND={modified_network_density(r['edges'], r['v'], M):.3f} "
                  f"{r['secs']:.1f}s  order_acc={r['order_acc']:.2f} struct_acc={r['struct_acc']:.2f}")
        mean = sum(r["secs"] for r in res) / len(res)
        n = t_last - t_first + 1
        print(f"mean {mean:.1f}s/day -> {n} days on {workers} workers ~ {mean * n / workers / 3600:.1f} h "
              f"(rough: assumes {workers} full-speed cores)")
        return

    pending = [t for t in range(t_first, t_last + 1) if a.force or not net_path(out_dir, t).exists()]
    print(f"{len(pending)} of {t_last - t_first + 1} days to compute -> {out_dir}")
    if not pending:
        print("nothing to do")
        return

    done, t_start = 0, time.time()
    dens, secs = [], []
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=runner.init_worker,
                                 initargs=(cfg, a.profile)) as ex:
            futs = [ex.submit(runner.run_day, t) for t in pending]
            for f in as_completed(futs):
                r = f.result()
                done += 1
                dens.append(modified_network_density(r["edges"], r["v"], M))
                secs.append(r["secs"])
                if done % 25 == 0 or done == len(pending):
                    el = time.time() - t_start
                    eta = el / done * (len(pending) - done)
                    print(f"[{done:5d}/{len(pending)}] elapsed {el/60:6.1f} min  ETA {eta/60:6.1f} min  "
                          f"mean MND {sum(dens)/len(dens):.3f}  mean {sum(secs)/len(secs):.1f}s/day")
    except KeyboardInterrupt:
        print("\ninterrupted - finished days are saved; re-run the same command to resume")
        sys.exit(1)
    print(f"done in {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
