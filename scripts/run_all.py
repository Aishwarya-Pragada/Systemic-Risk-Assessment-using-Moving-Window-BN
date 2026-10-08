#!/usr/bin/env python
"""Run the whole pipeline in order (every stage is resumable / reuses saved results).

  python scripts/run_all.py --profile faithful                 # everything
  python scripts/run_all.py --profile faithful --from-stage 3  # skip data + network learning
  python scripts/run_all.py --dry-run                          # just print the commands

Stages: 1 preprocess | 2 learn networks | 3 indicators | 4 Granger tests | 5 LASSO prediction |
        6 figures | 7 S3 volatility link
Typical run time on a 12-core laptop: 1: seconds, 2: ~30 min, 3: ~1 min, 4: ~20 s, 5: ~15-20 min, 6-7: seconds."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(a):
    py, S = sys.executable, ROOT / "scripts"
    c5 = [py, str(S / "05_lasso_prediction.py"), "--profile",
          a.profile] + ([] if a.refit else ["--reuse"])
    if a.workers:
        c5 += ["--workers", str(a.workers)]
    c2 = [py, str(S / "02_learn_networks.py"), "--profile", a.profile]
    if a.workers:
        c2 += ["--workers", str(a.workers)]
    return {
        1: ("preprocess", [py, str(S / "01_preprocess.py")]),
        2: ("learn networks", c2),
        3: ("indicators", [py, str(S / "03_compute_indicators.py"), "--profile", a.profile]),
        4: ("Granger tests", [py, str(S / "04_granger_tests.py"), "--profile", a.profile]),
        5: ("LASSO prediction", c5),
        6: ("figures", [py, str(S / "06_make_figures.py"), "--profile", a.profile]),
        7: ("S3 volatility link", [py, str(S / "07a_volatility_link.py"), "--profile", a.profile]),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="faithful")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--to-stage", type=int, default=7)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--refit", action="store_true",
                    help="stage 5: refit the LASSO models even if saved")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmds = commands(a)
    for k in range(a.from_stage, a.to_stage + 1):
        name, cmd = cmds[k]
        print(
            f"\n===== stage {k}: {name} =====\n$ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        print(
            f"----- stage {k} finished in {(time.time() - t0) / 60:.1f} min (exit code {r.returncode})")
        if r.returncode != 0:
            sys.exit(
                f"stage {k} failed - fix it and re-run with --from-stage {k}")


if __name__ == "__main__":
    main()
