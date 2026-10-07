# hsi-bn-systemic-risk

Reimplementation of **Chan, Chu & So (2023), "A moving-window Bayesian network model for assessing
systemic risk in financial markets", PLOS ONE 18(1): e0279888** on the provided HSI data.

## Stages
| Stage | Status | Script |
|-------|--------|--------|
| 1. Data + preprocessing (prices, log-returns, binary indicators, HSI Loss / abs-return) | done | `scripts/01_preprocess.py` |
| 2. Rolling-window Bayesian network learning (Structure MCMC + Order MCMC, BDeu, M=13, w=30) | done | `scripts/02_learn_networks.py` |
| 3. Network density MND (Eq. 4), topological orders (Kahn, K=100), RM (Eq. 5), NT (Eq. 6), order distance (Eq. 7) | | |
| 4. Granger-causality tests (w_GC=40, L=1..5, alpha=0.1; Eq. 9-10; Table 1) | | |
| 5. Rolling LASSO prediction + tail-event RMSE (m=40, m'=100; Eq. 11-12; Tables 2-3) | | |
| 6. Figures 6-9 and sensitivity appendices (S4-S7) | | |

## Run
```bash
pip install -r requirements.txt
python scripts/01_preprocess.py        # writes data/processed/*
python -m pytest -q                    # sanity tests against the paper's reported counts
```

## Preprocessing decisions (all verified against numbers in the paper)
1. **Universe = 60 stocks of S1 Appendix.** `HSI_constituents.csv` has 103 tickers (every stock that was
   in the HSI at any time 2006-2021). The paper's networks use the 60 listed in Table 4; the other 43 are
   dropped. With these 60, 46 have prices on 2 Jan 2008 and 14 list later, which reproduces the paper's
   "46 to 60 nodes" and the staircase in Fig 7.
2. **Calendar = 3,404 trading days (2 Jan 2008 - 4 Nov 2021).** A date is kept if it is in both files,
   the HSI close is not missing (8 dates), and at least one of the 60 stocks has a price (7 half-day or
   closed-market dates). This gives exactly the paper's 3,404 days and T = 3,403 returns.
3. **Returns / indicators.** R = log P_t - log P_{t-1}; X = 1 if R >= 0 else 0 (Eq. 8). Day 0 (2 Jan 2008)
   only feeds the first return, so t = 1 is 3 Jan 2008.
4. **Gap filling (my assumption; the paper is silent).** ~328 isolated missing prices of already-listed
   stocks (suspensions) are forward-filled, giving R = 0 and X = 1. Leading NaNs (before listing) are never
   filled. Switch off with `forward_fill_gaps: false` in `config/config.yaml`, in which case a stock
   with a gap simply drops out of any window that contains it, per the paper's "variables common to all data
   sets in the window" rule.
5. **Node set per window** `Dataset.window(t, w)` keeps only stocks with a valid indicator on every day of
   D_{(t-w+1):t}. Node count goes 46 -> 60 and is non-decreasing.
6. **HSI series.** Close from `HSI_index.csv`; `abs_ret` = |R_t|; `loss` = -min(R_t, 0).

## Stage 2: learning the networks
```bash
python scripts/02_learn_networks.py --profile light --benchmark   # times 3 days, writes nothing
python scripts/02_learn_networks.py --profile light               # all 3,374 days, resumable
python scripts/02_learn_networks.py --profile faithful            # final run
```
Output: `data/interim/networks/<profile>/net_<t>.npz` (one per day: nodes, adjacency, BDeu score).
Re-running skips finished days. The paper does not specify the sampler settings; the choices (candidate
parents C, BDeu equivalent sample size, iteration counts) are in `config/config.yaml` under `learning`.
Method: top-C candidate parents per node by pairwise BDeu gain; exact parent-set sums inside that space
(max M parents); Order MCMC, then Structure MCMC from the best DAG; keep the highest-BDeu DAG.

## Layout
See `config/config.yaml` for every parameter taken from the paper.
