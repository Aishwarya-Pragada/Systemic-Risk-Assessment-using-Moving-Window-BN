# Methodology and where it lives in the code

Reimplementation of Chan, Chu & So (2023), *A moving-window Bayesian network model for assessing systemic risk in
financial markets*, PLOS ONE 18(1): e0279888.

## 1. Pipeline

| Step | What it does | Paper | Code |
|---|---|---|---|
| Data | 60 HSI stocks (S1 Appendix) + HSI index, 2 Jan 2008 - 4 Nov 2021, 3,404 days, T = 3,403 returns | Data | `data/universe.py`, `data/preprocessing.py` |
| Binary returns | X_it = 1 if R_it >= 0 else 0, R_it = log P_it - log P_i(t-1) | Eq. 8 | `data/preprocessing.py` |
| Responses | Loss_t = -min(R_t, 0); abs. return = abs(R_t) | Eq. 9-10 | `data/preprocessing.py` |
| Rolling windows | D_(t-w+1):t, w = 30, only stocks present on every day of the window | Structural learning | `data/dataset.py` |
| BDeu score | local scores of all parent subsets, at most M = 13 parents | Eq. 1-2, Structural learning | `networks/bdeu.py` |
| Structure learning | Order MCMC + Structure MCMC; highest-BDeu DAG kept | Structural learning (refs 40, 41) | `networks/order_mcmc.py`, `structure_mcmc.py`, `candidates.py`, `learner.py` |
| Modified network density | e_t / (v_t M - M(M+1)/2) | Eq. 4 | `networks/stats.py` |
| Topological orders | Kahn's algorithm from K = 100 random initial orders, median position M_t(X_i) | Topological orders | `indicators/topological.py` |
| Relative order | RM_t = (v_t - M_t) / (v_t - 1) | Eq. 5 | `indicators/order_distance.py` |
| Normalised order | NT_t(V_s) = (v_t - M_t) / sum over V_s | Eq. 6 | `indicators/order_distance.py` |
| Order distance | OD_t = L1 distance of NT on V_(t-1) intersect V_t | Eq. 7 | `indicators/order_distance.py` |
| Granger tests | w_GC = 40, lags 1-5, F-tests, 10% level, H0 / H1a / H1b / H2 | Eq. 9-10, Table 1 | `evaluation/granger.py` |
| LASSO prediction | rolling m = 40, L = 5 (H1a, H1b), L = 3 (H2), one-day ahead | Eq. 9-11, Table 2 | `evaluation/lasso.py` |
| Tail-event RMSE | days whose predicted value exceeds the p-th percentile (99.9, 99, 98) of the last m' = 100 values | Eq. 12, Table 3 | `evaluation/metrics.py` |
| Figures | Figs 6-9; S3 Figs 11-12; S5 Fig 13 | | `viz/figures.py`, `scripts/06`, `07a`, `07b` |

## 2. Parameters taken from the paper
w = 30, M = 13, K = 100, w_GC = 40, lags 1-5, alpha = 0.1, m = 40, L = 5 / 3, m' = 100, percentiles 99.9 / 99 / 98.
All are in `config/config.yaml`.

## 3. Choices the paper does not specify (mine)
| Choice | Value | Why |
|---|---|---|
| Stock universe per window | the 60 stocks of S1 Appendix, entering when they have a full window | reproduces the paper's 46 -> 60 nodes and 3,404 days |
| Gaps in prices | forward-filled for already-listed stocks (return 0, indicator 1) | paper is silent |
| BDeu equivalent sample size | 500 | calibrated so the mean modified density (0.64) falls in the paper's Fig. 8 range of about 0.6-0.8 |
| Search space | 14 candidate parents per node (top pairwise BDeu gain); exact parent-set sums inside it | tractable; exact within the restricted space |
| MCMC length | 2 order chains x 500,000 + 500,000 structure steps | paper gives none |
| LASSO penalty | scikit-learn LassoCV, 5-fold expanding-window CV, 50-point log grid, standardised regressors | paper says "time series cross-validation" |
| Tail rule | the prediction for day t+1 is scored if y_(t+1) exceeds the percentile of y_(t-m'+1)..y_t | only reading that reproduces the paper's Tables 3 and 10 day counts and its H0 RMSE levels |

## 4. Checks against the paper
* Days, returns, node counts, sector sizes: exact.
* Tail-day counts (Tables 3 and 10, m' = 40, 60, 100): exact.
* H0 RMSE levels in Table 2 and all-days RMSE: within about 3%.
* Order distance relation to volatility (S3): reproduced in direction and peak level.

## 5. Known differences
* The order distance is smaller (mean 0.30) and more persistent (lag-1 autocorrelation 0.68) than in the paper
  (about 0.3-0.8 and about 0.35).
* Absolute-return gain from the order distance in Table 2 is about 1%, not 5-7%. The loss gain from both indicators
  (about 2%) is statistically significant on the tail days (paired test p about 0.02-0.05).
* Density alone does not hurt the prediction here, whereas it does in the paper.
* Granger-test counts are similar to the paper's but only modestly above a time-shifted placebo.
* Not done: sensitivity runs over M, w, m (S4, S6) and the Dow Jones study (S8).