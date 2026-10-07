"""Figures 6-9 of the paper (matplotlib; each function returns the Figure)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates              # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import networkx as nx                          # noqa: E402
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402

SECTORS = ["Finance", "Information technology", "Commerce", "Properties", "Utilities"]
COLORS = {"Finance": "#66c2a5", "Information technology": "#fc8d62", "Commerce": "#8da0cb",
          "Properties": "#e78ac3", "Utilities": "#a6d854"}
STYLES = {"Finance": "-", "Information technology": (0, (3, 2)), "Commerce": (0, (6, 3)),
          "Properties": (0, (6, 3)), "Utilities": (0, (1, 2))}
FIG6_DATES = [("2008-10-27", "(a) 27 Oct 2008 (global financial crisis)"),
              ("2018-01-26", "(b) 26 Jan 2018 (HSI at its highest)"),
              ("2020-03-23", "(c) 23 Mar 2020 (COVID-19)")]


def moving_average_order(rm: pd.DataFrame, window=60) -> pd.DataFrame:
    """60-day moving average of the relative topological order of every stock."""
    return rm.rolling(window, min_periods=window).mean()


def fig7_sector_order(rm, hsi, ind, sector_of, window=60):
    ma = moving_average_order(rm, window)
    by_sector = {s: ma[[c for c in ma.columns if sector_of[c] == s]].mean(axis=1) for s in SECTORS}
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1]})
    for s in SECTORS:
        ax[0].plot(pd.to_datetime(ma.index), by_sector[s], ls=STYLES[s], color=COLORS[s], lw=1.3, label=s)
    ax[0].set_ylabel("Relative topological order\n(60-day moving average)")
    ax[0].legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax[1].plot(hsi.index, hsi["close"], color="black", lw=0.8)
    ax[1].set_ylabel("HSI closing price")
    ax[2].step(pd.to_datetime(ind["date"]), ind["v"], color="black", lw=1)
    ax[2].set_ylabel("Number of variables")
    ax[2].xaxis.set_major_locator(mdates.YearLocator(2))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Fig 7. Relative topological orders by sector, HSI price and number of nodes", y=0.995)
    fig.tight_layout()
    return fig


def fig8_series(hsi, ind):
    d = hsi.iloc[1:]
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(d.index, d["loss"], color="black", lw=0.5)
    ax[0].set_ylabel("Loss")
    ax[1].plot(d.index, d["abs_ret"], color="black", lw=0.5)
    ax[1].set_ylabel("Absolute return")
    dates = pd.to_datetime(ind["date"])
    ax[2].plot(dates, ind["OD"], color="black", lw=0.5)
    ax[2].set_ylabel("Order distance")
    ax[3].plot(dates, ind["MND"], color="black", lw=0.5)
    ax[3].set_ylabel("Modified network\ndensity")
    ax[3].xaxis.set_major_locator(mdates.YearLocator(1))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("Fig 8. Responses (loss, absolute return) and indicators (order distance, modified network density)",
                 y=0.995)
    fig.tight_layout()
    return fig


def significant(gr: pd.DataFrame, resp, hyp, alpha=0.1, lags=(1, 2, 3, 4, 5)):
    p = gr[[f"{resp}_{hyp}_L{L}" for L in lags]].to_numpy()
    return np.nanmin(np.where(np.isnan(p), np.inf, p), axis=1) < alpha


def fig9_granger(hsi, gr, alpha=0.1):
    fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    dates = pd.to_datetime(gr["date"])
    for r, (resp, name) in enumerate((("loss", "Loss"), ("abs_ret", "Absolute return"))):
        y = hsi.loc[hsi["t"].isin(gr["t"]), resp].to_numpy()
        for c, (hyp, title) in enumerate((("H1a", "Modified network density"), ("H1b", "Order distance"),
                                          ("H2", "Both"))):
            sig = significant(gr, resp, hyp, alpha)
            ax[r, c].plot(dates, y, color="0.7", lw=0.6)
            ax[r, c].scatter(dates[sig], y[sig], s=3, color="royalblue", zorder=3)
            ax[r, c].set_title(f"{title}: {sig.sum()} significant days ({sig.mean():.0%})", fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel(name)
            ax[r, c].grid(alpha=0.3)
    fig.suptitle(f"Fig 9. Days with a significant Granger-causality result (>= 1 lag, {alpha:.0%} level)", y=0.995)
    fig.tight_layout()
    return fig


def fig6_networks(nets, ma_rows, sector_of, titles, seed=7):
    """nets: list of dicts from load_network; ma_rows: list of Series (60-day MA relative order by stock)."""
    fig, axes = plt.subplots(1, len(nets), figsize=(6.2 * len(nets), 6.2))
    axes = np.atleast_1d(axes)
    for ax, net, ma, title in zip(axes, nets, ma_rows, titles):
        nodes, adj = net["nodes"], net["adj"]
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        for j, i in zip(*np.nonzero(adj)):
            G.add_edge(int(j), int(i))
        pos = nx.spring_layout(G.to_undirected(), seed=seed, k=1.6 / np.sqrt(len(nodes)), iterations=200)
        size = np.array([ma.get(s, np.nan) for s in nodes], dtype=float)
        size = np.where(np.isnan(size), np.nanmean(size) if np.isfinite(size).any() else 0.5, size)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=5, width=0.25, alpha=0.35,
                               edge_color="black", node_size=30)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60 + 900 * size ** 2,
                               node_color=[COLORS[sector_of[s]] for s in nodes],
                               edgecolors="0.3", linewidths=0.4)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[s], label=s) for s in SECTORS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Fig 6. Learned networks on three days (node size = 60-day moving average of relative order)",
                 y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig
