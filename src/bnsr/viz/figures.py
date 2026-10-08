"""Figures 6-9 of the paper (matplotlib; each function returns the Figure)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates              # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import networkx as nx                          # noqa: E402
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402

SECTORS = ["Finance", "Information technology",
           "Commerce", "Properties", "Utilities"]
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
    by_sector = {s: ma[[c for c in ma.columns if sector_of[c] == s]].mean(
        axis=1) for s in SECTORS}
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={
                           "height_ratios": [2.2, 1.4, 1]})
    for s in SECTORS:
        ax[0].plot(pd.to_datetime(ma.index), by_sector[s],
                   ls=STYLES[s], color=COLORS[s], lw=1.3, label=s)
    ax[0].set_ylabel("Relative topological order\n(60-day moving average)")
    ax[0].legend(ncol=5, fontsize=8, loc="upper center",
                 bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax[1].plot(hsi.index, hsi["close"], color="black", lw=0.8)
    ax[1].set_ylabel("HSI closing price")
    ax[2].step(pd.to_datetime(ind["date"]), ind["v"], color="black", lw=1)
    ax[2].set_ylabel("Number of variables")
    ax[2].xaxis.set_major_locator(mdates.YearLocator(2))
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(
        "Fig 7. Relative topological orders by sector, HSI price and number of nodes", y=0.995)
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
            ax[r, c].scatter(dates[sig], y[sig], s=3,
                             color="royalblue", zorder=3)
            ax[r, c].set_title(
                f"{title}: {sig.sum()} significant days ({sig.mean():.0%})", fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel(name)
            ax[r, c].grid(alpha=0.3)
    fig.suptitle(
        f"Fig 9. Days with a significant Granger-causality result (>= 1 lag, {alpha:.0%} level)", y=0.995)
    fig.tight_layout()
    return fig


# Sector regions of the paper's Fig. 6 (unit square, y up): finance top-left, utilities top-middle, other commerce
# bottom-left, properties right, information technology in between.
ANCHORS = {"Finance": (0.20, 0.80), "Utilities": (0.60, 0.80), "Commerce": (0.22, 0.28),
           "Properties": (0.80, 0.42), "Information technology": (0.72, 0.64)}
LABELS = {"Commerce": "Other commerce"}


def _spread(xy, min_dist, radius, iters=150):
    """Push apart nodes that are closer than min_dist (keeps the sector blob compact and the circles visible)."""
    xy = xy.copy()
    rng = np.random.default_rng(0)
    xy += 1e-4 * rng.standard_normal(xy.shape)
    for _ in range(iters):
        d = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + np.eye(len(xy))
        push = np.where((dist < min_dist) & (dist > 0),
                        (min_dist - dist) / dist, 0.0)
        np.fill_diagonal(push, 0.0)
        xy += 0.5 * (push[:, :, None] * d).sum(axis=1) / 2
        norm = np.sqrt((xy ** 2).sum(-1, keepdims=True))
        xy = np.where(norm > radius * 1.15, xy / norm * radius * 1.15, xy)
    return xy


def sector_layout(nodes, adj, sector_of, seed=7):
    """Sector-clustered layout: every sector is a blob around its anchor, nodes inside a blob are placed by a
    force-directed layout of the edges within that sector."""
    pos = {}
    for sec in SECTORS:
        idx = [k for k, s_ in enumerate(nodes) if sector_of[s_] == sec]
        if not idx:
            continue
        ax_, ay_ = ANCHORS[sec]
        radius = 0.045 * np.sqrt(len(idx)) + 0.02
        if len(idx) == 1:
            pos[idx[0]] = np.array([ax_, ay_])
            continue
        H = nx.Graph()
        H.add_nodes_from(idx)
        sub = adj[np.ix_(idx, idx)]
        for a, b in zip(*np.nonzero(sub | sub.T)):
            H.add_edge(idx[a], idx[b])
        p = nx.spring_layout(H, seed=seed, iterations=100)
        xy = np.array([p[k] for k in idx])
        xy = xy - xy.mean(axis=0)
        xy = xy / max(np.abs(xy).max(), 1e-9) * radius
        xy = _spread(xy, min_dist=min(0.032, 1.6 * radius /
                     np.sqrt(len(idx))), radius=radius)
        for k, (x, y) in zip(idx, xy):
            pos[k] = np.array([ax_ + x, ay_ + y])
    return pos


def fig6_networks(nets, ma_rows, sector_of, titles, seed=7):
    """nets: list of dicts from load_network; ma_rows: list of Series (60-day MA relative order by stock).
    Layout as in the paper: (a) top-left, (b) bottom-left, (c) bottom-right, legend top-right."""
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.06, wspace=0.04)
    cells = [gs[0, 0], gs[1, 0], gs[1, 1]]
    for cell, net, ma, title in zip(cells, nets, ma_rows, titles):
        ax = fig.add_subplot(cell)
        nodes, adj = net["nodes"], net["adj"]
        pos = sector_layout(nodes, adj, sector_of, seed)
        size = np.array([ma.get(s, np.nan) for s in nodes], dtype=float)
        size = np.where(np.isnan(size), np.nanmean(
            size) if np.isfinite(size).any() else 0.5, size)
        node_size = 30 + 2200 * np.clip(size, 0, 1) ** 3
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        for j, i in zip(*np.nonzero(adj)):
            G.add_edge(int(j), int(i))
        # node_size is passed to the edge drawing so that arrows stop at the circle's border
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=7, width=0.35,
                               alpha=0.65, edge_color="0.15", node_size=node_size, min_source_margin=0,
                               min_target_margin=0)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=[COLORS[sector_of[s]] for s in nodes],
                               edgecolors="0.25", linewidths=0.5)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=COLORS[s], label=LABELS.get(s, s))
               for s in SECTORS]
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11)
    lax.text(0.0, 0.35, "Arrows point from a stock to the stocks that depend on it\n"
                        "(parent -> child in the learned DAG).\nNode size = 60-day moving average of the\n"
                        "relative topological order (larger = nearer the head of the order).",
             fontsize=9, va="top", transform=lax.transAxes)
    return fig
