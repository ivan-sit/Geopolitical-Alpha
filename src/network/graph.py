"""
Bipartite network construction and analysis.

Graph structure:
  Left nodes:  Commodities (WTI, Brent, NatGas, HeatOil)
  Right nodes: Energy stocks (tickers)
  Directed edges: commodity → stock (if any HAR horizon beta is non-zero)
  Edge attributes: weight (Σ|β|), dominant_horizon (d/w/m), betas per horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


COMMODITIES = ["WTI", "Brent", "NatGas", "HeatOil"]
HORIZONS = ["d", "w", "m"]
HORIZON_COLORS = {"d": "#E74C3C", "w": "#3498DB", "m": "#2ECC71"}
COMMODITY_COLORS = {
    "WTI":     "#2C3E50",
    "Brent":   "#2980B9",
    "NatGas":  "#E67E22",
    "HeatOil": "#27AE60",
}


def build_bipartite_graph(beta_matrix: pd.DataFrame,
                           threshold: float = 0.0,
                           commodities: list[str] | None = None) -> nx.DiGraph:
    """
    Build a directed bipartite graph from the HAR Lasso beta matrix.

    Args:
        beta_matrix: DataFrame (features × stocks), e.g. from lasso.fit_insample()
                     Index should contain entries like "WTI_d", "Brent_w", etc.
        threshold:   Minimum |β| to draw an edge
        commodities: List of commodity names (default: WTI, Brent, NatGas, HeatOil)

    Returns:
        nx.DiGraph with commodity and stock nodes, edges for non-zero betas
    """
    if commodities is None:
        commodities = COMMODITIES

    G = nx.DiGraph()

    for c in commodities:
        G.add_node(c, bipartite=0, node_type="commodity")
    for s in beta_matrix.columns:
        G.add_node(s, bipartite=1, node_type="stock")

    for commodity in commodities:
        for stock in beta_matrix.columns:
            b = {}
            for h in HORIZONS:
                feat = f"{commodity}_{h}"
                if feat in beta_matrix.index:
                    b[h] = beta_matrix.loc[feat, stock]

            active = {h: v for h, v in b.items() if abs(v) > threshold}
            if active:
                weight = sum(abs(v) for v in active.values())
                dominant = max(active, key=lambda h: abs(active[h]))
                G.add_edge(commodity, stock,
                           weight=weight,
                           betas=b,
                           active_horizons=list(active.keys()),
                           dominant_horizon=dominant)

    return G


def compute_network_statistics(G: nx.DiGraph) -> dict:
    """
    Compute summary statistics for the bipartite commodity → stock graph.

    Returns:
        dict with: n_edges, density, horizon_counts, commodity_degrees,
                   stock_degrees, dominant_horizon
    """
    commodities = [n for n, d in G.nodes(data=True) if d.get("node_type") == "commodity"]
    stocks      = [n for n, d in G.nodes(data=True) if d.get("node_type") == "stock"]

    horizon_counts = {h: 0 for h in HORIZONS}
    dominant_counts = {h: 0 for h in HORIZONS}
    for _, _, data in G.edges(data=True):
        for h in data.get("active_horizons", []):
            horizon_counts[h] += 1
        dom = data.get("dominant_horizon")
        if dom:
            dominant_counts[dom] += 1

    return {
        "n_nodes":         G.number_of_nodes(),
        "n_edges":         G.number_of_edges(),
        "density":         G.number_of_edges() / max(len(commodities) * len(stocks), 1),
        "horizon_counts":  horizon_counts,
        "dominant_counts": dominant_counts,
        "dominant_horizon": max(dominant_counts, key=dominant_counts.get),
        "commodity_outdegree": {c: G.out_degree(c) for c in commodities},
        "stock_indegree":      {s: G.in_degree(s)  for s in stocks},
    }


def visualize_network(G: nx.DiGraph,
                      title: str = "Commodity → Stock Bipartite Network",
                      figsize: tuple[int, int] = (16, 10),
                      output_path: str | None = None) -> plt.Figure:
    """
    Visualize the bipartite graph.

    Layout: commodities on left, stocks on right.
    Edge color = dominant HAR horizon (d=red, w=blue, m=green).
    Edge width ∝ |β| magnitude.
    Stock node size ∝ in-degree (number of commodity predictors).

    Args:
        G:           nx.DiGraph from build_bipartite_graph()
        title:       Figure title
        figsize:     Matplotlib figure size
        output_path: If given, save figure to this path

    Returns:
        Matplotlib Figure object
    """
    commodities = [n for n, d in G.nodes(data=True) if d.get("node_type") == "commodity"]
    stocks      = [n for n, d in G.nodes(data=True) if d.get("node_type") == "stock"]

    fig, ax = plt.subplots(figsize=figsize)

    # Positions
    n_comm  = len(commodities)
    n_stock = len(stocks)
    max_y   = max(n_comm, n_stock)

    pos = {}
    for i, c in enumerate(commodities):
        pos[c] = (0.0, (n_comm - 1 - i) * max_y / max(n_comm - 1, 1))
    for i, s in enumerate(stocks):
        pos[s] = (2.0, (n_stock - 1 - i) * max_y / max(n_stock - 1, 1))

    # Edges
    for u, v, data in G.edges(data=True):
        color = HORIZON_COLORS.get(data.get("dominant_horizon", "d"), "gray")
        width = 0.5 + data.get("weight", 0.0) * 30
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="->", color=color, lw=width,
                            alpha=0.65, connectionstyle="arc3,rad=0.1"),
        )

    # Commodity nodes
    for c in commodities:
        x, y = pos[c]
        circle = plt.Circle((x, y), 0.25, color=COMMODITY_COLORS.get(c, "gray"),
                             ec="black", lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x - 0.4, y, c, ha="right", va="center",
                fontsize=11, fontweight="bold")

    # Stock nodes (size by in-degree)
    for s in stocks:
        x, y = pos[s]
        r = 0.15 + G.in_degree(s) * 0.04
        circle = plt.Circle((x, y), r, color="#E74C3C", ec="black",
                             lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x + 0.35, y, s, ha="left", va="center",
                fontsize=9, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=HORIZON_COLORS["d"], label="Daily dominant"),
        mpatches.Patch(facecolor=HORIZON_COLORS["w"], label="Weekly dominant"),
        mpatches.Patch(facecolor=HORIZON_COLORS["m"], label="Monthly dominant"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=10, framealpha=0.9)

    ax.set_xlim(-1.0, 3.5)
    ax.set_ylim(-1.0, max_y + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "\n(edge color = dominant HAR horizon, width = |β| magnitude)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
