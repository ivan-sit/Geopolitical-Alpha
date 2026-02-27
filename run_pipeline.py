#!/usr/bin/env python3
"""
Geopolitical Alpha Pipeline
============================
Run the full pipeline from raw data to results with a single command:

    python run_pipeline.py

Options
-------
    --fast          Skip rolling OOS step (use in-sample only, for quick checks)
    --notebooks     Execute Jupyter notebooks (slower but produces full HTML output)
    --output DIR    Directory to save outputs (default: outputs/)
    --data PATH     Path to CRSP data directory (auto-detects format)
    --start DATE    Start date (default: 2000-01-01)
    --end DATE      End date (default: 2024-12-31)

Example
-------
    # Quick run (in-sample only, no notebooks):
    python run_pipeline.py --fast

    # Full pipeline with notebook execution:
    python run_pipeline.py --notebooks

    # Use new CRSP per-date format:
    python run_pipeline.py --data /path/to/crsp_perdate/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless plotting
import matplotlib.pyplot as plt

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.loader import (
    load_crsp_matrix, load_crsp_auto,
    load_commodity_futures, load_market_proxy,
)
from src.features.har import (
    build_full_feature_matrix, residualize_against_market,
)
from src.models.lasso import fit_insample, rolling_predict
from src.network.graph import (
    build_bipartite_graph, compute_network_statistics, visualize_network,
)
from src.strategy.backtest import (
    generate_signals, run_backtest, compute_metrics, rolling_sharpe,
)

# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Geopolitical Alpha Pipeline")
    p.add_argument("--fast",      action="store_true", help="Skip rolling OOS")
    p.add_argument("--notebooks", action="store_true", help="Execute Jupyter notebooks")
    p.add_argument("--output",    default="outputs",   help="Output directory")
    p.add_argument("--data",      default="US_CRSP_NYSE/Matrix_Format_SubsetUniverse",
                   help="CRSP data directory (matrix or per-date format)")
    p.add_argument("--sectors",   default="US_CRSP_NYSE/Sectors/Sectors_SP500_YahooNWikipedia.csv",
                   help="Sectors CSV (for matrix format)")
    p.add_argument("--start",     default="2000-01-01")
    p.add_argument("--end",       default="2024-12-31")
    p.add_argument("--train-window", type=int, default=252,
                   help="Rolling training window (days)")
    p.add_argument("--lasso-alpha",  type=float, default=0.001,
                   help="Lasso regularization alpha")
    p.add_argument("--tc",           type=float, default=0.0005,
                   help="Transaction cost one-way (default 5bps)")
    return p.parse_args()


def banner(msg: str):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def step(n: int, msg: str):
    print(f"\n[{n}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────


def run(args):
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    banner("Geopolitical Alpha — HAR-Lasso Bipartite Pipeline")
    print(f"  Data:    {args.data}")
    print(f"  Period:  {args.start} → {args.end}")
    print(f"  Output:  {out}/")

    # ── 1. Load CRSP stock returns ────────────────────────────────────────
    step(1, "Loading CRSP stock returns")
    data_path = ROOT / args.data
    sectors_path = ROOT / args.sectors if Path(ROOT / args.sectors).exists() else None

    stock_ret = load_crsp_auto(
        data_path,
        sector_csv=sectors_path,
        sector="Energy",
    )
    equity_calendar = stock_ret.index
    print(f"   {stock_ret.shape[1]} energy stocks × {len(equity_calendar)} trading days")

    # ── 2. Load & align commodity futures ─────────────────────────────────
    step(2, "Loading commodity futures (yfinance, forward-filling to equity calendar)")
    comm_prices = load_commodity_futures(equity_calendar, start=args.start, end=args.end)

    # ── 3. Load market proxy (SPY) ─────────────────────────────────────────
    step(3, "Loading market proxy (SPY) for residualization")
    spy_prices = load_market_proxy(equity_calendar, start=args.start, end=args.end)
    spy_ret = spy_prices.pct_change(fill_method=None)

    # ── 4. Build feature matrix ────────────────────────────────────────────
    step(4, "Building HAR + derived features")
    X_full = build_full_feature_matrix(comm_prices, clip=1.0)
    print(f"   Feature matrix: {X_full.shape[0]} days × {X_full.shape[1]} features")

    # ── 5. Market residualization ──────────────────────────────────────────
    step(5, "Market residualization (rolling 252d beta vs SPY)")
    common_dates = X_full.index.intersection(stock_ret.index).intersection(spy_ret.dropna().index)
    X_aligned   = X_full.loc[common_dates]
    Y_raw_all   = stock_ret.loc[common_dates]
    spy_aligned  = spy_ret.loc[common_dates]

    Y_idio_all = residualize_against_market(Y_raw_all, spy_aligned, window=252)

    # Drop burn-in
    valid = Y_idio_all.dropna(how="all").index
    X = X_aligned.loc[valid]
    Y_idio = Y_idio_all.loc[valid]
    Y_raw  = Y_raw_all.loc[valid]
    print(f"   After burn-in: {len(valid)} days  ({valid[0].date()} → {valid[-1].date()})")

    # ── 6. In-sample Lasso (for network visualization) ────────────────────
    step(6, "Fitting in-sample LassoCV (bipartite network)")
    beta_matrix, is_models = fit_insample(X, Y_idio)
    avg_r2 = np.mean([
        is_models[s].score(
            __import__("sklearn.preprocessing", fromlist=["StandardScaler"])
            .StandardScaler().fit_transform(X.values),
            Y_idio[s].values
        ) for s in is_models
    ]) if is_models else 0
    nz = (beta_matrix != 0).sum().sum()
    print(f"   Non-zero betas: {nz}/{beta_matrix.size}  "
          f"({100*nz/beta_matrix.size:.1f}%)")
    beta_matrix.to_csv(out / "beta_matrix.csv")

    # ── 7. Bipartite network ───────────────────────────────────────────────
    step(7, "Building bipartite graph")
    har_feat_names = [c for c in X.columns
                      if any(c.startswith(cm) for cm in ["WTI", "Brent", "NatGas", "HeatOil"])
                      and c.endswith(("_d", "_w", "_m"))]
    beta_har = beta_matrix.loc[har_feat_names]
    G = build_bipartite_graph(beta_har)
    stats = compute_network_statistics(G)
    print(f"   Nodes: {stats['n_nodes']}  Edges: {stats['n_edges']}  "
          f"Density: {stats['density']:.1%}")
    fig = visualize_network(G, output_path=str(out / "bipartite_network.png"))
    plt.close(fig)
    print(f"   Saved → {out}/bipartite_network.png")

    # ── 8. Rolling OOS predictions ────────────────────────────────────────
    if not args.fast:
        step(8, f"Rolling OOS Lasso (window={args.train_window}d, alpha={args.lasso_alpha})")
        oos_pred = rolling_predict(X, Y_idio,
                                   train_window=args.train_window,
                                   alpha=args.lasso_alpha)
        oos_pred.to_csv(out / "oos_predictions.csv")
    else:
        print("[8] Skipped (--fast mode) — using zeros as predictions")
        oos_pred = pd.DataFrame(0.0, index=X.index, columns=Y_idio.columns)

    # ── 9. Backtest ────────────────────────────────────────────────────────
    step(9, "Running long-short backtest")
    signals   = generate_signals(oos_pred, top_pct=0.25, bottom_pct=0.25)
    strat_ret = run_backtest(signals, Y_raw, transaction_cost=args.tc)
    bench_ret = Y_raw.mean(axis=1)

    oos_start = oos_pred.dropna(how="all").index[0] if not args.fast else X.index[0]
    strat_m = compute_metrics(strat_ret, start=oos_start)
    bench_m = compute_metrics(bench_ret, start=oos_start)

    results = pd.DataFrame({"Strategy L/S": strat_m, "Benchmark EW": bench_m})

    banner("RESULTS")
    print(results.to_string())

    results.to_csv(out / "backtest_results.csv")

    # ── 10. Cumulative returns plot ────────────────────────────────────────
    strat_cum = (1 + strat_ret.loc[oos_start:].fillna(0)).cumprod()
    bench_cum = (1 + bench_ret.loc[oos_start:]).cumprod()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(strat_cum, color="steelblue", lw=1.5, label="Strategy L/S")
    ax.plot(bench_cum, color="gray",      lw=1.2, label="Benchmark EW", alpha=0.7)
    ax.axhline(1, color="black", ls="--", lw=0.5)
    ax.set_title("Cumulative Returns: HAR-Lasso Strategy vs Benchmark", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out}/cumulative_returns.png")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    # ── 11. Optional: execute notebooks ───────────────────────────────────
    if args.notebooks:
        _run_notebooks(out)

    return results


def _run_notebooks(out: Path):
    """Execute all project notebooks via nbconvert, saving HTML outputs."""
    try:
        import subprocess
    except ImportError:
        print("Cannot import subprocess — skipping notebook execution.")
        return

    notebooks = sorted((ROOT / "notebooks").glob("[0-9]*.ipynb"))
    print(f"\n[Notebooks] Executing {len(notebooks)} notebooks...")

    for nb in notebooks:
        nb_out = out / (nb.stem + ".html")
        print(f"  Running {nb.name}...", end=" ", flush=True)
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "html",
             "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=600",
             str(nb)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("done")
        else:
            print(f"FAILED\n{result.stderr[:300]}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
