#!/usr/bin/env python3
"""Run a full backtest of the Polymarket Momentum Sniper strategy.

Usage:
    # Download data first (one-time):
    python -m backtest.download_data 6

    # Run backtest with default config:
    python -m backtest.run_backtest

    # Run with custom parameters:
    python -m backtest.run_backtest --data backtest/data/BTCUSDT_1m_6mo.csv --bankroll 200

    # Run parameter sweep:
    python -m backtest.run_backtest --sweep
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backtest.download_data import download_binance_candles, load_candles_from_csv
from backtest.replay_engine import BacktestConfig, ReplayEngine
from backtest.report import full_report, print_summary
from logging_db.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtest")


def find_data_file() -> str | None:
    """Find the most recent downloaded data file."""
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        return None
    csvs = sorted(data_dir.glob("BTCUSDT_1m_*.csv"), key=os.path.getmtime, reverse=True)
    return str(csvs[0]) if csvs else None


def run_single(
    data_path: str,
    config: BacktestConfig | None = None,
    db_path: str | None = None,
    export_csv: str | None = None,
):
    """Run a single backtest with the given config."""
    if config is None:
        config = BacktestConfig()

    # Load data
    logger.info(f"Loading candles from {data_path}...")
    candles = load_candles_from_csv(data_path)
    logger.info(f"Loaded {len(candles):,} candles")

    # Database (optional)
    db = None
    if db_path:
        db = Database(db_path)
        db.connect()

    # Run
    engine = ReplayEngine(config=config, db=db)
    start = time.time()
    result = engine.run(candles)
    elapsed = time.time() - start

    logger.info(f"Backtest complete in {elapsed:.1f}s")

    # Report
    full_report(result, export_csv_path=export_csv)

    if db:
        db.close()

    return result


def run_parameter_sweep(data_path: str):
    """Run backtests over a grid of key parameters to find optimal values."""
    logger.info("Loading candles for parameter sweep...")
    candles = load_candles_from_csv(data_path)
    logger.info(f"Loaded {len(candles):,} candles")

    # Parameter grid
    sweeps = {
        "max_adjustment": [0.10, 0.15, 0.20, 0.25],
        "min_edge": [0.01, 0.02, 0.03, 0.05],
        "max_edge": [0.05, 0.08, 0.10, 0.15],
        "kelly_multiplier": [0.10, 0.15, 0.25, 0.35],
        "oracle_lag_mean_sec": [15.0, 30.0, 45.0, 60.0, 90.0],
        "market_efficiency": [0.15, 0.25, 0.35, 0.50, 0.65],
    }

    print()
    print("=" * 90)
    print("  PARAMETER SWEEP")
    print("=" * 90)

    # Baseline
    baseline_cfg = BacktestConfig()
    engine = ReplayEngine(config=baseline_cfg)
    baseline = engine.run(candles)
    print(f"\n  BASELINE: Trades={baseline.trades_placed} WR={baseline.win_rate:.1%} "
          f"P&L=${baseline.total_pnl:+.2f} ROI={baseline.roi:+.1%} "
          f"MaxDD={baseline.max_drawdown:.1%}")

    best_overall = {"config": "baseline", "pnl": baseline.total_pnl, "wr": baseline.win_rate}

    for param_name, values in sweeps.items():
        print(f"\n  -- Sweeping: {param_name} --")
        print(f"  {'Value':>10s}  {'Trades':>6s}  {'WR':>6s}  {'P&L':>10s}  {'ROI':>8s}  {'MaxDD':>6s}  {'AvgPnL':>8s}")

        for val in values:
            cfg = BacktestConfig()
            setattr(cfg, param_name, val)
            engine = ReplayEngine(config=cfg)
            r = engine.run(candles)

            marker = " <-- best" if r.total_pnl > best_overall["pnl"] else ""
            print(
                f"  {val:>10}  "
                f"{r.trades_placed:>6d}  "
                f"{r.win_rate:>5.1%}  "
                f"${r.total_pnl:>+9.2f}  "
                f"{r.roi:>+7.1%}  "
                f"{r.max_drawdown:>5.1%}  "
                f"${r.avg_pnl_per_trade:>+7.4f}"
                f"{marker}"
            )

            if r.total_pnl > best_overall["pnl"]:
                best_overall = {
                    "config": f"{param_name}={val}",
                    "pnl": r.total_pnl,
                    "wr": r.win_rate,
                }

    print(f"\n  BEST: {best_overall['config']} -> "
          f"P&L=${best_overall['pnl']:+.2f}, WR={best_overall['wr']:.1%}")
    print()


def run_multi_sweep(data_path: str):
    """Run a focused multi-parameter optimisation."""
    logger.info("Loading candles for multi-parameter sweep...")
    candles = load_candles_from_csv(data_path)
    logger.info(f"Loaded {len(candles):,} candles")

    # Key combinations to test — all use realistic defaults
    configs = [
        ("Baseline", BacktestConfig()),
        ("Conservative", BacktestConfig(
            max_adjustment=0.10, min_edge=0.03, max_edge=0.12,
            kelly_multiplier=0.15,
        )),
        ("Aggressive Edge", BacktestConfig(
            max_adjustment=0.20, min_edge=0.01, max_edge=0.08,
            kelly_multiplier=0.25,
        )),
        ("High Confidence", BacktestConfig(
            max_adjustment=0.15, min_edge=0.05, max_edge=0.15,
            kelly_multiplier=0.35,
        )),
        ("Fast Oracle", BacktestConfig(
            oracle_lag_mean_sec=20.0, oracle_lag_std_sec=10.0,
        )),
        ("Slow Oracle", BacktestConfig(
            oracle_lag_mean_sec=75.0, oracle_lag_std_sec=40.0,
        )),
        ("Wide Adjustment", BacktestConfig(
            max_adjustment=0.25,
        )),
        ("Tight Stops", BacktestConfig(
            daily_loss_cap_pct=0.10, streak_reduce_at=2,
        )),
        ("Efficient Mkt", BacktestConfig(
            market_efficiency=0.55,
        )),
        ("Dumb Market", BacktestConfig(
            market_efficiency=0.15,
        )),
        ("Wide Spread", BacktestConfig(
            market_base_spread=0.10,
        )),
        ("Worst Case", BacktestConfig(
            market_efficiency=0.55, market_base_spread=0.08,
            oracle_lag_mean_sec=75.0, fill_rate_gtc=0.70, fill_rate_fok=0.55,
        )),
    ]

    print()
    print("=" * 100)
    print("  MULTI-PARAMETER COMPARISON")
    print("=" * 100)
    print(f"  {'Name':>20s}  {'Trades':>6s}  {'WR':>6s}  {'P&L':>10s}  {'ROI':>8s}  "
          f"{'MaxDD':>6s}  {'AvgPnL':>8s}  {'Skip%':>6s}")

    for name, cfg in configs:
        engine = ReplayEngine(config=cfg)
        r = engine.run(candles)
        print(
            f"  {name:>20s}  "
            f"{r.trades_placed:>6d}  "
            f"{r.win_rate:>5.1%}  "
            f"${r.total_pnl:>+9.2f}  "
            f"{r.roi:>+7.1%}  "
            f"{r.max_drawdown:>5.1%}  "
            f"${r.avg_pnl_per_trade:>+7.4f}  "
            f"{r.skip_rate:>5.1%}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest the Polymarket Momentum Sniper")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument("--download", type=int, default=0, help="Download N months of data first")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Initial bankroll")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--multi", action="store_true", help="Run multi-parameter comparison")
    parser.add_argument("--db", type=str, help="SQLite path to log backtest trades")
    parser.add_argument("--export", type=str, help="Export trades to CSV")
    parser.add_argument("--max-adjustment", type=float, default=0.15)
    parser.add_argument("--min-edge", type=float, default=0.02)
    parser.add_argument("--max-edge", type=float, default=0.10)
    parser.add_argument("--kelly", type=float, default=0.25)
    parser.add_argument("--oracle-lag-sec", type=float, default=45.0, help="Mean oracle lag in seconds")
    parser.add_argument("--market-efficiency", type=float, default=0.35, help="Market efficiency 0-1")
    args = parser.parse_args()

    # Download if requested
    if args.download > 0:
        data_path = download_binance_candles(months_back=args.download)
    elif args.data:
        data_path = args.data
    else:
        data_path = find_data_file()
        if not data_path:
            print("No data file found. Run with --download 6 first.")
            print("  python -m backtest.run_backtest --download 6")
            sys.exit(1)

    print(f"Using data: {data_path}")

    if args.sweep:
        run_parameter_sweep(data_path)
    elif args.multi:
        run_multi_sweep(data_path)
    else:
        config = BacktestConfig(
            initial_bankroll=args.bankroll,
            max_adjustment=args.max_adjustment,
            min_edge=args.min_edge,
            max_edge=args.max_edge,
            kelly_multiplier=args.kelly,
            oracle_lag_mean_sec=args.oracle_lag_sec,
            market_efficiency=args.market_efficiency,
        )
        run_single(
            data_path=data_path,
            config=config,
            db_path=args.db,
            export_csv=args.export,
        )


if __name__ == "__main__":
    main()
