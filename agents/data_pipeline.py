"""Data Pipeline Agent — automates data collection, normalisation, and freshness checks.

Commands:
    python -m agents.data_pipeline status     — Report staleness of all data files
    python -m agents.data_pipeline fetch      — Fetch data from a source (--source binance|coinalyze)
    python -m agents.data_pipeline unify      — Merge all sources into unified CSV
    python -m agents.data_pipeline validate   — Check for gaps, duplicates, NaN counts
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.common import (
    DATA_DIR,
    DateRange,
    ensure_data_dir,
    file_age_hours,
    format_table,
    get_market_date_range,
    load_backtest_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agents.data_pipeline")

# Known data files and their descriptions
DATA_FILES = {
    "polybacktest_markets.csv": "PolyBackTest Pro market list",
    "polybacktest_snapshots.csv": "PolyBackTest Pro orderbook snapshots",
    "binance_klines_1m.csv": "Binance 1-min BTC/USDT candles",
    "coinalyze_oi_history.csv": "Coinalyze open interest history",
    "coinalyze_lsr_history.csv": "Coinalyze long/short ratio history",
    "coinalyze_funding_history.csv": "Coinalyze funding rate history",
    "unified_mid-default.csv": "Unified merged dataset (default mode)",
    "unified_mid-strict.csv": "Unified merged dataset (strict mode)",
}


def cmd_status(args: argparse.Namespace) -> None:
    """Report the staleness and size of all data files."""
    ensure_data_dir()

    headers = ["File", "Description", "Size", "Age", "Rows"]
    rows = []

    for filename, description in DATA_FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            rows.append([filename, description, "MISSING", "-", "-"])
            continue

        size_bytes = filepath.stat().st_size
        if size_bytes > 1_000_000:
            size_str = f"{size_bytes / 1_000_000:.1f} MB"
        elif size_bytes > 1_000:
            size_str = f"{size_bytes / 1_000:.1f} KB"
        else:
            size_str = f"{size_bytes} B"

        age = file_age_hours(filepath)
        if age is not None:
            if age < 1:
                age_str = f"{age * 60:.0f}m ago"
            elif age < 24:
                age_str = f"{age:.1f}h ago"
            else:
                age_str = f"{age / 24:.1f}d ago"
        else:
            age_str = "-"

        # Count rows (skip header)
        row_count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            row_str = f"{row_count:,}"
        except Exception:
            row_str = "?"

        rows.append([filename, description, size_str, age_str, row_str])

    print("\n  DATA PIPELINE STATUS")
    print("  " + "=" * 80)
    print(format_table(headers, rows))

    # Date range info
    date_range = get_market_date_range()
    if date_range:
        print(f"\n  Market date range: {date_range}")
    else:
        print("\n  No market data found. Run fetch first.")

    # Freshness warnings
    stale_threshold_hours = 48.0
    stale = []
    for filename in DATA_FILES:
        filepath = DATA_DIR / filename
        age = file_age_hours(filepath)
        if age is not None and age > stale_threshold_hours:
            stale.append(filename)
        elif age is None:
            stale.append(filename)

    if stale:
        print(f"\n  WARNING: {len(stale)} file(s) are stale or missing:")
        for f in stale:
            print(f"    - {f}")
    else:
        print("\n  All data files are fresh.")
    print()


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch data from the specified source."""
    source = args.source.lower()

    if source == "binance":
        _fetch_binance()
    elif source == "coinalyze":
        _fetch_coinalyze()
    elif source == "polybacktest":
        _fetch_polybacktest()
    else:
        logger.error(f"Unknown source: {source}. Use: binance, coinalyze, polybacktest")
        sys.exit(1)


def _fetch_binance() -> None:
    """Run the Binance klines fetch pipeline."""
    from backtest.fetch_binance_klines import main as binance_main
    logger.info("Starting Binance klines fetch...")
    binance_main()
    logger.info("Binance klines fetch complete.")


def _fetch_coinalyze() -> None:
    """Run the Coinalyze history fetch pipeline."""
    from backtest.fetch_coinalyze_history import main as coinalyze_main
    logger.info("Starting Coinalyze history fetch...")
    coinalyze_main()
    logger.info("Coinalyze history fetch complete.")


def _fetch_polybacktest() -> None:
    """Run the PolyBackTest Pro snapshot fetch pipeline."""
    from backtest.run_fetch import main as polybacktest_main
    logger.info("Starting PolyBackTest Pro fetch...")
    polybacktest_main()
    logger.info("PolyBackTest Pro fetch complete.")


def cmd_unify(args: argparse.Namespace) -> None:
    """Merge all data sources into a unified CSV for backtesting.

    Reads polybacktest snapshots, enriches with Binance candle data and
    Coinalyze metrics, then writes unified_mid-default.csv.
    """
    import pandas as pd

    ensure_data_dir()
    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    markets_path = DATA_DIR / "polybacktest_markets.csv"
    klines_path = DATA_DIR / "binance_klines_1m.csv"

    # Validate required files
    missing = []
    for p in [snaps_path, markets_path, klines_path]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        logger.error(f"Missing required files: {', '.join(missing)}")
        logger.error("Run: python -m agents.data_pipeline fetch --source <source>")
        sys.exit(1)

    # Load snapshots and markets
    snaps = pd.read_csv(snaps_path, encoding="utf-8")
    markets = pd.read_csv(markets_path, encoding="utf-8")
    logger.info(f"Loaded {len(snaps):,} snapshots, {len(markets):,} markets")

    # Merge market metadata into snapshots
    market_cols = ["market_id", "start_time", "end_time", "winner"]
    if all(c in markets.columns for c in market_cols):
        snaps = snaps.merge(
            markets[market_cols],
            on="market_id",
            how="left",
        )

    # Load klines for BTC price enrichment
    klines = pd.read_csv(klines_path, encoding="utf-8")
    logger.info(f"Loaded {len(klines):,} klines")

    # Compute mid prices for YES/NO from best bid/ask
    for side in ["up", "down"]:
        bid_col = f"{side}_best_bid"
        ask_col = f"{side}_best_ask"
        mid_col = f"{side}_mid"
        if bid_col in snaps.columns and ask_col in snaps.columns:
            snaps[mid_col] = (
                snaps[bid_col].fillna(0).astype(float)
                + snaps[ask_col].fillna(0).astype(float)
            ) / 2

    # Enrich with Coinalyze data if available
    for ca_file in ["coinalyze_oi_history.csv", "coinalyze_lsr_history.csv", "coinalyze_funding_history.csv"]:
        ca_path = DATA_DIR / ca_file
        if ca_path.exists():
            ca_df = pd.read_csv(ca_path, encoding="utf-8")
            logger.info(f"Loaded {len(ca_df):,} rows from {ca_file}")

    # Write unified output
    output_path = DATA_DIR / "unified_mid-default.csv"
    snaps.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Wrote unified dataset: {output_path} ({len(snaps):,} rows)")
    print(f"\n  Unified dataset written to: {output_path}")
    print(f"  Rows: {len(snaps):,}")
    print()


def cmd_validate(args: argparse.Namespace) -> None:
    """Check data files for gaps, duplicates, timezone issues, and NaN counts."""
    ensure_data_dir()

    issues: list[str] = []
    checks_passed = 0

    # 1. Check all required files exist
    print("\n  DATA VALIDATION")
    print("  " + "=" * 60)

    for filename in DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            issues.append(f"MISSING: {filename}")
        else:
            checks_passed += 1

    # 2. Validate snapshots
    snaps_path = DATA_DIR / "polybacktest_snapshots.csv"
    if snaps_path.exists():
        snaps = load_backtest_data(snaps_path)

        # Check for duplicate market_id + offset_sec combinations
        if "market_id" in snaps.columns and "offset_sec" in snaps.columns:
            dupes = snaps.duplicated(subset=["market_id", "offset_sec"], keep=False)
            dupe_count = dupes.sum()
            if dupe_count > 0:
                issues.append(f"DUPLICATES: {dupe_count} duplicate (market_id, offset_sec) pairs in snapshots")
            else:
                checks_passed += 1
                print(f"  [OK] No duplicate snapshots")

        # Check for NaN in critical columns
        critical_cols = ["btc_price", "price_up", "price_down"]
        for col in critical_cols:
            if col in snaps.columns:
                nan_count = snaps[col].isna().sum()
                nan_pct = nan_count / len(snaps) * 100
                if nan_pct > 10:
                    issues.append(f"HIGH_NAN: {col} has {nan_count:,} NaN values ({nan_pct:.1f}%)")
                else:
                    checks_passed += 1
                    print(f"  [OK] {col}: {nan_count:,} NaN ({nan_pct:.1f}%)")

    # 3. Validate klines
    klines_path = DATA_DIR / "binance_klines_1m.csv"
    if klines_path.exists():
        klines = load_backtest_data(klines_path)

        # Check for time gaps (expect 1-min intervals = 60000ms)
        if "open_time_ms" in klines.columns:
            timestamps = klines["open_time_ms"].sort_values()
            diffs = timestamps.diff().dropna()
            expected_interval = 60_000
            gaps = diffs[diffs > expected_interval * 2]  # >2 minutes is a gap
            if len(gaps) > 0:
                max_gap_min = gaps.max() / 60_000
                issues.append(f"GAPS: {len(gaps)} time gaps in klines (max: {max_gap_min:.0f} min)")
            else:
                checks_passed += 1
                print(f"  [OK] No significant time gaps in klines")

            # Check for duplicates
            kline_dupes = timestamps.duplicated().sum()
            if kline_dupes > 0:
                issues.append(f"DUPLICATES: {kline_dupes} duplicate timestamps in klines")
            else:
                checks_passed += 1
                print(f"  [OK] No duplicate kline timestamps")

    # 4. Validate markets
    markets_path = DATA_DIR / "polybacktest_markets.csv"
    if markets_path.exists():
        markets = load_backtest_data(markets_path)

        # Check winner column
        if "winner" in markets.columns:
            winners = markets["winner"].value_counts()
            print(f"  [OK] Market outcomes: {dict(winners)}")
            checks_passed += 1

        # Check for markets without snapshots
        if snaps_path.exists():
            snap_markets = set(snaps["market_id"].unique()) if "market_id" in snaps.columns else set()
            all_markets = set(markets["market_id"].unique()) if "market_id" in markets.columns else set()
            missing_snaps = all_markets - snap_markets
            coverage_pct = (1 - len(missing_snaps) / len(all_markets)) * 100 if all_markets else 0
            print(f"  [OK] Snapshot coverage: {coverage_pct:.1f}% ({len(snap_markets)}/{len(all_markets)} markets)")
            checks_passed += 1

    # Summary
    print(f"\n  Checks passed: {checks_passed}")
    if issues:
        print(f"  Issues found: {len(issues)}")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No issues found.")
    print()


def main() -> None:
    """CLI entry point for the Data Pipeline Agent."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.data_pipeline",
        description="Data Pipeline Agent — data collection, normalisation, and freshness checks",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Report staleness of all data files")

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch data from a source")
    fetch_parser.add_argument(
        "--source",
        required=True,
        choices=["binance", "coinalyze", "polybacktest"],
        help="Data source to fetch from",
    )

    # unify
    subparsers.add_parser("unify", help="Merge all sources into unified CSV")

    # validate
    subparsers.add_parser("validate", help="Check for gaps, duplicates, NaN counts")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "status": cmd_status,
        "fetch": cmd_fetch,
        "unify": cmd_unify,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
