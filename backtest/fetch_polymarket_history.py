"""Fetch historical BTC 5-min market data from Polymarket APIs.

Strategy:
1. Walk backwards through 5-minute window timestamps
2. For each window, fetch market info from Gamma API (token IDs, resolution)
3. For each window, fetch all trades from Data API
4. Reconstruct mid-prices at each second within the window
5. Output CSV with per-window efficiency data

This gives us thousands of windows of real Polymarket data to compute
the actual market efficiency parameter.

Usage:
    python -m backtest.fetch_polymarket_history [--days 7] [--output backtest/data/poly_history.csv]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def fetch_market_info(window_ts: int) -> dict | None:
    """Fetch market info for a BTC 5-min window from Gamma API."""
    slug = f"btc-updown-5m-{window_ts}"
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data or not data[0].get("markets"):
            return None

        m = data[0]["markets"][0]
        tokens = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
        prices = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]

        return {
            "slug": slug,
            "window_ts": window_ts,
            "yes_token": tokens[0],
            "no_token": tokens[1],
            "up_price": float(prices[0]),
            "down_price": float(prices[1]),
            "condition_id": m.get("conditionId", ""),
            "resolved": float(prices[0]) in (0, 1),
            "resolution": "UP" if float(prices[0]) == 1 else "DOWN" if float(prices[0]) == 0 else "ACTIVE",
        }
    except Exception as e:
        logger.debug(f"Failed to fetch {slug}: {e}")
        return None


def fetch_trades_for_market(condition_id: str) -> list:
    """Fetch all trades for a specific market using its conditionId."""
    all_trades = []
    offset = 0
    max_pages = 20  # Safety: max 20 pages x 10000 = 200k trades

    while True:
        try:
            resp = requests.get(
                f"{DATA_API}/trades",
                params={
                    "market": condition_id,
                    "limit": 10000,
                    "offset": offset,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or not data:
                break

            all_trades.extend(data)
            if len(data) < 10000:
                break  # Last page
            offset += len(data)
            max_pages -= 1
            if max_pages <= 0:
                break
            time.sleep(0.1)
        except Exception as e:
            logger.debug(f"Trade fetch error for {condition_id}: {e}")
            break

    return all_trades


def reconstruct_window_prices(trades: list, window_ts: int) -> dict:
    """Reconstruct mid-prices at different timepoints from trade data.

    Groups trades into time buckets and computes average trade prices
    to approximate the market's view at each point in the window.
    """
    window_end = window_ts + 300

    # Buckets: every 30 seconds within the window
    # Also track specific timepoints for efficiency analysis
    timepoints = {
        "4m": window_ts + 60,      # 1 min into window (4 min remaining)
        "3m": window_ts + 120,     # 2 min into window (3 min remaining)
        "2m": window_ts + 180,     # 3 min into window (2 min remaining)
        "1m": window_ts + 240,     # 4 min into window (1 min remaining)
        "30s": window_ts + 270,    # 4.5 min into window (30s remaining)
    }

    result = {
        "num_trades": len(trades),
        "total_volume": sum(t.get("size", 0) for t in trades),
    }

    # Filter to trades within the window
    window_trades = [
        t for t in trades
        if window_ts <= t.get("timestamp", 0) < window_end
    ]
    result["window_trades"] = len(window_trades)

    if not window_trades:
        return result

    # For each timepoint, find trades near that time and compute implied mid
    for label, target_ts in timepoints.items():
        # Use trades within +/- 30 seconds of the target
        nearby = [
            t for t in window_trades
            if abs(t["timestamp"] - target_ts) <= 30
        ]

        if nearby:
            # Compute volume-weighted average price for "Up" outcome
            up_trades = [t for t in nearby if t.get("outcome") == "Up"]
            down_trades = [t for t in nearby if t.get("outcome") == "Down"]

            if up_trades:
                total_size = sum(t["size"] for t in up_trades)
                if total_size > 0:
                    vwap = sum(t["price"] * t["size"] for t in up_trades) / total_size
                    result[f"up_vwap_{label}"] = vwap
                else:
                    result[f"up_vwap_{label}"] = None
            else:
                result[f"up_vwap_{label}"] = None

            if down_trades:
                total_size = sum(t["size"] for t in down_trades)
                if total_size > 0:
                    vwap = sum(t["price"] * t["size"] for t in down_trades) / total_size
                    result[f"down_vwap_{label}"] = vwap
                else:
                    result[f"down_vwap_{label}"] = None
            else:
                result[f"down_vwap_{label}"] = None

            # Implied UP probability from both sides
            up_p = result.get(f"up_vwap_{label}")
            down_p = result.get(f"down_vwap_{label}")
            if up_p is not None and down_p is not None:
                result[f"implied_up_{label}"] = up_p / (up_p + down_p) if (up_p + down_p) > 0 else 0.5
            elif up_p is not None:
                result[f"implied_up_{label}"] = up_p
            elif down_p is not None:
                result[f"implied_up_{label}"] = 1.0 - down_p
            else:
                result[f"implied_up_{label}"] = None

            result[f"trades_{label}"] = len(nearby)
        else:
            result[f"up_vwap_{label}"] = None
            result[f"down_vwap_{label}"] = None
            result[f"implied_up_{label}"] = None
            result[f"trades_{label}"] = 0

    return result


def fetch_history(days: int = 7, output_path: str = "backtest/data/poly_history.csv"):
    """Main fetcher: walk backwards through windows, collect trade data."""

    now = int(time.time())
    current_window = now - (now % 300)
    start_window = current_window - (days * 24 * 3600)
    total_windows = (current_window - start_window) // 300

    logger.info(f"Fetching {total_windows} windows ({days} days) of BTC 5-min market data")
    logger.info(f"Time range: {datetime.fromtimestamp(start_window, tz=timezone.utc)} to {datetime.fromtimestamp(current_window, tz=timezone.utc)}")

    # Phase 1: Discover all windows and get resolution data from Gamma API
    logger.info("Phase 1: Discovering markets from Gamma API...")
    markets = {}
    discovered = 0
    failed = 0

    for i in range(total_windows):
        window_ts = current_window - ((i + 1) * 300)  # Walk backwards, skip current

        if i % 100 == 0 and i > 0:
            logger.info(f"  Discovered {discovered}/{i} markets ({failed} not found)")

        info = fetch_market_info(window_ts)
        if info and info["resolved"]:
            markets[window_ts] = info
            discovered += 1
        elif info:
            # Active/unresolved — skip
            pass
        else:
            failed += 1

        # Rate limit: Gamma API
        time.sleep(0.15)

    logger.info(f"Phase 1 complete: {discovered} resolved markets found")

    if not markets:
        logger.error("No markets found. Exiting.")
        return

    # Phase 2: Fetch trades per market from Data API using conditionId
    logger.info(f"Phase 2: Fetching trades for {discovered} markets from Data API...")

    btc_trades_by_slug = {}
    total_trades = 0
    markets_with_trades = 0

    market_list = sorted(markets.keys())
    for idx, window_ts in enumerate(market_list):
        info = markets[window_ts]
        condition_id = info["condition_id"]
        slug = info["slug"]

        if not condition_id:
            continue

        trades = fetch_trades_for_market(condition_id)
        btc_trades_by_slug[slug] = trades
        total_trades += len(trades)
        if trades:
            markets_with_trades += 1

        if (idx + 1) % 50 == 0:
            logger.info(
                f"  {idx + 1}/{len(market_list)} markets fetched, "
                f"{total_trades} trades, {markets_with_trades} with activity"
            )

        time.sleep(0.15)  # Rate limit Data API

    logger.info(
        f"Phase 2 complete: {total_trades} trades across "
        f"{markets_with_trades} windows"
    )

    # Phase 3: Reconstruct prices and compute results
    logger.info("Phase 3: Reconstructing prices and writing output...")

    rows = []
    for window_ts in sorted(markets.keys()):
        info = markets[window_ts]
        slug = info["slug"]
        trades = btc_trades_by_slug.get(slug, [])

        prices = reconstruct_window_prices(trades, window_ts)

        row = {
            "slug": slug,
            "window_ts": window_ts,
            "window_time": datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "resolution": info["resolution"],
            "num_trades": prices["num_trades"],
            "window_trades": prices.get("window_trades", 0),
            "total_volume": f"{prices.get('total_volume', 0):.2f}",
        }

        for label in ["4m", "3m", "2m", "1m", "30s"]:
            imp = prices.get(f"implied_up_{label}")
            row[f"implied_up_{label}"] = f"{imp:.4f}" if imp is not None else ""
            row[f"trades_{label}"] = prices.get(f"trades_{label}", 0)

        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "slug", "window_ts", "window_time", "resolution",
        "num_trades", "window_trades", "total_volume",
        "implied_up_4m", "trades_4m",
        "implied_up_3m", "trades_3m",
        "implied_up_2m", "trades_2m",
        "implied_up_1m", "trades_1m",
        "implied_up_30s", "trades_30s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} windows to {output_path}")

    # Quick stats
    windows_with_trades = sum(1 for r in rows if int(r["window_trades"]) > 0)
    total_trades = sum(int(r["window_trades"]) for r in rows)
    up_count = sum(1 for r in rows if r["resolution"] == "UP")
    down_count = sum(1 for r in rows if r["resolution"] == "DOWN")

    print(f"\n{'='*60}")
    print(f"  POLYMARKET HISTORICAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Windows:        {len(rows)}")
    print(f"  With trades:    {windows_with_trades} ({100*windows_with_trades/len(rows):.0f}%)")
    print(f"  Total trades:   {total_trades}")
    print(f"  UP/DOWN:        {up_count}/{down_count}")
    print(f"  Output:         {output_path}")
    print(f"{'='*60}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket BTC 5-min historical data")
    parser.add_argument("--days", type=int, default=7, help="Days of history to fetch (default: 7)")
    parser.add_argument("--output", type=str, default="backtest/data/poly_history.csv", help="Output CSV path")
    args = parser.parse_args()

    fetch_history(days=args.days, output_path=args.output)


if __name__ == "__main__":
    main()
