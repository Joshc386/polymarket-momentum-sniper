"""Fetch historical BTC 5m market data from PolyBackTest Pro API.

Downloads market metadata + order book snapshots at key entry timepoints
for backtesting with real Polymarket pricing.

Usage:
    python -m backtest.fetch_polybacktest_pro

Output:
    backtest/data/polybacktest_markets.csv   — market metadata + resolutions
    backtest/data/polybacktest_snapshots.csv — order book snapshots at entry timepoints
    backtest/data/polybacktest_spot.csv      — Binance spot trade data (taker buy ratio)
"""

import csv
import logging
import os
import time
from collections import deque
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
# Ensure logs are flushed immediately (critical for background processes)
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

API_BASE = "https://api.polybacktest.com"

# Load API key from environment or .env file
API_KEY = os.environ.get("POLYBACKTEST_API_KEY", "")
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("POLYBACKTEST_API_KEY="):
                    API_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

COIN = "btc"
MARKET_TYPE = "5m"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Seconds into the 5-min window to sample snapshots.
SNAPSHOT_OFFSETS_SEC = [30, 60, 120, 180, 240]

SNAPSHOT_FIELDS = [
    "market_id", "offset_sec", "seconds_remaining", "snapshot_time",
    "btc_price", "price_up", "price_down",
    "up_best_ask", "up_best_bid", "down_best_ask", "down_best_bid",
    "up_bid_depth_5", "up_ask_depth_5", "down_bid_depth_5", "down_ask_depth_5",
]


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Make authenticated GET request. Simple: 0.5s between calls, long wait on 429."""
    url = f"{API_BASE}{endpoint}"
    headers = {"X-API-Key": API_KEY}

    time.sleep(0.5)  # Hard 2 req/s cap — never exceed burst limit

    for attempt in range(10):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                wait = max(retry_after + 5, 15)  # Always wait generously on 429
                logger.warning(f"429 (attempt {attempt+1}) — waiting {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return None

            return resp.json()

        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(5)

    logger.error(f"Failed after 10 attempts: {endpoint}")
    return None


def fetch_all_markets() -> list[dict]:
    """Fetch all resolved BTC 5m markets."""
    markets = []
    offset = 0
    batch_size = 100
    consecutive_failures = 0

    while True:
        data = api_get(
            "/v2/markets",
            params={
                "coin": COIN,
                "market_type": MARKET_TYPE,
                "limit": batch_size,
                "offset": offset,
                "resolved": "true",
            },
        )

        if not data or not data.get("markets"):
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning(f"3 consecutive failures at offset {offset}, stopping")
                break
            logger.warning(f"Retry market listing at offset {offset}")
            time.sleep(5)
            continue

        consecutive_failures = 0
        batch = data["markets"]
        markets.extend(batch)
        offset += len(batch)

        total = data.get("total", "?")
        logger.info(f"  Markets: {len(markets)}/{total}")

        if len(batch) < batch_size:
            break

    return markets


def fetch_snapshots_for_market(market: dict) -> list[dict]:
    """Fetch order book snapshots at key timepoints for one market."""
    market_id = market["market_id"]
    start_time = datetime.fromisoformat(market["start_time"].replace("Z", "+00:00"))
    rows = []

    for offset_sec in SNAPSHOT_OFFSETS_SEC:
        ts = start_time.timestamp() + offset_sec
        iso_ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        data = api_get(
            f"/v2/markets/{market_id}/snapshot-at/{iso_ts}",
            params={"coin": COIN},
        )

        if not data or not data.get("snapshots"):
            continue

        snap = data["snapshots"][0]
        ob_up = snap.get("orderbook_up", {})
        ob_down = snap.get("orderbook_down", {})

        up_asks = ob_up.get("asks", [])
        up_bids = ob_up.get("bids", [])
        down_asks = ob_down.get("asks", [])
        down_bids = ob_down.get("bids", [])

        up_best_ask = up_asks[0]["price"] if up_asks else None
        up_best_bid = up_bids[0]["price"] if up_bids else None
        down_best_ask = down_asks[0]["price"] if down_asks else None
        down_best_bid = down_bids[0]["price"] if down_bids else None

        up_bid_depth = sum(level["size"] for level in up_bids[:5])
        up_ask_depth = sum(level["size"] for level in up_asks[:5])
        down_bid_depth = sum(level["size"] for level in down_bids[:5])
        down_ask_depth = sum(level["size"] for level in down_asks[:5])

        rows.append({
            "market_id": market_id,
            "offset_sec": offset_sec,
            "seconds_remaining": 300 - offset_sec,
            "snapshot_time": snap["time"],
            "btc_price": snap.get("btc_price"),
            "price_up": snap.get("price_up"),
            "price_down": snap.get("price_down"),
            "up_best_ask": up_best_ask,
            "up_best_bid": up_best_bid,
            "down_best_ask": down_best_ask,
            "down_best_bid": down_best_bid,
            "up_bid_depth_5": up_bid_depth,
            "up_ask_depth_5": up_ask_depth,
            "down_bid_depth_5": down_bid_depth,
            "down_ask_depth_5": down_ask_depth,
        })

    return rows


def fetch_spot_trades(start_ts: str, end_ts: str) -> list[dict]:
    """Fetch Binance spot trade aggregates for signal reconstruction."""
    trades = []
    offset = 0
    batch_size = 1000
    consecutive_failures = 0

    while True:
        data = api_get(
            "/v2/spot/trades",
            params={
                "coin": COIN,
                "start_time": start_ts,
                "end_time": end_ts,
                "limit": batch_size,
                "offset": offset,
            },
        )

        if not data or not data.get("trades"):
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
            time.sleep(5)
            continue

        consecutive_failures = 0
        batch = data["trades"]
        trades.extend(batch)
        offset += len(batch)

        total = data.get("total", "?")
        if len(trades) % 10000 < batch_size:
            logger.info(f"  Spot trades: {len(trades)}/{total}")

        if len(batch) < batch_size:
            break

    return trades


def save_markets_csv(markets: list[dict], path: str) -> None:
    """Save market metadata to CSV."""
    fields = [
        "market_id", "slug", "start_time", "end_time",
        "btc_price_start", "btc_price_end", "winner",
        "final_volume", "final_liquidity",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for m in markets:
            writer.writerow(m)
    logger.info(f"Saved {len(markets)} markets to {path}")


def append_snapshots(snapshots: list[dict], path: str, write_header: bool = False) -> None:
    """Append snapshot rows to CSV (supports incremental saves)."""
    if not snapshots:
        return
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(snapshots)


def save_spot_csv(trades: list[dict], path: str) -> None:
    """Save spot trade data to CSV."""
    if not trades:
        logger.warning("No spot trades to save")
        return
    fields = [
        "timestamp", "price_open", "price_high", "price_low", "price_close",
        "total_volume", "aggressive_buy_volume", "aggressive_sell_volume",
        "num_trades",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
    logger.info(f"Saved {len(trades)} spot trades to {path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch PolyBackTest Pro data")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch markets list to pick up newly resolved markets"
    )
    args = parser.parse_args()

    if not API_KEY:
        logger.error("Set POLYBACKTEST_API_KEY environment variable (or add to .env)")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    snapshots_path = os.path.join(DATA_DIR, "polybacktest_snapshots.csv")

    # Phase 1: Fetch/refresh markets list
    if os.path.exists(markets_path) and not args.refresh:
        with open(markets_path, "r", encoding="utf-8") as f:
            resolved = list(csv.DictReader(f))
        logger.info(f"Phase 1: Loaded {len(resolved)} markets from cache")
    else:
        old_count = 0
        if os.path.exists(markets_path):
            with open(markets_path, "r", encoding="utf-8") as f:
                old_count = sum(1 for _ in f) - 1
        logger.info("Phase 1: Fetching all resolved BTC 5m markets...")
        markets = fetch_all_markets()
        resolved = [m for m in markets if m.get("winner")]
        new_count = len(resolved) - old_count
        logger.info(
            f"Found {len(resolved)} resolved markets (of {len(markets)} total)"
            f"{f' — {new_count} new since last fetch' if old_count else ''}"
        )
        save_markets_csv(resolved, markets_path)

    # Phase 2: Fetch snapshots — check for resume
    fetched_ids: set[str] = set()
    if os.path.exists(snapshots_path):
        with open(snapshots_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fetched_ids.add(row["market_id"])
        logger.info(f"Resuming — {len(fetched_ids)} markets already fetched")

    remaining = [m for m in resolved if m["market_id"] not in fetched_ids]
    logger.info(f"Phase 2: Fetching snapshots for {len(remaining)} markets...")

    buffer: list[dict] = []
    completed = 0
    save_every = 200
    t0 = time.time()

    for market in remaining:
        completed += 1
        try:
            rows = fetch_snapshots_for_market(market)
            buffer.extend(rows)
        except Exception as e:
            logger.warning(f"Error for {market['market_id']}: {e}")

        if completed % 100 == 0 or completed == len(remaining):
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - completed) / rate / 60 if rate > 0 else 0
            logger.info(
                f"  Phase 2: {completed}/{len(remaining)} "
                f"({len(buffer)} buffered, {rate:.2f} mkts/s, ETA {eta:.0f}m)"
            )

        if completed % save_every == 0:
            is_first = len(fetched_ids) == 0 and completed == save_every
            append_snapshots(buffer, snapshots_path, write_header=is_first)
            logger.info(f"  Checkpoint: {len(buffer)} snapshots saved")
            buffer = []

    if buffer:
        is_first = len(fetched_ids) == 0 and completed <= save_every
        append_snapshots(buffer, snapshots_path, write_header=is_first)

    # Phase 3: Spot trades
    if resolved:
        oldest = min(m["start_time"] for m in resolved)
        newest = max(m["end_time"] for m in resolved)
        logger.info(f"Phase 3: Fetching spot trades {oldest} → {newest}...")
        spot = fetch_spot_trades(oldest, newest)
        save_spot_csv(spot, os.path.join(DATA_DIR, "polybacktest_spot.csv"))
    else:
        spot = []

    # Count total snapshots
    snap_count = 0
    if os.path.exists(snapshots_path):
        with open(snapshots_path, "r", encoding="utf-8") as f:
            snap_count = sum(1 for _ in f) - 1  # minus header

    logger.info("Done!")
    logger.info(f"  Markets:   {len(resolved)}")
    logger.info(f"  Snapshots: {snap_count}")
    logger.info(f"  Spot:      {len(spot)}")


if __name__ == "__main__":
    main()
