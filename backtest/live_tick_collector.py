"""Live tick data collector — continuously fetches 8Hz snapshots as markets resolve.

Runs alongside multi_runner.py to keep tick_data.db always up to date.
Every cycle it checks for newly resolved markets, fetches their full
snapshot history, and appends to the database.

Usage:
    python -m backtest.live_tick_collector [--interval 300] [--include-orderbook]

Designed to run 24/7 as a background process.
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

API_BASE = "https://api.polybacktest.com"
COIN = "btc"
MARKET_TYPE = "5m"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "tick_data.db")

# Load API key
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


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Authenticated GET with retries."""
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                time.sleep(max(retry_after + 2, 10))
                continue

            if resp.status_code != 200:
                return None

            return resp.json()

        except requests.RequestException as e:
            logger.debug(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(3)

    return None


def init_db() -> sqlite3.Connection:
    """Open database (created by fetch_tick_data.py)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Ensure tables exist (in case this runs before the batch fetcher)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            btc_price_start REAL,
            btc_price_end REAL,
            winner TEXT,
            final_volume REAL,
            final_liquidity REAL,
            snapshots_fetched INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            market_id TEXT NOT NULL,
            time TEXT NOT NULL,
            seconds_into_window REAL,
            btc_price REAL,
            price_up REAL,
            price_down REAL,
            up_best_bid REAL,
            up_best_ask REAL,
            down_best_bid REAL,
            down_best_ask REAL,
            up_bid_depth_5 REAL,
            up_ask_depth_5 REAL,
            down_bid_depth_5 REAL,
            down_ask_depth_5 REAL,
            PRIMARY KEY (market_id, time)
        );

        CREATE TABLE IF NOT EXISTS spot_trades (
            timestamp TEXT PRIMARY KEY,
            price_open REAL,
            price_close REAL,
            price_high REAL,
            price_low REAL,
            total_volume REAL,
            aggressive_buy_volume REAL,
            aggressive_sell_volume REAL,
            num_trades INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_market
            ON snapshots(market_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_time
            ON snapshots(time);
        CREATE INDEX IF NOT EXISTS idx_spot_timestamp
            ON spot_trades(timestamp);
    """)
    conn.commit()
    return conn


def get_known_market_ids(conn: sqlite3.Connection) -> set[str]:
    """Get all market IDs already in the database."""
    cursor = conn.execute(
        "SELECT market_id FROM markets WHERE snapshots_fetched > 0"
    )
    return {row[0] for row in cursor.fetchall()}


def fetch_recent_resolved_markets(limit: int = 50) -> list[dict]:
    """Fetch the most recently resolved markets."""
    data = api_get(
        "/v2/markets",
        params={
            "coin": COIN,
            "market_type": MARKET_TYPE,
            "resolved": "true",
            "limit": limit,
        },
    )
    if not data:
        return []
    return data.get("markets", [])


def fetch_and_store_snapshots(
    conn: sqlite3.Connection,
    market: dict,
    include_orderbook: bool = True,
) -> int:
    """Fetch all snapshots for one market and store in DB."""
    market_id = market["market_id"]
    start_time = market["start_time"]
    end_time = market["end_time"]

    try:
        window_start = datetime.fromisoformat(
            start_time.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        window_start = None

    # Upsert market
    conn.execute(
        """INSERT OR REPLACE INTO markets
           (market_id, start_time, end_time, btc_price_start,
            btc_price_end, winner, final_volume, final_liquidity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            market_id, start_time, end_time,
            market.get("btc_price_start"),
            market.get("btc_price_end"),
            market.get("winner"),
            market.get("final_volume"),
            market.get("final_liquidity"),
        ),
    )

    total_rows = 0
    offset = 0

    while True:
        data = api_get(
            f"/v2/markets/{market_id}/snapshots",
            params={
                "coin": COIN,
                "start_time": start_time,
                "end_time": end_time,
                "limit": 1000,
                "offset": offset,
                "include_orderbook": str(include_orderbook).lower(),
            },
        )

        if not data or not data.get("snapshots"):
            break

        snaps = data["snapshots"]
        rows = []
        for s in snaps:
            snap_time = s.get("time", "")

            secs_into = None
            if window_start and snap_time:
                try:
                    snap_dt = datetime.fromisoformat(
                        snap_time.replace("Z", "+00:00")
                    )
                    secs_into = (snap_dt - window_start).total_seconds()
                except (ValueError, TypeError):
                    pass

            ob_up = s.get("orderbook_up") or {}
            ob_down = s.get("orderbook_down") or {}
            up_bids = ob_up.get("bids", [])
            up_asks = ob_up.get("asks", [])
            down_bids = ob_down.get("bids", [])
            down_asks = ob_down.get("asks", [])

            rows.append((
                market_id, snap_time, secs_into,
                s.get("btc_price"),
                s.get("price_up"), s.get("price_down"),
                up_bids[0]["price"] if up_bids else None,
                up_asks[0]["price"] if up_asks else None,
                down_bids[0]["price"] if down_bids else None,
                down_asks[0]["price"] if down_asks else None,
                sum(l["size"] for l in up_bids[:5]) if up_bids else 0,
                sum(l["size"] for l in up_asks[:5]) if up_asks else 0,
                sum(l["size"] for l in down_bids[:5]) if down_bids else 0,
                sum(l["size"] for l in down_asks[:5]) if down_asks else 0,
            ))

        conn.executemany(
            """INSERT OR IGNORE INTO snapshots
               (market_id, time, seconds_into_window, btc_price,
                price_up, price_down,
                up_best_bid, up_best_ask,
                down_best_bid, down_best_ask,
                up_bid_depth_5, up_ask_depth_5,
                down_bid_depth_5, down_ask_depth_5)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()

        total_rows += len(snaps)
        offset += len(snaps)

        if len(snaps) < 1000:
            break

        # Brief pause between pages
        time.sleep(0.3)

    # Mark as fetched
    conn.execute(
        "UPDATE markets SET snapshots_fetched = ? WHERE market_id = ?",
        (total_rows, market_id),
    )
    conn.commit()
    return total_rows


def run_collection_cycle(
    conn: sqlite3.Connection,
    include_orderbook: bool = True,
) -> int:
    """Run one collection cycle: find new markets, fetch their snapshots.

    Returns number of new markets processed.
    """
    known_ids = get_known_market_ids(conn)
    recent_markets = fetch_recent_resolved_markets(limit=50)

    if not recent_markets:
        return 0

    new_markets = [m for m in recent_markets if m["market_id"] not in known_ids]

    if not new_markets:
        return 0

    logger.info(f"Found {len(new_markets)} new resolved markets")

    total_snaps = 0
    for i, market in enumerate(new_markets):
        n = fetch_and_store_snapshots(conn, market, include_orderbook)
        total_snaps += n
        logger.info(
            f"  [{i+1}/{len(new_markets)}] Market {market['market_id']}: "
            f"{n:,} snapshots ({market.get('winner', '?')})"
        )

    logger.info(
        f"Cycle complete: {len(new_markets)} markets, "
        f"{total_snaps:,} snapshots added"
    )
    return len(new_markets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live tick data collector for PolyBackTest Pro"
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between collection cycles (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--include-orderbook", action="store_true", default=True,
        help="Include orderbook data (default: True)",
    )
    parser.add_argument(
        "--no-orderbook", action="store_true",
        help="Skip orderbook data",
    )
    args = parser.parse_args()

    if not API_KEY:
        logger.error("No POLYBACKTEST_API_KEY found")
        sys.exit(1)

    include_ob = not args.no_orderbook
    interval = args.interval

    logger.info(f"{'=' * 60}")
    logger.info(f"  Live Tick Data Collector")
    logger.info(f"  Interval: {interval}s, Orderbook: {include_ob}")
    logger.info(f"  DB: {DB_PATH}")
    logger.info(f"{'=' * 60}")

    conn = init_db()

    # Stats
    total_markets = 0
    start_time = time.time()

    while True:
        try:
            n = run_collection_cycle(conn, include_orderbook=include_ob)
            total_markets += n

            # Periodic stats
            elapsed_hrs = (time.time() - start_time) / 3600
            total_snaps = conn.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0]
            total_mkts = conn.execute(
                "SELECT COUNT(*) FROM markets WHERE snapshots_fetched > 0"
            ).fetchone()[0]

            if n > 0:
                logger.info(
                    f"DB totals: {total_mkts:,} markets, "
                    f"{total_snaps:,} snapshots | "
                    f"Running {elapsed_hrs:.1f}h"
                )
            else:
                logger.debug("No new markets this cycle")

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

        time.sleep(interval)

    conn.close()
    logger.info("Collector stopped")


if __name__ == "__main__":
    main()
