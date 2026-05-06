"""Fetch 8Hz tick-level snapshot data from PolyBackTest Pro API.

Downloads CLOB snapshots (~2,400/market) and Binance spot trades (1-sec bars)
for all resolved BTC 5m markets. Stores in SQLite for efficient querying.

Usage:
    python -m backtest.fetch_tick_data [--days 7] [--include-orderbook]

Output:
    backtest/data/tick_data.db — SQLite database with markets, snapshots, spot_trades
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

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


# ── Rate limiter ──────────────────────────────────────────────────
class RateLimiter:
    """Enforces rate limits: max N requests per minute with min gap."""

    def __init__(self, max_per_minute: int = 250, min_gap_secs: float = 0.25):
        self._max_per_minute = max_per_minute
        self._min_gap = min_gap_secs
        self._timestamps: list[float] = []
        self._last_request = 0.0

    def wait(self) -> None:
        """Block until it's safe to make the next request."""
        now = time.time()

        # Min gap between requests
        elapsed = now - self._last_request
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)

        # Sliding window: drop timestamps older than 60s
        cutoff = time.time() - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        # If at capacity, wait until oldest falls out of window
        if len(self._timestamps) >= self._max_per_minute:
            wait_until = self._timestamps[0] + 60.0
            wait_secs = wait_until - time.time()
            if wait_secs > 0:
                logger.info(f"Rate limit: waiting {wait_secs:.1f}s")
                time.sleep(wait_secs)

        self._timestamps.append(time.time())
        self._last_request = time.time()


rate_limiter = RateLimiter(max_per_minute=250, min_gap_secs=0.25)


def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    """Make authenticated GET request with rate limiting and retries."""
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for attempt in range(10):
        rate_limiter.wait()

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                wait = max(retry_after + 5, 15)
                logger.warning(f"429 (attempt {attempt+1}) -- waiting {wait}s")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.warning(
                    f"HTTP {resp.status_code} for {endpoint} "
                    f"(attempt {attempt+1})"
                )
                if resp.status_code in (401, 402, 403):
                    return None  # Auth issue, don't retry
                time.sleep(2)
                continue

            return resp.json()

        except requests.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(5)

    logger.error(f"Failed after 10 attempts: {endpoint}")
    return None


# ── Database setup ────────────────────────────────────────────────
def init_db(db_path: str) -> sqlite3.Connection:
    """Create database and tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

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


# ── Fetch markets ─────────────────────────────────────────────────
def fetch_markets(
    conn: sqlite3.Connection,
    days: int = 7,
) -> list[dict]:
    """Fetch all resolved BTC 5m markets within the date range."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    logger.info(
        f"Fetching markets from {start_time.date()} to {end_time.date()} "
        f"({days} days)"
    )

    markets: list[dict] = []
    offset = 0
    batch_size = 100

    while True:
        data = api_get(
            "/v2/markets",
            params={
                "coin": COIN,
                "market_type": MARKET_TYPE,
                "resolved": "true",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "limit": batch_size,
                "offset": offset,
            },
        )

        if not data or not data.get("markets"):
            if not markets:
                logger.warning("No markets returned — check API key and date range")
            break

        batch = data["markets"]
        markets.extend(batch)
        offset += len(batch)

        total = data.get("total", "?")
        logger.info(f"  Markets: {len(markets)}/{total}")

        if len(batch) < batch_size:
            break

    # Upsert into database
    for m in markets:
        conn.execute(
            """INSERT OR IGNORE INTO markets
               (market_id, start_time, end_time, btc_price_start,
                btc_price_end, winner, final_volume, final_liquidity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                m["market_id"], m["start_time"], m["end_time"],
                m.get("btc_price_start"), m.get("btc_price_end"),
                m.get("winner"), m.get("final_volume"),
                m.get("final_liquidity"),
            ),
        )
    conn.commit()

    logger.info(f"Fetched {len(markets)} markets")
    return markets


# ── Fetch snapshots for one market ────────────────────────────────
def fetch_market_snapshots(
    conn: sqlite3.Connection,
    market: dict,
    include_orderbook: bool = True,
) -> int:
    """Fetch all 8Hz snapshots for a single market. Returns row count."""
    market_id = market["market_id"]
    start_time = market["start_time"]
    end_time = market["end_time"]

    # Parse start time to compute seconds_into_window
    try:
        window_start = datetime.fromisoformat(
            start_time.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        window_start = None

    total_rows = 0
    offset = 0
    batch_size = 1000

    while True:
        params = {
            "coin": COIN,
            "start_time": start_time,
            "end_time": end_time,
            "limit": batch_size,
            "offset": offset,
            "include_orderbook": str(include_orderbook).lower(),
        }

        data = api_get(f"/v2/markets/{market_id}/snapshots", params=params)

        if not data or not data.get("snapshots"):
            break

        snaps = data["snapshots"]
        rows = []
        for s in snaps:
            snap_time = s.get("time", "")

            # Compute seconds into window
            secs_into = None
            if window_start and snap_time:
                try:
                    snap_dt = datetime.fromisoformat(
                        snap_time.replace("Z", "+00:00")
                    )
                    secs_into = (snap_dt - window_start).total_seconds()
                except (ValueError, TypeError):
                    pass

            # Extract best bid/ask from orderbook
            ob_up = s.get("orderbook_up") or {}
            ob_down = s.get("orderbook_down") or {}
            up_bids = ob_up.get("bids", [])
            up_asks = ob_up.get("asks", [])
            down_bids = ob_down.get("bids", [])
            down_asks = ob_down.get("asks", [])

            up_best_bid = up_bids[0]["price"] if up_bids else None
            up_best_ask = up_asks[0]["price"] if up_asks else None
            down_best_bid = down_bids[0]["price"] if down_bids else None
            down_best_ask = down_asks[0]["price"] if down_asks else None

            up_bid_depth = sum(l["size"] for l in up_bids[:5]) if up_bids else 0
            up_ask_depth = sum(l["size"] for l in up_asks[:5]) if up_asks else 0
            down_bid_depth = sum(l["size"] for l in down_bids[:5]) if down_bids else 0
            down_ask_depth = sum(l["size"] for l in down_asks[:5]) if down_asks else 0

            rows.append((
                market_id, snap_time, secs_into,
                s.get("btc_price"),
                s.get("price_up"), s.get("price_down"),
                up_best_bid, up_best_ask,
                down_best_bid, down_best_ask,
                up_bid_depth, up_ask_depth,
                down_bid_depth, down_ask_depth,
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

        if len(snaps) < batch_size:
            break

    # Mark market as fetched
    conn.execute(
        "UPDATE markets SET snapshots_fetched = ? WHERE market_id = ?",
        (total_rows, market_id),
    )
    conn.commit()
    return total_rows


# ── Fetch Binance spot trades ─────────────────────────────────────
def fetch_spot_trades(
    conn: sqlite3.Connection,
    days: int = 7,
) -> int:
    """Fetch 1-second Binance spot trade bars."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    logger.info(f"Fetching Binance spot trades ({days} days)...")

    total_rows = 0
    # Fetch in 1-hour chunks to avoid hitting limits
    chunk_start = start_time

    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(hours=1), end_time)
        offset = 0

        while True:
            data = api_get(
                "/v2/spot/trades",
                params={
                    "coin": COIN,
                    "start_time": chunk_start.isoformat(),
                    "end_time": chunk_end.isoformat(),
                    "limit": 1000,
                    "offset": offset,
                },
            )

            if not data or not data.get("trades"):
                break

            trades = data["trades"]
            rows = []
            for t in trades:
                rows.append((
                    t.get("timestamp", ""),
                    t.get("price_open"),
                    t.get("price_close"),
                    t.get("price_high"),
                    t.get("price_low"),
                    t.get("total_volume"),
                    t.get("aggressive_buy_volume"),
                    t.get("aggressive_sell_volume"),
                    t.get("num_trades"),
                ))

            conn.executemany(
                """INSERT OR IGNORE INTO spot_trades
                   (timestamp, price_open, price_close, price_high, price_low,
                    total_volume, aggressive_buy_volume, aggressive_sell_volume,
                    num_trades)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

            total_rows += len(trades)
            offset += len(trades)

            if len(trades) < 1000:
                break

        chunk_start = chunk_end
        elapsed_days = (chunk_start - start_time).total_seconds() / 86400
        logger.info(
            f"  Spot trades: {total_rows:,} rows "
            f"({elapsed_days:.1f}/{days} days)"
        )

    logger.info(f"Fetched {total_rows:,} spot trade rows")
    return total_rows


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch 8Hz tick data from PolyBackTest Pro"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Number of days to fetch (default: 7, max: 31)",
    )
    parser.add_argument(
        "--include-orderbook", action="store_true", default=True,
        help="Include full orderbook in snapshots (default: True)",
    )
    parser.add_argument(
        "--no-orderbook", action="store_true",
        help="Skip orderbook data (faster, smaller DB)",
    )
    parser.add_argument(
        "--skip-spot", action="store_true",
        help="Skip Binance spot trade fetching",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip markets that already have snapshots (default: True)",
    )
    args = parser.parse_args()

    if not API_KEY:
        logger.error(
            "No API key found. Set POLYBACKTEST_API_KEY in .env or environment"
        )
        sys.exit(1)

    include_ob = not args.no_orderbook
    days = min(args.days, 31)

    logger.info(f"{'=' * 60}")
    logger.info(f"  PolyBackTest Pro Tick Data Fetcher")
    logger.info(f"  Days: {days}, Orderbook: {include_ob}")
    logger.info(f"  DB: {DB_PATH}")
    logger.info(f"{'=' * 60}")

    conn = init_db(DB_PATH)

    # Step 1: Fetch market list
    markets = fetch_markets(conn, days=days)
    if not markets:
        logger.error("No markets found — exiting")
        conn.close()
        return

    # Step 2: Fetch snapshots for each market
    # Check which markets already have data (resume support)
    already_fetched = set()
    if args.resume:
        cursor = conn.execute(
            "SELECT market_id FROM markets WHERE snapshots_fetched > 0"
        )
        already_fetched = {row[0] for row in cursor.fetchall()}
        if already_fetched:
            logger.info(
                f"Resuming: {len(already_fetched)} markets already fetched, "
                f"{len(markets) - len(already_fetched)} remaining"
            )

    total_snaps = 0
    fetched_count = 0
    skipped_count = 0
    start_ts = time.time()

    for i, market in enumerate(markets):
        mid = market["market_id"]
        if mid in already_fetched:
            skipped_count += 1
            continue

        n_snaps = fetch_market_snapshots(conn, market, include_orderbook=include_ob)
        total_snaps += n_snaps
        fetched_count += 1

        # Progress update every 10 markets
        if fetched_count % 10 == 0:
            elapsed = time.time() - start_ts
            rate = fetched_count / elapsed * 60 if elapsed > 0 else 0
            remaining = len(markets) - i - 1 - skipped_count
            eta_mins = remaining / rate if rate > 0 else 0
            logger.info(
                f"  Progress: {i+1}/{len(markets)} markets | "
                f"{total_snaps:,} snapshots | "
                f"{rate:.0f} markets/min | "
                f"ETA: {eta_mins:.0f}min"
            )

    logger.info(
        f"Snapshot fetch complete: {fetched_count} markets, "
        f"{total_snaps:,} snapshots"
    )

    # Step 3: Fetch Binance spot trades
    if not args.skip_spot:
        fetch_spot_trades(conn, days=days)

    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM snapshots")
    total_snap_rows = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM spot_trades")
    total_spot_rows = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM markets WHERE snapshots_fetched > 0")
    total_markets_done = cursor.fetchone()[0]

    # DB file size
    conn.close()
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FETCH COMPLETE")
    logger.info(f"  Markets with data: {total_markets_done}")
    logger.info(f"  Total snapshots:   {total_snap_rows:,}")
    logger.info(f"  Total spot trades: {total_spot_rows:,}")
    logger.info(f"  Database size:     {db_size_mb:.1f} MB")
    logger.info(f"  Path:              {DB_PATH}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
