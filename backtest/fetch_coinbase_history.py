"""Fetch historical 1-minute BTC/USD candles from Coinbase Exchange API.

Needed for:
  - Bot B: Kalman filter's third measurement source
  - Bot D: Cross-exchange divergence signal (Binance vs Coinbase spread)

Usage:
    python -m backtest.fetch_coinbase_history

Output:
    backtest/data/coinbase_btc_1m.csv

Coinbase Exchange API:
    - GET /products/BTC-USD/candles — no auth required
    - Max 300 candles per request
    - Rate limit: 10 req/sec (public endpoints)
"""

import csv
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "coinbase_btc_1m.csv")

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY = 60  # 1 minute
MAX_CANDLES = 300  # Per request


def fetch_coinbase_candles(
    start_ts: int,
    end_ts: int,
    output_path: str = OUTPUT_PATH,
) -> int:
    """Fetch 1-minute BTC/USD candles from Coinbase and save to CSV.

    Args:
        start_ts: Start Unix timestamp.
        end_ts: End Unix timestamp.
        output_path: CSV output path.

    Returns:
        Total candles fetched.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check for existing data to resume
    existing_end_ts = 0
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = int(row["timestamp"])
                if ts > existing_end_ts:
                    existing_end_ts = ts
        if existing_end_ts > start_ts:
            logger.info(
                f"Resuming from "
                f"{datetime.fromtimestamp(existing_end_ts, tz=timezone.utc).isoformat()}"
            )
            start_ts = existing_end_ts + GRANULARITY

    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
    total_fetched = 0
    current_start = start_ts

    # Coinbase returns candles newest-first, and expects ISO timestamps
    # We'll request in forward chunks of MAX_CANDLES * GRANULARITY seconds
    chunk_seconds = MAX_CANDLES * GRANULARITY  # 300 minutes = 5 hours

    with httpx.Client(
        timeout=15.0,
        headers={"User-Agent": "polymarket-backtest/1.0"},
    ) as client:
        while current_start < end_ts:
            chunk_end = min(current_start + chunk_seconds, end_ts)

            start_iso = datetime.fromtimestamp(
                current_start, tz=timezone.utc
            ).isoformat()
            end_iso = datetime.fromtimestamp(
                chunk_end, tz=timezone.utc
            ).isoformat()

            params = {
                "start": start_iso,
                "end": end_iso,
                "granularity": GRANULARITY,
            }

            try:
                resp = client.get(COINBASE_URL, params=params)

                if resp.status_code == 429:
                    logger.warning("Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue

                if resp.status_code != 200:
                    logger.warning(
                        f"Coinbase HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    current_start = chunk_end
                    time.sleep(2)
                    continue

                data = resp.json()
            except Exception as e:
                logger.error(f"Coinbase API error: {e}. Retrying in 5s...")
                time.sleep(5)
                continue

            if not data:
                logger.debug(f"No data for chunk ending {end_iso}")
                current_start = chunk_end
                time.sleep(0.5)
                continue

            # Coinbase returns: [[timestamp, low, high, open, close, volume], ...]
            # Sorted newest-first — reverse for chronological order
            data.sort(key=lambda c: c[0])

            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "timestamp_utc",
                        "open", "high", "low", "close", "volume",
                    ])
                    write_header = False

                for candle in data:
                    ts = int(candle[0])
                    if ts < start_ts or ts > end_ts:
                        continue

                    utc_str = datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")

                    writer.writerow([
                        ts,
                        utc_str,
                        candle[3],  # open
                        candle[2],  # high
                        candle[1],  # low
                        candle[4],  # close
                        candle[5],  # volume
                    ])
                    total_fetched += 1

            last_time = datetime.fromtimestamp(
                data[-1][0], tz=timezone.utc
            )
            logger.info(
                f"Fetched {len(data)} candles "
                f"(total: {total_fetched:,}, last: {last_time.isoformat()})"
            )

            current_start = chunk_end
            time.sleep(0.5)  # Stay well under 10 req/sec

    logger.info(f"Done. Total candles: {total_fetched:,}")
    return total_fetched


def main() -> None:
    """Fetch Coinbase candles covering the PolyBackTest Pro date range."""
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    if not os.path.exists(markets_path):
        logger.error(f"Markets file not found: {markets_path}")
        logger.error("Run: python -m backtest.fetch_polybacktest_pro first")
        return

    earliest = None
    latest = None
    with open(markets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["start_time", "end_time"]:
                t = row.get(key, "")
                if t:
                    try:
                        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                        if earliest is None or dt < earliest:
                            earliest = dt
                        if latest is None or dt > latest:
                            latest = dt
                    except ValueError:
                        pass

    if not earliest or not latest:
        logger.error("Could not determine date range from markets CSV")
        return

    # Buffer: 30 min before and 5 min after (match Binance fetch)
    start_ts = int((earliest - timedelta(minutes=30)).timestamp())
    end_ts = int((latest + timedelta(minutes=5)).timestamp())

    days = (end_ts - start_ts) / 86400
    expected_candles = (end_ts - start_ts) / GRANULARITY
    expected_requests = expected_candles / MAX_CANDLES

    logger.info(f"Date range: {earliest.isoformat()} to {latest.isoformat()}")
    logger.info(
        f"Expected: ~{expected_candles:,.0f} candles in "
        f"~{expected_requests:.0f} requests ({days:.1f} days)"
    )

    fetch_coinbase_candles(start_ts, end_ts)


if __name__ == "__main__":
    main()
