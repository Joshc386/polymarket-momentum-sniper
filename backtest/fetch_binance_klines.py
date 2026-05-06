"""Fetch Binance BTC/USDT klines across multiple timeframes.

Downloads 2 years of historical data for all timeframes needed by
the multi-timeframe regime detector: 1d, 4h, 1h, 15m, 5m, 1m.

Usage:
    python -m backtest.fetch_binance_klines              # fetch all missing
    python -m backtest.fetch_binance_klines --force       # re-fetch everything
    python -m backtest.fetch_binance_klines --only 4h 1h  # fetch specific timeframes

Output:
    backtest/data/binance_klines_{timeframe}.csv

Binance API:
    - GET /api/v3/klines -- no auth required
    - Max 1000 candles per request
    - Rate limit: 1200 req/min
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLUMNS = [
    "open_time_ms", "open_time_utc", "open", "high", "low", "close",
    "volume", "close_time_ms", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]

TIMEFRAMES = {
    "1d":  {"interval": "1d",  "ms_per_candle": 86_400_000},
    "4h":  {"interval": "4h",  "ms_per_candle": 14_400_000},
    "1h":  {"interval": "1h",  "ms_per_candle": 3_600_000},
    "15m": {"interval": "15m", "ms_per_candle": 900_000},
    "5m":  {"interval": "5m",  "ms_per_candle": 300_000},
    "1m":  {"interval": "1m",  "ms_per_candle": 60_000},
}


def fetch_klines(
    interval: str,
    start_ms: int,
    end_ms: int,
    ms_per_candle: int,
) -> list[dict]:
    """Fetch all klines for a given interval between start and end timestamps.

    Args:
        interval: Binance interval string (1m, 5m, 15m, 1h, 4h, 1d).
        start_ms: Start timestamp in milliseconds.
        end_ms: End timestamp in milliseconds.
        ms_per_candle: Duration of one candle in milliseconds.

    Returns:
        List of candle dicts ready for CSV writing.
    """
    all_rows: list[dict] = []
    current = start_ms
    request_count = 0
    batch_start = time.time()

    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }

        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt == 2:
                    logger.error("Failed after 3 attempts at %s: %s", current, e)
                    raise
                wait = 2 ** attempt
                logger.warning("Retry %d for %s: %s (waiting %ds)",
                             attempt + 1, interval, e, wait)
                time.sleep(wait)

        if not data:
            break

        for k in data:
            ts_utc = datetime.fromtimestamp(
                k[0] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            all_rows.append({
                "open_time_ms": k[0],
                "open_time_utc": ts_utc,
                "open": k[1],
                "high": k[2],
                "low": k[3],
                "close": k[4],
                "volume": k[5],
                "close_time_ms": k[6],
                "quote_volume": k[7],
                "num_trades": k[8],
                "taker_buy_base_vol": k[9],
                "taker_buy_quote_vol": k[10],
            })

        current = data[-1][0] + ms_per_candle
        request_count += 1

        # Progress logging every 50 requests
        if request_count % 50 == 0:
            elapsed = time.time() - batch_start
            rate = request_count / elapsed * 60 if elapsed > 0 else 0
            logger.info(
                "  %s: %d candles (%d requests, %.0f req/min)",
                interval, len(all_rows), request_count, rate,
            )
            # Throttle if approaching rate limit
            if rate > 800:
                time.sleep(1)

    return all_rows


def get_existing_end_ms(filepath: str) -> int:
    """Find the latest timestamp in an existing CSV file.

    Returns 0 if file doesn't exist or is empty.
    """
    if not os.path.exists(filepath):
        return 0

    max_ms = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["open_time_ms"])
            if ts > max_ms:
                max_ms = ts
    return max_ms


def save_klines(rows: list[dict], filepath: str) -> None:
    """Write klines to CSV, overwriting any existing file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_klines(rows: list[dict], filepath: str) -> None:
    """Append new klines to an existing CSV file."""
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerows(rows)


def count_rows(filepath: str) -> int:
    """Count data rows in a CSV file (excluding header)."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    """Fetch 2 years of klines across all timeframes."""
    os.makedirs(DATA_DIR, exist_ok=True)

    force = "--force" in sys.argv
    only_tfs = []
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        only_tfs = sys.argv[idx + 1:]

    # 2 years: Apr 10 2024 to Apr 10 2026
    end_dt = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    start_dt = datetime(2024, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info("Fetching Binance BTC/USDT klines")
    logger.info("  Range: %s to %s (2 years)", start_dt.date(), end_dt.date())

    timeframes_to_fetch = only_tfs if only_tfs else list(TIMEFRAMES.keys())

    for tf in timeframes_to_fetch:
        if tf not in TIMEFRAMES:
            logger.warning("Unknown timeframe: %s, skipping", tf)
            continue

        cfg = TIMEFRAMES[tf]
        filepath = os.path.join(DATA_DIR, f"binance_klines_{tf}.csv")
        expected = (end_ms - start_ms) // cfg["ms_per_candle"]

        if not force:
            existing = count_rows(filepath)
            if existing > expected * 0.9:
                logger.info(
                    "  %s: already have %d/%d candles, skipping (use --force to re-fetch)",
                    tf, existing, expected,
                )
                continue

            # Resume: fetch only from where we left off
            existing_end = get_existing_end_ms(filepath)
            if existing_end > start_ms:
                resume_ms = existing_end + cfg["ms_per_candle"]
                logger.info(
                    "  %s: resuming from %s (%d existing candles)",
                    tf,
                    datetime.fromtimestamp(resume_ms / 1000, tz=timezone.utc).isoformat(),
                    existing,
                )
                t0 = time.time()
                new_rows = fetch_klines(cfg["interval"], resume_ms, end_ms, cfg["ms_per_candle"])
                append_klines(new_rows, filepath)
                total = count_rows(filepath)
                logger.info(
                    "  %s: +%d new candles (%d total, %.1fs)",
                    tf, len(new_rows), total, time.time() - t0,
                )
                continue

        # Full fetch
        logger.info("  %s: fetching ~%d candles...", tf, expected)
        t0 = time.time()
        rows = fetch_klines(cfg["interval"], start_ms, end_ms, cfg["ms_per_candle"])
        save_klines(rows, filepath)
        elapsed = time.time() - t0
        logger.info(
            "  %s: %d candles saved (%.1fs, %.0f candles/s)",
            tf, len(rows), elapsed, len(rows) / elapsed if elapsed > 0 else 0,
        )

    # Summary
    logger.info("")
    logger.info("=== FETCH COMPLETE ===")
    for tf in TIMEFRAMES:
        filepath = os.path.join(DATA_DIR, f"binance_klines_{tf}.csv")
        if os.path.exists(filepath):
            rows = count_rows(filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            logger.info("  %s: %d candles (%.1f MB)", tf, rows, size_mb)
        else:
            logger.info("  %s: not fetched", tf)


if __name__ == "__main__":
    main()
