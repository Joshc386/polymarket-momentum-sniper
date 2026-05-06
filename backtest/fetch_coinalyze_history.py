"""Fetch historical Coinalyze data for backtest L3/L5 signal reconstruction.

Downloads:
- Open interest history (5-min intervals)
- Long/short ratio history (5-min intervals)
- Funding rate history (8-hour intervals)

Usage:
    python -m backtest.fetch_coinalyze_history

Output:
    backtest/data/coinalyze_oi_history.csv
    backtest/data/coinalyze_lsr_history.csv
    backtest/data/coinalyze_funding_history.csv

API: https://api.coinalyze.net/v1
Rate limit: 40 calls/minute (free tier)
"""

import csv
import logging
import math
import os
import time
from datetime import datetime, timezone, timedelta

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_URL = "https://api.coinalyze.net/v1"

# Same symbols as the live bot
BTC_PERP_SYMBOLS = [
    "BTCUSDT_PERP.A",   # Binance USDT-M
    "BTCUSD_PERP.A",    # Binance COIN-M
    "BTCUSDC_PERP.A",   # Binance USDC-M
    "BTCUSDT.6",         # Bybit USDT
    "BTCUSD.6",          # Bybit Inverse
    "BTCUSDT_PERP.0",   # BitMEX USDT
    "BTCUSD_PERP.4",    # Deribit
]

BTC_LSR_SYMBOLS = [
    "BTCUSDT_PERP.A",
    "BTCUSD_PERP.A",
    "BTCUSDC_PERP.A",
    "BTCUSDT.6",
    "BTCUSD.6",
]


def fetch_oi_history(
    api_key: str,
    start_ts: int,
    end_ts: int,
    output_path: str,
) -> int:
    """Fetch open interest history at 5-min intervals.

    Args:
        api_key: Coinalyze API key.
        start_ts: Start Unix timestamp.
        end_ts: End Unix timestamp.
        output_path: CSV output path.

    Returns:
        Total data points fetched.
    """
    symbols_str = ",".join(BTC_PERP_SYMBOLS)
    total_points = 0

    # Coinalyze limits data per request, so we chunk by day
    chunk_size = 86400  # 1 day in seconds

    write_header = True
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "timestamp_utc", "total_oi_usd"])

    current = start_ts
    with httpx.Client(timeout=30.0, headers={"api_key": api_key}) as client:
        while current < end_ts:
            chunk_end = min(current + chunk_size, end_ts)
            try:
                resp = client.get(
                    f"{BASE_URL}/open-interest-history",
                    params={
                        "symbols": symbols_str,
                        "interval": "5min",
                        "from": str(current),
                        "to": str(chunk_end),
                        "convert_to_usd": "true",
                    },
                )

                if resp.status_code == 429:
                    retry_after = math.ceil(float(resp.headers.get("Retry-After", "60"))) + 1
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"OI history HTTP {resp.status_code}: {resp.text[:200]}")
                    current = chunk_end
                    time.sleep(3.0)
                    continue

                data = resp.json()

                # Aggregate OI across all symbols per timestamp
                # Response: [{symbol, history: [{t, c}]}]
                ts_to_oi: dict[int, float] = {}
                for item in data:
                    for point in item.get("history", []):
                        ts = point.get("t", 0)
                        val = float(point.get("c", 0))
                        ts_to_oi[ts] = ts_to_oi.get(ts, 0) + val

                if ts_to_oi:
                    with open(output_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for ts in sorted(ts_to_oi.keys()):
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            writer.writerow([ts, dt.strftime("%Y-%m-%dT%H:%M:%SZ"), ts_to_oi[ts]])
                    total_points += len(ts_to_oi)

                dt_str = datetime.fromtimestamp(chunk_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                logger.info(f"OI: {len(ts_to_oi)} points up to {dt_str} (total: {total_points:,})")

            except Exception as e:
                logger.error(f"OI fetch error: {e}")

            current = chunk_end
            time.sleep(3.0)  # Stay well under 40/min rate limit

    return total_points


def fetch_lsr_history(
    api_key: str,
    start_ts: int,
    end_ts: int,
    output_path: str,
) -> int:
    """Fetch long/short ratio history at 5-min intervals."""
    symbols_str = ",".join(BTC_LSR_SYMBOLS)
    total_points = 0
    chunk_size = 86400

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "timestamp_utc", "avg_ls_ratio", "avg_long_pct", "avg_short_pct"])

    current = start_ts
    with httpx.Client(timeout=30.0, headers={"api_key": api_key}) as client:
        while current < end_ts:
            chunk_end = min(current + chunk_size, end_ts)
            try:
                resp = client.get(
                    f"{BASE_URL}/long-short-ratio-history",
                    params={
                        "symbols": symbols_str,
                        "interval": "5min",
                        "from": str(current),
                        "to": str(chunk_end),
                    },
                )

                if resp.status_code == 429:
                    retry_after = math.ceil(float(resp.headers.get("Retry-After", "60"))) + 1
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"LSR history HTTP {resp.status_code}: {resp.text[:200]}")
                    current = chunk_end
                    time.sleep(3.0)
                    continue

                data = resp.json()

                # Aggregate L/S ratio across symbols per timestamp
                # Response: [{symbol, history: [{t, r, l, s}]}]
                ts_data: dict[int, list[dict]] = {}
                for item in data:
                    for point in item.get("history", []):
                        ts = point.get("t", 0)
                        if ts not in ts_data:
                            ts_data[ts] = []
                        ts_data[ts].append({
                            "ratio": float(point.get("r", 1.0)),
                            "long_pct": float(point.get("l", 50.0)),
                            "short_pct": float(point.get("s", 50.0)),
                        })

                if ts_data:
                    with open(output_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for ts in sorted(ts_data.keys()):
                            points = ts_data[ts]
                            avg_ratio = sum(p["ratio"] for p in points) / len(points)
                            avg_long = sum(p["long_pct"] for p in points) / len(points)
                            avg_short = sum(p["short_pct"] for p in points) / len(points)
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            writer.writerow([
                                ts,
                                dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                round(avg_ratio, 4),
                                round(avg_long, 2),
                                round(avg_short, 2),
                            ])
                    total_points += len(ts_data)

                dt_str = datetime.fromtimestamp(chunk_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                logger.info(f"LSR: {len(ts_data)} points up to {dt_str} (total: {total_points:,})")

            except Exception as e:
                logger.error(f"LSR fetch error: {e}")

            current = chunk_end
            time.sleep(3.0)

    return total_points


def fetch_funding_history(
    api_key: str,
    start_ts: int,
    end_ts: int,
    output_path: str,
) -> int:
    """Fetch funding rate history at 1-hour intervals."""
    symbols_str = ",".join(BTC_PERP_SYMBOLS)
    total_points = 0
    chunk_size = 86400 * 7  # 7 days per chunk (funding is less granular)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "timestamp_utc", "avg_funding_rate"])

    current = start_ts
    with httpx.Client(timeout=30.0, headers={"api_key": api_key}) as client:
        while current < end_ts:
            chunk_end = min(current + chunk_size, end_ts)
            try:
                resp = client.get(
                    f"{BASE_URL}/funding-rate-history",
                    params={
                        "symbols": symbols_str,
                        "interval": "1hour",
                        "from": str(current),
                        "to": str(chunk_end),
                    },
                )

                if resp.status_code == 429:
                    retry_after = math.ceil(float(resp.headers.get("Retry-After", "60"))) + 1
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"Funding history HTTP {resp.status_code}: {resp.text[:200]}")
                    current = chunk_end
                    time.sleep(3.0)
                    continue

                data = resp.json()

                # Aggregate funding across symbols per timestamp
                ts_rates: dict[int, list[float]] = {}
                for item in data:
                    for point in item.get("history", []):
                        ts = point.get("t", 0)
                        val = float(point.get("c", 0))
                        if ts not in ts_rates:
                            ts_rates[ts] = []
                        ts_rates[ts].append(val)

                if ts_rates:
                    with open(output_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for ts in sorted(ts_rates.keys()):
                            rates = ts_rates[ts]
                            avg = sum(rates) / len(rates)
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            writer.writerow([
                                ts,
                                dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                round(avg, 6),
                            ])
                    total_points += len(ts_rates)

                dt_str = datetime.fromtimestamp(chunk_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                logger.info(f"Funding: {len(ts_rates)} points up to {dt_str} (total: {total_points:,})")

            except Exception as e:
                logger.error(f"Funding fetch error: {e}")

            current = chunk_end
            time.sleep(3.0)

    return total_points


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.getenv("COINALYZE_API_KEY", "")
    if not api_key:
        logger.error("Set COINALYZE_API_KEY environment variable")
        logger.error("Get one at: https://coinalyze.net/api/")
        return

    # Read date range from markets CSV
    markets_path = os.path.join(DATA_DIR, "polybacktest_markets.csv")
    earliest = latest = None
    with open(markets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["start_time", "end_time"]:
                t = row.get(key, "")
                if t:
                    dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if earliest is None or dt < earliest:
                        earliest = dt
                    if latest is None or dt > latest:
                        latest = dt

    if not earliest or not latest:
        logger.error("Could not determine date range")
        return

    # Add buffer
    start_ts = int((earliest - timedelta(hours=1)).timestamp())
    end_ts = int((latest + timedelta(minutes=5)).timestamp())

    logger.info(f"Date range: {earliest.isoformat()} to {latest.isoformat()}")
    days = (end_ts - start_ts) / 86400
    logger.info(f"Fetching {days:.1f} days of data...")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Fetch all three data types
    oi_path = os.path.join(DATA_DIR, "coinalyze_oi_history.csv")
    lsr_path = os.path.join(DATA_DIR, "coinalyze_lsr_history.csv")
    funding_path = os.path.join(DATA_DIR, "coinalyze_funding_history.csv")

    logger.info("=== Fetching Open Interest History ===")
    oi_count = fetch_oi_history(api_key, start_ts, end_ts, oi_path)

    logger.info("=== Fetching Long/Short Ratio History ===")
    lsr_count = fetch_lsr_history(api_key, start_ts, end_ts, lsr_path)

    logger.info("=== Fetching Funding Rate History ===")
    funding_count = fetch_funding_history(api_key, start_ts, end_ts, funding_path)

    logger.info(f"\nDone! OI: {oi_count:,}, LSR: {lsr_count:,}, Funding: {funding_count:,}")


if __name__ == "__main__":
    main()
