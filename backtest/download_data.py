"""Download historical 1-minute BTC/USDT candles from Binance REST API.

Binance /api/v3/klines returns up to 1000 candles per request.
6 months of 1m data ~= 262,800 candles ~= 263 requests.

No API key needed for public market data endpoints.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
CANDLES_PER_REQUEST = 1000
RATE_LIMIT_DELAY = 0.25  # seconds between requests (conservative)


def download_binance_candles(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    months_back: int = 6,
    output_dir: str = None,
) -> str:
    """Download historical candles and save to CSV.

    Args:
        symbol: Trading pair (default BTCUSDT).
        interval: Candle interval (default 1m).
        months_back: How many months of history to fetch.
        output_dir: Directory to save the CSV. Defaults to backtest/data/.

    Returns:
        Path to the saved CSV file.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "data")
    os.makedirs(output_dir, exist_ok=True)

    end_time = int(time.time() * 1000)  # Now, in milliseconds
    start_time = int((datetime.now(timezone.utc) - timedelta(days=months_back * 30)).timestamp() * 1000)

    output_file = os.path.join(
        output_dir,
        f"{symbol}_{interval}_{months_back}mo.csv"
    )

    print(f"Downloading {symbol} {interval} candles — {months_back} months")
    print(f"  From: {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    print(f"  To:   {datetime.fromtimestamp(end_time / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

    all_candles = []
    current_start = start_time
    request_count = 0
    total_expected = ((end_time - start_time) / (60 * 1000)) / CANDLES_PER_REQUEST

    client = httpx.Client(timeout=15.0)

    try:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": CANDLES_PER_REQUEST,
            }

            try:
                resp = client.get(BINANCE_KLINES_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print("  Rate limited — waiting 60s...")
                    time.sleep(60)
                    continue
                raise
            except Exception as e:
                print(f"  Request failed: {e}. Retrying in 5s...")
                time.sleep(5)
                continue

            if not data:
                break

            all_candles.extend(data)
            request_count += 1

            # Next batch starts after the last candle
            last_close_time = data[-1][6]  # Close time field
            current_start = last_close_time + 1

            # Progress
            pct = min(100, (len(all_candles) / (months_back * 30 * 24 * 60)) * 100)
            print(
                f"  Request {request_count}: {len(all_candles):,} candles "
                f"({pct:.1f}%) — "
                f"{datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}",
                end="\r",
            )

            if len(data) < CANDLES_PER_REQUEST:
                break  # Reached the end

            time.sleep(RATE_LIMIT_DELAY)

    finally:
        client.close()

    print(f"\n  Downloaded {len(all_candles):,} candles in {request_count} requests")

    # Write to CSV
    # Binance kline fields:
    # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
    # [6] Close time, [7] Quote asset volume, [8] Number of trades,
    # [9] Taker buy base vol, [10] Taker buy quote vol, [11] Ignore
    headers = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for candle in all_candles:
            writer.writerow(candle[:11])

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved to: {output_file} ({file_size_mb:.1f} MB)")
    return output_file


def load_candles_from_csv(csv_path: str) -> list[dict]:
    """Load candles from a downloaded CSV file.

    Returns:
        List of dicts with keys: open_time, open, high, low, close, volume, close_time.
    """
    candles = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                "open_time": int(row["open_time"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "close_time": int(row["close_time"]),
            })
    return candles


if __name__ == "__main__":
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    download_binance_candles(months_back=months)
