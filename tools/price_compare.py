"""Live 3-exchange BTC/USD price comparison monitor.

Standalone diagnostic — does NOT interact with the running bots.
Connects to Coinbase, Kraken, and Bitstamp in parallel — all three
are USD-native (no USDT basis distortion). Displays the live aggregate
and logs every sample to a CSV for offline analysis.

Note: Binance.US was previously included but is geo-blocked from
UK/EU IPs (only ~1 price/min reaches us), so it's been removed.
Bitstamp replaces it as the third USD-native venue.

Purpose: compare the USD-native average to the Polymarket UI BTC
price. Polymarket settles in USD via Chainlink BTC/USD, so a
USD-native aggregate should track it within ~$5-$15.

Usage:
    python -m tools.price_compare

Output:
    - Live terminal display, refreshed every 0.5s
    - CSV at data_runtime/price_compare_YYYYMMDD_HHMMSS.csv
"""

import asyncio
import csv
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone

# Silence the noisy feed reconnect logs - we have our own connection display
logging.basicConfig(level=logging.ERROR)

# Import after logging is configured
from data.bitstamp_feed import BitstampFeed  # noqa: E402
from data.coinbase_feed import CoinbaseFeed  # noqa: E402
from data.kraken_feed import KrakenFeed  # noqa: E402


CSV_DIR = "data_runtime"
CSV_PATH = os.path.join(
    CSV_DIR,
    f"price_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
)


def fmt_price(p: float) -> str:
    if p <= 0:
        return "      ---"
    return f"${p:>10,.2f}"


def fmt_age(seconds: float) -> str:
    if seconds > 999:
        return "  stale"
    return f"{seconds:>5.1f}s"


async def main() -> None:
    coinbase = CoinbaseFeed()
    kraken = KrakenFeed()
    bitstamp = BitstampFeed()

    # Start feeds in background tasks
    coinbase_task = asyncio.create_task(coinbase.start())
    kraken_task = asyncio.create_task(kraken.start())
    bitstamp_task = asyncio.create_task(bitstamp.start())

    # CSV setup
    os.makedirs(CSV_DIR, exist_ok=True)
    csv_file = open(CSV_PATH, "w", newline="", buffering=1)
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp_utc",
        "coinbase_price", "coinbase_age_s",
        "kraken_price", "kraken_age_s",
        "bitstamp_price", "bitstamp_age_s",
        "mean_3", "median_3", "spread_3", "n_feeds_active",
    ])

    samples = 0
    spreads: list[float] = []
    csv_writes = 0

    # Give feeds time to connect
    print("\nConnecting to Coinbase, Kraken, Bitstamp...")
    print(f"CSV: {CSV_PATH}")
    print("Wait ~5 seconds for initial connections...\n")
    await asyncio.sleep(5)

    # Hide cursor for cleaner display
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        while True:
            now = time.time()
            ts_display = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            ts_iso = datetime.now(timezone.utc).isoformat()

            feeds = [
                ("Coinbase   BTC/USD", coinbase, coinbase.price,
                 (now - coinbase.received_at) if coinbase.received_at > 0 else 999),
                ("Kraken     BTC/USD", kraken, kraken.price,
                 (now - kraken.received_at) if kraken.received_at > 0 else 999),
                ("Bitstamp   BTC/USD", bitstamp, bitstamp.price,
                 (now - bitstamp.received_at) if bitstamp.received_at > 0 else 999),
            ]

            active_prices = [p for _, _, p, age in feeds if p > 0 and age < 10]
            n_active = len(active_prices)

            if n_active >= 2:
                mean_p = sum(active_prices) / n_active
                median_p = statistics.median(active_prices)
                spread = max(active_prices) - min(active_prices)
                samples += 1
                spreads.append(spread)
            else:
                mean_p = median_p = spread = 0.0

            # All three feeds are USD-native (Coinbase, Kraken, Bitstamp).
            # No USDT basis distortion. Aggregate should track Polymarket
            # Chainlink BTC/USD within ~$5-$15 if the feeds are healthy.

            # Write CSV row every tick
            writer.writerow([
                ts_iso,
                f"{coinbase.price:.2f}" if coinbase.price > 0 else "",
                f"{(now - coinbase.received_at):.2f}" if coinbase.received_at > 0 else "",
                f"{kraken.price:.2f}" if kraken.price > 0 else "",
                f"{(now - kraken.received_at):.2f}" if kraken.received_at > 0 else "",
                f"{bitstamp.price:.2f}" if bitstamp.price > 0 else "",
                f"{(now - bitstamp.received_at):.2f}" if bitstamp.received_at > 0 else "",
                f"{mean_p:.2f}" if mean_p > 0 else "",
                f"{median_p:.2f}" if median_p > 0 else "",
                f"{spread:.2f}" if mean_p > 0 else "",
                n_active,
            ])
            csv_writes += 1

            # Aggregate stats
            avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
            max_spread = max(spreads) if spreads else 0.0
            med_spread = statistics.median(spreads) if spreads else 0.0

            # Render display (cursor home + clear)
            sys.stdout.write("\033[H\033[2J")

            print("=" * 75)
            print(f"  3-EXCHANGE BTC/USD PRICE MONITOR (all USD-native)   {ts_display}")
            print("=" * 75)
            print()

            for name, _, p, age in feeds:
                status = "[OK]  " if p > 0 and age < 10 else "[OFF] "
                print(f"  {status} {name}:  {fmt_price(p)}   ({fmt_age(age)})")

            print()
            print("  -----  AGGREGATE (equal weights, all USD-native)  -----")
            if n_active >= 2:
                print(f"  Mean of {n_active}:    {fmt_price(mean_p)}")
                print(f"  Median:       {fmt_price(median_p)}")
                print(f"  Inter-spread: {fmt_price(spread)}   (max - min)")
            else:
                print(f"  Only {n_active} feed(s) active - need >=2 to aggregate")

            print()
            print("  -----  COMPARE TO POLYMARKET UI  -----")
            print(f"  Target: gap < $20 sustained vs Polymarket displayed price")
            print(f"  Polymarket uses Chainlink BTC/USD, so a USD-native")
            print(f"  aggregate should track within ~$5-15.")
            print()
            print(f"  Samples logged:  {csv_writes:>6,}")
            print(f"  3-feed spread:   avg ${avg_spread:>6.2f}  max ${max_spread:>6.2f}")
            print()
            print(f"  CSV: {CSV_PATH}")
            print(f"  Press Ctrl+C to stop")
            print("=" * 67)
            sys.stdout.flush()

            await asyncio.sleep(0.5)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    finally:
        # Restore cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("\n\nShutting down feeds...")
        try:
            await coinbase.stop()
            await kraken.stop()
            await bitstamp.stop()
        except Exception:
            pass

        for task in (coinbase_task, kraken_task, bitstamp_task):
            task.cancel()
        try:
            await asyncio.gather(
                coinbase_task, kraken_task, bitstamp_task,
                return_exceptions=True,
            )
        except Exception:
            pass

        csv_file.close()
        print(f"\n  CSV saved: {CSV_PATH}")
        print(f"  Total samples: {csv_writes:,}")
        if spreads:
            print(f"  Avg spread:    ${sum(spreads)/len(spreads):.2f}")
            print(f"  Median spread: ${statistics.median(spreads):.2f}")
            print(f"  Max spread:    ${max(spreads):.2f}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone.")
