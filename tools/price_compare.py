"""Live 3-exchange BTC price comparison monitor.

Standalone diagnostic — does NOT interact with the running bots.
Connects to Binance, Coinbase, and Kraken in parallel, displays the
live aggregated price, and logs every sample to a CSV for offline
analysis.

Purpose: compare the 3-exchange average to the Polymarket UI BTC price
to determine if the current 2-exchange (Binance + Coinbase) average is
the cause of the ~$40 gap user is observing. Target: sustained gap
under $20 vs Polymarket UI.

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
from data.binance_feed import BinanceFeed  # noqa: E402
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
    binance = BinanceFeed()
    coinbase = CoinbaseFeed()
    kraken = KrakenFeed()

    # Start all 3 feeds in background tasks
    binance_task = asyncio.create_task(binance.start())
    coinbase_task = asyncio.create_task(coinbase.start())
    kraken_task = asyncio.create_task(kraken.start())

    # CSV setup
    os.makedirs(CSV_DIR, exist_ok=True)
    csv_file = open(CSV_PATH, "w", newline="", buffering=1)
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp_utc",
        "binance_price", "binance_age_s",
        "coinbase_price", "coinbase_age_s",
        "kraken_price", "kraken_age_s",
        "mean_3", "median_3", "spread_3", "n_feeds_active",
        # USD-native subset (Coinbase + Kraken only)
        "usd_native_mean", "usd_native_spread", "usdt_basis_vs_3",
    ])

    samples = 0
    spreads: list[float] = []
    csv_writes = 0

    # Give feeds time to connect
    print("\nConnecting to Binance, Coinbase, Kraken...")
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
                ("Binance ", binance, binance.price,
                 (now - binance.received_at) if binance.received_at > 0 else 999),
                ("Coinbase", coinbase, coinbase.price,
                 (now - coinbase.received_at) if coinbase.received_at > 0 else 999),
                ("Kraken  ", kraken, kraken.price,
                 (now - kraken.received_at) if kraken.received_at > 0 else 999),
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

            # USD-native subset (Coinbase + Kraken). Binance is BTC/USDT
            # which carries USDT-USD basis distortion (~$10-30 typical),
            # so this subset should track Polymarket's Chainlink BTC/USD
            # reference more closely.
            usd_prices = []
            if coinbase.price > 0 and (now - coinbase.received_at) < 10:
                usd_prices.append(coinbase.price)
            if kraken.price > 0 and (now - kraken.received_at) < 10:
                usd_prices.append(kraken.price)
            if len(usd_prices) >= 2:
                usd_mean = sum(usd_prices) / len(usd_prices)
                usd_spread = max(usd_prices) - min(usd_prices)
                # Compute basis vs all-3 (how much does Binance USDT pull up?)
                if mean_p > 0:
                    usdt_basis = mean_p - usd_mean
                else:
                    usdt_basis = 0.0
            else:
                usd_mean = 0.0
                usd_spread = 0.0
                usdt_basis = 0.0

            # Write CSV row every tick
            writer.writerow([
                ts_iso,
                f"{binance.price:.2f}" if binance.price > 0 else "",
                f"{(now - binance.received_at):.2f}" if binance.received_at > 0 else "",
                f"{coinbase.price:.2f}" if coinbase.price > 0 else "",
                f"{(now - coinbase.received_at):.2f}" if coinbase.received_at > 0 else "",
                f"{kraken.price:.2f}" if kraken.price > 0 else "",
                f"{(now - kraken.received_at):.2f}" if kraken.received_at > 0 else "",
                f"{mean_p:.2f}" if mean_p > 0 else "",
                f"{median_p:.2f}" if median_p > 0 else "",
                f"{spread:.2f}" if mean_p > 0 else "",
                n_active,
                f"{usd_mean:.2f}" if usd_mean > 0 else "",
                f"{usd_spread:.2f}" if usd_mean > 0 else "",
                f"{usdt_basis:.2f}" if usd_mean > 0 and mean_p > 0 else "",
            ])
            csv_writes += 1

            # Aggregate stats
            avg_spread = sum(spreads) / len(spreads) if spreads else 0.0
            max_spread = max(spreads) if spreads else 0.0
            med_spread = statistics.median(spreads) if spreads else 0.0

            # Render display (cursor home + clear)
            sys.stdout.write("\033[H\033[2J")

            print("=" * 67)
            print(f"  3-EXCHANGE BTC PRICE MONITOR              {ts_display}")
            print("=" * 67)
            print()

            for name, _, p, age in feeds:
                status = "[OK]  " if p > 0 and age < 10 else "[OFF] "
                print(f"  {status} {name}:  {fmt_price(p)}   ({fmt_age(age)})")

            print()
            print("  -----  ALL 3 (Binance USDT + Coinbase USD + Kraken USD)  -----")
            if n_active >= 2:
                print(f"  Mean of {n_active}:    {fmt_price(mean_p)}")
                print(f"  Median:       {fmt_price(median_p)}")
                print(f"  Inter-spread: {fmt_price(spread)}   (max - min)")
            else:
                print(f"  Only {n_active} feed(s) active - need >=2 to aggregate")

            print()
            print("  -----  USD-NATIVE ONLY (Coinbase + Kraken)  -----")
            if usd_mean > 0:
                print(f"  Mean:         {fmt_price(usd_mean)}")
                print(f"  Inter-spread: {fmt_price(usd_spread)}")
                basis_sign = "+" if usdt_basis >= 0 else ""
                print(f"  USDT basis:   {basis_sign}{usdt_basis:.2f}   (all-3 minus USD-native)")
                print(f"                ^ if Binance USDT pulls aggregate up, this is +ve")
            else:
                print(f"  Need both Coinbase and Kraken connected")

            print()
            print("  -----  COMPARE TO POLYMARKET UI  -----")
            print(f"  Target: gap < $20 sustained vs Polymarket displayed price")
            print(f"  Watch: USD-native should track Polymarket much more closely")
            print(f"         If it does -> drop Binance from prod oracle")
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
            await binance.stop()
            await coinbase.stop()
            await kraken.stop()
        except Exception:
            pass

        for task in (binance_task, coinbase_task, kraken_task):
            task.cancel()
        try:
            await asyncio.gather(
                binance_task, coinbase_task, kraken_task,
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
