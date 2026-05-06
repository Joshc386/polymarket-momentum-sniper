#!/usr/bin/env python3
"""Market Efficiency Observer — collects real Polymarket data without trading.

Watches every 5-minute BTC Up/Down window and records:
- Market mid-price (YES midpoint) at multiple timepoints within the window
- BTC exchange price at each sample
- Oracle price at window open and close
- Actual resolution (UP or DOWN)

Run for 24-48 hours, then analyse with efficiency_analyser.py.

Usage:
    python -m backtest.market_observer
    python -m backtest.market_observer --output market_data.csv --interval 5
"""

import asyncio
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")

from core.config import Config
from core.market_discovery import MarketDiscovery
from data.binance_feed import BinanceFeed
from data.chainlink_oracle import ChainlinkOracle
from data.orderbook import OrderbookManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("observer")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

# ANSI
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


class WindowObservation:
    """Tracks observations for a single 5-minute window."""

    def __init__(self, slug: str, market_question: str):
        self.slug = slug
        self.market_question = market_question
        self.start_time = time.time()

        # Samples: list of (timestamp, secs_remaining, btc_price, oracle_price,
        #                    yes_mid, yes_bid, yes_ask, spread)
        self.samples: list[dict] = []

        # Resolution
        self.oracle_open_price: float = 0.0
        self.oracle_close_price: float = 0.0
        self.btc_open_price: float = 0.0
        self.btc_close_price: float = 0.0
        self.resolution: str = ""  # "UP" or "DOWN"

    def add_sample(
        self,
        secs_remaining: float,
        btc_price: float,
        oracle_price: float,
        yes_bid: float,
        yes_ask: float,
        no_bid: float,
        no_ask: float,
    ):
        yes_mid = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else 0.5
        spread = yes_ask - yes_bid if yes_ask > 0 and yes_bid > 0 else 0.0

        self.samples.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "secs_remaining": round(secs_remaining, 1),
            "btc_price": btc_price,
            "oracle_price": oracle_price,
            "yes_mid": round(yes_mid, 4),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "spread": round(spread, 4),
        })

    def resolve(self, oracle_close: float):
        self.oracle_close_price = oracle_close
        if self.oracle_close_price >= self.oracle_open_price:
            self.resolution = "UP"
        else:
            self.resolution = "DOWN"

    @property
    def avg_yes_mid(self) -> float:
        if not self.samples:
            return 0.5
        return sum(s["yes_mid"] for s in self.samples) / len(self.samples)

    @property
    def avg_spread(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s["spread"] for s in self.samples) / len(self.samples)


def write_observations_to_csv(observations: list[WindowObservation], output_path: str):
    """Append completed observations to CSV."""
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "slug", "resolution", "oracle_open", "oracle_close",
                "btc_open", "btc_close", "num_samples", "avg_yes_mid",
                "avg_spread",
                # Snapshot at specific time buckets
                "yes_mid_at_4m", "yes_mid_at_3m", "yes_mid_at_2m",
                "yes_mid_at_1m", "yes_mid_at_30s",
                "spread_at_4m", "spread_at_3m", "spread_at_2m",
                "spread_at_1m", "spread_at_30s",
            ])

        for obs in observations:
            if not obs.resolution:
                continue

            # Find samples closest to each time bucket
            def sample_near(target_secs):
                best = None
                best_dist = 999
                for s in obs.samples:
                    dist = abs(s["secs_remaining"] - target_secs)
                    if dist < best_dist:
                        best_dist = dist
                        best = s
                return best

            s4 = sample_near(240)
            s3 = sample_near(180)
            s2 = sample_near(120)
            s1 = sample_near(60)
            s30 = sample_near(30)

            writer.writerow([
                obs.slug,
                obs.resolution,
                f"{obs.oracle_open_price:.2f}",
                f"{obs.oracle_close_price:.2f}",
                f"{obs.btc_open_price:.2f}",
                f"{obs.btc_close_price:.2f}",
                len(obs.samples),
                f"{obs.avg_yes_mid:.4f}",
                f"{obs.avg_spread:.4f}",
                f"{s4['yes_mid']:.4f}" if s4 else "",
                f"{s3['yes_mid']:.4f}" if s3 else "",
                f"{s2['yes_mid']:.4f}" if s2 else "",
                f"{s1['yes_mid']:.4f}" if s1 else "",
                f"{s30['yes_mid']:.4f}" if s30 else "",
                f"{s4['spread']:.4f}" if s4 else "",
                f"{s3['spread']:.4f}" if s3 else "",
                f"{s2['spread']:.4f}" if s2 else "",
                f"{s1['spread']:.4f}" if s1 else "",
                f"{s30['spread']:.4f}" if s30 else "",
            ])


def write_raw_samples_csv(observations: list[WindowObservation], output_path: str):
    """Write every individual sample for detailed analysis."""
    file_exists = os.path.exists(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "slug", "timestamp", "secs_remaining", "btc_price",
                "oracle_price", "yes_mid", "yes_bid", "yes_ask",
                "spread", "resolution",
            ])
        for obs in observations:
            for s in obs.samples:
                writer.writerow([
                    obs.slug,
                    s["timestamp"],
                    s["secs_remaining"],
                    f"{s['btc_price']:.2f}",
                    f"{s['oracle_price']:.2f}",
                    f"{s['yes_mid']:.4f}",
                    f"{s['yes_bid']:.4f}",
                    f"{s['yes_ask']:.4f}",
                    f"{s['spread']:.4f}",
                    obs.resolution,
                ])


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Observe Polymarket market efficiency")
    parser.add_argument("--output", default="backtest/data/market_observations.csv")
    parser.add_argument("--raw-output", default="backtest/data/market_samples_raw.csv")
    parser.add_argument("--interval", type=int, default=5, help="Sample interval in seconds")
    args = parser.parse_args()

    config = Config.load()
    logger.info("Starting Market Efficiency Observer")

    # Data feeds
    binance = BinanceFeed()
    oracle = ChainlinkOracle()
    market_discovery = MarketDiscovery()
    orderbook_mgr = OrderbookManager()

    # Shutdown
    shutdown_event = asyncio.Event()
    def handle_shutdown():
        logger.info("Shutting down observer...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, handle_shutdown)
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown)

    # Start feeds
    binance_task = asyncio.create_task(binance.start())
    oracle_task = asyncio.create_task(oracle.start(poll_interval=2.0))

    logger.info("Waiting for data feeds...")
    for _ in range(50):
        if binance.price > 0:
            break
        await asyncio.sleep(0.1)

    # State
    last_market_refresh = 0.0
    last_orderbook_refresh = 0.0
    last_window_slug = ""
    current_obs: WindowObservation | None = None
    completed_observations: list[WindowObservation] = []
    total_windows_observed = 0
    total_up = 0
    total_down = 0
    flush_interval = 10  # Write to disk every N windows

    try:
        while not shutdown_event.is_set():
            now = time.time()

            # Refresh market — only when needed
            # If we have an active market, don't spam the API
            needs_refresh = False
            if market_discovery.current_market is None:
                needs_refresh = now - last_market_refresh > 5
            elif market_discovery.current_market.time_remaining <= 0:
                needs_refresh = now - last_market_refresh > 2
            else:
                # We have an active market — only refresh near window boundaries
                needs_refresh = now - last_market_refresh > 30

            if needs_refresh:
                try:
                    await market_discovery.refresh()
                except Exception as e:
                    logger.debug("Market discovery refresh failed: %s", e)
                last_market_refresh = now

            mkt = market_discovery.current_market

            # Window transition
            if mkt and mkt.slug != last_window_slug:
                # Resolve previous window
                if current_obs and last_window_slug:
                    await oracle.fetch_once()
                    current_obs.resolve(oracle.price)
                    current_obs.btc_close_price = binance.price

                    if current_obs.resolution == "UP":
                        total_up += 1
                    else:
                        total_down += 1
                    total_windows_observed += 1
                    completed_observations.append(current_obs)

                    # Flush to disk periodically
                    if len(completed_observations) >= flush_interval:
                        write_observations_to_csv(completed_observations, args.output)
                        write_raw_samples_csv(completed_observations, args.raw_output)
                        logger.info(
                            f"Flushed {len(completed_observations)} windows to disk "
                            f"(total: {total_windows_observed})"
                        )
                        completed_observations = []

                # Start new window observation
                last_window_slug = mkt.slug
                current_obs = WindowObservation(slug=mkt.slug, market_question=mkt.question)

                # Capture oracle open
                if oracle.price > 0:
                    current_obs.oracle_open_price = oracle.price
                else:
                    await oracle.fetch_once()
                    current_obs.oracle_open_price = oracle.price
                current_obs.btc_open_price = binance.price

            # Refresh orderbook
            if mkt and mkt.yes_token_id and now - last_orderbook_refresh > args.interval:
                try:
                    orderbook_mgr.fetch(mkt.yes_token_id, mkt.no_token_id)
                except Exception as e:
                    logger.debug("Orderbook fetch failed: %s", e)
                last_orderbook_refresh = now

                # Record sample
                if current_obs and mkt.is_active:
                    ob = orderbook_mgr.last_summary
                    if ob:
                        current_obs.add_sample(
                            secs_remaining=mkt.seconds_remaining,
                            btc_price=binance.price,
                            oracle_price=oracle.price,
                            yes_bid=ob.yes_best_bid,
                            yes_ask=ob.yes_best_ask,
                            no_bid=ob.no_best_bid,
                            no_ask=ob.no_best_ask,
                        )

            # Display
            up_pct = (total_up / total_windows_observed * 100) if total_windows_observed > 0 else 0
            obs_samples = len(current_obs.samples) if current_obs else 0

            status_lines = [
                CLEAR,
                f"{BOLD}{CYAN}  MARKET EFFICIENCY OBSERVER{RESET}",
                f"  {DIM}{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}{RESET}",
                "",
                f"  {BOLD}BTC:{RESET} ${binance.price:,.2f}" if binance.price > 0 else f"  {BOLD}BTC:{RESET} Connecting...",
            ]

            if mkt:
                r = mkt.time_remaining
                mi, se = int(r // 60), int(r % 60)
                status_lines.append(
                    f"  {BOLD}Market:{RESET} {mkt.question}  {GREEN}{mi}m{se:02d}s{RESET}"
                )
                ob = orderbook_mgr.last_summary
                if ob:
                    mid = (ob.yes_best_bid + ob.yes_best_ask) / 2
                    status_lines.append(
                        f"  YES mid: {BOLD}{mid:.4f}{RESET}  "
                        f"Spread: {ob.spread:.4f}  "
                        f"Samples this window: {obs_samples}"
                    )
            else:
                status_lines.append(f"  {BOLD}Market:{RESET} {YELLOW}Searching...{RESET}")

            status_lines.extend([
                "",
                f"  {BOLD}Windows observed:{RESET} {total_windows_observed}",
                f"  UP: {total_up}  DOWN: {total_down}  ({up_pct:.1f}% UP)",
                f"  Pending flush: {len(completed_observations)}",
                f"  Output: {args.output}",
                "",
                f"  {DIM}Collecting data... Ctrl+C to stop and save{RESET}",
            ])

            print("\n".join(status_lines), flush=True)

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        # Flush remaining
        if current_obs and current_obs.samples:
            await oracle.fetch_once()
            current_obs.resolve(oracle.price)
            current_obs.btc_close_price = binance.price
            completed_observations.append(current_obs)

        if completed_observations:
            write_observations_to_csv(completed_observations, args.output)
            write_raw_samples_csv(completed_observations, args.raw_output)
            logger.info(f"Final flush: {len(completed_observations)} windows")

        logger.info(f"Total windows observed: {total_windows_observed + len(completed_observations)}")

        await binance.stop()
        await oracle.stop()
        binance_task.cancel()
        oracle_task.cancel()
        for t in [binance_task, oracle_task]:
            try:
                await t
            except asyncio.CancelledError:
                pass
        await market_discovery.close()
        orderbook_mgr.close()

        logger.info("Observer stopped.")
        print(f"\n  Data saved to: {args.output}")
        print(f"  Raw samples: {args.raw_output}")
        print(f"  Run: python -m backtest.efficiency_analyser {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
