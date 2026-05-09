#!/usr/bin/env python3
"""Multi-Bot Runner — run N strategies side-by-side on shared data feeds.

Entry point: python multi_runner.py

Shares all data feeds (Binance, Oracle, Coinbase, CoinGlass, Coinalyze,
Binance Liquidations, Orderbook, Market Discovery) across bots. Each bot
owns its own signal processing, entry logic, sizing, risk, execution
engine, and database.

main.py is untouched — this is a parallel entry point.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Fix Windows console encoding for Unicode/ANSI output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")  # Enable ANSI escape codes on Windows

# Install fastest available event loop (uvloop/winloop) before any asyncio use
from core.event_loop import setup_event_loop
setup_event_loop()

from core.polymarket_client import PolymarketClient
from core.market_discovery import MarketDiscovery
from core.health_monitor import HealthMonitor
from core.data_snapshot import build_snapshot
from core.multi_dashboard import render_multi_dashboard
from data.binance_feed import BinanceFeed
from data.polymarket_oracle import PolymarketOracle
from data.orderbook import OrderbookManager
from data.polymarket_ws import PolymarketWebSocket
from data.coinglass_scraper import CoinGlassScraper
from data.coinbase_feed import CoinbaseFeed
from data.coinalyze_feed import CoinalyzeFeed
from data.binance_liquidations import BinanceLiquidationFeed
from notifications.telegram import TelegramNotifier
from strategies.registry import create_bot
from tools.latency_logger import LatencyLogger
from data.sm_wallets import SMWalletRegistry
from data.sm_trade_monitor import SMTradeMonitor

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("multi_runner")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)


CLEAR_SCREEN = "\033[2J\033[H"


def load_config(path: str = "config_multi.yaml") -> dict:
    """Load the multi-bot YAML config.

    Args:
        path: Path to config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def detect_resolution(
    market_discovery: MarketDiscovery,
    slug: str,
    oracle: PolymarketOracle,
) -> str:
    """Determine market resolution — preferring actual Polymarket result.

    First tries the authoritative source: Polymarket's own resolution
    from the Gamma API (outcomePrices). Falls back to oracle price
    comparison only if the API resolution is unavailable.

    Args:
        market_discovery: MarketDiscovery instance for API lookups.
        slug: Slug of the market to resolve.
        oracle: PolymarketOracle as fallback for price comparison.

    Returns:
        'UP' or 'DOWN'.
    """
    # Primary: actual market resolution from Polymarket
    resolution = await market_discovery.fetch_resolution(slug)
    if resolution in ("UP", "DOWN"):
        logger.info(f"Resolution from Polymarket API: {resolution}")
        return resolution

    # Fallback: oracle price comparison (may not match exact resolution)
    logger.warning(
        f"Market {slug} resolution unknown from API — "
        f"falling back to oracle price comparison"
    )
    await oracle.fetch_once()
    if oracle.price >= oracle.window_open_price:
        return "UP"
    return "DOWN"


async def main() -> None:
    """Main entry point for the multi-bot runner."""
    cfg = load_config()
    logger.info("Loaded config_multi.yaml")

    # Ensure data_runtime directory exists
    Path("data_runtime").mkdir(exist_ok=True)

    # ── Shared data feeds ─────────────────────────────────────────────
    binance = BinanceFeed()
    oracle = PolymarketOracle()
    coinbase = CoinbaseFeed()
    health = HealthMonitor()

    # Polymarket client (for orderbook + market discovery)
    # Build a minimal config object for PolymarketClient
    from core.config import Config
    base_config = Config.load()
    poly_client = PolymarketClient(base_config)
    await poly_client.connect()
    market_discovery = MarketDiscovery()
    orderbook_mgr = OrderbookManager()  # REST fallback
    ws_feed = PolymarketWebSocket()     # Primary: real-time WS

    # Liquidation data
    coinglass = CoinGlassScraper(
        refresh_interval=cfg.get("liquidation", {}).get("refresh_interval_sec", 60)
    )

    # Coinalyze cross-exchange sentiment
    coinalyze = CoinalyzeFeed(
        refresh_interval=cfg.get("coinalyze", {}).get("refresh_interval", 60),
        api_key=base_config.coinalyze_api_key,
    )

    # Binance live liquidation stream
    binance_liqs = BinanceLiquidationFeed(window_seconds=300.0)

    # ── Telegram notifications ────────────────────────────────────────
    tg_cfg = cfg.get("telegram", {})
    telegram = TelegramNotifier(
        bot_token=base_config.telegram_bot_token,
        chat_id=base_config.telegram_chat_id,
        enabled=tg_cfg.get("enabled", False),
        paper_prefix=tg_cfg.get("paper_prefix", "[MULTI] "),
    )

    # ── Shared SM Trade Monitor (L9) ────────────────────────────────
    # Create a single shared SM monitor if any bot has SM confirmation
    # enabled. All bots watch the same market, so one RPC poller suffices.
    sm_monitor = None
    bots_cfg = cfg.get("bots", {})
    any_sm_enabled = any(
        bot_cfg.get("sm_confirmation", {}).get("enabled", False)
        for bot_cfg in bots_cfg.values()
        if bot_cfg.get("enabled", True)
    )
    if any_sm_enabled:
        sm_registry = SMWalletRegistry()
        sm_poll_interval = 3.0
        # Find the poll interval from the first SM-enabled bot
        for bot_cfg in bots_cfg.values():
            sm_cfg = bot_cfg.get("sm_confirmation", {})
            if sm_cfg.get("enabled", False):
                sm_poll_interval = sm_cfg.get("poll_interval", 3.0)
                break
        sm_monitor = SMTradeMonitor(
            _sm_registry=sm_registry,
            _rpcs=[base_config.polygon_rpc_url] if base_config.polygon_rpc_url else [],
        )
        logger.info(
            "Shared SM trade monitor created (poll every %.1fs, %d wallets)",
            sm_poll_interval, sm_registry.wallet_count,
        )

    # ── Instantiate bots ──────────────────────────────────────────────
    bots = []
    for bot_name, bot_cfg in bots_cfg.items():
        if not bot_cfg.get("enabled", True):
            logger.info(f"Skipping disabled bot: {bot_name}")
            continue
        strategy_name = bot_cfg.get("strategy", "")
        if not strategy_name:
            logger.error(f"Bot '{bot_name}' has no strategy defined, skipping")
            continue
        # Inject shared SM monitor reference for bots that need it
        if sm_monitor and bot_cfg.get("sm_confirmation", {}).get("enabled", False):
            bot_cfg["_sm_monitor"] = sm_monitor
        try:
            bot = create_bot(name=bot_name, strategy_name=strategy_name, cfg=bot_cfg)
            bots.append(bot)
            logger.info(f"Created bot: {bot_name} (strategy={strategy_name})")
        except KeyError as e:
            logger.error(f"Failed to create bot '{bot_name}': {e}")

    if not bots:
        logger.error("No bots configured. Check config_multi.yaml")
        return

    # ── Latency measurement logger ───────────────────────────────────
    latency_logger = LatencyLogger(
        output_dir="data_runtime",
        move_threshold_bps=4.0,     # 4bps (~$32 at $80K BTC)
        move_window_secs=5.0,       # Detect moves within 5-second windows
        catchup_threshold_pct=0.60, # 60% gap closure = caught up
        max_tracking_secs=120.0,    # Track for full 2 minutes
        min_gap_usd=10.0,          # Oracle must be $10+ behind to track
        cooldown_secs=15.0,         # No double-counting within 15s
    )

    logger.info(f"Running {len(bots)} bot(s): {[b.name for b in bots]}")
    await telegram.notify_bot_event(
        "Multi-Bot Started",
        f"Bots: {', '.join(b.name for b in bots)} | Mode: paper",
    )

    # ── Shutdown handler ──────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def handle_shutdown() -> None:
        logger.info("Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, handle_shutdown)
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown)

    # Register taker ratio callbacks from each bot on shared Binance feed
    for bot in bots:
        if hasattr(bot, "_taker_ratio"):
            binance.trade_callbacks.append(bot._taker_ratio.on_trade)

    # ── Start background feeds ────────────────────────────────────────
    binance_task = asyncio.create_task(binance.start())
    oracle_task = asyncio.create_task(oracle.start())  # Price fed from Binance/Coinbase average
    coinbase_task = asyncio.create_task(coinbase.start())
    coinglass_task = asyncio.create_task(
        coinglass.start(current_price_getter=lambda: binance.price)
    )
    coinalyze_task = asyncio.create_task(coinalyze.start())
    binance_liq_task = asyncio.create_task(binance_liqs.start())
    ws_feed_task = asyncio.create_task(ws_feed.start())
    sm_monitor_task = None
    if sm_monitor:
        sm_monitor_task = asyncio.create_task(
            sm_monitor.start(poll_interval=sm_poll_interval)
        )

    logger.info("Waiting for data feeds...")
    for _ in range(50):
        if binance.price > 0:
            break
        await asyncio.sleep(0.1)

    # ── Main loop state ───────────────────────────────────────────────
    market_refresh_interval = 10
    orderbook_refresh_interval = 2
    last_market_refresh = 0.0
    last_orderbook_refresh = 0.0
    last_window_slug = ""
    startup_skip = True  # Skip first window (may be mid-progress)
    startup_window_slug = ""  # Track which window we started in
    window_open_price = 0.0
    window_high_price = 0.0
    window_low_price = 0.0

    # Clear terminal, hide cursor
    sys.stdout.write(CLEAR_SCREEN + "\033[?25l")
    sys.stdout.flush()

    try:
        while not shutdown_event.is_set():
            now = time.time()

            # ── Market Discovery ──────────────────────────────────────
            if now - last_market_refresh > market_refresh_interval:
                try:
                    await market_discovery.refresh()
                except Exception as e:
                    logger.debug(f"Market refresh: {e}")
                last_market_refresh = now

            mkt = market_discovery.current_market

            # ── Orderbook: WS primary, REST fallback ─────────────────
            if mkt and mkt.yes_token_id:
                # Keep WS subscribed to current market tokens
                ws_feed.subscribe(mkt.yes_token_id, mkt.no_token_id)

                # REST fallback: poll when WS data isn't fresh
                if not ws_feed.is_fresh and now - last_orderbook_refresh > orderbook_refresh_interval:
                    try:
                        orderbook_mgr.fetch(mkt.yes_token_id, mkt.no_token_id)
                    except Exception as e:
                        logger.debug(f"Orderbook REST fallback: {e}")
                    last_orderbook_refresh = now

            # ── Oracle price (Binance/Coinbase average) ─────────────
            oracle.update_price(
                binance_price=binance.price,
                coinbase_price=coinbase.price if coinbase.is_connected else 0.0,
            )

            # ── Health monitor ────────────────────────────────────────
            if binance.price > 0:
                health.update_feed("binance")
            if oracle.price > 0:
                health.update_feed("oracle")
            if coinbase.is_connected:
                health.update_feed("coinbase")
            if ws_feed.last_summary or orderbook_mgr.last_summary:
                health.update_feed("orderbook")
            if mkt:
                health.update_feed("market_discovery")
            if coinglass.data.is_valid:
                health.update_feed("coinglass")
            if coinalyze.snapshot.is_valid:
                health.update_feed("coinalyze")
            if binance_liqs.is_connected:
                health.update_feed("binance_liqs")

            for alert in health.get_alerts():
                await telegram.notify_bot_event("Health Alert", alert)

            # ── Build snapshot ────────────────────────────────────────
            # Prefer WS orderbook (sub-second) over REST (2s polling)
            ob_source = ws_feed if ws_feed.is_fresh else orderbook_mgr
            snapshot = build_snapshot(
                binance=binance,
                oracle=oracle,
                coinbase=coinbase,
                coinglass=coinglass,
                coinalyze=coinalyze,
                binance_liqs=binance_liqs,
                orderbook_mgr=ob_source,
                market=mkt,
                health=health,
                clob_trade_flow=ws_feed.trade_flow if ws_feed.is_fresh else None,
            )

            # ── Latency logging (every tick) ─────────────────────────
            try:
                latency_logger.log_tick(
                    snapshot=snapshot,
                    binance_price_time=getattr(binance, "price_time", 0.0),
                    coinbase_price_time=getattr(coinbase, "price_time", 0.0),
                )
            except Exception as e:
                logger.debug(f"Latency logger error: {e}")

            # ── Window transition detection ───────────────────────────
            if mkt and mkt.slug != last_window_slug:
                # First window on startup: observe but don't trade
                if startup_skip:
                    last_window_slug = mkt.slug
                    startup_window_slug = mkt.slug
                    startup_skip = False
                    if sm_monitor and mkt.yes_token_id and mkt.no_token_id:
                        sm_monitor.set_market(
                            mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                        )
                    logger.info(
                        f"Startup: observing in-progress window {mkt.slug}, "
                        f"will trade from next window"
                    )
                    # Set exchange avg immediately, poll PolyBackTest in background
                    oracle.start_window_open_fetch(expected_slug=mkt.slug)
                    window_open_price = oracle.window_open_price
                    window_high_price = oracle.price
                    window_low_price = oracle.price

                    # Tell all bots about the window but mark it as startup
                    for bot in bots:
                        try:
                            bot.on_new_window(snapshot)
                        except Exception as e:
                            logger.error(f"Bot {bot.name} on_new_window error: {e}")
                else:
                    # ── Resolve previous window ───────────────────────
                    resolution = await detect_resolution(
                        market_discovery, last_window_slug, oracle
                    )
                    logger.info(
                        f"Window transition: {last_window_slug} -> {mkt.slug} | "
                        f"Resolution: {resolution}"
                    )
                    for bot in bots:
                        try:
                            bot.on_window_end(resolution)
                        except Exception as e:
                            logger.error(f"Bot {bot.name} on_window_end error: {e}")

                    # ── New window setup ──────────────────────────────
                    last_window_slug = mkt.slug
                    binance_liqs.reset()
                    if sm_monitor and mkt.yes_token_id and mkt.no_token_id:
                        sm_monitor.set_market(
                            mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                        )
                    # Set exchange avg immediately, poll PolyBackTest in background
                    oracle.start_window_open_fetch(expected_slug=mkt.slug)
                    window_open_price = oracle.window_open_price
                    window_high_price = oracle.price
                    window_low_price = oracle.price

                    # Rebuild snapshot with updated oracle window open price
                    ob_source = ws_feed if ws_feed.is_fresh else orderbook_mgr
                    snapshot = build_snapshot(
                        binance=binance,
                        oracle=oracle,
                        coinbase=coinbase,
                        coinglass=coinglass,
                        coinalyze=coinalyze,
                        binance_liqs=binance_liqs,
                        orderbook_mgr=ob_source,
                        market=mkt,
                        health=health,
                    )

                    for bot in bots:
                        try:
                            bot.on_new_window(snapshot)
                        except Exception as e:
                            logger.error(f"Bot {bot.name} on_new_window error: {e}")

            # ── Force-resolve stale pending trades ────────────────────
            # If the window has ended (market expired or discovery stalled),
            # force-resolve any pending trades across all bots.
            if mkt is None or (mkt and mkt.time_remaining <= 0):
                for bot in bots:
                    if hasattr(bot, "executor") and bot.executor.pending_trade:
                        pending_age = now - bot.executor.pending_trade_time
                        if pending_age > 30:
                            logger.warning(
                                f"Force-resolving {bot.name} trade after "
                                f"window end ({pending_age:.0f}s)"
                            )
                            resolution = await detect_resolution(
                                market_discovery, last_window_slug, oracle
                            )
                            bot.on_window_end(resolution)
            elif mkt:
                # Safety fallback: 600s stuck trade
                for bot in bots:
                    if hasattr(bot, "executor") and bot.executor.pending_trade:
                        pending_age = now - bot.executor.pending_trade_time
                        if pending_age > 600:
                            logger.error(
                                f"Force-resolving {bot.name} stuck trade "
                                f"after {pending_age:.0f}s"
                            )
                            resolution = await detect_resolution(
                                market_discovery, last_window_slug, oracle
                            )
                            bot.on_window_end(resolution)

            # ── Update window open from background fetch ─────────────
            # oracle.window_open_price updates in-place when PolyBackTest
            # background poll succeeds — keep local var in sync
            if oracle.window_open_price > 0:
                window_open_price = oracle.window_open_price

            # ── Update window high/low ────────────────────────────────
            if oracle.price > 0 and window_open_price > 0:
                if oracle.price > window_high_price:
                    window_high_price = oracle.price
                if oracle.price < window_low_price:
                    window_low_price = oracle.price

            # ── Tick all bots ─────────────────────────────────────────
            # Don't tick on the startup window — we may have joined mid-progress
            if (mkt and mkt.is_active and binance.price > 0
                    and not startup_skip
                    and mkt.slug != startup_window_slug):
                for bot in bots:
                    try:
                        bot.on_tick(snapshot)
                    except Exception as e:
                        logger.error(f"Bot {bot.name} on_tick error: {e}")

            # ── Render dashboard ──────────────────────────────────────
            try:
                bot_states = []
                for bot in bots:
                    try:
                        bot_states.append(bot.get_dashboard_state())
                    except Exception as e:
                        logger.debug(f"Dashboard state error for {bot.name}: {e}")
                        bot_states.append({"name": bot.name, "signals": {}})

                # Regime info from first bot (shared data, same detection)
                regime_label = ""
                regime_conf = 0.0
                for bs in bot_states:
                    if bs.get("regime"):
                        regime_label = bs["regime"]
                        regime_conf = bs.get("regime_confidence", 0.0)
                        break

                # Prefer WS orderbook for dashboard display
                ob_display = ws_feed.last_summary if ws_feed.is_fresh else orderbook_mgr.last_summary
                render_multi_dashboard(
                    bot_states=bot_states,
                    binance_price=binance.price,
                    oracle_price=oracle.price,
                    coinbase_price=coinbase.price if coinbase.is_connected else 0.0,
                    market_question=mkt.question if mkt else "",
                    seconds_remaining=mkt.seconds_remaining if mkt else 0.0,
                    orderbook=ob_display,
                    regime=regime_label,
                    regime_confidence=regime_conf,
                    window_open=window_open_price,
                    window_high=window_high_price,
                    window_low=window_low_price,
                    window_open_source=oracle.window_open_source,
                    ws_connected=ws_feed.is_connected,
                    ws_updates=ws_feed.updates_received,
                    ws_last_update=ws_feed.last_update_time,
                )
            except Exception as e:
                logger.warning(f"Dashboard render error (non-fatal): {e}")

            # ── Sleep until next tick ─────────────────────────────────
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        # Restore cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        logger.info("Cleaning up...")

        # Resolve any final pending trades
        if last_window_slug:
            try:
                resolution = await detect_resolution(
                    market_discovery, last_window_slug, oracle
                )
                for bot in bots:
                    try:
                        bot.on_window_end(resolution)
                    except Exception:
                        pass
            except Exception:
                pass

        # Shutdown all bots (close DB connections)
        for bot in bots:
            try:
                bot.shutdown()
            except Exception as e:
                logger.warning(f"Bot {bot.name} shutdown error: {e}")

        # Latency summary
        try:
            latency_logger.print_summary()
            latency_logger.close()
        except Exception as e:
            logger.warning(f"Latency logger shutdown error: {e}")

        # Summary
        for bot in bots:
            try:
                state = bot.get_dashboard_state()
                logger.info(
                    f"{bot.name}: {state.get('total_trades', 0)} trades, "
                    f"PnL: ${state.get('session_pnl', 0):.2f}, "
                    f"WR: {state.get('win_rate', 0):.0%}"
                )
            except Exception:
                pass

        await telegram.notify_bot_event(
            "Multi-Bot Stopped",
            f"Bots: {', '.join(b.name for b in bots)}",
        )

        # Stop feeds
        await binance.stop()
        await oracle.stop()
        await coinbase.stop()
        await coinglass.stop()
        await coinalyze.stop()
        await binance_liqs.stop()
        await ws_feed.stop()
        if sm_monitor:
            await sm_monitor.stop()

        binance_task.cancel()
        oracle_task.cancel()
        coinbase_task.cancel()
        coinglass_task.cancel()
        coinalyze_task.cancel()
        binance_liq_task.cancel()
        ws_feed_task.cancel()
        if sm_monitor_task:
            sm_monitor_task.cancel()

        _cleanup_tasks = [binance_task, oracle_task, coinbase_task,
                          coinglass_task, coinalyze_task, binance_liq_task,
                          ws_feed_task]
        if sm_monitor_task:
            _cleanup_tasks.append(sm_monitor_task)
        for t in _cleanup_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

        await market_discovery.close()
        await coinglass.close()
        await coinalyze.close()
        await telegram.close()
        orderbook_mgr.close()
        logger.info("Multi-bot runner stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
