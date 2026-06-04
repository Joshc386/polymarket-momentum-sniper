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
from core.kill_switch_io import halt_active, write_heartbeat
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
from data.kraken_feed import KrakenFeed
from data.bitstamp_feed import BitstampFeed
from data.price_aggregator import PriceAggregator
from notifications.telegram import TelegramNotifier
from strategies.registry import create_bot
from data.sm_wallets import SMWalletRegistry
from data.sm_trade_monitor import SMTradeMonitor
from data.wallet_flow_monitor import WalletFlowMonitor

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
    price_aggregator=None,
) -> str:
    """Determine market resolution — preferring actual Polymarket result.

    First tries the authoritative source: Polymarket's own resolution
    from the Gamma API (outcomePrices). Falls back to comparing the
    bot's reference price (3-feed aggregated) vs the PolyBackTest
    window-open snapshot only if the API resolution is unavailable.

    Args:
        market_discovery: MarketDiscovery instance for API lookups.
        slug: Slug of the market to resolve.
        oracle: PolymarketOracle (used for window_open_price).
        price_aggregator: PriceAggregator for the live BTC reference price.

    Returns:
        'UP' or 'DOWN'.
    """
    # Primary: actual market resolution from Polymarket
    resolution = await market_discovery.fetch_resolution(slug)
    if resolution in ("UP", "DOWN"):
        logger.info(f"Resolution from Polymarket API: {resolution}")
        return resolution

    # Fallback: live aggregated price vs window-open snapshot.
    # (oracle.price was the old 2-feed avg; superseded by aggregated_price
    # which uses 3 USD-native feeds.)
    logger.warning(
        f"Market {slug} resolution unknown from API — "
        f"falling back to aggregated price comparison"
    )
    fallback_price = (
        price_aggregator.price if price_aggregator and price_aggregator.price > 0
        else 0.0
    )
    if fallback_price >= oracle.window_open_price:
        return "UP"
    return "DOWN"


async def main() -> None:
    """Main entry point for the multi-bot runner."""
    cfg = load_config()
    logger.info("Loaded config_multi.yaml")

    # Ensure data_runtime directory exists
    Path("data_runtime").mkdir(exist_ok=True)

    # ── Shared data feeds ─────────────────────────────────────────────
    binance = BinanceFeed()  # still required for L2 momentum + L7 taker + liqs
    oracle = PolymarketOracle()
    coinbase = CoinbaseFeed()
    kraken = KrakenFeed()
    bitstamp = BitstampFeed()
    # PriceAggregator wraps the 3 USD-native feeds — exposes .price as
    # mean-of-healthy and is drop-in for binance.price as the BTC reference
    # price (used by L1, L3, risk stops, trade logging, TUI). Binance keeps
    # streaming for its other roles (candles, taker ratio, liquidations).
    price_aggregator = PriceAggregator(coinbase, kraken, bitstamp)
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

    # ── Shared Wallet Flow Monitor (L12) ────────────────────────────
    # Create a single shared wallet flow monitor if any bot has
    # wallet_flow_weight > 0. Polls Polygon RPC for all on-chain trades.
    wallet_flow_monitor = None
    any_wf_enabled = any(
        bot_cfg.get("signals", {}).get("wallet_flow_weight", 0.0) > 0
        for bot_cfg in bots_cfg.values()
        if bot_cfg.get("enabled", True)
    )
    if any_wf_enabled:
        wallet_flow_monitor = WalletFlowMonitor()
        logger.info("Shared wallet flow monitor created for L12 signal")

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
        # Inject shared wallet flow monitor for bots with L12 enabled
        if wallet_flow_monitor and bot_cfg.get("signals", {}).get("wallet_flow_weight", 0.0) > 0:
            bot_cfg["_wallet_flow_monitor"] = wallet_flow_monitor
        try:
            bot = create_bot(name=bot_name, strategy_name=strategy_name, cfg=bot_cfg)
            bots.append(bot)
            logger.info(f"Created bot: {bot_name} (strategy={strategy_name})")
        except KeyError as e:
            logger.error(f"Failed to create bot '{bot_name}': {e}")

    if not bots:
        logger.error("No bots configured. Check config_multi.yaml")
        return

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
    kraken_task = asyncio.create_task(kraken.start())
    bitstamp_task = asyncio.create_task(bitstamp.start())
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
    wf_monitor_task = None
    if wallet_flow_monitor:
        wf_monitor_task = asyncio.create_task(
            wallet_flow_monitor.start(poll_interval=3.0)
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

            # ── Kill switch: honour the sticky HALT flag ──────────────
            # If the kill switch (manual or watchdog) has fired, stop trading
            # immediately and exit gracefully. HALT is sticky and human-cleared,
            # so we do not resume on our own. The per-order guard in the live
            # executor backstops the mid-tick race.
            if halt_active():
                logger.critical("HALT flag detected -- kill switch fired; stopping bot.")
                try:
                    await telegram.notify_bot_event(
                        "KILL SWITCH", "HALT flag detected -- bot stopping."
                    )
                except Exception:
                    pass
                shutdown_event.set()
                break

            # ── Market Discovery ──────────────────────────────────────
            if now - last_market_refresh > market_refresh_interval:
                try:
                    await market_discovery.refresh()
                except Exception as e:
                    logger.debug(f"Market refresh: {e}")
                last_market_refresh = now

            mkt = market_discovery.current_market

            # ── Kill switch: write the heartbeat (atomic) ─────────────
            # The watchdog reads ts for staleness and window_end_ts/token_ids
            # for the flatten guard. Written every loop so a hang shows up as
            # a stale ts within ~1 tick. Best-effort: never let it break trading.
            try:
                write_heartbeat(
                    window_end_ts=mkt.end_time if mkt else None,
                    token_ids=[mkt.yes_token_id, mkt.no_token_id] if mkt else [],
                )
            except Exception as e:
                logger.debug(f"Heartbeat write failed: {e}")

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

            # NOTE: oracle.update_price() removed — the 2-feed (Binance/Coinbase)
            # average was a worse proxy for the same thing aggregated_price now
            # provides via 3-feed USD-native mean. oracle.window_open_price still
            # populated separately via the PolyBackTest API (different code path).

            # ── Health monitor ────────────────────────────────────────
            if binance.price > 0:
                health.update_feed("binance")
            if price_aggregator.n_healthy_feeds > 0:
                health.update_feed("aggregated_price")
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
                price_aggregator=price_aggregator,
            )

            # ── Window transition detection ───────────────────────────
            if mkt and mkt.slug != last_window_slug:
                # First window on startup: observe but don't trade
                if startup_skip:
                    last_window_slug = mkt.slug
                    startup_window_slug = mkt.slug
                    startup_skip = False
                    if mkt.yes_token_id and mkt.no_token_id:
                        if sm_monitor:
                            sm_monitor.set_market(
                                mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                            )
                        if wallet_flow_monitor:
                            wallet_flow_monitor.set_market(
                                mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                            )
                    logger.info(
                        f"Startup: observing in-progress window {mkt.slug}, "
                        f"will trade from next window"
                    )
                    # Set exchange avg immediately, poll PolyBackTest in background
                    oracle.start_window_open_fetch(expected_slug=mkt.slug)
                    window_open_price = oracle.window_open_price
                    # Track window high/low from the bot's actual reference
                    # price (3-feed aggregated). oracle.price is no longer
                    # maintained as of 2026-05-21.
                    window_high_price = price_aggregator.price
                    window_low_price = price_aggregator.price

                    # Tell all bots about the window but mark it as startup
                    for bot in bots:
                        try:
                            bot.on_new_window(snapshot)
                        except Exception as e:
                            logger.error(f"Bot {bot.name} on_new_window error: {e}")
                else:
                    # ── Resolve previous window ───────────────────────
                    resolution = await detect_resolution(
                        market_discovery, last_window_slug, oracle, price_aggregator
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
                    if mkt.yes_token_id and mkt.no_token_id:
                        if sm_monitor:
                            sm_monitor.set_market(
                                mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                            )
                        if wallet_flow_monitor:
                            wallet_flow_monitor.set_market(
                                mkt.yes_token_id, mkt.no_token_id, mkt.slug,
                            )
                    # Set exchange avg immediately, poll PolyBackTest in background
                    oracle.start_window_open_fetch(expected_slug=mkt.slug)
                    window_open_price = oracle.window_open_price
                    # Track window high/low from the bot's actual reference
                    # price (3-feed aggregated). oracle.price is no longer
                    # maintained as of 2026-05-21.
                    window_high_price = price_aggregator.price
                    window_low_price = price_aggregator.price

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
                        price_aggregator=price_aggregator,
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
                                market_discovery, last_window_slug, oracle, price_aggregator
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
                                market_discovery, last_window_slug, oracle, price_aggregator
                            )
                            bot.on_window_end(resolution)

            # ── Update window open from background fetch ─────────────
            # oracle.window_open_price updates in-place when PolyBackTest
            # background poll succeeds — keep local var in sync
            if oracle.window_open_price > 0:
                window_open_price = oracle.window_open_price

            # ── Update window high/low ────────────────────────────────
            # Use aggregated price (what the bot trades on), not the dead
            # oracle.price proxy.
            agg_price = price_aggregator.price
            if agg_price > 0 and window_open_price > 0:
                if agg_price > window_high_price:
                    window_high_price = agg_price
                if agg_price < window_low_price:
                    window_low_price = agg_price

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
                    aggregated_price=price_aggregator.price,
                    aggregated_n_feeds=price_aggregator.n_healthy_feeds,
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
                    market_discovery, last_window_slug, oracle, price_aggregator
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
        await kraken.stop()
        await bitstamp.stop()
        await coinglass.stop()
        await coinalyze.stop()
        await binance_liqs.stop()
        await ws_feed.stop()
        if sm_monitor:
            await sm_monitor.stop()
        if wallet_flow_monitor:
            await wallet_flow_monitor.stop()

        binance_task.cancel()
        oracle_task.cancel()
        coinbase_task.cancel()
        kraken_task.cancel()
        bitstamp_task.cancel()
        coinglass_task.cancel()
        coinalyze_task.cancel()
        binance_liq_task.cancel()
        ws_feed_task.cancel()
        if sm_monitor_task:
            sm_monitor_task.cancel()
        if wf_monitor_task:
            wf_monitor_task.cancel()

        _cleanup_tasks = [binance_task, oracle_task, coinbase_task,
                          kraken_task, bitstamp_task,
                          coinglass_task, coinalyze_task, binance_liq_task,
                          ws_feed_task]
        if sm_monitor_task:
            _cleanup_tasks.append(sm_monitor_task)
        if wf_monitor_task:
            _cleanup_tasks.append(wf_monitor_task)
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
