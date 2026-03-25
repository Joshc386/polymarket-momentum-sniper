#!/usr/bin/env python3
"""Polymarket Momentum Sniper Bot — Phase 7: Optimised"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

# Fix Windows console encoding for Unicode/ANSI output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")  # Enable ANSI escape codes on Windows

from core.config import Config
from core.polymarket_client import PolymarketClient
from core.market_discovery import MarketDiscovery
from core.execution import PaperExecutionEngine, LiveExecutionEngine
from data.binance_feed import BinanceFeed
from data.chainlink_oracle import ChainlinkOracle
from data.orderbook import OrderbookManager
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.liquidation import LiquidationSignal
from signals.orderbook_signal import OrderbookSignal
from signals.combiner import SignalCombiner
from data.coinglass_scraper import CoinGlassScraper
from data.coinbase_feed import CoinbaseFeed
from core.health_monitor import HealthMonitor
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from notifications.telegram import TelegramNotifier
from logging_db.database import Database

# ── Logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)


# ── Terminal display ───────────────────────────────────────────────────
CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"


def _sc(val: float) -> str:
    if val > 0.1: return GREEN
    elif val < -0.1: return RED
    return YELLOW


def _bar(val: float, w: int = 16) -> str:
    mid = w // 2
    filled = int(abs(val) * mid)
    b = list("." * w)
    b[mid] = "|"
    if val > 0:
        for i in range(mid + 1, min(mid + 1 + filled, w)): b[i] = "+"
    elif val < 0:
        for i in range(max(mid - filled, 0), mid): b[i] = "-"
    return "".join(b)


def render_dashboard(
    config, binance, oracle, coinbase, market, orderbook, db,
    poly_connected, executor, risk_mgr, risk_state, health,
    oracle_lag_val, momentum_val, liquidation_val, ob_signal_val, raw_signal, est_prob_up,
    weights, entry_decision, last_trade_msg, coinglass_status="",
    coinbase_dir=0.0,
):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    L = []
    L.append(CLEAR)
    mode_label = "LIVE TRADING" if config.mode == "live" else "Paper Trading"
    mode_color = RED if config.mode == "live" else CYAN
    L.append(f"{BOLD}{mode_color}  POLYMARKET MOMENTUM SNIPER -- {mode_label}{RESET}")
    L.append(f"  {DIM}{now}  |  Mode: {config.mode.upper()}{RESET}")

    # ── Prices ──
    if binance.price > 0:
        price_line = f"  {BOLD}BTC:{RESET} {GREEN}${binance.price:,.2f}{RESET}"
        if oracle.price > 0:
            d = ((binance.price - oracle.price) / oracle.price) * 100
            dc = GREEN if d > 0 else RED
            price_line += f"  Oracle: ${oracle.price:,.2f} {dc}({d:+.3f}%){RESET}"
        if coinbase.is_connected:
            cb_dir = coinbase_dir
            cb_c = GREEN if cb_dir > 0.1 else RED if cb_dir < -0.1 else DIM
            price_line += f"  CB: ${coinbase.price:,.2f} {cb_c}({cb_dir:+.2f}){RESET}"
        L.append(price_line)
    else:
        L.append(f"  {BOLD}BTC:{RESET} {YELLOW}Connecting...{RESET}")

    # Candle
    if binance.current_candle:
        c = binance.current_candle
        ch = c.close - c.open
        cc = GREEN if ch >= 0 else RED
        L.append(
            f"  {DIM}1m:{RESET} O:{c.open:,.0f} H:{c.high:,.0f} L:{c.low:,.0f} "
            f"C:{cc}{c.close:,.0f}{RESET} Vol:{c.volume:,.1f} "
            f"{DIM}[{len(binance.candles)} closed]{RESET}"
        )

    # ── Market ──
    mkt = market.current_market
    if mkt:
        r = mkt.time_remaining
        mi, se = int(r // 60), int(r % 60)
        L.append(
            f"  {BOLD}Market:{RESET} {mkt.question}  "
            f"{GREEN}{mi}m{se:02d}s{RESET}"
        )
        ob = orderbook.last_summary
        if ob:
            L.append(
                f"  {DIM}YES{RESET} {GREEN}{ob.yes_best_bid:.2f}{RESET}/"
                f"{RED}{ob.yes_best_ask:.2f}{RESET}  "
                f"{DIM}NO{RESET} {GREEN}{ob.no_best_bid:.2f}{RESET}/"
                f"{RED}{ob.no_best_ask:.2f}{RESET}  "
                f"Spr:{ob.spread:.3f}"
            )
    else:
        L.append(f"  {BOLD}Market:{RESET} {YELLOW}Searching...{RESET}")

    # ── Signals ──
    L.append(f"  {MAGENTA}-- Signals --{RESET}")
    w1, w2, w3, w4 = weights
    L.append(
        f"  L1 Oracle:  {_sc(oracle_lag_val)}{oracle_lag_val:+.3f}{RESET} [{_bar(oracle_lag_val)}] "
        f"L2 Mom: {_sc(momentum_val)}{momentum_val:+.3f}{RESET} [{_bar(momentum_val)}]"
    )
    liq_status = coinglass_status if coinglass_status else ""
    L.append(
        f"  L3 Liq:     {_sc(liquidation_val)}{liquidation_val:+.3f}{RESET} [{_bar(liquidation_val)}] "
        f"{DIM}{liq_status}{RESET}"
    )
    L.append(
        f"  L4 Book:    {_sc(ob_signal_val)}{ob_signal_val:+.3f}{RESET} [{_bar(ob_signal_val)}] "
        f"{DIM}Polymarket CLOB{RESET}"
    )
    rc = _sc(raw_signal)
    L.append(
        f"  Combined:   {rc}{raw_signal:+.3f}{RESET} [{_bar(raw_signal)}]  "
        f"{DIM}w=[{w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f}]{RESET}"
    )

    ob = orderbook.last_summary
    mp = ob.market_implied_prob_up if ob else 0.0
    pc = GREEN if est_prob_up > 0.55 else RED if est_prob_up < 0.45 else YELLOW
    edge = est_prob_up - mp
    L.append(
        f"  P(Up): {pc}{est_prob_up:.1%}{RESET}  Mkt: {mp:.1%}  "
        f"Edge: {BOLD}{edge:+.1%}{RESET}"
    )

    # ── Entry / Risk ──
    L.append(f"  {MAGENTA}-- Trading --{RESET}")

    # Risk state
    rs = risk_state
    if rs:
        risk_line = "  Risk: "
        if rs.can_trade:
            if rs.size_multiplier < 1.0:
                risk_line += f"{YELLOW}{rs.reason}{RESET}"
            else:
                risk_line += f"{GREEN}OK{RESET}"
        else:
            risk_line += f"{RED}BLOCKED - {rs.reason}{RESET}"
        if rs.consecutive_losses > 0:
            risk_line += f"  {DIM}Losses: {rs.consecutive_losses}{RESET}"
        L.append(risk_line)

    # Entry decision
    ed = entry_decision
    if ed:
        if ed.should_enter:
            L.append(
                f"  Entry: {GREEN}SIGNAL! {ed.side} @ ${ed.price:.4f} "
                f"EV={ed.best_ev:.4f} > {ed.required_edge:.4f}{RESET}"
            )
        else:
            L.append(f"  Entry: {DIM}{ed.reason}{RESET}")

    # Pending trade
    if executor.pending_trade:
        t = executor.pending_trade
        L.append(
            f"  {YELLOW}PENDING:{RESET} {t.side} @ ${t.entry_price:.4f} "
            f"${t.size_usdc:.2f}  Waiting for resolution..."
        )

    # ── Session stats ──
    L.append(f"  {MAGENTA}-- Session --{RESET}")
    pnl_c = GREEN if executor.session_pnl >= 0 else RED
    L.append(
        f"  Bankroll: {BOLD}${executor.bankroll:.2f}{RESET}  "
        f"P&L: {pnl_c}${executor.session_pnl:+.2f}{RESET}  "
        f"Trades: {executor.total_trades}  "
        f"W/L: {executor.session_wins}/{executor.session_losses}  "
        f"WR: {executor.win_rate:.0%}"
    )

    # Last trade message
    if last_trade_msg:
        L.append(f"  {last_trade_msg}")

    # Health
    if health:
        L.append(f"  {DIM}{health.summary_str()}{RESET}")

    L.append(f"  {DIM}Ctrl+C to stop{RESET}")
    print("\n".join(L), flush=True)


# ── Resolution detection ──────────────────────────────────────────────
async def detect_resolution(oracle: ChainlinkOracle) -> str:
    """Determine if BTC went UP or DOWN by comparing oracle price to window open."""
    await oracle.fetch_once()
    if oracle.price >= oracle.window_open_price:
        return "UP"
    return "DOWN"


# ── Main loop ──────────────────────────────────────────────────────────
async def main():
    config = Config.load()
    logger.info(f"Loaded config - mode: {config.mode}")

    # Database
    db = Database(config.db_path)
    db.connect()

    # Data feeds
    binance = BinanceFeed()
    oracle = ChainlinkOracle()
    coinbase = CoinbaseFeed()
    health = HealthMonitor()

    # Polymarket
    poly_client = PolymarketClient(config)
    await poly_client.connect()
    market_discovery = MarketDiscovery()
    orderbook_mgr = OrderbookManager()

    # Signal engine
    oracle_lag_signal = OracleLagSignal(max_expected_lag=config.oracle_max_expected_lag)
    momentum_signal = MomentumSignal(
        roc_weight=config.momentum_roc_weight,
        direction_weight=config.momentum_direction_weight,
        volume_weight=config.momentum_volume_weight,
        body_ratio_weight=config.momentum_body_ratio_weight,
        rsi_weight=config.momentum_rsi_weight,
        rsi_period=config.momentum_rsi_period,
        lookback_candles=config.momentum_lookback_candles,
    )
    ob_signal = OrderbookSignal()
    combiner = SignalCombiner(max_adjustment=config.max_adjustment)

    # Liquidation layer
    coinglass = CoinGlassScraper(refresh_interval=config.liquidation_refresh_interval_sec)
    liq_signal = LiquidationSignal()

    # Strategy
    entry_logic = EntryLogic(
        min_edge=config.min_edge,
        max_edge=config.max_edge,
        fee_adjustment=config.fee_adjustment,
    )
    sizer = PositionSizer(
        kelly_multiplier=config.kelly_multiplier,
        min_bet_usdc=config.min_bet_usdc,
        max_bet_usdc=config.max_bet_usdc,
    )
    risk_mgr = RiskManager(
        daily_loss_cap_pct=config.daily_loss_cap_pct,
        min_bankroll=config.min_bankroll,
        streak_reduce_at=config.streak_reduce_at,
        streak_reduce_factor=config.streak_reduce_factor,
        streak_pause_at=config.streak_pause_at,
        streak_pause_minutes=config.streak_pause_minutes,
        streak_reset_wins=config.streak_reset_wins,
        low_volatility_threshold=config.low_volatility_threshold,
        high_volatility_threshold=config.high_volatility_threshold,
        high_volatility_size_factor=config.high_volatility_size_factor,
    )

    # Execution — select engine based on mode
    is_live = config.mode == "live"
    if is_live:
        if not poly_client.is_authenticated:
            logger.error(
                "FATAL: Live mode requires authentication. "
                "Set POLYMARKET_PRIVATE_KEY in .env"
            )
            sys.exit(1)

        # Set up allowances for EOA wallets
        await poly_client.check_and_set_allowance()

        executor = LiveExecutionEngine(
            db=db,
            poly_client=poly_client,
            initial_bankroll=config.initial_bankroll,
            fok_slippage=config.fok_slippage,
            gtc_timeout_sec=config.gtc_timeout_sec,
        )

        # Sync bankroll from on-chain balance
        await executor.sync_bankroll()
        logger.warning(">>> LIVE MODE ACTIVE — Real money at risk <<<")
    else:
        executor = PaperExecutionEngine(
            db=db,
            initial_bankroll=config.initial_bankroll,
            slippage=config.fok_slippage,
        )

    # Notifications
    telegram = TelegramNotifier(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        enabled=config.telegram_enabled,
        paper_prefix=config.telegram_paper_prefix,
    )
    await telegram.notify_bot_event("Bot Started", f"Mode: {config.mode} | Bankroll: ${config.initial_bankroll:.2f}")

    # Shutdown
    shutdown_event = asyncio.Event()
    def handle_shutdown():
        logger.info("Shutting down...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, handle_shutdown)
        loop.add_signal_handler(signal.SIGTERM, handle_shutdown)

    # Start background feeds
    binance_task = asyncio.create_task(binance.start())
    oracle_task = asyncio.create_task(oracle.start(poll_interval=2.0))
    coinbase_task = asyncio.create_task(coinbase.start())
    coinglass_task = asyncio.create_task(
        coinglass.start(current_price_getter=lambda: binance.price)
    )

    logger.info("Waiting for data feeds...")
    for _ in range(50):
        if binance.price > 0:
            break
        await asyncio.sleep(0.1)

    # Timing
    market_refresh_interval = 10
    orderbook_refresh_interval = 2
    last_market_refresh = 0.0
    last_orderbook_refresh = 0.0
    last_window_slug = ""
    has_position_this_window = False

    # Signal state
    oracle_lag_val = 0.0
    momentum_val = 0.0
    liquidation_val = 0.0
    ob_signal_val = 0.0
    coinbase_dir = 0.0
    raw_signal_val = 0.0
    est_prob_up = 0.5
    current_weights = (0.0, 0.0, 0.0, 0.0)
    entry_decision = EntryDecision()
    risk_state = None
    last_trade_msg = ""

    try:
        while not shutdown_event.is_set():
            now = time.time()

            # ── Market Discovery ──
            if now - last_market_refresh > market_refresh_interval:
                try:
                    await market_discovery.refresh()
                except Exception as e:
                    logger.debug(f"Market refresh: {e}")
                last_market_refresh = now

            mkt = market_discovery.current_market

            # ── Window transition ──
            if mkt and mkt.slug != last_window_slug:
                # Resolve any pending trade from previous window
                if executor.pending_trade and last_window_slug:
                    resolution = await detect_resolution(oracle)
                    trade = executor.resolve_pending_trade(resolution)
                    if trade:
                        won = trade.pnl is not None and trade.pnl > 0
                        risk_mgr.record_result(trade.pnl or 0)
                        tag = "LIVE" if is_live else "PAPER"
                        last_trade_msg = (
                            f"{'V' if won else 'X'} [{tag}] {trade.side} "
                            f"{'WIN' if won else 'LOSS'} ${trade.pnl:+.2f}  "
                            f"Resolved: {resolution}"
                        )
                        await telegram.notify_resolution(
                            is_paper=not is_live, side=trade.side,
                            resolution=resolution, won=won,
                            pnl=trade.pnl or 0, bankroll=executor.bankroll,
                            session_pnl=executor.session_pnl,
                            win_rate=executor.win_rate,
                            total_trades=executor.total_trades,
                        )
                        if risk_mgr.consecutive_losses >= risk_mgr.streak_reduce_at:
                            action = "Pausing" if risk_mgr.consecutive_losses >= risk_mgr.streak_pause_at else "Reducing size"
                            await telegram.notify_streak_alert(risk_mgr.consecutive_losses, action)

                # New window setup
                last_window_slug = mkt.slug
                has_position_this_window = False
                ob_signal.reset()  # Reset L4 history for new window
                if oracle.price > 0:
                    oracle.set_window_open_price()
                else:
                    await oracle.fetch_once()
                    oracle.set_window_open_price()

            # ── Orderbook refresh ──
            if mkt and mkt.yes_token_id and now - last_orderbook_refresh > orderbook_refresh_interval:
                try:
                    orderbook_mgr.fetch(mkt.yes_token_id, mkt.no_token_id)
                except Exception as e:
                    logger.debug(f"Orderbook: {e}")
                last_orderbook_refresh = now

            # ── Update health monitor ──
            if binance.price > 0:
                health.update_feed("binance")
            if oracle.price > 0:
                health.update_feed("oracle")
            if coinbase.is_connected:
                health.update_feed("coinbase")
            if orderbook_mgr.last_summary:
                health.update_feed("orderbook")
            if mkt:
                health.update_feed("market_discovery")
            if coinglass.data.is_valid:
                health.update_feed("coinglass")

            # Send health alerts
            for alert in health.get_alerts():
                await telegram.notify_bot_event("Health Alert", alert)

            # ── Compute signals ──
            if mkt and mkt.is_active and binance.price > 0:
                secs_remaining = mkt.seconds_remaining

                oracle_lag_val = oracle_lag_signal.compute(
                    exchange_price=binance.price,
                    oracle_price=oracle.price,
                    oracle_open_price=oracle.window_open_price,
                )
                momentum_val = momentum_signal.compute(
                    candles=binance.candles,
                    current_candle=binance.current_candle,
                )
                # Layer 3: Liquidation
                liquidation_val = liq_signal.compute(
                    current_price=binance.price,
                    liq_data=coinglass.data,
                    price_momentum=momentum_val,
                )
                # Coinbase cross-confirmation
                coinbase_dir = coinbase.price_direction if coinbase.is_connected else 0.0

                # Layer 4: Polymarket orderbook signal
                ob = orderbook_mgr.last_summary
                if ob and ob.yes_bid_total_size > 0:
                    ob_signal_val = ob_signal.compute(
                        yes_bid_total=ob.yes_bid_total_size,
                        yes_ask_total=ob.yes_ask_total_size,
                        no_bid_total=ob.no_bid_total_size,
                        no_ask_total=ob.no_ask_total_size,
                        yes_best_bid=ob.yes_best_bid,
                        yes_best_ask=ob.yes_best_ask,
                        yes_bid_depth_top5=ob.yes_bid_depth,
                        yes_ask_depth_top5=ob.yes_ask_depth,
                        size_weighted_mid=ob.yes_size_weighted_mid,
                        simple_mid=ob.yes_midpoint,
                        yes_bid_delta=orderbook_mgr.yes_bid_delta,
                        yes_ask_delta=orderbook_mgr.yes_ask_delta,
                        no_bid_delta=orderbook_mgr.no_bid_delta,
                        no_ask_delta=orderbook_mgr.no_ask_delta,
                        num_bid_levels=ob.num_yes_bid_levels,
                        num_ask_levels=ob.num_yes_ask_levels,
                    )
                else:
                    ob_signal_val = 0.0

                current_weights = combiner.get_weights(secs_remaining)
                raw_signal_val, est_prob_up = combiner.combine(
                    oracle_lag_signal=oracle_lag_val,
                    momentum_signal=momentum_val,
                    liquidation_signal=liquidation_val,
                    seconds_remaining=secs_remaining,
                    coinbase_direction=coinbase_dir,
                    orderbook_signal=ob_signal_val,
                )

                # Check if critical feeds are healthy before trading
                feeds_healthy, health_issues = health.check_health()
                if not feeds_healthy:
                    risk_state_override = f"Unhealthy feeds: {', '.join(health_issues)}"

                # ── Risk check ──
                recent_vol = risk_mgr.compute_recent_volatility(binance.candles)
                risk_state = risk_mgr.evaluate(executor.bankroll, recent_vol)

                # Override risk if feeds unhealthy
                if not feeds_healthy and risk_state.can_trade:
                    risk_state.can_trade = False
                    risk_state.reason = risk_state_override

                # ── Entry evaluation ──
                ob = orderbook_mgr.last_summary
                if ob and risk_state.can_trade and not has_position_this_window:
                    entry_decision = entry_logic.evaluate(
                        est_prob_up=est_prob_up,
                        yes_best_ask=ob.yes_best_ask,
                        no_best_ask=ob.no_best_ask,
                        yes_best_bid=ob.yes_best_bid,
                        no_best_bid=ob.no_best_bid,
                        seconds_remaining=secs_remaining,
                        has_position=has_position_this_window,
                    )

                    # ── Execute if signal fires ──
                    if entry_decision.should_enter:
                        win_prob = est_prob_up if entry_decision.side == "YES" else (1.0 - est_prob_up)
                        bet_size = sizer.compute(
                            est_prob=win_prob,
                            share_price=entry_decision.price,
                            bankroll=executor.bankroll,
                            size_multiplier=risk_state.size_multiplier,
                        )

                        if bet_size > 0:
                            market_prob = ob.market_implied_prob_up
                            trade_kwargs = dict(
                                side=entry_decision.side,
                                price=entry_decision.price,
                                size_usdc=bet_size,
                                market_id=mkt.condition_id,
                                market_slug=mkt.slug,
                                oracle_lag_signal=oracle_lag_val,
                                momentum_signal=momentum_val,
                                liquidation_signal=liquidation_val,
                                combined_signal=raw_signal_val,
                                estimated_prob_up=est_prob_up,
                                market_implied_prob=market_prob,
                                edge=entry_decision.best_ev,
                                time_remaining_secs=secs_remaining,
                                btc_price=binance.price,
                                oracle_price=oracle.price,
                                oracle_open_price=oracle.window_open_price,
                            )

                            if is_live:
                                trade_kwargs["yes_token_id"] = mkt.yes_token_id
                                trade_kwargs["no_token_id"] = mkt.no_token_id
                                trade = await executor.execute_trade(**trade_kwargs)
                            else:
                                trade = executor.execute_trade(**trade_kwargs)

                            if trade:
                                has_position_this_window = True
                                tag = "LIVE" if is_live else "PAPER"
                                last_trade_msg = (
                                    f">> [{tag}] {entry_decision.side} @ "
                                    f"${entry_decision.price:.4f} ${bet_size:.2f}"
                                )
                                await telegram.notify_trade(
                                    is_paper=not is_live,
                                    side=entry_decision.side,
                                    price=entry_decision.price,
                                    size_usdc=bet_size,
                                    edge=entry_decision.best_ev,
                                    est_prob=est_prob_up,
                                    market_prob=market_prob,
                                    time_remaining=secs_remaining,
                                    oracle_lag=oracle_lag_val,
                                    momentum=momentum_val,
                                    liquidation=liquidation_val,
                                    bankroll=executor.bankroll,
                                    session_pnl=executor.session_pnl,
                                )
                            else:
                                if is_live:
                                    last_trade_msg = (
                                        f">> [LIVE] Order NOT filled: "
                                        f"{entry_decision.side} @ ${entry_decision.price:.4f}"
                                    )
                elif ob and not risk_state.can_trade:
                    entry_decision = EntryDecision(reason=f"Risk: {risk_state.reason}")
                elif has_position_this_window:
                    entry_decision = EntryDecision(reason="Position held this window")
                else:
                    entry_decision = EntryDecision(reason="No orderbook data")
            else:
                oracle_lag_val = 0.0
                momentum_val = 0.0
                liquidation_val = 0.0
                ob_signal_val = 0.0
                coinbase_dir = 0.0
                raw_signal_val = 0.0
                est_prob_up = 0.5
                current_weights = (0.0, 0.0, 0.0, 0.0)
                entry_decision = EntryDecision(reason="No active market")

            # ── Log signal every tick ──
            if mkt and mkt.is_active:
                try:
                    ob = orderbook_mgr.last_summary
                    db.insert_signal(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        estimated_prob_up=est_prob_up,
                        market_implied_prob=ob.market_implied_prob_up if ob else None,
                        oracle_lag_signal=oracle_lag_val,
                        momentum_signal=momentum_val,
                        liquidation_signal=liquidation_val,
                        combined_signal=raw_signal_val,
                        time_remaining_secs=mkt.seconds_remaining,
                        trade_placed=1 if (entry_decision and entry_decision.should_enter) else 0,
                        btc_price=binance.price,
                        oracle_price=oracle.price,
                        market_id=mkt.condition_id,
                    )
                except Exception:
                    pass

            # ── Render ──
            cg_status = ""
            if coinglass.data.is_valid:
                age = int(time.time() - coinglass.last_fetch_time)
                cg_status = f"L:{len(coinglass.data.long_levels)} S:{len(coinglass.data.short_levels)} [{age}s ago]"
            elif coinglass.consecutive_failures > 0:
                cg_status = f"(offline x{coinglass.consecutive_failures})"
            else:
                cg_status = "(loading...)"
            render_dashboard(
                config, binance, oracle, coinbase, market_discovery, orderbook_mgr, db,
                poly_client.is_authenticated, executor, risk_mgr, risk_state, health,
                oracle_lag_val, momentum_val, liquidation_val, ob_signal_val, raw_signal_val, est_prob_up,
                current_weights, entry_decision, last_trade_msg, cg_status,
                coinbase_dir=coinbase_dir,
            )

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Cleaning up...")

        # Live mode: cancel any open orders first
        if is_live and hasattr(executor, 'cancel_pending_order'):
            await executor.cancel_pending_order()
            await poly_client.cancel_all_orders()

        # Resolve any final pending trade
        if executor.pending_trade:
            resolution = await detect_resolution(oracle)
            trade = executor.resolve_pending_trade(resolution)
            if trade:
                risk_mgr.record_result(trade.pnl or 0)

        await telegram.notify_bot_event(
            "Bot Stopped",
            f"Trades: {executor.total_trades} | P&L: ${executor.session_pnl:+.2f} | "
            f"Bankroll: ${executor.bankroll:.2f}"
        )

        await binance.stop()
        await oracle.stop()
        await coinbase.stop()
        await coinglass.stop()
        binance_task.cancel()
        oracle_task.cancel()
        coinbase_task.cancel()
        coinglass_task.cancel()
        for t in [binance_task, oracle_task, coinbase_task, coinglass_task]:
            try:
                await t
            except asyncio.CancelledError:
                pass
        await market_discovery.close()
        await coinglass.close()
        await telegram.close()
        orderbook_mgr.close()
        db.close()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
