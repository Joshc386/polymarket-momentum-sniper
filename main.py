#!/usr/bin/env python3
"""Polymarket Momentum Sniper Bot — Phase 10: Regime Detection + Event Calendar"""

import asyncio
import json
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

# Install fastest available event loop (uvloop/winloop) before any asyncio use
from core.event_loop import setup_event_loop
setup_event_loop()

from core.config import Config
from core.polymarket_client import PolymarketClient
from core.market_discovery import MarketDiscovery
from core.execution import PaperExecutionEngine, LiveExecutionEngine
from data.binance_feed import BinanceFeed
from data.polymarket_oracle import PolymarketOracle
from data.orderbook import OrderbookManager
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.liquidation import LiquidationSignal
from signals.orderbook_signal import OrderbookSignal
from signals.orderbook_fade import OrderbookFadeSignal
from signals.sentiment_signal import SentimentSignal
from signals.taker_ratio import TakerRatioSignal
from signals.combiner import SignalCombiner
from data.coinglass_scraper import CoinGlassScraper
from data.coinbase_feed import CoinbaseFeed
from data.coinalyze_feed import CoinalyzeFeed
from data.binance_liquidations import BinanceLiquidationFeed
from core.health_monitor import HealthMonitor
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from strategy.regime_detector import RegimeDetector, Regime
from strategy.mtf_regime_detector import MTFRegimeDetector
from strategy.event_calendar import EventCalendar
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
CURSOR_HOME = "\033[H"   # Move cursor to top-left (no clear, no scroll)
CLEAR_SCREEN = "\033[2J\033[H"  # Full clear + home (used once at startup)
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
    oracle_lag_val, momentum_val, liquidation_val, ob_signal_val, sentiment_val, raw_signal, est_prob_up,
    weights, entry_decision, last_trade_msg, coinglass_status="",
    coinbase_dir=0.0, coinalyze_status="", regime_state=None, event_status="",
):
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    L = []
    mode_label = "LIVE TRADING" if config.mode == "live" else "Paper Trading"
    mode_color = RED if config.mode == "live" else CYAN
    L.append(f"{BOLD}{mode_color}  POLYMARKET MOMENTUM SNIPER -- {mode_label}{RESET}")
    L.append(f"  {DIM}{now}  |  Mode: {config.mode.upper()}{RESET}")

    # ── Prices ──
    # Oracle price is primary — Polymarket resolves on Chainlink, not Binance
    if oracle.price > 0:
        price_line = f"  {BOLD}BTC (Oracle):{RESET} {GREEN}${oracle.price:,.2f}{RESET}"
        if binance.price > 0:
            d = ((binance.price - oracle.price) / oracle.price) * 100
            dc = GREEN if d > 0 else RED
            price_line += f"  Binance: ${binance.price:,.2f} {dc}({d:+.3f}%){RESET}"
        if coinbase.is_connected:
            cb_dir = coinbase_dir
            cb_c = GREEN if cb_dir > 0.1 else RED if cb_dir < -0.1 else DIM
            price_line += f"  CB: ${coinbase.price:,.2f} {cb_c}({cb_dir:+.2f}){RESET}"
        oracle_age = int(time.time() - oracle.last_fetch_time) if oracle.last_fetch_time > 0 else 0
        if oracle_age > 30:
            price_line += f"  {RED}[stale {oracle_age}s]{RESET}"
        L.append(price_line)
    elif binance.price > 0:
        L.append(f"  {BOLD}BTC:{RESET} {GREEN}${binance.price:,.2f}{RESET} {YELLOW}(Oracle connecting...){RESET}")
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
    w1, w2, w3, w4, w5 = weights
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
    ca_status = coinalyze_status if coinalyze_status else ""
    L.append(
        f"  L5 Sent:    {_sc(sentiment_val)}{sentiment_val:+.3f}{RESET} [{_bar(sentiment_val)}] "
        f"{DIM}{ca_status}{RESET}"
    )
    rc = _sc(raw_signal)
    L.append(
        f"  Combined:   {rc}{raw_signal:+.3f}{RESET} [{_bar(raw_signal)}]  "
        f"{DIM}w=[{w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f},{w5:.2f}]{RESET}"
    )

    ob = orderbook.last_summary
    mp = ob.market_implied_prob_up if ob else 0.0
    pc = GREEN if est_prob_up > 0.55 else RED if est_prob_up < 0.45 else YELLOW
    edge = est_prob_up - mp
    L.append(
        f"  P(Up): {pc}{est_prob_up:.1%}{RESET}  Mkt: {mp:.1%}  "
        f"Edge: {BOLD}{edge:+.1%}{RESET}"
    )

    # ── Regime + Events ──
    if regime_state:
        regime_colors = {
            "trending_up": GREEN, "trending_down": RED,
            "ranging": YELLOW, "high_vol": RED, "low_vol": DIM,
        }
        rc = regime_colors.get(regime_state.label, DIM)
        L.append(
            f"  Regime: {rc}{regime_state.label.upper()}{RESET} "
            f"(conf={regime_state.confidence:.2f} trend={regime_state.trend_strength:.2f} "
            f"vol={regime_state.volatility_pct:.2f} chop={regime_state.choppiness:.2f})"
        )
    if event_status:
        L.append(f"  {DIM}Events: {event_status}{RESET}")

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
            f"${t.size_usdc:.2f} ({t.num_shares:.2f} shares)  "
            f"Waiting for resolution..."
        )

    # ── Session stats ──
    L.append(f"  {MAGENTA}-- Session --{RESET}")
    pnl_c = GREEN if executor.session_pnl >= 0 else RED
    bankroll_line = (
        f"  Bankroll: {BOLD}${executor.bankroll:.2f}{RESET}  "
        f"P&L: {pnl_c}${executor.session_pnl:+.2f}{RESET}  "
        f"Trades: {executor.total_trades}  "
        f"W/L: {executor.session_wins}/{executor.session_losses}  "
        f"WR: {executor.win_rate:.0%}"
    )
    if executor.pending_trade:
        bankroll_line += f"  {YELLOW}(${executor.pending_trade.size_usdc:.2f} in play){RESET}"
    L.append(bankroll_line)

    # Last trade event
    if last_trade_msg:
        L.append(f"  {DIM}Last: {last_trade_msg}{RESET}")

    # ── Equity Curve (ASCII sparkline) ──
    L.append(f"  {MAGENTA}-- Equity Curve --{RESET}")
    resolved = [t for t in executor.session_trades if t.pnl is not None]
    if resolved:
        cumulative = []
        running = 0.0
        for t in resolved:
            running += t.pnl
            cumulative.append(running)
        # Show last 50 data points, scale to 30 chars wide
        points = cumulative[-50:]
        mn, mx = min(points), max(points)
        span = mx - mn if mx != mn else 1.0
        SPARK_CHARS = " _.,:-=!#@"
        spark = ""
        for p in points:
            idx = int((p - mn) / span * (len(SPARK_CHARS) - 1))
            idx = max(0, min(idx, len(SPARK_CHARS) - 1))
            spark += SPARK_CHARS[idx]
        pnl_c = GREEN if cumulative[-1] >= 0 else RED
        L.append(
            f"  {DIM}[{spark}]{RESET} "
            f"{pnl_c}${cumulative[-1]:+.2f}{RESET}"
        )
    else:
        L.append(f"  {DIM}No trades yet{RESET}")

    # ── Recent Trades ──
    L.append(f"  {MAGENTA}-- Recent Trades --{RESET}")
    recent = resolved[-7:] if resolved else []
    if recent or executor.pending_trade:
        # Show pending trade at the top of the list
        if executor.pending_trade:
            pt = executor.pending_trade
            pt_ts = pt.created_at[:19].split("T")[1] if "T" in pt.created_at else pt.created_at[-8:]
            L.append(
                f"  {DIM}{pt_ts}{RESET} "
                f"{YELLOW}P{RESET} "
                f"{pt.side:>3s} @ ${pt.entry_price:.3f} "
                f"-> {YELLOW}{'PEND':>4s}{RESET} "
                f"{DIM}{'$' + f'{pt.size_usdc:.2f}':>8s}{RESET}"
            )
        for t in recent:
            # Show resolution time if available, otherwise entry time
            ts_raw = (t.resolved_at if hasattr(t, "resolved_at") and t.resolved_at
                      else t.created_at if hasattr(t, "created_at") and t.created_at
                      else "??:??:??")
            ts = ts_raw[:19].split("T")[1] if "T" in ts_raw else ts_raw[-8:]
            won = t.pnl > 0 if t.pnl is not None else False
            wc = GREEN if won else RED
            wl = "W" if won else "L"
            res = t.resolution or "?"
            pnl_str = f"${t.pnl:+.2f}" if t.pnl is not None else "..."
            L.append(
                f"  {DIM}{ts}{RESET} "
                f"{wc}{wl}{RESET} "
                f"{t.side:>3s} @ ${t.entry_price:.3f} "
                f"-> {res:>4s} "
                f"{wc}{pnl_str:>8s}{RESET}"
            )
    else:
        L.append(f"  {DIM}No trades yet{RESET}")

    # ── Market Activity (large orderbook moves) ──
    L.append(f"  {MAGENTA}-- Market Activity --{RESET}")
    if hasattr(orderbook, '_activity_log') and orderbook._activity_log:
        for entry in orderbook._activity_log[-5:]:
            L.append(f"  {DIM}{entry}{RESET}")
    else:
        # Show current depth deltas as activity indicator
        if hasattr(orderbook, 'yes_bid_delta'):
            bd = orderbook.yes_bid_delta
            ad = orderbook.yes_ask_delta
            nbd = orderbook.no_bid_delta
            nad = orderbook.no_ask_delta
            if abs(bd) > 0 or abs(ad) > 0:
                bc = GREEN if bd > 0 else RED if bd < 0 else DIM
                ac = GREEN if ad > 0 else RED if ad < 0 else DIM
                L.append(
                    f"  YES bids: {bc}{bd:+.0f}{RESET}  "
                    f"YES asks: {ac}{ad:+.0f}{RESET}  "
                    f"{DIM}NO bids: {nbd:+.0f}  NO asks: {nad:+.0f}{RESET}"
                )
            else:
                L.append(f"  {DIM}Orderbook stable{RESET}")
        else:
            L.append(f"  {DIM}No activity data{RESET}")

    # Health
    if health:
        L.append(f"  {DIM}{health.summary_str()}{RESET}")

    L.append(f"  {DIM}Ctrl+C to stop{RESET}")

    # Render in-place: move cursor to each row and overwrite.
    # Uses absolute cursor positioning to avoid scrolling — never
    # writes a trailing newline on the last row, so the terminal
    # buffer doesn't grow.
    CLEAR_LINE = "\033[K"
    buf = []
    for row_idx, line in enumerate(L):
        # \033[{row};1H = move to row (1-based), column 1
        buf.append(f"\033[{row_idx + 1};1H{line}{CLEAR_LINE}")
    # Clear any leftover rows below our content (up to row 50)
    for row_idx in range(len(L), 50):
        buf.append(f"\033[{row_idx + 1};1H{CLEAR_LINE}")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


# ── Resolution detection ──────────────────────────────────────────────
async def detect_resolution(
    market_discovery: MarketDiscovery,
    slug: str,
    oracle: PolymarketOracle,
) -> str:
    """Determine market resolution — preferring actual Polymarket result.

    First tries the authoritative source: Polymarket's own resolution
    from the Gamma API. Falls back to oracle price comparison.
    """
    resolution = await market_discovery.fetch_resolution(slug)
    if resolution in ("UP", "DOWN"):
        return resolution

    # Fallback: oracle price comparison
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
    oracle = PolymarketOracle()
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
    ob_fade_signal = OrderbookFadeSignal()
    sent_signal = SentimentSignal()
    taker_signal = TakerRatioSignal()
    combiner = SignalCombiner(max_adjustment=config.max_adjustment)

    # Liquidation layer
    coinglass = CoinGlassScraper(refresh_interval=config.liquidation_refresh_interval_sec)
    liq_signal = LiquidationSignal()

    # Coinalyze cross-exchange sentiment
    coinalyze = CoinalyzeFeed(
        refresh_interval=60,
        api_key=config.coinalyze_api_key,
    )

    # Real-time Binance liquidation stream
    binance_liqs = BinanceLiquidationFeed(window_seconds=300.0)

    # Strategy
    entry_logic = EntryLogic(
        min_edge=config.min_edge,
        max_edge=config.max_edge,
        fee_adjustment=config.fee_adjustment,
        min_confidence=config.min_confidence,
        preferred_entry_secs=config.preferred_entry_secs,
        latest_entry_secs=config.latest_entry_secs,
        earliest_entry_secs=config.earliest_entry_secs,
    )
    sizer = PositionSizer(
        kelly_multiplier=config.kelly_multiplier,
        min_bet_usdc=config.min_bet_usdc,
        max_bet_usdc=config.max_bet_usdc,
    )
    risk_mgr = RiskManager(
        daily_loss_cap_pct=config.daily_loss_cap_pct,
        daily_loss_warn_pct=config.daily_loss_warn_pct,
        min_bankroll=config.min_bankroll,
        streak_reduce_at=config.streak_reduce_at,
        streak_reduce_factor=config.streak_reduce_factor,
        streak_heavy_reduce_at=config.streak_heavy_reduce_at,
        streak_heavy_reduce_factor=config.streak_heavy_reduce_factor,
        streak_pause_at=config.streak_pause_at,
        streak_pause_minutes=config.streak_pause_minutes,
        streak_reset_wins=config.streak_reset_wins,
        low_volatility_threshold=config.low_volatility_threshold,
        high_volatility_threshold=config.high_volatility_threshold,
        high_volatility_size_factor=config.high_volatility_size_factor,
    )

    # Regime detection + Event calendar
    # Load regime config directly from YAML (Config object doesn't parse this section)
    import yaml as _yaml
    with open("config.yaml", "r") as _f:
        _raw_cfg = _yaml.safe_load(_f) or {}
    regime_cfg = _raw_cfg.get("regime", {})
    if regime_cfg.get("detector") == "mtf":
        regime_detector = MTFRegimeDetector(
            trend_threshold=regime_cfg.get("trend_threshold", 0.30),
            high_vol_percentile=regime_cfg.get("high_vol_percentile", 0.90),
            low_vol_percentile=regime_cfg.get("low_vol_percentile", 0.20),
            alignment_weight=regime_cfg.get("alignment_weight", 0.55),
            stickiness=regime_cfg.get("stickiness", 3),
        )
    else:
        regime_detector = RegimeDetector()
    regime_allowed = set(regime_cfg.get("allowed_regimes", []))
    regime_blocked = set(regime_cfg.get("block_regimes", []))
    event_calendar = EventCalendar(buffer_minutes=15)

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
        executor.restore_from_db()

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

    # Register taker ratio callback on Binance trade stream
    binance.trade_callbacks.append(taker_signal.on_trade)

    # Start background feeds
    binance_task = asyncio.create_task(binance.start())
    oracle_task = asyncio.create_task(oracle.start())  # Price fed from Binance/Coinbase average
    coinbase_task = asyncio.create_task(coinbase.start())
    coinglass_task = asyncio.create_task(
        coinglass.start(current_price_getter=lambda: binance.price)
    )
    coinalyze_task = asyncio.create_task(coinalyze.start())
    binance_liq_task = asyncio.create_task(binance_liqs.start())

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
    fade_signal_val = 0.0
    sentiment_val = 0.0
    taker_ratio_val = 0.0
    coinbase_dir = 0.0
    raw_signal_val = 0.0
    est_prob_up = 0.5
    current_weights = (0.0, 0.0, 0.0, 0.0, 0.0)
    entry_decision = EntryDecision()
    risk_state = None
    last_trade_msg = ""
    current_regime = None
    regime_params = {}

    # Clear terminal once at startup, hide cursor, then updates are in-place
    sys.stdout.write(CLEAR_SCREEN + "\033[?25l")  # Hide cursor
    sys.stdout.flush()

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

            # ── Force-resolve stale pending trades ──
            # Only force-resolve if the window has actually ended (market
            # expired or discovery stalled). The 180s timeout is a fallback
            # for when market discovery is completely stuck — it must NOT
            # fire during a normal 5-minute window, which would cause
            # double-entry on the same market.
            if executor.pending_trade:
                pending_age = now - executor.pending_trade_time
                window_ended = mkt is None or (mkt and mkt.time_remaining <= 0)
                if window_ended and pending_age > 30:
                    logger.warning(
                        f"Force-resolving trade after window end ({pending_age:.0f}s pending, "
                        f"market={'expired' if mkt else 'stalled'})"
                    )
                    resolution = await detect_resolution(
                        market_discovery, last_window_slug, oracle
                    )
                    trade = executor.resolve_pending_trade(resolution)
                    if trade:
                        won = trade.pnl is not None and trade.pnl > 0
                        risk_mgr.record_result(trade.pnl or 0)
                        tag = "LIVE" if is_live else "PAPER"
                        last_trade_msg = (
                            f"{'V' if won else 'X'} [{tag}] {trade.side} "
                            f"{'WIN' if won else 'LOSS'} ${trade.pnl:+.2f}  "
                            f"Resolved: {resolution} (force)"
                        )
                    has_position_this_window = False
                elif not window_ended and pending_age > 600:
                    # Ultimate fallback: if somehow stuck for 10+ minutes
                    # (2 full windows), force-resolve regardless
                    logger.error(
                        f"Force-resolving stuck trade after {pending_age:.0f}s "
                        f"(safety fallback)"
                    )
                    resolution = await detect_resolution(
                        market_discovery, last_window_slug, oracle
                    )
                    trade = executor.resolve_pending_trade(resolution)
                    if trade:
                        won = trade.pnl is not None and trade.pnl > 0
                        risk_mgr.record_result(trade.pnl or 0)
                        tag = "LIVE" if is_live else "PAPER"
                        last_trade_msg = (
                            f"{'V' if won else 'X'} [{tag}] {trade.side} "
                            f"{'WIN' if won else 'LOSS'} ${trade.pnl:+.2f}  "
                            f"Resolved: {resolution} (force-stuck)"
                        )
                    has_position_this_window = False

            # ── Window transition ──
            if mkt and mkt.slug != last_window_slug:
                # First iteration: observe the in-progress window but don't trade it.
                # We only trade windows we've seen from the start.
                if not last_window_slug:
                    last_window_slug = mkt.slug
                    has_position_this_window = True  # Block trading this window
                    logger.info(
                        f"Startup: observing in-progress window {mkt.slug}, "
                        f"will trade from next window"
                    )
                    if oracle.price > 0:
                        oracle.set_window_open_price()
                    else:
                        await oracle.fetch_once()
                        oracle.set_window_open_price()
                    continue

                # Resolve any pending trade from previous window
                if executor.pending_trade and last_window_slug:
                    resolution = await detect_resolution(
                        market_discovery, last_window_slug, oracle
                    )
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
                ob_fade_signal.reset()  # Reset L6 fade history
                binance_liqs.reset()  # Reset live liq stats for new window
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

            # ── Oracle price (Binance/Coinbase average) ──
            oracle.update_price(
                binance_price=binance.price,
                coinbase_price=coinbase.price if coinbase.is_connected else 0.0,
            )

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
            if coinalyze.snapshot.is_valid:
                health.update_feed("coinalyze")
            if binance_liqs.is_connected:
                health.update_feed("binance_liqs")

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
                # Layer 3: Liquidation (Binance live + Coinalyze positioning + CoinGlass)
                liquidation_val = liq_signal.compute(
                    current_price=binance.price,
                    liq_data=coinglass.data,
                    price_momentum=momentum_val,
                    live_stats=binance_liqs.stats,
                    coinalyze_snapshot=coinalyze.snapshot,
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

                # Layer 6: Orderbook fade (contrarian on extreme imbalances)
                if ob and ob.yes_bid_depth > 0:
                    fade_signal_val = ob_fade_signal.compute(
                        yes_bid_depth=ob.yes_bid_depth,
                        yes_ask_depth=ob.yes_ask_depth,
                        seconds_remaining=secs_remaining,
                    )
                else:
                    fade_signal_val = 0.0

                # Layer 5: Cross-exchange sentiment (Coinalyze)
                if coinalyze.snapshot.is_valid:
                    sentiment_val = sent_signal.compute(
                        snapshot=coinalyze.snapshot,
                        price_momentum=momentum_val,
                    )
                else:
                    sentiment_val = 0.0

                # Layer 7: Taker buy ratio (from Binance trade stream)
                taker_ratio_val = taker_signal.compute()

                # Regime detection
                current_regime = regime_detector.detect(binance.candles)
                regime_params = regime_detector.get_params(current_regime.regime)

                # Select weight schedule based on regime
                _schedule_override = ""
                if current_regime and current_regime.regime == Regime.RANGING:
                    _schedule_override = "ranging"

                current_weights = combiner.get_weights(secs_remaining, _schedule_override)
                raw_signal_val, est_prob_up = combiner.combine(
                    oracle_lag_signal=oracle_lag_val,
                    momentum_signal=momentum_val,
                    liquidation_signal=liquidation_val,
                    seconds_remaining=secs_remaining,
                    coinbase_direction=coinbase_dir,
                    orderbook_signal=ob_signal_val,
                    sentiment_signal=sentiment_val,
                    regime_weight_adjustments=regime_params.get("weight_adjustments"),
                    fade_signal=fade_signal_val,
                    taker_ratio_signal=taker_ratio_val,
                    schedule_override=_schedule_override,
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

                # Event calendar check — block trading around FOMC, CPI, etc.
                event_blocked, event_name, event_minutes = event_calendar.is_event_window()
                if event_blocked and risk_state.can_trade:
                    risk_state.can_trade = False
                    risk_state.reason = f"Event: {event_name} ({event_minutes:+.0f}m)"

                # Regime affinity gating
                if current_regime and risk_state.can_trade:
                    regime_label = current_regime.regime.value
                    blocked = False
                    if regime_blocked and regime_label in regime_blocked:
                        blocked = True
                    elif regime_allowed and regime_label not in regime_allowed:
                        blocked = True
                    elif current_regime.regime == Regime.LOW_VOLATILITY:
                        blocked = True  # Fallback safety net
                    if blocked:
                        risk_state.can_trade = False
                        risk_state.reason = (
                            f"Regime blocked: {regime_label} "
                            f"(conf={current_regime.confidence:.2f})"
                        )

                # Apply regime size multiplier
                if current_regime and regime_params.get("size_multiplier", 1.0) < 1.0:
                    risk_state.size_multiplier *= regime_params["size_multiplier"]

                # ── Entry evaluation ──
                ob = orderbook_mgr.last_summary
                regime_edge_mult = regime_params.get("edge_multiplier", 1.0) if regime_params else 1.0
                if ob and risk_state.can_trade and not has_position_this_window:
                    entry_decision = entry_logic.evaluate(
                        est_prob_up=est_prob_up,
                        yes_best_ask=ob.yes_best_ask,
                        no_best_ask=ob.no_best_ask,
                        yes_best_bid=ob.yes_best_bid,
                        no_best_bid=ob.no_best_bid,
                        seconds_remaining=secs_remaining,
                        has_position=has_position_this_window,
                        regime_edge_multiplier=regime_edge_mult,
                        signal_aligned=_schedule_override == "ranging",
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
                            weights_json = json.dumps({
                                "L1": current_weights[0], "L2": current_weights[1],
                                "L3": current_weights[2], "L4": current_weights[3],
                                "L5": current_weights[4],
                            })
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
                                # Enriched snapshot
                                orderbook_signal=ob_signal_val,
                                sentiment_signal=sentiment_val,
                                coinbase_direction=coinbase_dir,
                                regime=current_regime.regime.value if current_regime else "",
                                regime_confidence=current_regime.confidence if current_regime else 0.0,
                                risk_size_multiplier=risk_state.size_multiplier,
                                consecutive_losses=risk_mgr.consecutive_losses,
                                signal_weights=weights_json,
                                ob_spread=ob.spread if ob else 0.0,
                                ob_yes_depth=ob.yes_bid_depth if ob else 0.0,
                                ob_ask_depth=ob.yes_ask_depth if ob else 0.0,
                                session_trade_num=executor.total_trades + 1,
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
                                    f">> [{tag}] {trade.side} @ "
                                    f"${trade.entry_price:.4f} ${trade.size_usdc:.2f}"
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
                fade_signal_val = 0.0
                sentiment_val = 0.0
                taker_ratio_val = 0.0
                coinbase_dir = 0.0
                raw_signal_val = 0.0
                est_prob_up = 0.5
                current_weights = (0.0, 0.0, 0.0, 0.0, 0.0)
                entry_decision = EntryDecision(reason="No active market")

            # ── Log signal every tick ──
            if mkt and mkt.is_active:
                try:
                    ob = orderbook_mgr.last_summary
                    # Build entry reason string
                    if entry_decision and entry_decision.should_enter:
                        _entry_reason = f"ENTERED: {entry_decision.side}"
                    elif entry_decision and entry_decision.reason:
                        _entry_reason = entry_decision.reason
                    elif has_position_this_window:
                        _entry_reason = "Position held this window"
                    else:
                        _entry_reason = "No signal"

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
                        # Enriched fields
                        orderbook_signal=ob_signal_val,
                        sentiment_signal=sentiment_val,
                        coinbase_direction=coinbase_dir,
                        regime=current_regime.regime.value if current_regime else "",
                        regime_confidence=current_regime.confidence if current_regime else 0.0,
                        entry_reason=_entry_reason,
                        signal_weights=json.dumps({
                            "L1": current_weights[0], "L2": current_weights[1],
                            "L3": current_weights[2], "L4": current_weights[3],
                            "L5": current_weights[4],
                        }),
                    )
                except Exception as e:
                    logger.debug("Signal logging failed: %s", e)

            # ── Render ──
            # L3 status: show all three data sources
            liq_parts = []
            # Coinalyze positioning (primary replacement for CoinGlass)
            if coinalyze.snapshot.is_valid:
                s = coinalyze.snapshot
                liq_parts.append(f"CA:L/S={s.ls_ratio:.2f} FR={s.funding_rate:+.4f}%")
            else:
                liq_parts.append("CA:(offline)")
            # CoinGlass static levels (bonus if available)
            if coinglass.data.is_valid:
                age = int(time.time() - coinglass.last_fetch_time)
                liq_parts.append(f"CG:L{len(coinglass.data.long_levels)}/S{len(coinglass.data.short_levels)} [{age}s]")
            # Binance live liquidation stream
            if binance_liqs.is_connected:
                liq_parts.append(f"Live:{binance_liqs.status_str}")
            else:
                liq_parts.append("Live:(disconnected)")
            cg_status = " | ".join(liq_parts)
            try:
                render_dashboard(
                    config, binance, oracle, coinbase, market_discovery, orderbook_mgr, db,
                    poly_client.is_authenticated, executor, risk_mgr, risk_state, health,
                    oracle_lag_val, momentum_val, liquidation_val, ob_signal_val, sentiment_val, raw_signal_val, est_prob_up,
                    current_weights, entry_decision, last_trade_msg, cg_status,
                    coinbase_dir=coinbase_dir,
                    coinalyze_status=coinalyze.status_str,
                    regime_state=current_regime,
                    event_status=event_calendar.status_str(),
                )
            except Exception as dash_err:
                logger.warning(f"Dashboard render error (non-fatal): {dash_err}")

            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")  # Restore cursor visibility
        sys.stdout.flush()
        logger.info("Cleaning up...")

        # Live mode: cancel any open orders first
        if is_live and hasattr(executor, 'cancel_pending_order'):
            await executor.cancel_pending_order()
            await poly_client.cancel_all_orders()

        # Resolve any final pending trade
        if executor.pending_trade:
            resolution = await detect_resolution(
                market_discovery, last_window_slug, oracle
            )
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
        await coinalyze.stop()
        await binance_liqs.stop()
        binance_task.cancel()
        oracle_task.cancel()
        coinbase_task.cancel()
        coinglass_task.cancel()
        coinalyze_task.cancel()
        binance_liq_task.cancel()
        for t in [binance_task, oracle_task, coinbase_task, coinglass_task, coinalyze_task, binance_liq_task]:
            try:
                await t
            except asyncio.CancelledError:
                pass
        await market_discovery.close()
        await coinglass.close()
        await coinalyze.close()
        await telegram.close()
        orderbook_mgr.close()
        db.close()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
