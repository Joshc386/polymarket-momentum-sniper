"""Execution engines for paper and live trading.

Both engines share the same interface (execute_trade / resolve_pending_trade)
so main.py can swap between them based on config.mode.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from core.execution_rounds import best_ask, best_bid, log_round
from core.kill_switch_io import halt_active
from logging_db.database import Database
from strategy.feature_snapshot import FEE_RATE, fee_per_share, fee_per_share_v2

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A completed (or pending) trade with full condition snapshot."""
    market_id: str
    market_slug: str
    side: str               # "YES" or "NO"
    entry_price: float
    size_usdc: float
    num_shares: float
    oracle_lag_signal: float
    momentum_signal: float
    liquidation_signal: float
    combined_signal: float
    estimated_prob_up: float
    market_implied_prob: float
    edge: float
    time_remaining_secs: float
    btc_price_at_entry: float
    oracle_price_at_entry: float
    oracle_price_at_open: float
    is_paper: bool
    # Enriched snapshot fields
    orderbook_signal: float = 0.0
    sentiment_signal: float = 0.0
    coinbase_direction: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    risk_size_multiplier: float = 1.0
    consecutive_losses: int = 0
    signal_weights: str = ""           # JSON string
    ob_spread: float = 0.0
    ob_yes_depth: float = 0.0
    ob_ask_depth: float = 0.0
    session_trade_num: int = 0
    # Full feature snapshot at entry (flat dict of all L1–L12 + sub-components,
    # prob_edge, net_ev_per_share, secs_into_window, …). NULL-not-zero: a key
    # whose value is None records as SQL NULL. See ADR-0001.
    feature_snapshot: dict = field(default_factory=dict)
    # Outcome fields
    resolution: str | None = None  # "UP" or "DOWN" or None
    pnl: float | None = None
    db_id: int | None = None
    order_id: str | None = None
    created_at: str = field(default="", init=False)
    resolved_at: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Auto-populate created_at with current UTC timestamp."""
        self.created_at = datetime.now(timezone.utc).isoformat()


# ── Shared trade parameters (used by both engines) ────────────────────

_TRADE_FIELDS = [
    "side", "price", "size_usdc", "market_id", "market_slug",
    "oracle_lag_signal", "momentum_signal", "liquidation_signal",
    "combined_signal", "estimated_prob_up", "market_implied_prob",
    "edge", "time_remaining_secs", "btc_price", "oracle_price",
    "oracle_open_price",
]

# Enriched feature-snapshot columns merged from trade.feature_snapshot at
# insert time. Kept separate from the explicit columns above so a None value
# records as SQL NULL (layer absent), never 0.0. See ADR-0001.
_ENRICHED_COLUMNS = [
    "l1_lag_component", "l1_open_component",
    "l4_imbalance", "l4_flow", "l4_mid_dev", "l4_top_pressure", "l4_thickness",
    "l6_fade", "l7_taker_ratio", "l8_clob_flow", "l9b_absorption",
    "l10_exhaustion", "l11_trade_size", "l12_wallet_flow",
    "prob_edge", "net_ev_per_share", "required_edge",
    "secs_into_window", "schedule_override",
    "size_bumped", "kelly_raw_usdc",
]


def _log_to_db(db, trade, is_paper: bool) -> int:
    """Insert a trade into the database. Returns row ID."""
    now_str = datetime.now(timezone.utc).isoformat()
    params = dict(
        timestamp=now_str,
        market_id=trade.market_id,
        market_slug=trade.market_slug,
        side=trade.side,
        entry_price=trade.entry_price,
        size_usdc=trade.size_usdc,
        num_shares=trade.num_shares,
        oracle_lag_signal=trade.oracle_lag_signal,
        momentum_signal=trade.momentum_signal,
        liquidation_signal=trade.liquidation_signal,
        combined_signal=trade.combined_signal,
        estimated_prob_up=trade.estimated_prob_up,
        market_implied_prob=trade.market_implied_prob,
        edge=trade.edge,
        time_remaining_secs=trade.time_remaining_secs,
        is_paper=1 if is_paper else 0,
        btc_price_at_entry=trade.btc_price_at_entry,
        oracle_price_at_entry=trade.oracle_price_at_entry,
        oracle_price_at_open=trade.oracle_price_at_open,
        fill_price=trade.entry_price,
        order_id=trade.order_id,
        # Enriched snapshot
        orderbook_signal=trade.orderbook_signal,
        sentiment_signal=trade.sentiment_signal,
        coinbase_direction=trade.coinbase_direction,
        regime=trade.regime,
        regime_confidence=trade.regime_confidence,
        risk_size_multiplier=trade.risk_size_multiplier,
        consecutive_losses=trade.consecutive_losses,
        signal_weights=trade.signal_weights,
        ob_spread=trade.ob_spread,
        ob_yes_depth=trade.ob_yes_depth,
        ob_ask_depth=trade.ob_ask_depth,
        session_trade_num=trade.session_trade_num,
    )
    # Merge the full feature snapshot (NULL-not-zero preserved: a None value
    # writes as SQL NULL). Only the enriched columns are pulled in; keys that
    # duplicate explicit columns above are ignored.
    snap = getattr(trade, "feature_snapshot", None) or {}
    for col in _ENRICHED_COLUMNS:
        if col in snap:
            params[col] = snap[col]
    return db.insert_trade(**params)


def _resolve_trade(trade: TradeRecord, resolution: str, fee_fn=fee_per_share) -> bool:
    """Compute P&L for a trade. Returns True if won.

    ``fee_fn`` maps entry price -> fee per share (default: the validated 0.07
    basis). The live engine passes its per-window market fee so recorded LIVE
    P&L reflects the real CLOB V2 fee; paper keeps the default.
    """
    trade.resolution = resolution
    trade.resolved_at = datetime.now(timezone.utc).isoformat()
    won = (trade.side == "YES" and resolution == "UP") or \
          (trade.side == "NO" and resolution == "DOWN")

    # Polymarket crypto taker fee, charged once at ENTRY on every trade
    # (win or lose): fee_fn(p) * shares, p = entry price.
    entry_fee = fee_fn(trade.entry_price) * trade.num_shares

    if won:
        payout = trade.num_shares * 1.0
        gross_profit = payout - trade.size_usdc
        trade.pnl = gross_profit - entry_fee
    else:
        trade.pnl = -trade.size_usdc - entry_fee

    return won


# ══════════════════════════════════════════════════════════════════════
#  PAPER EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════

class PaperExecutionEngine:
    """Simulates order execution for paper trading.

    Fills at current best ask + configurable slippage.
    Tracks virtual bankroll and logs all trades to SQLite.
    """

    def __init__(
        self,
        db: Database,
        initial_bankroll: float = 100.0,
        slippage: float = 0.005,
        bankroll_epoch: str | None = None,
    ):
        self.db = db
        self.bankroll = initial_bankroll
        self.slippage = slippage
        self.bankroll_epoch = bankroll_epoch
        self.pending_trade: TradeRecord | None = None
        self.pending_trade_time: float = 0.0
        self.session_trades: list[TradeRecord] = []
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
        self.is_paper = True

    def restore_from_db(self) -> None:
        """Restore state on startup.

        With `bankroll_epoch` set, the wallet persists across restarts:
        bankroll = initial + realised PnL since the epoch (so the
        wallet-proportional sizer sees a compounding wallet). Without an
        epoch, bankroll starts fresh at initial (legacy behaviour).
        Session trades and equity curve always start fresh; all trades
        remain stored in the database for historical analysis.
        """
        if not self.db.conn:
            return

        # Log all-time stats for reference only
        row = self.db.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total_pnl, "
            "       COUNT(*) as total_trades, "
            "       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
            "       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses "
            "FROM trades WHERE is_paper = 1 AND pnl IS NOT NULL"
        ).fetchone()

        if row and row[1] > 0:
            logger.info(
                f"All-time paper stats: {row[1]} trades, "
                f"{row[2]}W/{row[3]}L, P&L=${row[0]:+.2f}"
            )

        if self.bankroll_epoch:
            epoch_pnl = self.db.conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                "WHERE is_paper = 1 AND pnl IS NOT NULL AND timestamp >= ?",
                (self.bankroll_epoch,),
            ).fetchone()[0]
            self.bankroll += epoch_pnl
            logger.info(
                f"Wallet restored from epoch {self.bankroll_epoch}: "
                f"${self.bankroll:.2f} (epoch PnL ${epoch_pnl:+.2f})"
            )

        logger.info(f"New session started: bankroll=${self.bankroll:.2f}")

    def execute_paper_trade(
        self,
        side: str,
        price: float,
        size_usdc: float,
        market_id: str,
        market_slug: str,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        combined_signal: float,
        estimated_prob_up: float,
        market_implied_prob: float,
        edge: float,
        time_remaining_secs: float,
        btc_price: float,
        oracle_price: float,
        oracle_open_price: float,
        # Enriched snapshot fields
        orderbook_signal: float = 0.0,
        sentiment_signal: float = 0.0,
        coinbase_direction: float = 0.0,
        regime: str = "",
        regime_confidence: float = 0.0,
        risk_size_multiplier: float = 1.0,
        consecutive_losses: int = 0,
        signal_weights: str = "",
        ob_spread: float = 0.0,
        ob_yes_depth: float = 0.0,
        ob_ask_depth: float = 0.0,
        session_trade_num: int = 0,
        feature_snapshot: dict | None = None,
    ) -> TradeRecord | None:
        """Simulate placing a trade. Returns TradeRecord if 'filled'."""
        fill_price = min(price + self.slippage, 0.99)
        num_shares = size_usdc / fill_price
        self.bankroll -= size_usdc

        trade = TradeRecord(
            market_id=market_id, market_slug=market_slug, side=side,
            entry_price=fill_price, size_usdc=size_usdc, num_shares=num_shares,
            oracle_lag_signal=oracle_lag_signal, momentum_signal=momentum_signal,
            liquidation_signal=liquidation_signal, combined_signal=combined_signal,
            estimated_prob_up=estimated_prob_up, market_implied_prob=market_implied_prob,
            edge=edge, time_remaining_secs=time_remaining_secs,
            btc_price_at_entry=btc_price, oracle_price_at_entry=oracle_price,
            oracle_price_at_open=oracle_open_price, is_paper=True,
            orderbook_signal=orderbook_signal, sentiment_signal=sentiment_signal,
            coinbase_direction=coinbase_direction, regime=regime,
            regime_confidence=regime_confidence,
            risk_size_multiplier=risk_size_multiplier,
            consecutive_losses=consecutive_losses, signal_weights=signal_weights,
            ob_spread=ob_spread, ob_yes_depth=ob_yes_depth, ob_ask_depth=ob_ask_depth,
            session_trade_num=session_trade_num,
            feature_snapshot=feature_snapshot or {},
        )

        trade.db_id = _log_to_db(self.db, trade, is_paper=True)
        self.pending_trade = trade
        self.pending_trade_time = time.time()

        logger.info(
            f"[PAPER] {side} @ ${fill_price:.4f} | "
            f"Size: ${size_usdc:.2f} | Shares: {num_shares:.2f} | "
            f"Edge: {edge:+.4f}"
        )
        return trade

    # Alias so main.py can call executor.execute_trade() uniformly
    execute_trade = execute_paper_trade

    def resolve_pending_trade(self, resolution: str) -> TradeRecord | None:
        """Resolve a pending trade."""
        trade = self.pending_trade
        if not trade:
            return None

        won = _resolve_trade(trade, resolution)

        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ?, resolved_at = ? WHERE id = ?",
                (resolution, trade.pnl, trade.resolved_at, trade.db_id),
            )
            self.db.conn.commit()

        self.pending_trade = None

        logger.info(
            f"[PAPER] Resolved {resolution} | {trade.side} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    def close_position_early(
        self, exit_price: float, reason: str = ""
    ) -> TradeRecord | None:
        """Close pending position before resolution by selling shares.

        Used by latency arb strategy for smart exits — sell when the
        CLOB bid has moved above our entry price, locking in profit
        without waiting for the 5-min window to resolve.

        Args:
            exit_price: Bid price we're selling at.
            reason: Why we're exiting (e.g. "smart_exit_profit").

        Returns:
            Resolved TradeRecord with PnL, or None if no position.
        """
        trade = self.pending_trade
        if not trade:
            return None

        trade.resolved_at = datetime.now(timezone.utc).isoformat()
        trade.resolution = f"EARLY_EXIT:{reason}" if reason else "EARLY_EXIT"

        # PnL = (exit - entry) * shares, minus the single entry fee charged
        # at position open (p = entry price), on every trade.
        gross_pnl = (exit_price - trade.entry_price) * trade.num_shares
        entry_fee = fee_per_share(trade.entry_price) * trade.num_shares
        trade.pnl = gross_pnl - entry_fee

        won = trade.pnl > 0
        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ?, resolved_at = ? "
                "WHERE id = ?",
                (trade.resolution, trade.pnl, trade.resolved_at, trade.db_id),
            )
            self.db.conn.commit()

        self.pending_trade = None

        logger.info(
            f"[PAPER] Early exit ({reason}) | {trade.side} @ "
            f"${trade.entry_price:.4f} -> ${exit_price:.4f} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    @property
    def win_rate(self) -> float:
        total = self.session_wins + self.session_losses
        return self.session_wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.session_wins + self.session_losses


# ══════════════════════════════════════════════════════════════════════
#  LIVE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════

class LiveExecutionEngine:
    """Real order execution on Polymarket CLOB.

    Order strategy:
    - >60s remaining: GTC limit at best bid/ask. Cancel and re-evaluate if
      not filled within gtc_timeout_sec.
    - <=60s remaining: FOK limit slightly above best ask (accept slippage
      for fill certainty).
    - Never market orders (orderbook can be thin).

    Tracks real bankroll via on-chain balance and logs to SQLite.
    """

    def __init__(
        self,
        db: Database,
        poly_client,              # PolymarketClient instance
        initial_bankroll: float = 100.0,
        fok_slippage: float = 0.005,
        gtc_timeout_sec: int = 10,
        bot_id: str = "",
        min_order_shares: float = 5.0,
        fee_rate: float = FEE_RATE,
        fee_exponent: float = 1.0,
    ):
        self.db = db
        self.poly = poly_client
        self.bankroll = initial_bankroll
        self.fok_slippage = fok_slippage
        self.gtc_timeout_sec = gtc_timeout_sec
        self.bot_id = bot_id
        # Live CLOB V2 taker fee for recorded PnL (rate/exponent). Defaults to
        # the validated 0.07/exp-1 basis until set_market_fee() supplies the
        # market's live values (fetched once per window by the runner). The
        # entry EV gate is NOT affected — PnL-only (CLOB V2 migration decision).
        self.fee_rate = fee_rate
        self.fee_exponent = fee_exponent
        # Exchange minimum order size (shares). The 5-min BTC market reports
        # min_order_size = 5 (observed live 2026-06-12). A sized order below
        # this is skipped, not posted — a rejected post would otherwise spin
        # the re-post loop. Counted for the dashboard / telemetry.
        self.min_order_shares = min_order_shares
        self.below_min_skips = 0
        self.pending_trade: TradeRecord | None = None
        self.pending_trade_time: float = 0.0
        self.pending_order_id: str | None = None
        self.pending_token_id: str | None = None
        self.session_trades: list[TradeRecord] = []
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
        self.is_paper = False
        # J27 telemetry: GTC re-post round counter (per token).
        self._gtc_round_token: str = ""
        self._gtc_round_num: int = 0

    def set_market_fee(self, rate: float, exponent: float) -> None:
        """Set the live CLOB V2 taker fee (rate fraction + exponent) used for
        recorded PnL. Called once per window by the runner from the market's
        get_fee_rate_bps / get_fee_exponent. PnL-only — never the EV gate."""
        self.fee_rate = rate
        self.fee_exponent = exponent
        logger.info(
            f"[LIVE] market fee set: rate={rate:.4f} exponent={exponent} "
            f"(fee@p=0.50 = ${fee_per_share_v2(0.50, rate, exponent):.4f}/share)"
        )

    def _fee_per_share(self, price: float) -> float:
        """Entry fee per share at this market's live rate/exponent."""
        return fee_per_share_v2(price, self.fee_rate, self.fee_exponent)

    async def execute_trade(
        self,
        side: str,
        price: float,
        size_usdc: float,
        market_id: str,
        market_slug: str,
        oracle_lag_signal: float,
        momentum_signal: float,
        liquidation_signal: float,
        combined_signal: float,
        estimated_prob_up: float,
        market_implied_prob: float,
        edge: float,
        time_remaining_secs: float,
        btc_price: float,
        oracle_price: float,
        oracle_open_price: float,
        yes_token_id: str = "",
        no_token_id: str = "",
        # Enriched snapshot fields
        orderbook_signal: float = 0.0,
        sentiment_signal: float = 0.0,
        coinbase_direction: float = 0.0,
        regime: str = "",
        regime_confidence: float = 0.0,
        risk_size_multiplier: float = 1.0,
        consecutive_losses: int = 0,
        signal_weights: str = "",
        ob_spread: float = 0.0,
        ob_yes_depth: float = 0.0,
        ob_ask_depth: float = 0.0,
        session_trade_num: int = 0,
        feature_snapshot: dict | None = None,
    ) -> TradeRecord | None:
        """Place a real order on Polymarket.

        Chooses between GTC and FOK based on time remaining.
        Handles order monitoring and cancellation.
        """
        # Kill switch: never place a NEW order once HALT is set. The kill
        # switch owns cancellation/flattening; a recovering bot must not race
        # it. Checked here (not only at the loop top) to close the mid-tick
        # race between HALT being written and an order being placed.
        if halt_active():
            logger.warning(
                f"[LIVE] HALT active -- refusing entry order: {side} @ ${price:.4f}"
            )
            return None

        # Determine which token to buy
        token_id = yes_token_id if side == "YES" else no_token_id
        if not token_id:
            logger.error(f"No token ID for side {side}")
            return None

        # Exchange-minimum guard: a sized order below min_order_shares would
        # be rejected by the CLOB and spin the re-post loop. Skip + count it
        # rather than over-bet (user decision 2026-06-12, ADR-0003).
        est_shares = size_usdc / price if price > 0 else 0.0
        if est_shares < self.min_order_shares:
            self.below_min_skips += 1
            logger.info(
                f"[LIVE] skip {side} @ ${price:.4f}: {est_shares:.1f} shares "
                f"(${size_usdc:.2f}) < {self.min_order_shares:.0f}-share "
                f"exchange minimum"
            )
            log_round({
                "bot_id": self.bot_id, "outcome": "below_min", "side": side,
                "price": price, "size_usdc": size_usdc,
                "est_shares": est_shares, "min_shares": self.min_order_shares,
            })
            return None

        # Choose order type based on time remaining
        if time_remaining_secs > 60:
            fill_result = await self._execute_gtc(
                token_id, price, size_usdc, time_remaining_secs,
                spread=ob_spread,
            )
        else:
            fill_result = await self._execute_fok(
                token_id, price, size_usdc
            )

        if not fill_result:
            logger.warning(f"[LIVE] Order NOT filled: {side} @ ${price:.4f}")
            return None

        fill_price, num_shares, order_id = fill_result
        actual_cost = fill_price * num_shares

        # Deduct from tracked bankroll
        self.bankroll -= actual_cost

        trade = TradeRecord(
            market_id=market_id, market_slug=market_slug, side=side,
            entry_price=fill_price, size_usdc=actual_cost, num_shares=num_shares,
            oracle_lag_signal=oracle_lag_signal, momentum_signal=momentum_signal,
            liquidation_signal=liquidation_signal, combined_signal=combined_signal,
            estimated_prob_up=estimated_prob_up, market_implied_prob=market_implied_prob,
            edge=edge, time_remaining_secs=time_remaining_secs,
            btc_price_at_entry=btc_price, oracle_price_at_entry=oracle_price,
            oracle_price_at_open=oracle_open_price, is_paper=False,
            orderbook_signal=orderbook_signal, sentiment_signal=sentiment_signal,
            coinbase_direction=coinbase_direction, regime=regime,
            regime_confidence=regime_confidence,
            risk_size_multiplier=risk_size_multiplier,
            consecutive_losses=consecutive_losses, signal_weights=signal_weights,
            ob_spread=ob_spread, ob_yes_depth=ob_yes_depth, ob_ask_depth=ob_ask_depth,
            session_trade_num=session_trade_num, order_id=order_id,
            feature_snapshot=feature_snapshot or {},
        )

        trade.db_id = _log_to_db(self.db, trade, is_paper=False)
        self.pending_trade = trade
        self.pending_trade_time = time.time()
        self.pending_order_id = order_id
        self.pending_token_id = token_id  # held token, needed to SELL

        logger.info(
            f"[LIVE] {side} @ ${fill_price:.4f} | "
            f"Size: ${actual_cost:.2f} | Shares: {num_shares:.2f} | "
            f"Edge: {edge:+.4f} | Order: {order_id[:16] if order_id else 'N/A'}..."
        )
        return trade

    async def _execute_gtc(
        self, token_id: str, price: float, size_usdc: float,
        time_remaining: float, spread: float = 0.0,
    ) -> tuple[float, float, str] | None:
        """Place a GTC limit order and wait for fill.

        Cancel and return None if not filled within gtc_timeout_sec.
        Returns (fill_price, num_shares, order_id) or None.

        One call = one posting round of the re-post loop; each call appends
        one J27 telemetry record to the execution_rounds sink.
        """
        num_shares = size_usdc / price

        # J27: round # within the re-post loop (same token = next round).
        if token_id == self._gtc_round_token:
            self._gtc_round_num += 1
        else:
            self._gtc_round_token = token_id
            self._gtc_round_num = 1
        post_ts = time.time()
        round_rec = {
            "bot_id": self.bot_id,
            "token_id": token_id,
            "round": self._gtc_round_num,
            "post_ts": post_ts,
            "posted_bid": price,
            "shares": num_shares,
            "spread": spread,
            "time_remaining_at_post": time_remaining,
        }

        def record(outcome: str, **extra) -> None:
            log_round({**round_rec, "outcome": outcome,
                       "fill_price": None, "fill_shares": None, **extra})

        resp = await self.poly.place_order(
            token_id=token_id,
            side="BUY",
            price=price,
            size=num_shares,
            order_type="GTC",
        )

        if not resp:
            record("post_failed")
            return None

        order_id = resp.get("orderID", resp.get("id", ""))
        if not order_id:
            logger.warning("GTC order placed but no order ID returned")
            record("post_failed")
            return None
        round_rec["order_id"] = order_id

        # Monitor for fill
        deadline = time.time() + self.gtc_timeout_sec
        while time.time() < deadline:
            await asyncio.sleep(1.0)

            status = await self.poly.get_order_status(order_id)
            if not status:
                continue

            order_status = status.get("status", "").upper()

            if order_status in ("MATCHED", "FILLED"):
                fill_px = float(status.get("price", price))
                fill_sz = float(status.get("size_matched", num_shares))
                logger.info(f"GTC order filled: {fill_sz:.2f} @ ${fill_px:.4f}")
                record("filled", fill_price=fill_px, fill_shares=fill_sz)
                return (fill_px, fill_sz, order_id)

            if order_status in ("CANCELED", "CANCELLED", "EXPIRED"):
                logger.info(f"GTC order {order_status}")
                record("cancelled")
                return None

        # Timeout — cancel the unfilled order
        logger.info(f"GTC order not filled in {self.gtc_timeout_sec}s, cancelling")
        await self.poly.cancel_order(order_id)

        # Orphaned-shares check (ADR-0003): shares matched just before the
        # cancel are real money in the wallet — query the final state once
        # and return any partial as a fill rather than reporting a miss.
        try:
            final = await self.poly.get_order_status(order_id)
            if not isinstance(final, dict):
                final = {}
            matched = float(final.get("size_matched", 0) or 0)
        except Exception:
            matched = 0.0
        if matched > 0:
            fill_px = float(final.get("price", price) or price)
            logger.info(
                f"GTC partial fill before cancel: {matched:.2f} @ ${fill_px:.4f}"
            )
            record("filled", fill_price=fill_px, fill_shares=matched)
            return (fill_px, matched, order_id)

        # <=60s left now -> execute_trade will not route to GTC again, so
        # this was the final round of the re-post loop (the floor stop).
        remaining_now = time_remaining - (time.time() - post_ts)
        try:
            chase_ask = best_ask(self.poly.get_orderbook(token_id))
        except Exception:  # telemetry must never break order flow
            chase_ask = None
        record(
            "timeout_floor" if remaining_now <= 60.0 else "timeout",
            best_ask_at_timeout=chase_ask,
        )
        return None

    async def _execute_fok(
        self, token_id: str, price: float, size_usdc: float,
    ) -> tuple[float, float, str] | None:
        """Place a FOK (fill-or-kill) order with slight slippage.

        Returns (fill_price, num_shares, order_id) or None.
        """
        # Accept slippage for fill certainty
        fok_price = min(price + self.fok_slippage, 0.99)
        num_shares = size_usdc / fok_price

        resp = await self.poly.place_order(
            token_id=token_id,
            side="BUY",
            price=fok_price,
            size=num_shares,
            order_type="FOK",
        )

        if not resp:
            return None

        order_id = resp.get("orderID", resp.get("id", ""))
        post_status = str(resp.get("status", "")).upper()

        # FOK is instant, but the POST body carries no size_matched/price
        # (verified live 2026-06-13: {errorMsg, orderID, taking/makingAmount,
        # status, success}). The order record is the source of truth for the
        # fill. If the POST didn't already confirm a match, give it a moment
        # to settle before reading.
        if post_status not in ("MATCHED", "FILLED"):
            await asyncio.sleep(0.5)
        status = await self.poly.get_order_status(order_id)

        state = post_status
        if isinstance(status, dict):
            state = str(status.get("status", post_status)).upper()
        if state in ("MATCHED", "FILLED"):
            src = status if isinstance(status, dict) else {}
            # FOK fills fully or not at all, so num_shares is the size; price
            # and size come from the record when present (fall back to the
            # request only if the record is unavailable).
            fill_px = float(src.get("price", fok_price) or fok_price)
            fill_sz = float(src.get("size_matched", num_shares) or num_shares)
            return (fill_px, fill_sz, order_id)

        logger.info(f"FOK order not filled at ${fok_price:.4f}")
        return None

    def resolve_pending_trade(self, resolution: str) -> TradeRecord | None:
        """Resolve a pending trade (market settled)."""
        trade = self.pending_trade
        if not trade:
            return None

        won = _resolve_trade(trade, resolution, self._fee_per_share)

        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        # For live: actual balance updated on-chain.
        # We track locally for dashboard display.
        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ?, resolved_at = ? WHERE id = ?",
                (resolution, trade.pnl, trade.resolved_at, trade.db_id),
            )
            self.db.conn.commit()

        self.pending_trade = None
        self.pending_order_id = None

        logger.info(
            f"[LIVE] Resolved {resolution} | {trade.side} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    # Best-effort flatten parameters (ADR-0003). One tick under the bid is
    # marketable against the resting bid even if the book moved slightly.
    EXIT_TICK = 0.01
    EXIT_PRICE_FLOOR = 0.01
    EXIT_MAX_ATTEMPTS = 3
    EXIT_FILL_WAIT_SECS = 1.0
    EXIT_DUST_SHARES = 0.01   # remainders at/below this count as flat

    async def close_position_early(
        self, exit_price: float, reason: str = ""
    ) -> TradeRecord | None:
        """Best-effort flatten of the pending position (ADR-0003).

        Places an aggressive marketable SELL one tick under the best bid,
        accepts partial fills, retries stepping the price down, and is
        backstopped by resolution: whatever cannot be sold stays pending
        and settles at window end. The ledger records only actual fills at
        actual prices.

        Returns the resolved TradeRecord (full exit), the resolved record
        for the sold portion (partial exit), or None (nothing sold / no
        position / halted).
        """
        # Kill switch: once HALT is set, defer flattening to the kill switch
        # rather than racing it with the bot's own exit. The position is left
        # untouched; the kill switch + manual reconciliation handle it.
        if halt_active():
            logger.warning(
                f"[LIVE] HALT active -- deferring early exit ({reason}) to kill switch"
            )
            return None

        trade = self.pending_trade
        if not trade:
            return None

        token_id = self.pending_token_id
        if not token_id:
            logger.error(
                f"[LIVE] Early exit ({reason}) impossible: no token id for the "
                "pending position -- riding to resolution"
            )
            return None

        # Fresh best bid as the price reference; the strategy's view as backup.
        try:
            ref_bid = best_bid(self.poly.get_orderbook(token_id))
        except Exception:
            ref_bid = None
        if not ref_bid:
            ref_bid = exit_price

        remaining = trade.num_shares
        sold = 0.0
        proceeds = 0.0
        for attempt in range(self.EXIT_MAX_ATTEMPTS):
            px = max(
                ref_bid - (attempt + 1) * self.EXIT_TICK,
                self.EXIT_PRICE_FLOOR,
            )
            resp = await self.poly.place_order(
                token_id=token_id, side="SELL", price=px,
                size=remaining, order_type="GTC",
            )
            order_id = (resp or {}).get("orderID", (resp or {}).get("id", ""))
            if not order_id:
                continue

            await asyncio.sleep(self.EXIT_FILL_WAIT_SECS)
            status = await self.poly.get_order_status(order_id)
            state = ""
            if isinstance(status, dict):
                state = str(status.get("status", "")).upper()
            if state not in ("MATCHED", "FILLED"):
                # Cancel the remainder, then trust the FINAL state for what
                # matched before the cancel landed (same orphan-shares rule
                # as entries).
                await self.poly.cancel_order(order_id)
                status = await self.poly.get_order_status(order_id)

            matched, fill_px = 0.0, px
            if isinstance(status, dict):
                try:
                    matched = float(status.get("size_matched", 0) or 0)
                    fill_px = float(status.get("price", px) or px)
                except (TypeError, ValueError):
                    matched = 0.0
            matched = min(matched, remaining)
            if matched > 0:
                sold += matched
                proceeds += fill_px * matched
                remaining -= matched
            if remaining <= self.EXIT_DUST_SHARES:
                break

        if sold <= 0:
            logger.warning(
                f"[LIVE] Early exit ({reason}) UNFILLED after "
                f"{self.EXIT_MAX_ATTEMPTS} attempts -- position rides to "
                f"resolution ({trade.side} {trade.num_shares:.2f} shares)"
            )
            return None

        if remaining <= self.EXIT_DUST_SHARES:
            # Full exit at the blended ACTUAL price.
            return self._book_early_exit(trade, sold, proceeds, reason)

        # Partial exit: split the record. The sold portion books now at the
        # actual blended price; the remainder stays pending and resolves at
        # settlement (proportional cost basis and entry fee).
        logger.warning(
            f"[LIVE] Early exit ({reason}) PARTIAL: sold {sold:.2f}, "
            f"{remaining:.2f} shares ride to resolution"
        )
        sold_part = replace(
            trade,
            num_shares=sold,
            size_usdc=trade.entry_price * sold,
        )
        sold_part.order_id = trade.order_id
        sold_part.db_id = _log_to_db(self.db, sold_part, is_paper=False)
        booked = self._book_early_exit(sold_part, sold, proceeds, reason)

        trade.num_shares = remaining
        trade.size_usdc = trade.entry_price * remaining
        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET num_shares = ?, size_usdc = ? WHERE id = ?",
                (trade.num_shares, trade.size_usdc, trade.db_id),
            )
            self.db.conn.commit()
        return booked

    def _book_early_exit(
        self, trade: TradeRecord, sold: float, proceeds: float, reason: str
    ) -> TradeRecord:
        """Book an early-exit fill into the ledger at the actual prices.

        ``trade`` must already carry the sold size (num_shares == sold for a
        full exit; the cloned sold-portion record for a partial).
        """
        trade.resolved_at = datetime.now(timezone.utc).isoformat()
        trade.resolution = f"EARLY_EXIT:{reason}" if reason else "EARLY_EXIT"

        # Entry fee on the sold shares (charged once, at entry price), at this
        # market's live rate/exponent.
        gross_pnl = proceeds - trade.entry_price * sold
        entry_fee = self._fee_per_share(trade.entry_price) * sold
        trade.pnl = gross_pnl - entry_fee

        won = trade.pnl > 0
        if won:
            self.session_wins += 1
        else:
            self.session_losses += 1

        self.bankroll += trade.size_usdc + trade.pnl
        self.session_pnl += trade.pnl
        self.session_trades.append(trade)

        if trade.db_id and self.db.conn:
            self.db.conn.execute(
                "UPDATE trades SET resolution = ?, pnl = ?, resolved_at = ? "
                "WHERE id = ?",
                (trade.resolution, trade.pnl, trade.resolved_at, trade.db_id),
            )
            self.db.conn.commit()

        # Full exit clears the pending slot; the partial-split caller passes
        # a clone, so its reduced remainder stays pending.
        if self.pending_trade is trade:
            self.pending_trade = None
            self.pending_order_id = None
            self.pending_token_id = None

        avg_px = proceeds / sold if sold else 0.0
        logger.info(
            f"[LIVE] Early exit ({reason}) | {trade.side} @ "
            f"${trade.entry_price:.4f} -> ${avg_px:.4f} x {sold:.2f} | "
            f"{'WIN' if won else 'LOSS'} | P&L: ${trade.pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:.2f}"
        )
        return trade

    async def cancel_pending_order(self) -> bool:
        """Cancel any pending order (used during shutdown)."""
        if self.pending_order_id:
            result = await self.poly.cancel_order(self.pending_order_id)
            if result:
                logger.info(f"Cancelled pending order on shutdown")
            return result
        return False

    async def sync_bankroll(self):
        """Sync bankroll with on-chain USDC balance."""
        balance = await self.poly.get_balance()
        if balance > 0:
            self.bankroll = balance
            logger.info(f"Bankroll synced: ${balance:.2f} USDC")

    @property
    def win_rate(self) -> float:
        total = self.session_wins + self.session_losses
        return self.session_wins / total if total > 0 else 0.0

    @property
    def total_trades(self) -> int:
        return self.session_wins + self.session_losses
