"""Bot F: Latency Arbitrage Strategy.

Exploits the delay between Binance price moves and Polymarket CLOB
odds adjustments. When Binance spikes and the CLOB hasn't repriced yet,
buy shares at stale (cheap) odds and either:
  - Smart exit: sell when the bid rises above entry (95%+ profitable)
  - Hold to resolution: wait for 5-min window to settle (71% WR)

Unlike Bots A-E, this strategy doesn't predict direction — Binance
tells us the direction. The edge is in speed, not prediction.

Backtested on 35 days of real PolyBackTest Pro data:
  - 398 trades, 71.4% WR holding to resolution
  - Smart exit: 98.7% profitable, $21/day at $5 sizing
  - Hold to resolution: $28/day at $5 sizing
"""

import json
import logging
import time
from datetime import datetime, timezone

from core.bot_strategy import BotStrategy
from core.data_snapshot import DataSnapshot
from core.execution import PaperExecutionEngine
from logging_db.database import Database
from signals.move_detector import MoveDetector, DetectedMove
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from tools.execution_latency import ExecutionLatencyTracker

logger = logging.getLogger(__name__)

# State machine states
STATE_SCANNING = "SCANNING"
STATE_ENTERED = "ENTERED"
STATE_EXITED_EARLY = "EXITED_EARLY"
STATE_RESOLVED = "RESOLVED"


class LatencyArbStrategy:
    """Latency arbitrage — buy stale CLOB odds after Binance moves.

    State machine per window (supports multiple trades):
        SCANNING  →  move detected + stale odds  →  ENTERED
        ENTERED   →  bid > entry (smart exit)     →  SCANNING (after cooldown)
        ENTERED   →  window end                   →  RESOLVED
    Re-entry after smart exit is allowed up to max_trades_per_window,
    with a configurable cooldown between trades.
    """

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self._cfg = cfg

        # Database (independent per bot)
        db_path = cfg.get("db_path", f"./data_runtime/{name}.db")
        self._db = Database(db_path)
        self._db.connect()

        # Move detector
        move_cfg = cfg.get("move_detection", {})
        self._move_detector = MoveDetector(
            threshold_bps=move_cfg.get("threshold_bps", 3.0),
            lookback_secs=move_cfg.get("lookback_secs", 10.0),
            cooldown_secs=move_cfg.get("cooldown_secs", 20.0),
            min_clob_gap_bps=move_cfg.get("min_clob_gap_bps", 1.0),
        )

        # Entry config
        entry_cfg = cfg.get("entry", {})
        self._max_entry_price = entry_cfg.get("max_entry_price", 0.55)
        self._earliest_entry_secs = entry_cfg.get("earliest_entry_secs", 270)
        self._latest_entry_secs = entry_cfg.get("latest_entry_secs", 30)

        # Entry config — multi-trade per window
        self._max_trades_per_window = entry_cfg.get("max_trades_per_window", 5)
        self._re_entry_cooldown_secs = entry_cfg.get("re_entry_cooldown_secs", 15.0)

        # Primary exchange for move detection.
        # "coinbase" — 8ms RTT to UK, much faster than Binance (234ms via Tokyo).
        # "binance" — kept for A/B testing.
        # Falls back to the other if primary feed is stale/disconnected.
        self._primary_exchange = entry_cfg.get("primary_exchange", "coinbase")

        # Exit config
        exit_cfg = cfg.get("exit", {})
        self._exit_mode = exit_cfg.get("mode", "smart")
        # Minimum profit target — require meaningful CLOB repricing, not
        # just a 1-tick bid wiggle. Old $0.01 target captured too many
        # tiny wins that didn't justify the $3 loss cuts.
        self._min_profit_ticks = exit_cfg.get("min_profit_ticks", 0.04)
        # Defensive loss cut: if unrealised loss exceeds this fraction of
        # entry price, the latency thesis has failed — exit immediately.
        # Tighter cut (0.30) caps losses at ~$1.50 per trade instead of $3.
        self._max_loss_pct = exit_cfg.get("max_loss_pct", 0.30)
        # Time-based exit: if neither profit nor loss trigger hits within
        # this many seconds, close at current bid. Latency arb is fast —
        # if it hasn't worked in 60s the thesis has failed.
        self._max_hold_secs = exit_cfg.get("max_hold_secs", 60.0)
        # Trailing profit lock: if price moves favourably by at least
        # trailing_lock_ticks, then pulls back, take profit at the high
        # minus trailing_giveback_ticks.
        self._trailing_lock_ticks = exit_cfg.get("trailing_lock_ticks", 0.06)
        self._trailing_giveback_ticks = exit_cfg.get(
            "trailing_giveback_ticks", 0.02
        )

        # Sizing
        sizing_cfg = cfg.get("sizing", {})
        self._sizer = PositionSizer(
            kelly_multiplier=sizing_cfg.get("kelly_multiplier", 0.25),
            min_bet_usdc=sizing_cfg.get("min_bet_usdc", 1.0),
            max_bet_usdc=sizing_cfg.get("max_bet_usdc", 5.0),
        )

        # Risk management
        risk_cfg = cfg.get("risk", {})
        self._risk_mgr = RiskManager(
            daily_loss_cap_pct=risk_cfg.get("daily_loss_cap_pct", 0.20),
            daily_loss_warn_pct=risk_cfg.get("daily_loss_warn_pct", 0.15),
            min_bankroll=risk_cfg.get("min_bankroll", 10.0),
            streak_reduce_at=risk_cfg.get("streak_reduce_at", 3),
            streak_reduce_factor=risk_cfg.get("streak_reduce_factor", 0.75),
            streak_heavy_reduce_at=risk_cfg.get("streak_heavy_reduce_at", 5),
            streak_heavy_reduce_factor=risk_cfg.get(
                "streak_heavy_reduce_factor", 0.5
            ),
            streak_pause_at=risk_cfg.get("streak_pause_at", 7),
            streak_pause_minutes=risk_cfg.get("streak_pause_minutes", 30),
            streak_reset_wins=risk_cfg.get("streak_reset_wins", 3),
        )

        # Execution engine
        self._executor = PaperExecutionEngine(
            db=self._db,
            initial_bankroll=sizing_cfg.get("initial_bankroll", 100.0),
            slippage=entry_cfg.get("fok_slippage", 0.005),
        )
        self._executor.restore_from_db()

        # Latency instrumentation — measures end-to-end execution timing
        self._latency_tracker = ExecutionLatencyTracker(output_dir="data_runtime")

        # Per-window state
        self._state = STATE_SCANNING
        self._entry_side = ""
        self._entry_price = 0.0
        self._entry_time = 0.0         # When the current position was opened
        self._current_bid = 0.0
        self._unrealised_pnl = 0.0
        self._trailing_high_bid = 0.0  # Highest bid seen since entry
        self._last_move: DetectedMove | None = None
        self._last_trade_msg = ""
        self._reason = "Scanning for moves"
        self._risk_state = None
        self._window_trades = 0        # Trades executed this window
        self._last_exit_time = 0.0     # Timestamp of last exit (for cooldown)
        self._equity_curve: list[float] = [
            sizing_cfg.get("initial_bankroll", 100.0)
        ]

    def on_new_window(self, snapshot: DataSnapshot) -> None:
        """Reset per-window state."""
        self._state = STATE_SCANNING
        self._entry_side = ""
        self._entry_price = 0.0
        self._entry_time = 0.0
        self._current_bid = 0.0
        self._unrealised_pnl = 0.0
        self._trailing_high_bid = 0.0
        self._last_move = None
        self._reason = "Scanning for moves"
        self._window_trades = 0
        self._last_exit_time = 0.0
        # Don't reset move_detector buffer — price history carries over

    def _get_exchange_price(self, snapshot: DataSnapshot) -> float:
        """Return the primary exchange price, with fallback if stale.

        Prefers Coinbase (8ms RTT) over Binance (234ms via Tokyo).
        Falls back to the other if primary is unavailable.
        """
        if self._primary_exchange == "coinbase":
            if snapshot.coinbase_price > 0:
                return snapshot.coinbase_price
            return snapshot.binance_price  # fallback
        # binance
        if snapshot.binance_price > 0:
            return snapshot.binance_price
        return snapshot.coinbase_price  # fallback

    def on_tick(self, snapshot: DataSnapshot) -> None:
        """Core tick logic — detect moves, enter, manage exits."""
        mkt = snapshot.market
        exchange_price = self._get_exchange_price(snapshot)
        if not mkt or not mkt.is_active or exchange_price <= 0:
            self._reason = "No active market"
            return

        secs_remaining = snapshot.seconds_remaining
        ob = snapshot.orderbook

        # ── Risk check ──
        recent_vol = self._risk_mgr.compute_recent_volatility(
            snapshot.binance_candles
        )
        self._risk_state = self._risk_mgr.evaluate(
            self._executor.bankroll, recent_vol
        )

        if not snapshot.feeds_healthy and self._risk_state.can_trade:
            self._risk_state.can_trade = False
            self._risk_state.reason = (
                f"Unhealthy feeds: {', '.join(snapshot.health_issues)}"
            )

        # ── State machine ──
        if self._state == STATE_SCANNING:
            self._tick_scanning(snapshot, ob, secs_remaining)
        elif self._state == STATE_ENTERED:
            self._tick_entered(snapshot, ob, secs_remaining)

    def _tick_scanning(
        self, snapshot: DataSnapshot, ob, secs_remaining: float
    ) -> None:
        """SCANNING state: look for Binance moves with stale CLOB odds."""
        # Time window check
        if secs_remaining > self._earliest_entry_secs:
            self._reason = (
                f"Window not open ({secs_remaining:.0f}s > "
                f"{self._earliest_entry_secs}s)"
            )
            return

        if secs_remaining < self._latest_entry_secs:
            self._reason = (
                f"Too late ({secs_remaining:.0f}s < "
                f"{self._latest_entry_secs}s)"
            )
            return

        # Max trades per window
        if self._window_trades >= self._max_trades_per_window:
            self._reason = (
                f"Max trades reached ({self._window_trades}/"
                f"{self._max_trades_per_window})"
            )
            return

        # Re-entry cooldown after an early exit
        if self._last_exit_time > 0:
            cooldown_remaining = (
                self._re_entry_cooldown_secs
                - (snapshot.timestamp - self._last_exit_time)
            )
            if cooldown_remaining > 0:
                self._reason = (
                    f"Re-entry cooldown ({cooldown_remaining:.0f}s)"
                )
                return

        # Risk gate
        if not self._risk_state or not self._risk_state.can_trade:
            self._reason = f"Risk: {self._risk_state.reason if self._risk_state else 'unknown'}"
            return

        # Detect exchange move (Coinbase by default — 8ms RTT vs Binance 234ms)
        # CLOB midpoint checks if odds are stale relative to the exchange.
        clob_mid = ob.yes_midpoint if ob else 0.0
        exchange_price = self._get_exchange_price(snapshot)
        move = self._move_detector.update(
            binance_price=exchange_price,  # param name kept for backward compat
            timestamp=snapshot.timestamp,
            clob_midpoint=clob_mid,
        )
        if not move:
            self._reason = "Scanning for moves"
            return

        self._last_move = move

        # Check if CLOB odds are stale
        if not ob:
            self._reason = "Move detected but no orderbook"
            return

        # Require two-sided orderbook — one-sided books during window
        # transitions produce dust prices that aren't real fills
        if ob.spread <= 0 or ob.yes_best_bid <= 0 or ob.yes_best_ask <= 0:
            self._reason = (
                f"Move {move.direction} {move.move_bps:.1f}bps but "
                f"one-sided book (bid={ob.yes_best_bid:.2f} "
                f"ask={ob.yes_best_ask:.2f})"
            )
            return

        if move.direction == "up":
            entry_price = ob.yes_best_ask
            side = "YES"
        else:
            entry_price = ob.no_best_ask
            side = "NO"

        # Price sanity: reject dust orders (< $0.05) and already-repriced
        # odds (> max_entry_price). Dust orders at $0.01-$0.02 appear when
        # real liquidity clears out — they're not real fills.
        min_entry_price = 0.05
        if entry_price < min_entry_price:
            self._reason = (
                f"Move {move.direction} {move.move_bps:.1f}bps but "
                f"dust price (${entry_price:.3f} < ${min_entry_price:.2f})"
            )
            return

        if entry_price > self._max_entry_price:
            self._reason = (
                f"Move {move.direction} {move.move_bps:.1f}bps but "
                f"odds adjusted (${entry_price:.3f} > "
                f"${self._max_entry_price:.2f})"
            )
            return

        # ── Execute trade ──
        # For latency arb, win probability is high (71% backtest)
        # but we use a conservative estimate for sizing
        est_win_prob = 0.70
        bet_size = self._sizer.compute(
            est_prob=est_win_prob,
            share_price=entry_price,
            bankroll=self._executor.bankroll,
            size_multiplier=self._risk_state.size_multiplier,
        )
        if bet_size <= 0:
            self._reason = "Sizer returned 0"
            return

        # ── Latency tracking — start the timer chain ──
        # T0 = exchange-side timestamp, T1 = when we received the WS msg.
        # Use the primary exchange's timing (coinbase by default).
        if self._primary_exchange == "coinbase":
            t0_exchange = snapshot.coinbase_price_time
            t1_received = snapshot.coinbase_received_at
        else:
            t0_exchange = snapshot.binance_price_time
            t1_received = snapshot.binance_received_at

        trade_id = f"{self.name}_{int(time.time()*1000)}"
        self._latency_tracker.start_trade(
            trade_id=trade_id,
            exchange_msg_time=t0_exchange,
            msg_received_time=t1_received,
            primary_exchange=self._primary_exchange,
            side=side,
            move_bps=move.move_bps,
            clob_gap_bps=move.clob_gap_bps,
        )
        self._latency_tracker.mark_decision(trade_id)
        self._latency_tracker.mark_submitted(trade_id)

        mkt = snapshot.market
        trade = self._executor.execute_trade(
            side=side,
            price=entry_price,
            size_usdc=bet_size,
            market_id=mkt.condition_id,
            market_slug=mkt.slug,
            oracle_lag_signal=0.0,
            momentum_signal=0.0,
            liquidation_signal=0.0,
            combined_signal=move.move_bps / 10.0,  # Normalise bps to ~signal range
            estimated_prob_up=0.70 if side == "YES" else 0.30,
            market_implied_prob=ob.market_implied_prob_up,
            edge=move.move_bps,
            time_remaining_secs=secs_remaining,
            # Log Binance price for cross-comparability with prior trades.
            # The actual signal is from self._primary_exchange (default coinbase).
            btc_price=snapshot.binance_price,
            oracle_price=snapshot.oracle_price,
            oracle_open_price=snapshot.oracle_window_open_price,
            regime="latency_arb",
            regime_confidence=move.clob_gap_bps / 10.0,
            risk_size_multiplier=self._risk_state.size_multiplier,
            consecutive_losses=self._risk_mgr.consecutive_losses,
            signal_weights=json.dumps({
                "move_bps": move.move_bps,
                "clob_gap_bps": move.clob_gap_bps,
                "clob_shift_bps": move.clob_shift_bps,
                "direction": move.direction,
            }),
            ob_spread=ob.spread if ob else 0.0,
            ob_yes_depth=ob.yes_bid_depth if ob else 0.0,
            ob_ask_depth=ob.yes_ask_depth if ob else 0.0,
            session_trade_num=self._executor.total_trades + 1,
        )

        if trade:
            self._latency_tracker.mark_acknowledged(trade_id)
            self._state = STATE_ENTERED
            self._window_trades += 1
            self._entry_side = side
            self._entry_price = trade.entry_price
            self._entry_time = snapshot.timestamp
            self._trailing_high_bid = trade.entry_price  # init at entry
            self._last_trade_msg = (
                f">> {side} @ ${trade.entry_price:.3f} "
                f"(move: {move.direction} {move.move_bps:.1f}bps, "
                f"gap: {move.clob_gap_bps:.1f}bps) "
                f"[trade {self._window_trades}/{self._max_trades_per_window}]"
            )
            self._reason = f"Entered {side} @ ${trade.entry_price:.3f}"
            logger.info(f"[{self.name}] {self._last_trade_msg}")
        else:
            self._latency_tracker.cancel(trade_id, reason="executor_returned_none")

    def _tick_entered(
        self, snapshot: DataSnapshot, ob, secs_remaining: float
    ) -> None:
        """ENTERED state: monitor for exit opportunities.

        Exit priority (first to trigger wins):
        1. Trailing stop — lock in profits if price reversed from peak
        2. Profit target — bid reached entry + min_profit_ticks
        3. Loss cut — bid dropped by max_loss_pct (tighter cap on losses)
        4. Time exit — held for max_hold_secs without profit/loss trigger
        """
        if not ob:
            return

        # Get current bid for our side
        if self._entry_side == "YES":
            self._current_bid = ob.yes_best_bid
        else:
            self._current_bid = ob.no_best_bid

        if self._current_bid <= 0:
            return

        # Track trailing high bid (for trailing stop)
        if self._current_bid > self._trailing_high_bid:
            self._trailing_high_bid = self._current_bid

        # Calculate unrealised PnL
        if self._executor.pending_trade:
            shares = self._executor.pending_trade.num_shares
            gross = (self._current_bid - self._entry_price) * shares
            fee = gross * 0.02 if gross > 0 else 0.0
            self._unrealised_pnl = gross - fee

        spread = self._current_bid - self._entry_price
        time_held = snapshot.timestamp - self._entry_time

        # ── 1. Trailing stop ──
        # If bid moved up by trailing_lock_ticks, protect profits
        # by exiting when bid drops trailing_giveback_ticks from peak.
        # This captures moves larger than min_profit_ticks without
        # exiting too early.
        trailing_gain = self._trailing_high_bid - self._entry_price
        if (trailing_gain >= self._trailing_lock_ticks
                and self._current_bid <= self._trailing_high_bid - self._trailing_giveback_ticks):
            self._exit_trade(
                snapshot, "trailing_stop",
                f"Trailing stop: peak ${self._trailing_high_bid:.3f}, "
                f"exit ${self._current_bid:.3f}",
            )
            return

        # ── 2. Profit target ──
        if self._exit_mode == "smart" and spread >= self._min_profit_ticks:
            self._exit_trade(
                snapshot, "smart_exit_profit",
                f"Profit target: ${self._entry_price:.3f} -> "
                f"${self._current_bid:.3f}",
            )
            return

        # ── 3. Loss cut ──
        if self._max_loss_pct > 0:
            loss_pct = (self._entry_price - self._current_bid) / self._entry_price
            if loss_pct >= self._max_loss_pct:
                self._exit_trade(
                    snapshot, f"loss_cut_{int(self._max_loss_pct * 100)}pct",
                    f"Loss cut: ${self._entry_price:.3f} -> "
                    f"${self._current_bid:.3f} ({loss_pct:.0%})",
                )
                return

        # ── 4. Time-based exit ──
        # If neither profit nor loss triggered within max_hold_secs, exit
        # at current bid. The latency thesis is "CLOB will reprice within
        # seconds" — if it hasn't in 60s, the thesis has failed.
        if self._max_hold_secs > 0 and time_held >= self._max_hold_secs:
            self._exit_trade(
                snapshot, "time_exit",
                f"Time exit ({time_held:.0f}s): ${self._entry_price:.3f} -> "
                f"${self._current_bid:.3f}",
            )
            return

        self._reason = (
            f"Holding {self._entry_side} @ ${self._entry_price:.3f} | "
            f"Bid: ${self._current_bid:.3f} (peak ${self._trailing_high_bid:.3f}) | "
            f"P&L: ${self._unrealised_pnl:+.2f} | {time_held:.0f}s"
        )

    def _exit_trade(self, snapshot, reason: str, display_reason: str) -> None:
        """Helper: close position and update state for re-entry."""
        trade = self._executor.close_position_early(
            exit_price=self._current_bid,
            reason=reason,
        )
        if trade:
            self._state = STATE_SCANNING  # Re-enter after cooldown
            self._last_exit_time = snapshot.timestamp
            won = trade.pnl is not None and trade.pnl > 0
            self._risk_mgr.record_result(trade.pnl or 0)
            self._last_trade_msg = (
                f"{'W' if won else 'L'} {reason} "
                f"P&L: ${trade.pnl:+.2f} "
                f"[{self._window_trades}/{self._max_trades_per_window}]"
            )
            self._reason = (
                f"{display_reason} "
                f"(cooldown {self._re_entry_cooldown_secs:.0f}s)"
            )
            logger.info(f"[{self.name}] {self._last_trade_msg}")

    def on_window_end(self, resolution: str) -> None:
        """Resolve pending trade if still holding."""
        if self._state == STATE_ENTERED:
            trade = self._executor.resolve_pending_trade(resolution)
            if trade:
                won = trade.pnl is not None and trade.pnl > 0
                self._risk_mgr.record_result(trade.pnl or 0)
                self._last_trade_msg = (
                    f"{'W' if won else 'L'} {trade.side} "
                    f"{'WIN' if won else 'LOSS'} ${trade.pnl:+.2f} "
                    f"Resolved: {resolution}"
                )
                logger.info(f"[{self.name}] {self._last_trade_msg}")

        self._state = STATE_SCANNING
        self._equity_curve.append(self._executor.bankroll)

    def get_dashboard_state(self) -> dict:
        """Return state dict for the multi-bot dashboard."""
        recent = self._executor.session_trades[-8:]
        recent_dicts = []
        for t in recent:
            recent_dicts.append({
                "side": t.side,
                "entry_price": t.entry_price,
                "size_usdc": t.size_usdc,
                "pnl": t.pnl,
                "resolution": t.resolution,
                "created_at": t.created_at,
                "resolved_at": getattr(t, "resolved_at", ""),
            })

        pending = None
        if self._executor.pending_trade:
            pt = self._executor.pending_trade
            pending = {
                "side": pt.side,
                "entry_price": pt.entry_price,
                "size_usdc": pt.size_usdc,
            }

        move_info = {}
        if self._last_move:
            move_info = {
                "direction": self._last_move.direction,
                "move_bps": self._last_move.move_bps,
                "clob_gap_bps": self._last_move.clob_gap_bps,
                "clob_shift_bps": self._last_move.clob_shift_bps,
            }

        return {
            "name": self.name,
            "strategy_type": "latency_arb",
            "state": self._state,
            "signals": {
                "Move": self._last_move.move_bps if self._last_move else 0.0,
                "Gap": (
                    self._last_move.clob_gap_bps if self._last_move else 0.0
                ),
            },
            "combined_signal": (
                self._last_move.move_bps / 10.0 if self._last_move else 0.0
            ),
            "prob_up": 0.70 if self._entry_side == "YES" else (
                0.30 if self._entry_side == "NO" else 0.50
            ),
            "entry_decision": self._reason,
            "side": self._entry_side if self._state == STATE_ENTERED else "",
            "regime": "latency_arb",
            "regime_confidence": 0.0,
            "bankroll": self._executor.bankroll,
            "session_pnl": self._executor.session_pnl,
            "session_wins": self._executor.session_wins,
            "session_losses": self._executor.session_losses,
            "win_rate": self._executor.win_rate,
            "total_trades": self._executor.total_trades,
            "equity_curve": self._equity_curve[-50:],
            "pending_trade": pending,
            "recent_trades": recent_dicts,
            "last_trade_msg": self._last_trade_msg,
            # Latency arb specific
            "move_info": move_info,
            "entry_price": self._entry_price,
            "current_bid": self._current_bid,
            "unrealised_pnl": self._unrealised_pnl,
            "exit_mode": self._exit_mode,
            "max_loss_pct": self._max_loss_pct,
            "window_trades": self._window_trades,
            "max_trades_per_window": self._max_trades_per_window,
        }

    def shutdown(self) -> None:
        """Close database connection and flush latency tracker."""
        self._db.close()
        try:
            self._latency_tracker.close()
        except Exception:
            pass
        logger.info(f"[{self.name}] Shut down")
