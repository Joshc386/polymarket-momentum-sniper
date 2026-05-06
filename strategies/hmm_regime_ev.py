"""Bot C: HMM Regime EV Strategy.

Identical to Bot A (Contrarian EV) except the regime detector uses a
Hidden Markov Model instead of ATR-based threshold rules. All 5 signal
layers, entry logic, sizing, and risk management are unchanged — only
the regime classification method differs.

This isolates the effect of probabilistic regime detection vs the
baseline threshold-based approach.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from core.bot_strategy import BotStrategy
from core.data_snapshot import DataSnapshot
from core.execution import PaperExecutionEngine
from logging_db.database import Database
from signals.oracle_lag import OracleLagSignal
from signals.momentum import MomentumSignal
from signals.liquidation import LiquidationSignal
from signals.orderbook_signal import OrderbookSignal
from signals.sentiment_signal import SentimentSignal
from signals.combiner import SignalCombiner
from signals.hmm_regime import HmmRegimeDetector
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from strategy.regime_detector import Regime
from strategy.event_calendar import EventCalendar

logger = logging.getLogger(__name__)


class HmmRegimeEvStrategy:
    """Contrarian EV strategy with HMM-based regime detection.

    Replaces the ATR-based RegimeDetector with a Hidden Markov Model
    that infers market regimes probabilistically using forward algorithm
    filtering on observable candle features (returns, volatility,
    directional conviction).

    All other logic (L1-L5 signals, entry, sizing, risk) is identical
    to Bot A (ContrarianEvStrategy).
    """

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self._cfg = cfg

        # Database (independent per bot)
        db_path = cfg.get("db_path", f"./data_runtime/{name}.db")
        self._db = Database(db_path)
        self._db.connect()

        # Signal layers (identical to Bot A)
        sig_cfg = cfg.get("signals", {})
        oracle_cfg = sig_cfg.get("oracle_lag", {})
        mom_cfg = sig_cfg.get("momentum", {})

        self._oracle_lag = OracleLagSignal(
            max_expected_lag=oracle_cfg.get("max_expected_lag", 0.001),
        )
        self._momentum = MomentumSignal(
            roc_weight=mom_cfg.get("roc_weight", 0.30),
            direction_weight=mom_cfg.get("direction_weight", 0.25),
            volume_weight=mom_cfg.get("volume_weight", 0.25),
            body_ratio_weight=mom_cfg.get("body_ratio_weight", 0.10),
            rsi_weight=mom_cfg.get("rsi_weight", 0.10),
            rsi_period=mom_cfg.get("rsi_period", 5),
            lookback_candles=mom_cfg.get("lookback_candles", 10),
        )
        self._liquidation = LiquidationSignal()
        self._orderbook = OrderbookSignal()
        self._sentiment = SentimentSignal()
        self._combiner = SignalCombiner(
            max_adjustment=sig_cfg.get("max_adjustment", 0.20),
        )

        # Strategy components (identical to Bot A)
        entry_cfg = cfg.get("entry", {})
        self._entry_logic = EntryLogic(
            min_edge=entry_cfg.get("min_edge", 0.003),
            max_edge=entry_cfg.get("max_edge", 0.025),
            fee_adjustment=entry_cfg.get("fee_adjustment", 0.02),
            min_confidence=entry_cfg.get("min_confidence", 0.02),
            preferred_entry_secs=entry_cfg.get("preferred_entry_secs", 180),
            latest_entry_secs=entry_cfg.get("latest_entry_secs", 60),
            earliest_entry_secs=entry_cfg.get("earliest_entry_secs", 270),
        )

        sizing_cfg = cfg.get("sizing", {})
        self._sizer = PositionSizer(
            kelly_multiplier=sizing_cfg.get("kelly_multiplier", 0.25),
            min_bet_usdc=sizing_cfg.get("min_bet_usdc", 1.0),
            max_bet_usdc=sizing_cfg.get("max_bet_usdc", 5.0),
        )

        risk_cfg = cfg.get("risk", {})
        self._risk_mgr = RiskManager(
            daily_loss_cap_pct=risk_cfg.get("daily_loss_cap_pct", 0.20),
            daily_loss_warn_pct=risk_cfg.get("daily_loss_warn_pct", 0.15),
            min_bankroll=risk_cfg.get("min_bankroll", 10.0),
            streak_reduce_at=risk_cfg.get("streak_reduce_at", 3),
            streak_reduce_factor=risk_cfg.get("streak_reduce_factor", 0.75),
            streak_heavy_reduce_at=risk_cfg.get("streak_heavy_reduce_at", 5),
            streak_heavy_reduce_factor=risk_cfg.get("streak_heavy_reduce_factor", 0.5),
            streak_pause_at=risk_cfg.get("streak_pause_at", 7),
            streak_pause_minutes=risk_cfg.get("streak_pause_minutes", 30),
            streak_reset_wins=risk_cfg.get("streak_reset_wins", 3),
            low_volatility_threshold=risk_cfg.get("low_volatility_threshold", 0.0001),
            high_volatility_threshold=risk_cfg.get("high_volatility_threshold", 0.02),
            high_volatility_size_factor=risk_cfg.get("high_volatility_size_factor", 0.5),
        )

        # ── HMM regime detector (the key difference from Bot A) ──
        hmm_cfg = sig_cfg.get("hmm_regime", {})
        self._regime_detector = HmmRegimeDetector(
            transition_stickiness=hmm_cfg.get("transition_stickiness", 0.90),
            adaptation_rate=hmm_cfg.get("adaptation_rate", 0.02),
            min_candles=hmm_cfg.get("min_candles", 16),
            feature_lookback=hmm_cfg.get("feature_lookback", 5),
        )

        self._event_calendar = EventCalendar(buffer_minutes=15)

        # Weekend signal flip — inverts strategy on Sat/Sun
        self._weekend_flip = cfg.get("weekend_flip", False)

        # Execution engine (independent bankroll + trade history)
        self._executor = PaperExecutionEngine(
            db=self._db,
            initial_bankroll=sizing_cfg.get("initial_bankroll", 100.0),
            slippage=entry_cfg.get("fok_slippage", 0.005),
        )
        self._executor.restore_from_db()

        # Per-window state
        self._has_position = False
        self._entry_decision = EntryDecision()
        self._last_trade_msg = ""

        # Signal state (for dashboard)
        self._oracle_lag_val = 0.0
        self._momentum_val = 0.0
        self._liquidation_val = 0.0
        self._ob_signal_val = 0.0
        self._sentiment_val = 0.0
        self._coinbase_dir = 0.0
        self._combined_signal = 0.0
        self._est_prob_up = 0.5
        self._current_weights = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._current_regime = None
        self._regime_params = {}
        self._risk_state = None
        self._equity_curve: list[float] = [sizing_cfg.get("initial_bankroll", 100.0)]

        # Bot C differentiator: only trade during trending regimes
        self._only_trending = entry_cfg.get("only_trending", False)
        self._min_regime_confidence = entry_cfg.get("min_regime_confidence", 0.50)

        # HMM-specific dashboard state
        self._state_probs: dict[str, float] = {}

    def on_new_window(self, snapshot: DataSnapshot) -> None:
        """Reset per-window state."""
        self._has_position = False
        self._entry_decision = EntryDecision()
        self._orderbook.reset()

    def on_tick(self, snapshot: DataSnapshot) -> None:
        """Compute signals, evaluate entry, execute trade if warranted."""
        mkt = snapshot.market
        if not mkt or not mkt.is_active or snapshot.binance_price <= 0:
            self._entry_decision = EntryDecision(reason="No active market")
            return

        secs_remaining = snapshot.seconds_remaining

        # ── Signal computation (identical to Bot A) ──
        self._oracle_lag_val = self._oracle_lag.compute(
            exchange_price=snapshot.binance_price,
            oracle_price=snapshot.oracle_price,
            oracle_open_price=snapshot.oracle_window_open_price,
        )
        self._momentum_val = self._momentum.compute(
            candles=snapshot.binance_candles,
            current_candle=snapshot.binance_current_candle,
        )
        self._liquidation_val = self._liquidation.compute(
            current_price=snapshot.binance_price,
            liq_data=snapshot.coinglass_data,
            price_momentum=self._momentum_val,
            live_stats=snapshot.binance_liq_stats,
            coinalyze_snapshot=snapshot.coinalyze_snapshot,
        )
        self._coinbase_dir = snapshot.coinbase_direction

        # L4: Orderbook
        ob = snapshot.orderbook
        if ob and ob.yes_bid_total_size > 0:
            self._ob_signal_val = self._orderbook.compute(
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
                yes_bid_delta=snapshot.ob_yes_bid_delta,
                yes_ask_delta=snapshot.ob_yes_ask_delta,
                no_bid_delta=snapshot.ob_no_bid_delta,
                no_ask_delta=snapshot.ob_no_ask_delta,
                num_bid_levels=ob.num_yes_bid_levels,
                num_ask_levels=ob.num_yes_ask_levels,
            )
        else:
            self._ob_signal_val = 0.0

        # L5: Sentiment
        if snapshot.coinalyze_snapshot and snapshot.coinalyze_snapshot.is_valid:
            self._sentiment_val = self._sentiment.compute(
                snapshot=snapshot.coinalyze_snapshot,
                price_momentum=self._momentum_val,
            )
        else:
            self._sentiment_val = 0.0

        # ── HMM regime detection (the key difference) ──
        self._current_regime = self._regime_detector.detect(snapshot.binance_candles)
        self._regime_params = self._regime_detector.get_params(
            self._current_regime.regime
        )
        self._state_probs = self._regime_detector.get_state_probabilities()

        # Combine signals
        is_weekend = datetime.now(timezone.utc).weekday() >= 5
        self._current_weights = self._combiner.get_weights(secs_remaining)
        self._combined_signal, self._est_prob_up = self._combiner.combine(
            oracle_lag_signal=self._oracle_lag_val,
            momentum_signal=self._momentum_val,
            liquidation_signal=self._liquidation_val,
            seconds_remaining=secs_remaining,
            coinbase_direction=self._coinbase_dir,
            orderbook_signal=self._ob_signal_val,
            sentiment_signal=self._sentiment_val,
            regime_weight_adjustments=self._regime_params.get("weight_adjustments"),
            flip_signal=self._weekend_flip and is_weekend,
        )

        # ── Risk check (identical to Bot A) ──
        recent_vol = self._risk_mgr.compute_recent_volatility(
            snapshot.binance_candles
        )
        self._risk_state = self._risk_mgr.evaluate(
            self._executor.bankroll, recent_vol
        )

        if not snapshot.feeds_healthy and self._risk_state.can_trade:
            self._risk_state.can_trade = False
            self._risk_state.reason = f"Unhealthy feeds: {', '.join(snapshot.health_issues)}"

        event_blocked, event_name, event_minutes = self._event_calendar.is_event_window()
        if event_blocked and self._risk_state.can_trade:
            self._risk_state.can_trade = False
            self._risk_state.reason = f"Event: {event_name} ({event_minutes:+.0f}m)"

        if (self._current_regime
                and self._current_regime.regime == Regime.LOW_VOLATILITY
                and self._risk_state.can_trade):
            self._risk_state.can_trade = False
            self._risk_state.reason = (
                f"Low vol regime (conf={self._current_regime.confidence:.2f})"
            )

        if (self._current_regime
                and self._regime_params.get("size_multiplier", 1.0) < 1.0):
            self._risk_state.size_multiplier *= self._regime_params["size_multiplier"]

        # ── Entry evaluation (identical to Bot A) ──
        regime_edge_mult = (
            self._regime_params.get("edge_multiplier", 1.0)
            if self._regime_params else 1.0
        )

        if ob and self._risk_state.can_trade and not self._has_position:
            # ── Bot C differentiator: only trade during trending regimes ──
            # Sits out ranging, low_vol, and high_vol windows entirely.
            # This means Bot C trades far less often than A or B, but
            # should have higher accuracy when it does trade.
            if self._only_trending and self._current_regime:
                regime = self._current_regime.regime
                conf = self._current_regime.confidence
                is_trending = regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
                if not is_trending or conf < self._min_regime_confidence:
                    self._entry_decision = EntryDecision(
                        reason=(
                            f"Regime {regime.value} "
                            f"(conf={conf:.2f}) — waiting for trend"
                        )
                    )
                    return

            self._entry_decision = self._entry_logic.evaluate(
                est_prob_up=self._est_prob_up,
                yes_best_ask=ob.yes_best_ask,
                no_best_ask=ob.no_best_ask,
                yes_best_bid=ob.yes_best_bid,
                no_best_bid=ob.no_best_bid,
                seconds_remaining=secs_remaining,
                has_position=self._has_position,
                regime_edge_multiplier=regime_edge_mult,
            )

            if self._entry_decision.should_enter:
                self._execute_trade(snapshot, ob, secs_remaining)
        elif ob and not self._risk_state.can_trade:
            self._entry_decision = EntryDecision(
                reason=f"Risk: {self._risk_state.reason}"
            )
        elif self._has_position:
            self._entry_decision = EntryDecision(
                reason="Position held this window"
            )
        elif not ob:
            self._entry_decision = EntryDecision(reason="No orderbook data")

    def _execute_trade(
        self, snapshot: DataSnapshot, ob, secs_remaining: float
    ) -> None:
        """Place a paper trade using the current entry decision."""
        ed = self._entry_decision
        win_prob = (
            self._est_prob_up if ed.side == "YES"
            else (1.0 - self._est_prob_up)
        )
        bet_size = self._sizer.compute(
            est_prob=win_prob,
            share_price=ed.price,
            bankroll=self._executor.bankroll,
            size_multiplier=self._risk_state.size_multiplier,
        )
        if bet_size <= 0:
            return

        weights_json = json.dumps({
            "L1": self._current_weights[0], "L2": self._current_weights[1],
            "L3": self._current_weights[2], "L4": self._current_weights[3],
            "L5": self._current_weights[4],
        })

        mkt = snapshot.market
        trade = self._executor.execute_trade(
            side=ed.side,
            price=ed.price,
            size_usdc=bet_size,
            market_id=mkt.condition_id,
            market_slug=mkt.slug,
            oracle_lag_signal=self._oracle_lag_val,
            momentum_signal=self._momentum_val,
            liquidation_signal=self._liquidation_val,
            combined_signal=self._combined_signal,
            estimated_prob_up=self._est_prob_up,
            market_implied_prob=ob.market_implied_prob_up,
            edge=ed.best_ev,
            time_remaining_secs=secs_remaining,
            btc_price=snapshot.binance_price,
            oracle_price=snapshot.oracle_price,
            oracle_open_price=snapshot.oracle_window_open_price,
            orderbook_signal=self._ob_signal_val,
            sentiment_signal=self._sentiment_val,
            coinbase_direction=self._coinbase_dir,
            regime=(
                self._current_regime.regime.value
                if self._current_regime else ""
            ),
            regime_confidence=(
                self._current_regime.confidence
                if self._current_regime else 0.0
            ),
            risk_size_multiplier=self._risk_state.size_multiplier,
            consecutive_losses=self._risk_mgr.consecutive_losses,
            signal_weights=weights_json,
            ob_spread=ob.spread if ob else 0.0,
            ob_yes_depth=ob.yes_bid_depth if ob else 0.0,
            ob_ask_depth=ob.yes_ask_depth if ob else 0.0,
            session_trade_num=self._executor.total_trades + 1,
        )

        if trade:
            self._has_position = True
            self._last_trade_msg = (
                f">> [PAPER] {trade.side} @ "
                f"${trade.entry_price:.4f} ${trade.size_usdc:.2f}"
            )
            logger.info(f"[{self.name}] {self._last_trade_msg}")

    def on_window_end(self, resolution: str) -> None:
        """Resolve pending trade and record result."""
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

        return {
            "name": self.name,
            "signals": {
                "L1 Oracle": self._oracle_lag_val,
                "L2 Mom": self._momentum_val,
                "L3 Liq": self._liquidation_val,
                "L4 Book": self._ob_signal_val,
                "L5 Sent": self._sentiment_val,
            },
            "combined_signal": self._combined_signal,
            "prob_up": self._est_prob_up,
            "entry_decision": (
                self._entry_decision.reason
                if self._entry_decision else "Waiting"
            ),
            "side": (
                self._entry_decision.side
                if self._entry_decision and self._entry_decision.should_enter
                else ""
            ),
            "regime": (
                self._current_regime.regime.value
                if self._current_regime else "unknown"
            ),
            "regime_confidence": (
                self._current_regime.confidence
                if self._current_regime else 0.0
            ),
            # HMM-specific: full state probability distribution
            "hmm_state_probs": self._state_probs,
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
        }

    def shutdown(self) -> None:
        """Close database connection."""
        self._db.close()
        logger.info(f"[{self.name}] Shut down")
