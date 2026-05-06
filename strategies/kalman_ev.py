"""Bot B: Kalman EV Strategy — Kalman Filter upgrade to L1 oracle lag.

Identical to ContrarianEvStrategy except L1 uses a Kalman Filter that
fuses Binance + Coinbase + Oracle prices to predict oracle landing,
instead of the simple exchange-minus-oracle difference.
"""

import json
import logging
from datetime import datetime, timezone

from core.bot_strategy import BotStrategy
from core.data_snapshot import DataSnapshot
from core.execution import PaperExecutionEngine
from logging_db.database import Database
from signals.kalman_filter import KalmanFilterSignal
from signals.momentum import MomentumSignal
from signals.liquidation import LiquidationSignal
from signals.orderbook_signal import OrderbookSignal
from signals.orderbook_fade import OrderbookFadeSignal
from signals.sentiment_signal import SentimentSignal
from signals.taker_ratio import TakerRatioSignal
from signals.combiner import SignalCombiner
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from strategy.regime_detector import RegimeDetector, Regime
from strategy.mtf_regime_detector import MTFRegimeDetector
from strategy.event_calendar import EventCalendar

logger = logging.getLogger(__name__)


class KalmanEvStrategy:
    """Contrarian EV strategy with Kalman Filter replacing L1 oracle lag.

    The Kalman Filter fuses three price sources (Binance, Coinbase,
    Chainlink oracle) into a smoothed price estimate with velocity
    tracking. This provides:
    - Better noise rejection than raw price difference
    - Leading indicator via velocity estimate
    - Breakout detection for confidence boosting
    - Adaptive tracking (faster in volatile periods, smoother in calm)
    """

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self._cfg = cfg

        # Database (independent per bot)
        db_path = cfg.get("db_path", f"./data_runtime/{name}.db")
        self._db = Database(db_path)
        self._db.connect()

        # Signal layers — L1 is Kalman Filter instead of OracleLagSignal
        sig_cfg = cfg.get("signals", {})
        kf_cfg = sig_cfg.get("kalman_filter", {})
        mom_cfg = sig_cfg.get("momentum", {})

        self._kalman = KalmanFilterSignal(
            process_noise_scale=kf_cfg.get("process_noise_scale", 1e-5),
            measurement_noise_binance=kf_cfg.get("measurement_noise_binance", 1e-4),
            measurement_noise_coinbase=kf_cfg.get("measurement_noise_coinbase", 2e-4),
            measurement_noise_oracle=kf_cfg.get("measurement_noise_oracle", 5e-4),
            adaptive_window=kf_cfg.get("adaptive_window", 30),
            breakout_velocity_threshold=kf_cfg.get("breakout_velocity_threshold", 0.5),
            max_expected_lag=kf_cfg.get("max_expected_lag", 0.001),
            velocity_confidence_boost=kf_cfg.get("velocity_confidence_boost", 1.15),
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
        self._orderbook_fade = OrderbookFadeSignal(
            imbalance_threshold=sig_cfg.get("fade_threshold", 0.35),
            min_secs_remaining=sig_cfg.get("fade_min_secs", 60.0),
            max_secs_remaining=sig_cfg.get("fade_max_secs", 200.0),
        )
        self._sentiment = SentimentSignal()
        self._taker_ratio = TakerRatioSignal(
            window_secs=sig_cfg.get("taker_window_secs", 30.0),
            min_trades=sig_cfg.get("taker_min_trades", 20),
            neutral_band=sig_cfg.get("taker_neutral_band", 0.05),
        )
        self._combiner = SignalCombiner(
            max_adjustment=sig_cfg.get("max_adjustment", 0.20),
            weight_schedule_name=sig_cfg.get("weight_schedule", "kf_dominant"),
        )

        # Strategy components (identical config interface to Bot A)
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

        regime_cfg = cfg.get("regime", {})
        if regime_cfg.get("detector") == "mtf":
            self._regime_detector = MTFRegimeDetector(
                trend_threshold=regime_cfg.get("trend_threshold", 0.30),
                high_vol_percentile=regime_cfg.get("high_vol_percentile", 0.90),
                low_vol_percentile=regime_cfg.get("low_vol_percentile", 0.20),
                alignment_weight=regime_cfg.get("alignment_weight", 0.55),
                stickiness=regime_cfg.get("stickiness", 3),
            )
        else:
            self._regime_detector = RegimeDetector()
        self._regime_allowed = set(regime_cfg.get("allowed_regimes", []))
        self._regime_blocked = set(regime_cfg.get("block_regimes", []))
        self._event_calendar = EventCalendar(buffer_minutes=15)

        # Weekend signal flip — inverts strategy on Sat/Sun
        self._weekend_flip = cfg.get("weekend_flip", False)

        # Execution engine
        self._executor = PaperExecutionEngine(
            db=self._db,
            initial_bankroll=sizing_cfg.get("initial_bankroll", 100.0),
            slippage=entry_cfg.get("fok_slippage", 0.005),
        )
        self._executor.restore_from_db()

        # Bot B differentiator: require KF velocity to confirm trade direction
        self._require_velocity_confirm = entry_cfg.get(
            "require_velocity_confirm", False
        )

        # Per-window state
        self._has_position = False
        self._entry_decision = EntryDecision()
        self._last_trade_msg = ""

        # Signal state
        self._kalman_signal_val = 0.0
        self._momentum_val = 0.0
        self._liquidation_val = 0.0
        self._ob_signal_val = 0.0
        self._fade_signal_val = 0.0
        self._sentiment_val = 0.0
        self._taker_ratio_val = 0.0
        self._coinbase_dir = 0.0
        self._combined_signal = 0.0
        self._est_prob_up = 0.5
        self._current_weights = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._current_regime = None
        self._regime_params = {}
        self._risk_state = None
        self._equity_curve: list[float] = [sizing_cfg.get("initial_bankroll", 100.0)]

    def on_new_window(self, snapshot: DataSnapshot) -> None:
        """Reset per-window state."""
        self._has_position = False
        self._entry_decision = EntryDecision()
        self._orderbook.reset()
        self._orderbook_fade.reset()

    def on_tick(self, snapshot: DataSnapshot) -> None:
        """Compute signals (KF for L1), evaluate entry, execute."""
        mkt = snapshot.market
        if not mkt or not mkt.is_active or snapshot.binance_price <= 0:
            self._entry_decision = EntryDecision(reason="No active market")
            return

        secs_remaining = snapshot.seconds_remaining

        # ── L1: Kalman Filter (replaces oracle lag) ──
        kf_output = self._kalman.update(
            binance_price=snapshot.binance_price,
            coinbase_price=snapshot.coinbase_price,
            oracle_price=snapshot.oracle_price,
            coinbase_connected=snapshot.coinbase_connected,
        )
        self._kalman_signal_val = kf_output.signal

        # ── L2-L5: identical to Bot A ──
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

        # L6: Orderbook fade (contrarian on extreme imbalances)
        if ob and ob.yes_bid_depth > 0:
            self._fade_signal_val = self._orderbook_fade.compute(
                yes_bid_depth=ob.yes_bid_depth,
                yes_ask_depth=ob.yes_ask_depth,
                seconds_remaining=secs_remaining,
            )
        else:
            self._fade_signal_val = 0.0

        if snapshot.coinalyze_snapshot and snapshot.coinalyze_snapshot.is_valid:
            self._sentiment_val = self._sentiment.compute(
                snapshot=snapshot.coinalyze_snapshot,
                price_momentum=self._momentum_val,
            )
        else:
            self._sentiment_val = 0.0

        # L7: Taker buy ratio (from Binance trade stream)
        self._taker_ratio_val = self._taker_ratio.compute()

        # Regime
        self._current_regime = self._regime_detector.detect(snapshot.binance_candles)
        self._regime_params = self._regime_detector.get_params(
            self._current_regime.regime
        )

        # ── Combine signals (KF signal goes into oracle_lag slot) ──
        is_weekend = datetime.now(timezone.utc).weekday() >= 5

        # Select weight schedule based on regime
        _schedule_override = ""
        if self._current_regime and self._current_regime.regime == Regime.RANGING:
            _schedule_override = "ranging"

        self._current_weights = self._combiner.get_weights(secs_remaining, _schedule_override)
        self._combined_signal, self._est_prob_up = self._combiner.combine(
            oracle_lag_signal=self._kalman_signal_val,  # KF replaces L1
            momentum_signal=self._momentum_val,
            liquidation_signal=self._liquidation_val,
            seconds_remaining=secs_remaining,
            coinbase_direction=self._coinbase_dir,
            orderbook_signal=self._ob_signal_val,
            sentiment_signal=self._sentiment_val,
            regime_weight_adjustments=self._regime_params.get("weight_adjustments"),
            flip_signal=self._weekend_flip and is_weekend,
            fade_signal=self._fade_signal_val,
            taker_ratio_signal=self._taker_ratio_val,
            schedule_override=_schedule_override,
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

        if self._current_regime and self._risk_state.can_trade:
            regime_label = self._current_regime.regime.value
            blocked = False
            if self._regime_blocked and regime_label in self._regime_blocked:
                blocked = True
            elif self._regime_allowed and regime_label not in self._regime_allowed:
                blocked = True
            elif self._current_regime.regime == Regime.LOW_VOLATILITY:
                blocked = True
            if blocked:
                self._risk_state.can_trade = False
                self._risk_state.reason = (
                    f"Regime blocked: {regime_label} "
                    f"(conf={self._current_regime.confidence:.2f})"
                )

        if (self._current_regime
                and self._regime_params.get("size_multiplier", 1.0) < 1.0):
            self._risk_state.size_multiplier *= self._regime_params["size_multiplier"]

        # ── Entry evaluation ──
        regime_edge_mult = (
            self._regime_params.get("edge_multiplier", 1.0)
            if self._regime_params else 1.0
        )

        if ob and self._risk_state.can_trade and not self._has_position:
            self._entry_decision = self._entry_logic.evaluate(
                est_prob_up=self._est_prob_up,
                yes_best_ask=ob.yes_best_ask,
                no_best_ask=ob.no_best_ask,
                yes_best_bid=ob.yes_best_bid,
                no_best_bid=ob.no_best_bid,
                seconds_remaining=secs_remaining,
                has_position=self._has_position,
                regime_edge_multiplier=regime_edge_mult,
                signal_aligned=_schedule_override == "ranging",
            )

            # ── Bot B differentiator: KF velocity must confirm direction ──
            # If entering YES, velocity must be positive (price rising).
            # If entering NO, velocity must be negative (price falling).
            # This filters out trades where signals say one thing but
            # price momentum (as tracked by the Kalman filter) disagrees.
            if self._entry_decision.should_enter and self._require_velocity_confirm:
                velocity = kf_output.velocity
                side = self._entry_decision.side
                if (side == "YES" and velocity <= 0) or (side == "NO" and velocity >= 0):
                    self._entry_decision.should_enter = False
                    self._entry_decision.reason = (
                        f"KF velocity {velocity:+.2f} opposes {side} | "
                        f"edge={self._entry_decision.prob_edge:.3f}"
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
        """Place a paper trade."""
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
            oracle_lag_signal=self._kalman_signal_val,
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
        kf = self._kalman.output

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
                "L1 Kalman": self._kalman_signal_val,
                "L2 Mom": self._momentum_val,
                "L3 Liq": self._liquidation_val,
                "L4 Book": self._ob_signal_val,
                "L5 Sent": self._sentiment_val,
                "L6 Fade": self._fade_signal_val,
                "L7 Taker": self._taker_ratio_val,
            },
            "kalman_extras": {
                "filtered_price": kf.filtered_price,
                "velocity": kf.velocity,
                "noise_ratio": kf.noise_ratio,
                "breakout": kf.breakout_flag,
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
