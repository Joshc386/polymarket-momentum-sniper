"""Bot A: Contrarian EV Strategy — baseline.

Mechanical extraction of the current main.py trading logic into a
BotStrategy-compatible class. This is the production strategy that has
been running in paper trading. No modifications from the original logic.
"""

import json
import logging
import os
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
from signals.orderbook_fade import OrderbookFadeSignal
from signals.sentiment_signal import SentimentSignal
from signals.taker_ratio import TakerRatioSignal
from signals.absorption import AbsorptionSignal
from signals.clob_flow import CLOBFlowSignal
from signals.level_exhaustion import LevelExhaustionSignal
from signals.trade_size import TradeSizeSignal
from signals.wallet_flow import WalletFlowSignal
from signals.combiner import SignalCombiner
from strategy.entry_logic import EntryLogic, EntryDecision
from strategy.sizing import PositionSizer
from strategy.risk_manager import RiskManager
from strategy.regime_detector import RegimeDetector, Regime
from strategy.mtf_regime_detector import MTFRegimeDetector
from strategy.event_calendar import EventCalendar
from strategy.sm_confirmation import (
    SMConfirmationConfig, SMDecision, PositionSide,
    check_sm_confirmation,
)
from strategy.sm_decision_log import SMDecisionLogger
from strategy.signal_diagnostic_log import SignalDiagnosticLogger, SignalTick

logger = logging.getLogger(__name__)


class ContrarianEvStrategy:
    """Baseline contrarian EV strategy — buys the cheap side when model
    estimates +EV after Polymarket's 2% winner fee.

    This is a direct extraction of the main.py inline logic. Every signal
    computation, entry evaluation, and execution call is identical to the
    single-bot version.
    """

    def __init__(self, name: str, cfg: dict) -> None:
        self.name = name
        self._cfg = cfg

        # Database (independent per bot)
        db_path = cfg.get("db_path", f"./data_runtime/{name}.db")
        self._db = Database(db_path)
        self._db.connect()

        # Signal layers
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

        # L4: Orderbook — sub-signal weights and normalization constants
        ob_cfg = sig_cfg.get("orderbook", {})
        self._orderbook = OrderbookSignal(
            imbalance_weight=ob_cfg.get("imbalance_weight", 0.30),
            flow_weight=ob_cfg.get("flow_weight", 0.25),
            weighted_mid_weight=ob_cfg.get("weighted_mid_weight", 0.20),
            top_pressure_weight=ob_cfg.get("top_pressure_weight", 0.15),
            thickness_weight=ob_cfg.get("thickness_weight", 0.10),
            imbalance_norm=ob_cfg.get("imbalance_norm", 0.3),
            flow_norm=ob_cfg.get("flow_norm", 200.0),
            mid_dev_norm=ob_cfg.get("mid_dev_norm", 0.02),
            top_pressure_norm=ob_cfg.get("top_pressure_norm", 0.3),
            thickness_norm=ob_cfg.get("thickness_norm", 0.3),
        )

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
        self._clob_flow = CLOBFlowSignal(
            min_trades=sig_cfg.get("clob_flow_min_trades", 5),
            large_trade_threshold=sig_cfg.get("clob_flow_large_trade", 100.0),
        )

        # L9b-L12: Optional signal layers — only instantiated when config
        # sets a non-zero weight. Bot G runs without these; Bot K enables
        # them via config. This keeps Bot G's hot path unchanged.
        self._absorption = None
        self._exhaustion = None
        self._trade_size = None
        self._wallet_flow = None

        if sig_cfg.get("absorption_weight", 0.0) > 0:
            abs_cfg = sig_cfg.get("absorption", {})
            self._absorption = AbsorptionSignal(
                min_pressure_volume=abs_cfg.get("min_pressure_volume", 50.0),
                pressure_imbalance_threshold=abs_cfg.get("pressure_imbalance_threshold", 0.30),
                resilience_threshold=abs_cfg.get("resilience_threshold", 0.0),
                smoothing_window=abs_cfg.get("smoothing_window", 10),
                decay_rate=abs_cfg.get("decay_rate", 0.3),
                strength_scale=abs_cfg.get("strength_scale", 2.0),
            )

        if sig_cfg.get("exhaustion_weight", 0.0) > 0:
            exh_cfg = sig_cfg.get("exhaustion", {})
            self._exhaustion = LevelExhaustionSignal(
                min_gap_cents=exh_cfg.get("min_gap_cents", 0.02),
                min_level_size=exh_cfg.get("min_level_size", 5.0),
                min_wall_size_floor=exh_cfg.get("min_wall_size_floor", 10.0),
                min_wall_fraction=exh_cfg.get("min_wall_fraction", 0.02),
                max_depth_levels=exh_cfg.get("max_depth_levels", 10),
                depletion_threshold=exh_cfg.get("depletion_threshold", 0.40),
                lookback_ticks=exh_cfg.get("lookback_ticks", 12),
                decay_rate=exh_cfg.get("decay_rate", 0.5),
                strength_scale=exh_cfg.get("strength_scale", 2.0),
                min_seconds_remaining=exh_cfg.get("min_seconds_remaining", 30.0),
            )

        if sig_cfg.get("trade_size_weight", 0.0) > 0:
            ts_cfg = sig_cfg.get("trade_size", {})
            self._trade_size = TradeSizeSignal(
                large_multiplier=ts_cfg.get("large_multiplier", 3.0),
                large_floor=ts_cfg.get("large_floor", 50.0),
                min_trades=ts_cfg.get("min_trades", 10),
                large_bias_weight=ts_cfg.get("large_bias_weight", 0.40),
                count_asym_weight=ts_cfg.get("count_asym_weight", 0.25),
                size_div_weight=ts_cfg.get("size_div_weight", 0.35),
                decay_rate=ts_cfg.get("decay_rate", 0.3),
            )

        if sig_cfg.get("wallet_flow_weight", 0.0) > 0:
            wf_cfg = sig_cfg.get("wallet_flow", {})
            self._wallet_flow = WalletFlowSignal(
                concentration_weight=wf_cfg.get("concentration_weight", 0.40),
                repeat_weight=wf_cfg.get("repeat_weight", 0.35),
                wallet_asym_weight=wf_cfg.get("wallet_asym_weight", 0.25),
                min_trades=wf_cfg.get("min_trades", 5),
                repeat_trade_min=wf_cfg.get("repeat_trade_min", 2),
                min_seconds_remaining=wf_cfg.get("min_seconds_remaining", 30.0),
                decay_rate=wf_cfg.get("decay_rate", 0.3),
            )

        self._combiner = SignalCombiner(
            max_adjustment=sig_cfg.get("max_adjustment", 0.20),
            weight_schedule_name=sig_cfg.get("weight_schedule", "default"),
            fade_weight=sig_cfg.get("fade_weight", 0.10),
            taker_ratio_weight=sig_cfg.get("taker_ratio_weight", 0.08),
            clob_flow_weight=sig_cfg.get("clob_flow_weight", 0.12),
            absorption_weight=sig_cfg.get("absorption_weight", 0.0),
            exhaustion_weight=sig_cfg.get("exhaustion_weight", 0.0),
            trade_size_weight=sig_cfg.get("trade_size_weight", 0.0),
            wallet_flow_weight=sig_cfg.get("wallet_flow_weight", 0.0),
        )

        # Strategy components
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
            # New features — backward compatible (disabled by default)
            streak_reduction_enabled=risk_cfg.get("streak_reduction_enabled", True),
            drawdown_pct=risk_cfg.get("drawdown_pct", 0.0),
            daily_trade_cap=risk_cfg.get("daily_trade_cap", 0),
            btc_distance_stop=risk_cfg.get("btc_distance_stop", 0.0),
            btc_distance_min_secs_remaining=risk_cfg.get(
                "btc_distance_min_secs_remaining", 60.0
            ),
            regime_size_overrides=risk_cfg.get("regime_size_overrides", {}),
        )

        # Regime detector — use MTF if configured, else fall back to base
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

        # Historical-performance filters (opt-in via config, default off).
        # Any bot can enable them by setting `filters:` in its config.
        # Bot G has them enabled based on its 1,321-trade review.
        filter_cfg = cfg.get("filters", {})
        self._yes_min_price = filter_cfg.get("yes_min_price", 0.0)
        self._skip_regimes = set(filter_cfg.get("skip_regimes", []))
        self._min_depth_multiplier = filter_cfg.get("min_depth_multiplier", 5.0)
        # High-EV early-window filter (added 2026-05-13). Blocks premature
        # high-EV entries: when the model estimates high EV very early in
        # the window, it's typically reacting to noise (no momentum history,
        # orderbook unsettled). Backtest on 2,388 Bot G trades: skipping
        # high-EV (>=0.15) trades in the first 60s saves -$213 PnL while
        # preserving the +$171 PnL from late high-EV winners.
        # Set high_ev_min_secs_into_window: 0 to disable.
        self._high_ev_threshold = filter_cfg.get("high_ev_threshold", 0.15)
        self._high_ev_min_secs_into_window = filter_cfg.get(
            "high_ev_min_secs_into_window", 0.0
        )
        self._last_orderbook = None  # stored each tick for filter access
        # Track how often each filter fires (for dashboard/diagnostics)
        self._filter_skipped = {
            "yes_low_price": 0, "regime": 0, "low_liquidity": 0,
            "high_ev_early": 0,
        }

        # Per-tick signal diagnostic logger (off by default; enable in config
        # with `signal_diagnostic: enabled: true`). Captures all signal values
        # every tick, not just on trades. Used to diagnose signal bias issues
        # like the 100% YES bias observed May 11+ 2026.
        diag_cfg = cfg.get("signal_diagnostic", {})
        self._signal_diag = None
        if diag_cfg.get("enabled", False):
            default_diag_path = os.path.join(
                "data_runtime", f"{self.name}_signal_diag.db"
            )
            diag_path = diag_cfg.get("db_path", default_diag_path)
            self._signal_diag = SignalDiagnosticLogger(
                db_path=diag_path,
                bot_name=self.name,
                enabled=True,
                sample_every_n=diag_cfg.get("sample_every_n", 1),
            )

        # Execution engine (independent bankroll + trade history)
        self._executor = PaperExecutionEngine(
            db=self._db,
            initial_bankroll=sizing_cfg.get("initial_bankroll", 100.0),
            slippage=entry_cfg.get("fok_slippage", 0.005),
        )
        self._executor.restore_from_db()

        # L9: SM Confirmation Layer (optional, off by default)
        sm_cfg = cfg.get("sm_confirmation", {})
        self._sm_enabled = sm_cfg.get("enabled", False)
        self._sm_monitor = cfg.get("_sm_monitor")  # injected by multi_runner
        self._sm_config = None
        self._sm_decision_logger = None
        self._sm_last_checked_minute = -1
        self._sm_l9_status = ""
        if self._sm_enabled and self._sm_monitor:
            self._sm_config = SMConfirmationConfig(
                price_floor=sm_cfg.get("price_floor", 0.65),
                price_ceiling=sm_cfg.get("price_ceiling", 0.80),
                agreement_threshold=sm_cfg.get("agreement_threshold", 0.60),
                min_sm_volume=sm_cfg.get("min_volume", 100.0),
                min_sm_wallets=sm_cfg.get("min_wallets", 2),
            )
            self._sm_decision_logger = SMDecisionLogger()
            self._sm_check_minutes = sm_cfg.get("check_minutes", [4])
            self._sm_exit_sides: set[str] = set(
                sm_cfg.get("exit_sides", ["YES"])
            )
            logger.info(
                "[%s] L9 SM confirmation enabled (check min %s, "
                "exit sides %s, threshold %.0f%%)",
                name, self._sm_check_minutes, self._sm_exit_sides,
                sm_cfg.get("agreement_threshold", 0.60) * 100,
            )
        elif self._sm_enabled and not self._sm_monitor:
            logger.warning(
                "[%s] SM confirmation enabled in config but no SM monitor "
                "available — L9 disabled", name,
            )
            self._sm_enabled = False

        # Per-window state
        self._has_position = False
        self._entry_decision = EntryDecision()
        self._last_trade_msg = ""

        # Signal state (for dashboard)
        self._oracle_lag_val = 0.0
        self._momentum_val = 0.0
        self._liquidation_val = 0.0
        self._ob_signal_val = 0.0
        self._fade_signal_val = 0.0
        self._sentiment_val = 0.0
        self._taker_ratio_val = 0.0
        self._clob_flow_val = 0.0
        self._absorption_val = 0.0
        self._exhaustion_val = 0.0
        self._trade_size_val = 0.0
        self._wallet_flow_val = 0.0
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
        self._taker_ratio.reset()
        self._clob_flow.reset()
        if self._absorption:
            self._absorption.reset()
        if self._exhaustion:
            self._exhaustion.reset()
        if self._trade_size:
            self._trade_size.reset()
        if self._wallet_flow:
            self._wallet_flow.reset()
        self._sm_last_checked_minute = -1
        self._sm_l9_status = ""

    def on_tick(self, snapshot: DataSnapshot) -> None:
        """Compute signals, evaluate entry, execute trade if warranted."""
        mkt = snapshot.market
        if not mkt or not mkt.is_active or snapshot.binance_price <= 0:
            self._entry_decision = EntryDecision(reason="No active market")
            return

        secs_remaining = snapshot.seconds_remaining

        # ── Signal computation (identical to main.py) ──
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
        self._last_orderbook = ob
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

        # L5: Sentiment
        if snapshot.coinalyze_snapshot and snapshot.coinalyze_snapshot.is_valid:
            self._sentiment_val = self._sentiment.compute(
                snapshot=snapshot.coinalyze_snapshot,
                price_momentum=self._momentum_val,
            )
        else:
            self._sentiment_val = 0.0

        # L7: Taker buy ratio (from Binance trade stream)
        self._taker_ratio_val = self._taker_ratio.compute()

        # L8: CLOB trade flow (from Polymarket WebSocket)
        self._clob_flow_val = self._clob_flow.compute(snapshot.clob_trade_flow)

        # L9b: Absorption detection (only when enabled via weight > 0)
        if self._absorption:
            self._absorption_val = self._absorption.compute(
                trade_flow=snapshot.clob_trade_flow,
                ob_yes_bid_delta=snapshot.ob_yes_bid_delta,
                ob_yes_ask_delta=snapshot.ob_yes_ask_delta,
                ob_no_bid_delta=snapshot.ob_no_bid_delta,
                ob_no_ask_delta=snapshot.ob_no_ask_delta,
            )

        # L10: Level exhaustion (only when enabled via weight > 0)
        if self._exhaustion:
            if ob and hasattr(ob, "yes_bid_levels") and ob.yes_bid_levels:
                # Window volume from CLOB trade flow for dynamic wall sizing
                window_vol = (
                    snapshot.clob_trade_flow.total_volume
                    if snapshot.clob_trade_flow
                    else 0.0
                )
                self._exhaustion_val = self._exhaustion.compute(
                    yes_bid_levels=ob.yes_bid_levels,
                    yes_ask_levels=ob.yes_ask_levels,
                    no_bid_levels=ob.no_bid_levels,
                    no_ask_levels=ob.no_ask_levels,
                    seconds_remaining=secs_remaining,
                    window_volume=window_vol,
                )
            else:
                self._exhaustion_val = 0.0

        # L11: Trade size conviction (only when enabled via weight > 0)
        if self._trade_size:
            if snapshot.clob_trade_flow and hasattr(snapshot.clob_trade_flow, "recent_trades"):
                self._trade_size_val = self._trade_size.compute(
                    snapshot.clob_trade_flow.recent_trades,
                )
            else:
                self._trade_size_val = 0.0

        # L12: On-chain wallet flow (only when enabled via weight > 0)
        if self._wallet_flow:
            wallet_monitor = self._cfg.get("_wallet_flow_monitor")
            if wallet_monitor:
                wf_state = wallet_monitor.get_flow_state()
                self._wallet_flow_val = self._wallet_flow.compute(
                    wallet_volumes=wf_state.wallet_volumes,
                    total_bull_volume=wf_state.total_bull_volume,
                    total_bear_volume=wf_state.total_bear_volume,
                    bull_wallets=wf_state.bull_wallets,
                    bear_wallets=wf_state.bear_wallets,
                    trades=wf_state.trades,
                    seconds_remaining=secs_remaining,
                )
            else:
                self._wallet_flow_val = 0.0

        # Regime detection
        self._current_regime = self._regime_detector.detect(snapshot.binance_candles)
        self._regime_params = self._regime_detector.get_params(
            self._current_regime.regime
        )

        # Combine signals
        is_weekend = datetime.now(timezone.utc).weekday() >= 5

        # Select weight schedule based on regime
        _schedule_override = ""
        if self._current_regime and self._current_regime.regime == Regime.RANGING:
            _schedule_override = "ranging"

        self._current_weights = self._combiner.get_weights(secs_remaining, _schedule_override)
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
            fade_signal=self._fade_signal_val,
            taker_ratio_signal=self._taker_ratio_val,
            clob_flow_signal=self._clob_flow_val,
            absorption_signal=self._absorption_val,
            exhaustion_signal=self._exhaustion_val,
            trade_size_signal=self._trade_size_val,
            wallet_flow_signal=self._wallet_flow_val,
            schedule_override=_schedule_override,
        )

        # ── Risk check ──
        recent_vol = self._risk_mgr.compute_recent_volatility(
            snapshot.binance_candles
        )
        regime_label = (
            self._current_regime.regime.value
            if self._current_regime else ""
        )
        self._risk_state = self._risk_mgr.evaluate(
            self._executor.bankroll, recent_vol, current_regime=regime_label
        )

        if not snapshot.feeds_healthy and self._risk_state.can_trade:
            self._risk_state.can_trade = False
            self._risk_state.reason = f"Unhealthy feeds: {', '.join(snapshot.health_issues)}"

        event_blocked, event_name, event_minutes = self._event_calendar.is_event_window()
        if event_blocked and self._risk_state.can_trade:
            self._risk_state.can_trade = False
            self._risk_state.reason = f"Event: {event_name} ({event_minutes:+.0f}m)"

        # Regime affinity gating — block trading in unfavourable regimes
        if self._current_regime and self._risk_state.can_trade:
            regime_label = self._current_regime.regime.value
            blocked = False
            if self._regime_blocked and regime_label in self._regime_blocked:
                blocked = True
            elif self._regime_allowed and regime_label not in self._regime_allowed:
                blocked = True
            elif self._current_regime.regime == Regime.LOW_VOLATILITY:
                blocked = True  # Always block low vol as fallback

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

            if self._entry_decision.should_enter:
                # ── Post-decision filters (data-driven exclusions) ──
                # These override the entry_logic decision when historical
                # evidence shows the specific trade shape is a loser.
                skip_reason = self._apply_entry_filters(
                    self._entry_decision, secs_remaining
                )
                if skip_reason:
                    self._entry_decision = EntryDecision(
                        reason=f"Filtered: {skip_reason}"
                    )
                else:
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

        # ── BTC distance stop ──
        if (
            self._has_position
            and self._executor.pending_trade
            and self._risk_mgr.btc_distance_stop > 0
        ):
            self._check_btc_distance_stop(snapshot, secs_remaining)

        # ── L9: SM Confirmation checkpoint ──
        if (
            self._sm_enabled
            and self._sm_monitor
            and self._has_position
            and self._executor.pending_trade
        ):
            self._check_sm_confirmation(snapshot)

        # ── Signal diagnostic logging (off by default) ──
        # Captures every tick's full signal state for offline analysis.
        # Used to diagnose signal bias issues. No effect on trading logic.
        if self._signal_diag is not None and ob:
            self._log_signal_diagnostic(
                snapshot, ob, secs_remaining, _schedule_override
            )

    def _log_signal_diagnostic(
        self,
        snapshot: DataSnapshot,
        ob,
        secs_remaining: float,
        schedule_override: str,
    ) -> None:
        """Write one row to the signal diagnostic DB. Best-effort, non-fatal."""
        try:
            mkt = snapshot.market
            decision = self._entry_decision
            # market_implied_prob from midpoint
            mkt_mid_yes = (ob.yes_best_bid + ob.yes_best_ask) / 2.0 if ob else 0.5

            # Determine "would_pick_side" mirroring entry_logic.signal_aligned
            if self._est_prob_up >= 0.5:
                would_side = "YES"
            else:
                would_side = "NO"

            tick = SignalTick(
                market_id=mkt.condition_id if mkt else "",
                market_slug=mkt.slug if mkt else "",
                secs_remaining=secs_remaining,
                secs_into_window=300.0 - secs_remaining,
                btc_price=snapshot.binance_price or 0.0,
                oracle_price=snapshot.oracle_price or 0.0,
                oracle_open_price=snapshot.oracle_window_open_price or 0.0,
                coinbase_price=getattr(snapshot, "coinbase_price", 0.0) or 0.0,
                regime=(
                    self._current_regime.regime.value
                    if self._current_regime else ""
                ),
                schedule_override=schedule_override or "",
                # Core L1-L5
                l1_oracle_lag=self._oracle_lag_val,
                l2_momentum=self._momentum_val,
                l3_liquidation=self._liquidation_val,
                l4_orderbook=self._ob_signal_val,
                l5_sentiment=self._sentiment_val,
                # L1 sub-components (exposed by oracle_lag.py)
                l1_lag_component=getattr(self._oracle_lag, "last_lag_component", 0.0),
                l1_open_component=getattr(self._oracle_lag, "last_open_component", 0.0),
                # L4 sub-components (exposed by orderbook_signal.py)
                l4_imbalance=getattr(self._orderbook, "last_imbalance", 0.0),
                l4_flow=getattr(self._orderbook, "last_flow", 0.0),
                l4_mid_dev=getattr(self._orderbook, "last_mid_dev", 0.0),
                l4_top_pressure=getattr(self._orderbook, "last_top_pressure", 0.0),
                l4_thickness=getattr(self._orderbook, "last_thickness", 0.0),
                # Additive signals
                l6_fade=self._fade_signal_val,
                l7_taker_ratio=self._taker_ratio_val,
                l8_clob_flow=self._clob_flow_val,
                l9b_absorption=self._absorption_val,
                l10_exhaustion=self._exhaustion_val,
                l11_trade_size=self._trade_size_val,
                l12_wallet_flow=self._wallet_flow_val,
                # Cross-exchange and combination
                coinbase_direction=self._coinbase_dir,
                combined_signal=self._combined_signal,
                est_prob_up=self._est_prob_up,
                market_implied_prob=mkt_mid_yes,
                prob_edge=decision.prob_edge if decision else 0.0,
                required_edge=decision.required_edge if decision else 0.0,
                # Weights actually used
                w_oracle=self._current_weights[0] if self._current_weights else 0.0,
                w_momentum=self._current_weights[1] if self._current_weights else 0.0,
                w_liquidation=self._current_weights[2] if self._current_weights else 0.0,
                w_orderbook=self._current_weights[3] if self._current_weights else 0.0,
                w_sentiment=self._current_weights[4] if self._current_weights else 0.0,
                # Entry decision
                would_pick_side=would_side,
                would_enter=1 if (decision and decision.should_enter) else 0,
                entry_reason=decision.reason if decision else "",
                # Filter/risk
                filter_blocked=(
                    decision.reason if decision and decision.reason.startswith("Filtered:")
                    else ""
                ),
                risk_can_trade=1 if self._risk_state.can_trade else 0,
                risk_reason=self._risk_state.reason or "",
                trade_placed=1 if (decision and decision.should_enter and
                                    not (decision.reason or "").startswith("Filtered:"))
                              else 0,
            )
            self._signal_diag.log(tick)
        except Exception as e:
            # Best-effort logging. Never let diagnostic logging crash the bot.
            logger.debug("Signal diag logging failed: %s", e)

    def _check_btc_distance_stop(
        self, snapshot: DataSnapshot, secs_remaining: float
    ) -> None:
        """Exit position if BTC has moved too far against us vs window open.

        Backtest validated: dist=$100, min_rem=60s → 91% helped rate,
        +$74 PnL improvement over 2,263 trades. Scales linearly with
        position size — at 10x sizing this becomes +$740.
        """
        trade = self._executor.pending_trade
        if not trade:
            return

        should_stop = self._risk_mgr.should_stop_btc_distance(
            side=trade.side,
            btc_price_now=snapshot.binance_price,
            btc_open_price=snapshot.oracle_window_open_price,
            secs_remaining=secs_remaining,
        )

        if not should_stop:
            return

        # Exit at current best bid for our side
        ob = snapshot.orderbook
        if ob:
            exit_price = (
                ob.yes_best_bid if trade.side == "YES"
                else ob.no_best_bid
            )
        else:
            exit_price = trade.entry_price

        if not exit_price or exit_price <= 0:
            return

        distance = abs(snapshot.binance_price - snapshot.oracle_window_open_price)
        logger.warning(
            "[%s] BTC distance stop: %s position, BTC moved $%.0f from "
            "open (threshold $%.0f), exiting at $%.4f with %.0fs remaining",
            self.name, trade.side, distance,
            self._risk_mgr.btc_distance_stop, exit_price, secs_remaining,
        )

        exit_trade = self._executor.close_position_early(
            exit_price=exit_price,
            reason=f"BTC_DISTANCE_STOP_${distance:.0f}",
        )
        if exit_trade:
            won = exit_trade.pnl is not None and exit_trade.pnl > 0
            self._risk_mgr.record_result(exit_trade.pnl or 0)
            self._last_trade_msg = (
                f"{'W' if won else 'L'} {exit_trade.side} "
                f"BTC DIST STOP ${exit_trade.pnl:+.2f} "
                f"(BTC moved ${distance:.0f})"
            )
            logger.info(f"[{self.name}] {self._last_trade_msg}")

    def _check_sm_confirmation(self, snapshot: DataSnapshot) -> None:
        """L9: Check SM wallet flow and exit if SM disagrees.

        Fires once per configured minute (default: 3, 4). Uses the shared
        SM monitor's flow state and calls check_sm_confirmation() to decide
        HOLD/EXIT/IGNORE. On EXIT, closes the position early.
        """
        secs_remaining = snapshot.seconds_remaining
        minutes_elapsed = (300 - secs_remaining) / 60.0
        current_minute = int(minutes_elapsed)

        if (
            current_minute not in self._sm_check_minutes
            or current_minute == self._sm_last_checked_minute
        ):
            return

        self._sm_last_checked_minute = current_minute
        try:
            sm_flow = self._sm_monitor.get_flow_state()
            trade = self._executor.pending_trade
            trade_side = trade.side
            position_side = (
                PositionSide.YES if trade_side == "YES" else PositionSide.NO
            )

            # Market price = current best bid for our side
            ob = snapshot.orderbook
            if ob:
                market_price = (
                    ob.yes_best_bid if trade_side == "YES"
                    else ob.no_best_bid
                )
            else:
                market_price = trade.entry_price

            sm_decision = check_sm_confirmation(
                position_side=position_side,
                market_price=market_price,
                flow=sm_flow,
                config=self._sm_config,
            )

            self._sm_l9_status = (
                f"L9@min{current_minute}: {sm_decision.value} "
                f"(Y${sm_flow.yes_volume:.0f}/N${sm_flow.no_volume:.0f}, "
                f"{sm_flow.num_wallets}w)"
            )
            logger.info(
                "[%s] L9 SM check at min %d: %s | flow Y=$%.0f N=$%.0f | "
                "%d wallets | price=$%.4f | position=%s",
                self.name, current_minute, sm_decision.value,
                sm_flow.yes_volume, sm_flow.no_volume,
                sm_flow.num_wallets, market_price, trade_side,
            )

            # Log for threshold monitoring
            if self._sm_decision_logger:
                mkt = snapshot.market
                self._sm_decision_logger.log_decision(
                    position_side=position_side,
                    market_price=market_price,
                    flow=sm_flow,
                    decision=sm_decision,
                    config=self._sm_config,
                    market_id=mkt.condition_id if mkt else None,
                    check_minute=current_minute,
                )

            # Execute early exit if SM disagrees AND side is eligible
            if sm_decision == SMDecision.EXIT and trade_side not in self._sm_exit_sides:
                logger.info(
                    "[%s] L9 EXIT suppressed: %s side not in exit_sides %s",
                    self.name, trade_side, self._sm_exit_sides,
                )
                self._sm_l9_status += " (side-filtered)"
                return

            if sm_decision == SMDecision.EXIT:
                exit_trade = self._executor.close_position_early(
                    exit_price=market_price,
                    reason=f"L9_SM_EXIT_min{current_minute}",
                )
                if exit_trade:
                    # Keep _has_position = True to block re-entry
                    won = exit_trade.pnl is not None and exit_trade.pnl > 0
                    self._risk_mgr.record_result(exit_trade.pnl or 0)
                    self._last_trade_msg = (
                        f"{'W' if won else 'L'} {exit_trade.side} "
                        f"L9 EXIT @ min{current_minute} "
                        f"${exit_trade.pnl:+.2f}"
                    )
                    logger.warning(
                        "[%s] L9 SM EXIT executed at min %d: %s P&L=$%.2f",
                        self.name, current_minute, exit_trade.side,
                        exit_trade.pnl or 0,
                    )
        except Exception as e:
            logger.error("[%s] L9 SM check failed: %s", self.name, e, exc_info=True)

    def _apply_entry_filters(self, decision, secs_remaining: float) -> str:
        """Check data-driven exclusion filters. Returns skip reason or ''.

        Filters (disabled by setting config values to 0 or empty list):
          - yes_min_price: Skip YES entries below this price. The dataset
            shows YES bets at low prices (market strongly implying DOWN)
            lose money consistently — likely catching falling knives.
          - skip_regimes: Skip trades in specified regimes. trending_down
            historically had 34% WR, clearly below break-even.
          - high_ev_min_secs_into_window: Skip if estimated EV >= threshold
            AND less than N seconds have elapsed in the window. Early-window
            high-EV signals are noisy (no momentum history, orderbook
            unsettled). Late-window high-EV signals reference real moves.
            Backtest: skipping high-EV at 0-60s into window saves -$213.
          - min_depth_multiplier: Skip if ask-side depth < N× our order
            size. Prevents walking the book on thin markets. At $5 stakes
            this rarely fires; becomes important when sizing up.
        """
        if (self._yes_min_price > 0
                and decision.side == "YES"
                and decision.price < self._yes_min_price):
            self._filter_skipped["yes_low_price"] += 1
            return f"YES @ ${decision.price:.3f} < ${self._yes_min_price:.2f}"

        if self._skip_regimes and self._current_regime:
            regime_label = self._current_regime.regime.value
            if regime_label in self._skip_regimes:
                self._filter_skipped["regime"] += 1
                return f"regime={regime_label}"

        # High-EV early-window filter: block premature high-confidence trades.
        # Late high-EV trades (60-120s remaining) are kept — those reference
        # real moves and are profitable in the backtest.
        if self._high_ev_min_secs_into_window > 0:
            secs_into_window = 300.0 - secs_remaining
            if (decision.best_ev >= self._high_ev_threshold
                    and secs_into_window < self._high_ev_min_secs_into_window):
                self._filter_skipped["high_ev_early"] += 1
                return (
                    f"high_ev_early: ev={decision.best_ev:.3f} "
                    f"at {secs_into_window:.0f}s into window "
                    f"(min={self._high_ev_min_secs_into_window:.0f}s)"
                )

        # Liquidity check: ensure ask-side depth can absorb our order
        if self._min_depth_multiplier > 0 and decision.price > 0:
            ob = self._last_orderbook
            if ob:
                # Use max bet size as conservative estimate (actual size
                # computed later by Kelly, but max is the worst case)
                intended_shares = self._sizer.max_bet_usdc / decision.price
                ask_depth = (
                    ob.yes_ask_depth if decision.side == "YES"
                    else ob.no_ask_depth
                )
                if ask_depth > 0 and ask_depth < intended_shares * self._min_depth_multiplier:
                    self._filter_skipped["low_liquidity"] += 1
                    return (
                        f"Low liquidity: {decision.side} ask depth "
                        f"{ask_depth:.0f} < {self._min_depth_multiplier}x "
                        f"order {intended_shares:.0f} shares"
                    )

        return ""

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
                "L6 Fade": self._fade_signal_val,
                "L7 Taker": self._taker_ratio_val,
                "L8 Flow": self._clob_flow_val,
                **({"L9b Abs": self._absorption_val} if self._absorption else {}),
                **({"L10 Exh": self._exhaustion_val} if self._exhaustion else {}),
                **({"L11 Size": self._trade_size_val} if self._trade_size else {}),
                **({"L12 Wallet": self._wallet_flow_val} if self._wallet_flow else {}),
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
            "filters_active": {
                "yes_min_price": self._yes_min_price,
                "skip_regimes": list(self._skip_regimes),
                "high_ev_threshold": self._high_ev_threshold,
                "high_ev_min_secs_into_window": (
                    self._high_ev_min_secs_into_window
                ),
            },
            "filter_skipped": dict(self._filter_skipped),
            "l9_status": self._sm_l9_status,
            "l9_enabled": self._sm_enabled,
            "risk_details": {
                "daily_trades": self._risk_mgr.daily_trades,
                "daily_trade_cap": self._risk_mgr.daily_trade_cap,
                "peak_bankroll": self._risk_mgr.peak_bankroll,
                "drawdown_pct": self._risk_mgr.drawdown_pct,
                "btc_distance_stop": self._risk_mgr.btc_distance_stop,
                "streak_reduction": self._risk_mgr.streak_reduction_enabled,
            },
        }

    def shutdown(self) -> None:
        """Close database connection."""
        self._db.close()
        if self._signal_diag is not None:
            self._signal_diag.close()
        logger.info(f"[{self.name}] Shut down")
