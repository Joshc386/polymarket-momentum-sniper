"""Tests for RiskManager — covers both legacy behavior (Bot G backward
compat) and new features (Bot K: drawdown breaker, daily trade cap,
streak reduction toggle, BTC distance stop, regime sizing).
"""

import pytest
from strategy.risk_manager import RiskManager, RiskState


# ── Helpers ──

def make_rm(**overrides) -> RiskManager:
    """Create a RiskManager with sensible test defaults."""
    defaults = dict(
        daily_loss_cap_pct=0.20,
        daily_loss_warn_pct=0.15,
        min_bankroll=10.0,
        streak_reduce_at=3,
        streak_reduce_factor=0.75,
        streak_heavy_reduce_at=5,
        streak_heavy_reduce_factor=0.5,
        streak_pause_at=7,
        streak_pause_minutes=30,
        streak_reset_wins=3,
        time_func=lambda: 1000.0,
        date_func=lambda: "2026-05-12",
    )
    defaults.update(overrides)
    return RiskManager(**defaults)


# ═══════════════════════════════════════════════════════════════
# 1. BACKWARD COMPATIBILITY — Bot G behavior unchanged
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Verify that the default RiskManager (no new params) behaves
    exactly like the old version — Bot G must not change."""

    def test_default_streak_reduction_enabled(self):
        rm = make_rm()
        assert rm.streak_reduction_enabled is True

    def test_streak_reduction_at_3_losses(self):
        rm = make_rm()
        rm.set_daily_start(100.0)
        for _ in range(3):
            rm.record_result(-1.0)
        state = rm.evaluate(97.0)
        assert state.can_trade is True
        assert state.size_multiplier == 0.75

    def test_streak_reduction_at_5_losses(self):
        rm = make_rm()
        rm.set_daily_start(100.0)
        for _ in range(5):
            rm.record_result(-1.0)
        state = rm.evaluate(95.0)
        assert state.can_trade is True
        assert state.size_multiplier == 0.5

    def test_no_drawdown_breaker_by_default(self):
        rm = make_rm()
        rm.peak_bankroll = 100.0
        state = rm.evaluate(60.0)  # 40% drawdown
        assert state.can_trade is True  # no drawdown breaker configured

    def test_no_daily_trade_cap_by_default(self):
        rm = make_rm()
        rm.set_daily_start(100.0)
        rm.daily_trades = 500
        state = rm.evaluate(100.0)
        assert state.can_trade is True  # no cap configured

    def test_no_btc_distance_stop_by_default(self):
        rm = make_rm()
        result = rm.should_stop_btc_distance("YES", 99000.0, 100000.0, 120.0)
        assert result is False

    def test_no_regime_sizing_by_default(self):
        rm = make_rm()
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="ranging")
        assert state.size_multiplier == 1.0


# ═══════════════════════════════════════════════════════════════
# 2. STREAK REDUCTION TOGGLE
# ═══════════════════════════════════════════════════════════════

class TestStreakReductionToggle:
    """When streak_reduction_enabled=False, size stays at 1.0
    regardless of consecutive losses (streak pause still works)."""

    def test_no_size_reduction_at_3_losses(self):
        rm = make_rm(streak_reduction_enabled=False)
        rm.set_daily_start(100.0)
        for _ in range(3):
            rm.record_result(-1.0)
        state = rm.evaluate(97.0)
        assert state.can_trade is True
        assert state.size_multiplier == 1.0

    def test_no_size_reduction_at_5_losses(self):
        rm = make_rm(streak_reduction_enabled=False)
        rm.set_daily_start(100.0)
        for _ in range(5):
            rm.record_result(-1.0)
        state = rm.evaluate(95.0)
        assert state.can_trade is True
        assert state.size_multiplier == 1.0

    def test_streak_pause_still_works(self):
        """Even with reduction disabled, 7 consecutive losses
        should still trigger the pause circuit breaker."""
        rm = make_rm(streak_reduction_enabled=False)
        rm.set_daily_start(100.0)
        for _ in range(7):
            rm.record_result(-1.0)
        state = rm.evaluate(93.0)
        assert state.can_trade is False
        assert "Streak pause" in state.reason


# ═══════════════════════════════════════════════════════════════
# 3. DRAWDOWN BREAKER
# ═══════════════════════════════════════════════════════════════

class TestDrawdownBreaker:
    def test_triggers_at_30pct(self):
        rm = make_rm(drawdown_pct=0.30)
        rm.peak_bankroll = 100.0
        rm.set_daily_start(100.0)
        state = rm.evaluate(69.0)  # 31% drawdown
        assert state.can_trade is False
        assert "Drawdown breaker" in state.reason

    def test_does_not_trigger_below_threshold(self):
        rm = make_rm(drawdown_pct=0.30)
        rm.peak_bankroll = 100.0
        rm.set_daily_start(100.0)
        state = rm.evaluate(75.0)  # 25% drawdown
        assert state.can_trade is True

    def test_peak_updates_on_new_high(self):
        rm = make_rm(drawdown_pct=0.30)
        rm.set_daily_start(100.0)
        rm.evaluate(100.0)
        assert rm.peak_bankroll == 100.0
        rm.evaluate(110.0)
        assert rm.peak_bankroll == 110.0

    def test_recovery_releases_breaker(self):
        rm = make_rm(drawdown_pct=0.30)
        rm.peak_bankroll = 100.0
        rm.set_daily_start(100.0)

        # Trigger drawdown
        state = rm.evaluate(65.0)
        assert state.can_trade is False
        assert rm._drawdown_paused is True

        # Recover
        state = rm.evaluate(75.0)
        assert state.can_trade is True
        assert rm._drawdown_paused is False


# ═══════════════════════════════════════════════════════════════
# 4. DAILY TRADE CAP
# ═══════════════════════════════════════════════════════════════

class TestDailyTradeCap:
    def test_blocks_at_cap(self):
        rm = make_rm(daily_trade_cap=150)
        rm.set_daily_start(100.0)
        rm.daily_trades = 150
        state = rm.evaluate(100.0)
        assert state.can_trade is False
        assert "Daily trade cap" in state.reason

    def test_allows_below_cap(self):
        rm = make_rm(daily_trade_cap=150)
        rm.set_daily_start(100.0)
        rm.daily_trades = 149
        state = rm.evaluate(100.0)
        assert state.can_trade is True

    def test_trade_count_increments(self):
        rm = make_rm(daily_trade_cap=150)
        rm.set_daily_start(100.0)
        assert rm.daily_trades == 0
        rm.record_result(1.0)
        assert rm.daily_trades == 1
        rm.record_result(-0.5)
        assert rm.daily_trades == 2

    def test_resets_on_new_day(self):
        day = "2026-05-12"
        rm = make_rm(daily_trade_cap=150, date_func=lambda: day)
        rm.set_daily_start(100.0)
        rm.daily_trades = 140

        # Simulate day change
        day = "2026-05-13"
        rm._today = lambda: day
        rm.set_daily_start(100.0)
        assert rm.daily_trades == 0


# ═══════════════════════════════════════════════════════════════
# 5. BTC DISTANCE STOP
# ═══════════════════════════════════════════════════════════════

class TestBTCDistanceStop:
    def test_yes_stops_on_btc_drop(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # YES position, BTC dropped $110 from open
        result = rm.should_stop_btc_distance("YES", 99890.0, 100000.0, 120.0)
        assert result is True

    def test_yes_no_stop_small_drop(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # BTC only dropped $50
        result = rm.should_stop_btc_distance("YES", 99950.0, 100000.0, 120.0)
        assert result is False

    def test_no_stops_on_btc_rise(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # NO position, BTC rose $120 from open
        result = rm.should_stop_btc_distance("NO", 100120.0, 100000.0, 120.0)
        assert result is True

    def test_no_no_stop_small_rise(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        result = rm.should_stop_btc_distance("NO", 100050.0, 100000.0, 120.0)
        assert result is False

    def test_respects_min_secs_remaining(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # BTC moved $150 against us but only 30s remaining — don't stop
        result = rm.should_stop_btc_distance("YES", 99850.0, 100000.0, 30.0)
        assert result is False

    def test_disabled_when_zero(self):
        rm = make_rm(btc_distance_stop=0.0)
        result = rm.should_stop_btc_distance("YES", 99000.0, 100000.0, 120.0)
        assert result is False

    def test_handles_zero_prices(self):
        rm = make_rm(btc_distance_stop=100.0)
        assert rm.should_stop_btc_distance("YES", 0.0, 100000.0, 120.0) is False
        assert rm.should_stop_btc_distance("YES", 99000.0, 0.0, 120.0) is False

    def test_exact_boundary_no_trigger(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # Exactly $100 drop — should NOT trigger (need to exceed, not equal)
        result = rm.should_stop_btc_distance("YES", 99900.0, 100000.0, 120.0)
        assert result is False

    def test_just_past_boundary_triggers(self):
        rm = make_rm(btc_distance_stop=100.0, btc_distance_min_secs_remaining=60.0)
        # $100.01 drop — should trigger
        result = rm.should_stop_btc_distance("YES", 99899.99, 100000.0, 120.0)
        assert result is True


# ═══════════════════════════════════════════════════════════════
# 6. REGIME SIZING
# ═══════════════════════════════════════════════════════════════

class TestRegimeSizing:
    def test_ranging_1_5x(self):
        rm = make_rm(regime_size_overrides={"ranging": 1.5, "trending_up": 0.75})
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="ranging")
        assert state.size_multiplier == 1.5

    def test_trending_up_0_75x(self):
        rm = make_rm(regime_size_overrides={"ranging": 1.5, "trending_up": 0.75})
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="trending_up")
        assert state.size_multiplier == 0.75

    def test_unknown_regime_1x(self):
        rm = make_rm(regime_size_overrides={"ranging": 1.5, "trending_up": 0.75})
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="trending_down")
        assert state.size_multiplier == 1.0

    def test_empty_regime_1x(self):
        rm = make_rm(regime_size_overrides={"ranging": 1.5})
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="")
        assert state.size_multiplier == 1.0

    def test_reason_includes_regime(self):
        rm = make_rm(regime_size_overrides={"ranging": 1.5})
        rm.set_daily_start(100.0)
        state = rm.evaluate(100.0, current_regime="ranging")
        assert "regime:ranging" in state.reason


# ═══════════════════════════════════════════════════════════════
# 7. COMBINED SCENARIOS
# ═══════════════════════════════════════════════════════════════

class TestCombinedScenarios:
    def test_bot_k_full_config(self):
        """Simulate Bot K's exact config — all new features together."""
        rm = make_rm(
            streak_reduction_enabled=False,
            drawdown_pct=0.30,
            daily_trade_cap=150,
            btc_distance_stop=100.0,
            btc_distance_min_secs_remaining=60.0,
            regime_size_overrides={"ranging": 1.5, "trending_up": 0.75},
        )
        rm.set_daily_start(100.0)

        # 4 consecutive losses — no size reduction (disabled)
        for _ in range(4):
            rm.record_result(-1.0)
        state = rm.evaluate(96.0, current_regime="ranging")
        assert state.can_trade is True
        assert state.size_multiplier == 1.5  # ranging boost only

    def test_regime_sizing_with_daily_warn(self):
        """Regime boost + daily loss warning should compound."""
        rm = make_rm(
            streak_reduction_enabled=False,
            regime_size_overrides={"ranging": 1.5},
        )
        rm.set_daily_start(100.0)
        rm.daily_pnl = -16.0  # past 15% warning

        state = rm.evaluate(84.0, current_regime="ranging")
        # 1.5 (regime) * 0.5 (daily warn) = 0.75
        assert state.can_trade is True
        assert abs(state.size_multiplier - 0.75) < 0.01

    def test_multiplier_floor_at_0_25(self):
        """Even with multiple reductions, floor is 0.25x."""
        rm = make_rm(
            streak_reduction_enabled=True,
            regime_size_overrides={"trending_up": 0.5},
            # Raise loss cap so it doesn't trigger before we test the floor
            daily_loss_cap_pct=0.50,
        )
        rm.set_daily_start(100.0)
        rm.daily_pnl = -16.0  # past 15% warning (0.5x multiplier)

        # 5 losses (0.5x) * daily warn (0.5x) * regime (0.5x) = 0.125
        # but floor is 0.25
        for _ in range(5):
            rm.record_result(-1.0)

        state = rm.evaluate(79.0, current_regime="trending_up")
        assert state.size_multiplier == 0.25

    def test_daily_trade_count_in_state(self):
        rm = make_rm(daily_trade_cap=150)
        rm.set_daily_start(100.0)
        rm.record_result(1.0)
        rm.record_result(-0.5)
        state = rm.evaluate(100.5)
        assert state.daily_trades == 2
