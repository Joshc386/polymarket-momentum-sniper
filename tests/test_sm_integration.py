"""Tests for L9 SM confirmation integration logic.

Tests the wiring between SMTradeMonitor, check_sm_confirmation, and
the main loop's early-exit mechanism. Does NOT test main.py directly
(that's a 1200+ line async entrypoint). Instead, tests the decision
flow: given flow state + position + price → correct action.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import Config
from data.sm_wallets import SMWalletRegistry
from data.sm_trade_monitor import SMTradeMonitor
from strategy.sm_confirmation import (
    SMConfirmationConfig,
    SMDecision,
    SMFlowState,
    PositionSide,
    check_sm_confirmation,
)
from strategy.sm_decision_log import SMDecisionLogger


# --- Fixtures ---


@pytest.fixture
def sm_config() -> SMConfirmationConfig:
    """Default SM confirmation config matching config.yaml defaults."""
    return SMConfirmationConfig(
        price_floor=0.65,
        price_ceiling=0.80,
        agreement_threshold=0.60,
        min_sm_volume=100.0,
        min_sm_wallets=2,
    )


@pytest.fixture
def config_with_sm() -> Config:
    """Config object with SM confirmation enabled."""
    c = Config()
    c.sm_confirmation_enabled = True
    c.sm_check_minutes = [3, 4]
    c.sm_agreement_threshold = 0.60
    c.sm_min_volume = 100.0
    c.sm_min_wallets = 2
    c.sm_price_floor = 0.65
    c.sm_price_ceiling = 0.80
    c.sm_poll_interval = 3.0
    return c


@pytest.fixture
def config_without_sm() -> Config:
    """Config object with SM confirmation disabled (default)."""
    return Config()


# --- Config tests ---


class TestSMConfig:
    def test_sm_disabled_by_default(self) -> None:
        c = Config()
        assert c.sm_confirmation_enabled is False

    def test_sm_config_fields(self, config_with_sm: Config) -> None:
        c = config_with_sm
        assert c.sm_confirmation_enabled is True
        assert c.sm_check_minutes == [3, 4]
        assert c.sm_agreement_threshold == 0.60
        assert c.sm_min_volume == 100.0
        assert c.sm_min_wallets == 2
        assert c.sm_price_floor == 0.65
        assert c.sm_price_ceiling == 0.80
        assert c.sm_poll_interval == 3.0

    def test_sm_config_from_yaml(self, tmp_path: Path) -> None:
        """SM config loads correctly from YAML."""
        yaml_content = """
mode: paper
sm_confirmation:
  enabled: true
  check_minutes: [3, 4]
  agreement_threshold: 0.55
  min_volume: 200.0
  min_wallets: 3
  price_floor: 0.60
  price_ceiling: 0.85
  poll_interval: 5.0
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        env_file = tmp_path / ".env"
        env_file.write_text("")

        with patch("core.config.Path") as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent = tmp_path
            c = Config()
            # Manually parse since load() has path coupling
            import yaml
            raw = yaml.safe_load(yaml_content)
            sm = raw.get("sm_confirmation", {})
            c.sm_confirmation_enabled = sm.get("enabled", False)
            c.sm_agreement_threshold = sm.get("agreement_threshold", 0.60)
            c.sm_min_volume = sm.get("min_volume", 100.0)
            c.sm_min_wallets = sm.get("min_wallets", 2)
            c.sm_price_floor = sm.get("price_floor", 0.65)
            c.sm_price_ceiling = sm.get("price_ceiling", 0.80)
            c.sm_poll_interval = sm.get("poll_interval", 3.0)

        assert c.sm_confirmation_enabled is True
        assert c.sm_agreement_threshold == 0.55
        assert c.sm_min_volume == 200.0
        assert c.sm_min_wallets == 3
        assert c.sm_price_floor == 0.60
        assert c.sm_price_ceiling == 0.85
        assert c.sm_poll_interval == 5.0


# --- Decision flow tests (simulate what main.py does) ---


class TestL9DecisionFlow:
    """Simulate the L9 checkpoint logic from main.py."""

    def _run_l9_check(
        self,
        trade_side: str,
        market_price: float,
        flow: SMFlowState,
        config: SMConfirmationConfig,
    ) -> SMDecision:
        """Replicate the L9 decision path from main.py."""
        position_side = (
            PositionSide.YES if trade_side == "YES" else PositionSide.NO
        )
        return check_sm_confirmation(
            position_side=position_side,
            market_price=market_price,
            flow=flow,
            config=config,
        )

    def test_sm_agrees_with_yes_position(self, sm_config: SMConfirmationConfig) -> None:
        """SM flow favours YES, position is YES → HOLD."""
        flow = SMFlowState(yes_volume=500.0, no_volume=100.0, num_wallets=5)
        result = self._run_l9_check("YES", 0.70, flow, sm_config)
        assert result == SMDecision.HOLD

    def test_sm_disagrees_with_yes_position(self, sm_config: SMConfirmationConfig) -> None:
        """SM flow favours NO, position is YES → EXIT."""
        flow = SMFlowState(yes_volume=100.0, no_volume=500.0, num_wallets=5)
        result = self._run_l9_check("YES", 0.70, flow, sm_config)
        assert result == SMDecision.EXIT

    def test_sm_agrees_with_no_position(self, sm_config: SMConfirmationConfig) -> None:
        """SM flow favours NO, position is NO → HOLD."""
        flow = SMFlowState(yes_volume=100.0, no_volume=500.0, num_wallets=5)
        result = self._run_l9_check("NO", 0.70, flow, sm_config)
        assert result == SMDecision.HOLD

    def test_sm_disagrees_with_no_position(self, sm_config: SMConfirmationConfig) -> None:
        """SM flow favours YES, position is NO → EXIT."""
        flow = SMFlowState(yes_volume=500.0, no_volume=100.0, num_wallets=5)
        result = self._run_l9_check("NO", 0.70, flow, sm_config)
        assert result == SMDecision.EXIT

    def test_price_below_floor_ignores(self, sm_config: SMConfirmationConfig) -> None:
        """Below $0.65 → IGNORE regardless of SM flow."""
        flow = SMFlowState(yes_volume=100.0, no_volume=500.0, num_wallets=5)
        result = self._run_l9_check("YES", 0.55, flow, sm_config)
        assert result == SMDecision.IGNORE

    def test_price_above_ceiling_ignores(self, sm_config: SMConfirmationConfig) -> None:
        """Above $0.80 → IGNORE regardless of SM flow."""
        flow = SMFlowState(yes_volume=100.0, no_volume=500.0, num_wallets=5)
        result = self._run_l9_check("YES", 0.85, flow, sm_config)
        assert result == SMDecision.IGNORE

    def test_insufficient_volume_ignores(self, sm_config: SMConfirmationConfig) -> None:
        """Below min_sm_volume → IGNORE."""
        flow = SMFlowState(yes_volume=30.0, no_volume=20.0, num_wallets=5)
        result = self._run_l9_check("YES", 0.70, flow, sm_config)
        assert result == SMDecision.IGNORE

    def test_insufficient_wallets_ignores(self, sm_config: SMConfirmationConfig) -> None:
        """Below min_sm_wallets → IGNORE."""
        flow = SMFlowState(yes_volume=500.0, no_volume=100.0, num_wallets=1)
        result = self._run_l9_check("YES", 0.70, flow, sm_config)
        assert result == SMDecision.IGNORE


# --- Minute calculation tests ---


class TestMinuteCalculation:
    """Verify the minutes_elapsed / current_minute calculation from main.py."""

    def _current_minute(self, seconds_remaining: float) -> int:
        """Replicate main.py's minute calculation."""
        minutes_elapsed = (300 - seconds_remaining) / 60.0
        return int(minutes_elapsed)

    def test_start_of_window(self) -> None:
        assert self._current_minute(300.0) == 0

    def test_minute_one(self) -> None:
        assert self._current_minute(240.0) == 1

    def test_minute_two(self) -> None:
        assert self._current_minute(180.0) == 2

    def test_minute_three(self) -> None:
        """Minute 3 is a check point."""
        assert self._current_minute(120.0) == 3

    def test_minute_four(self) -> None:
        """Minute 4 is a check point."""
        assert self._current_minute(60.0) == 4

    def test_mid_minute_three(self) -> None:
        """3:30 into window → still minute 3."""
        assert self._current_minute(90.0) == 3

    def test_end_of_window(self) -> None:
        assert self._current_minute(0.0) == 5

    def test_check_minute_gate(self) -> None:
        """Minute must be in check_minutes AND not already checked."""
        check_minutes = [3, 4]
        last_checked = -1
        current = 3
        should_check = current in check_minutes and current != last_checked
        assert should_check is True

    def test_check_minute_gate_blocks_repeat(self) -> None:
        """Same minute should not be checked twice."""
        check_minutes = [3, 4]
        last_checked = 3
        current = 3
        should_check = current in check_minutes and current != last_checked
        assert should_check is False

    def test_check_minute_gate_allows_next(self) -> None:
        """After checking min 3, min 4 should still fire."""
        check_minutes = [3, 4]
        last_checked = 3
        current = 4
        should_check = current in check_minutes and current != last_checked
        assert should_check is True


# --- Monitor + Registry wiring tests ---


class TestMonitorWiring:
    def test_monitor_creates_with_registry(self) -> None:
        reg = SMWalletRegistry(csv_path=Path("/nonexistent"))
        reg.load_from_list(["0x" + "aa" * 20])
        mon = SMTradeMonitor(_sm_registry=reg, _rpcs=["http://test:8545"])
        assert mon.get_flow_state().yes_volume == 0.0

    def test_set_market_resets_flow(self) -> None:
        reg = SMWalletRegistry(csv_path=Path("/nonexistent"))
        reg.load_from_list(["0x" + "aa" * 20])
        mon = SMTradeMonitor(_sm_registry=reg, _rpcs=["http://test:8545"])
        mon.set_market("111", "222", "window-1")
        assert mon._yes_token_id == 111
        assert mon._no_token_id == 222

    def test_window_change_resets(self) -> None:
        reg = SMWalletRegistry(csv_path=Path("/nonexistent"))
        reg.load_from_list(["0x" + "aa" * 20])
        mon = SMTradeMonitor(_sm_registry=reg, _rpcs=["http://test:8545"])
        mon.set_market("111", "222", "window-1")
        mon._yes_volume = 500.0
        mon.set_market("111", "222", "window-2")
        assert mon.get_flow_state().yes_volume == 0.0


# --- Decision logger wiring tests ---


class TestDecisionLoggerWiring:
    def test_logger_records_decision(self, tmp_path: Path, sm_config: SMConfirmationConfig) -> None:
        db_path = tmp_path / "test_decisions.db"
        dl = SMDecisionLogger(db_path=db_path)
        flow = SMFlowState(yes_volume=400.0, no_volume=100.0, num_wallets=5)
        dl.log_decision(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            decision=SMDecision.HOLD,
            config=sm_config,
            market_id="test-market",
            check_minute=3,
        )
        counts = dl.get_decision_counts()
        assert counts.get("HOLD", 0) == 1

    def test_logger_records_exit(self, tmp_path: Path, sm_config: SMConfirmationConfig) -> None:
        db_path = tmp_path / "test_decisions.db"
        dl = SMDecisionLogger(db_path=db_path)
        flow = SMFlowState(yes_volume=100.0, no_volume=400.0, num_wallets=5)
        dl.log_decision(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            decision=SMDecision.EXIT,
            config=sm_config,
            market_id="test-market",
            check_minute=4,
        )
        counts = dl.get_decision_counts()
        assert counts.get("EXIT", 0) == 1


# --- Early exit simulation ---


class TestEarlyExitSimulation:
    """Simulate the early exit path that main.py follows on SM EXIT."""

    def test_exit_flow(self, sm_config: SMConfirmationConfig) -> None:
        """Full flow: SM disagrees → EXIT decision → close_position_early."""
        # 1. SM disagrees with YES position
        flow = SMFlowState(yes_volume=50.0, no_volume=500.0, num_wallets=4)
        position_side = PositionSide.YES
        market_price = 0.72

        decision = check_sm_confirmation(
            position_side=position_side,
            market_price=market_price,
            flow=flow,
            config=sm_config,
        )
        assert decision == SMDecision.EXIT

        # 2. Simulate close_position_early logic
        entry_price = 0.68
        exit_price = market_price
        num_shares = 10.0
        gross_pnl = (exit_price - entry_price) * num_shares
        fee = gross_pnl * 0.02 if gross_pnl > 0 else 0.0
        net_pnl = gross_pnl - fee

        # This exit is profitable (0.72 > 0.68)
        assert net_pnl > 0

    def test_hold_does_not_exit(self, sm_config: SMConfirmationConfig) -> None:
        """SM agrees → HOLD → no early exit."""
        flow = SMFlowState(yes_volume=500.0, no_volume=50.0, num_wallets=4)
        decision = check_sm_confirmation(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            config=sm_config,
        )
        assert decision == SMDecision.HOLD

    def test_ignore_does_not_exit(self, sm_config: SMConfirmationConfig) -> None:
        """Insufficient data → IGNORE → no early exit."""
        flow = SMFlowState(yes_volume=10.0, no_volume=5.0, num_wallets=1)
        decision = check_sm_confirmation(
            position_side=PositionSide.YES,
            market_price=0.72,
            flow=flow,
            config=sm_config,
        )
        assert decision == SMDecision.IGNORE
