"""Tests for SM Confirmation Logic (Phase 3 of L9 signal layer)."""

import pytest

from strategy.sm_confirmation import (
    PositionSide,
    SMConfirmationConfig,
    SMDecision,
    SMFlowState,
    check_sm_confirmation,
)


@pytest.fixture
def default_config() -> SMConfirmationConfig:
    return SMConfirmationConfig()


class TestSMFlowState:
    def test_total_volume(self) -> None:
        flow = SMFlowState(yes_volume=300.0, no_volume=200.0, num_wallets=5)
        assert flow.total_volume == 500.0

    def test_yes_fraction(self) -> None:
        flow = SMFlowState(yes_volume=300.0, no_volume=200.0, num_wallets=5)
        assert flow.yes_fraction == pytest.approx(0.6)

    def test_no_fraction(self) -> None:
        flow = SMFlowState(yes_volume=300.0, no_volume=200.0, num_wallets=5)
        assert flow.no_fraction == pytest.approx(0.4)

    def test_zero_volume_fractions(self) -> None:
        flow = SMFlowState(yes_volume=0.0, no_volume=0.0, num_wallets=0)
        assert flow.yes_fraction == 0.0
        assert flow.no_fraction == 0.0
        assert flow.total_volume == 0.0

    def test_all_yes_volume(self) -> None:
        flow = SMFlowState(yes_volume=1000.0, no_volume=0.0, num_wallets=3)
        assert flow.yes_fraction == 1.0
        assert flow.no_fraction == 0.0

    def test_frozen(self) -> None:
        flow = SMFlowState(yes_volume=100.0, no_volume=100.0, num_wallets=2)
        with pytest.raises(AttributeError):
            flow.yes_volume = 999.0  # type: ignore[misc]


class TestPriceZoneFiltering:
    """SM signal should be IGNORED outside the $0.65-$0.80 zone."""

    def test_below_price_floor_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.50, flow, default_config,
        )
        assert result == SMDecision.IGNORE

    def test_at_price_floor_not_ignored(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.65, flow, default_config,
        )
        assert result != SMDecision.IGNORE

    def test_above_price_ceiling_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.85, flow, default_config,
        )
        assert result == SMDecision.IGNORE

    def test_at_price_ceiling_not_ignored(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.80, flow, default_config,
        )
        assert result != SMDecision.IGNORE

    def test_mid_zone_not_ignored(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result != SMDecision.IGNORE


class TestVolumeAndWalletMinimums:
    """IGNORE when SM data is insufficient."""

    def test_below_min_volume_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=40.0, no_volume=10.0, num_wallets=3)
        result = check_sm_confirmation(
            PositionSide.YES, 0.70, flow, default_config,
        )
        assert result == SMDecision.IGNORE

    def test_at_min_volume_not_ignored(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=80.0, no_volume=20.0, num_wallets=3)
        result = check_sm_confirmation(
            PositionSide.YES, 0.70, flow, default_config,
        )
        assert result != SMDecision.IGNORE

    def test_below_min_wallets_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=500.0, no_volume=100.0, num_wallets=1)
        result = check_sm_confirmation(
            PositionSide.YES, 0.70, flow, default_config,
        )
        assert result == SMDecision.IGNORE

    def test_at_min_wallets_not_ignored(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=500.0, no_volume=100.0, num_wallets=2)
        result = check_sm_confirmation(
            PositionSide.YES, 0.70, flow, default_config,
        )
        assert result != SMDecision.IGNORE

    def test_zero_volume_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=0.0, no_volume=0.0, num_wallets=0)
        result = check_sm_confirmation(
            PositionSide.YES, 0.70, flow, default_config,
        )
        assert result == SMDecision.IGNORE


class TestAgreementDisagreement:
    """HOLD when SM agrees, EXIT when SM disagrees."""

    def test_yes_position_sm_agrees_hold(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=700.0, no_volume=300.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result == SMDecision.HOLD

    def test_no_position_sm_agrees_hold(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=300.0, no_volume=700.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.NO, 0.72, flow, default_config,
        )
        assert result == SMDecision.HOLD

    def test_yes_position_sm_disagrees_exit(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=200.0, no_volume=800.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result == SMDecision.EXIT

    def test_no_position_sm_disagrees_exit(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.NO, 0.72, flow, default_config,
        )
        assert result == SMDecision.EXIT

    def test_split_flow_ignores(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=550.0, no_volume=450.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result == SMDecision.IGNORE

    def test_exact_threshold_is_agreement(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=600.0, no_volume=400.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result == SMDecision.HOLD

    def test_exact_threshold_opposite_is_exit(self, default_config: SMConfirmationConfig) -> None:
        flow = SMFlowState(yes_volume=400.0, no_volume=600.0, num_wallets=5)
        result = check_sm_confirmation(
            PositionSide.YES, 0.72, flow, default_config,
        )
        assert result == SMDecision.EXIT


class TestCustomConfig:
    def test_wider_price_zone(self) -> None:
        config = SMConfirmationConfig(price_floor=0.50, price_ceiling=0.90)
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)

        result = check_sm_confirmation(PositionSide.YES, 0.55, flow, config)
        assert result == SMDecision.HOLD

    def test_stricter_agreement_threshold(self) -> None:
        config = SMConfirmationConfig(agreement_threshold=0.75)
        flow = SMFlowState(yes_volume=700.0, no_volume=300.0, num_wallets=5)

        result = check_sm_confirmation(PositionSide.YES, 0.72, flow, config)
        assert result == SMDecision.IGNORE

    def test_higher_min_volume(self) -> None:
        config = SMConfirmationConfig(min_sm_volume=1000.0)
        flow = SMFlowState(yes_volume=400.0, no_volume=100.0, num_wallets=5)

        result = check_sm_confirmation(PositionSide.YES, 0.72, flow, config)
        assert result == SMDecision.IGNORE

    def test_higher_min_wallets(self) -> None:
        config = SMConfirmationConfig(min_sm_wallets=5)
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=3)

        result = check_sm_confirmation(PositionSide.YES, 0.72, flow, config)
        assert result == SMDecision.IGNORE


class TestDefaultConfig:
    def test_default_none_config(self) -> None:
        flow = SMFlowState(yes_volume=800.0, no_volume=200.0, num_wallets=5)
        result = check_sm_confirmation(PositionSide.YES, 0.72, flow)
        assert result == SMDecision.HOLD

    def test_config_is_frozen(self) -> None:
        config = SMConfirmationConfig()
        with pytest.raises(AttributeError):
            config.price_floor = 0.50  # type: ignore[misc]
