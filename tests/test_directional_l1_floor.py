"""Tests for the directional L1 floor entry filter (2026-05-29).

Background: L1 (oracle_lag_signal) = displacement from the window-open
resolution line (>0 price above the line, <0 below). Live + backtest
analysis showed the bot leaks money when it bets AGAINST the line:
buying YES (UP) while price is below the line, or NO (DOWN) while above.
See BTC_5min_Observations "L1 directional leak investigation (2026-05-29)".

The floor: optionally block entries that bet against the line beyond a
deadband. YES is against the line when L1 < -deadband; NO is against when
L1 > +deadband. Within [-deadband, +deadband] (the noise band near the
line) both sides are allowed. Default off → Bot G and Bot K unchanged.
"""

from strategies.contrarian_ev import (
    ContrarianEvStrategy,
    l1_directional_floor_blocks,
)
from strategy.entry_logic import EntryDecision


class TestYesSide:
    """YES bets UP — against the line when price is below it (L1 < 0)."""

    def test_yes_blocked_when_l1_below_negative_deadband(self) -> None:
        # price clearly below the line, deadband 0.1 → block the UP bet
        assert l1_directional_floor_blocks("YES", l1=-0.3, deadband=0.1) is True


class TestNoSide:
    """NO bets DOWN — against the line when price is above it (L1 > 0)."""

    def test_no_blocked_when_l1_above_positive_deadband(self) -> None:
        # price clearly above the line, deadband 0.1 → block the DOWN bet
        assert l1_directional_floor_blocks("NO", l1=0.3, deadband=0.1) is True


class TestWithTheLineAllowed:
    """Betting WITH the line is never blocked — that's the profitable side."""

    def test_yes_allowed_when_price_above_line(self) -> None:
        assert l1_directional_floor_blocks("YES", l1=0.5, deadband=0.1) is False

    def test_no_allowed_when_price_below_line(self) -> None:
        assert l1_directional_floor_blocks("NO", l1=-0.5, deadband=0.1) is False


class TestDeadbandNoiseBand:
    """Within [-deadband, +deadband] neither side is blocked (feed noise)."""

    def test_yes_allowed_inside_band(self) -> None:
        assert l1_directional_floor_blocks("YES", l1=-0.05, deadband=0.1) is False

    def test_no_allowed_inside_band(self) -> None:
        assert l1_directional_floor_blocks("NO", l1=0.05, deadband=0.1) is False

    def test_boundary_exactly_at_deadband_allowed(self) -> None:
        # Exactly at the band edge is allowed (strict < / >, inclusive band).
        assert l1_directional_floor_blocks("YES", l1=-0.1, deadband=0.1) is False
        assert l1_directional_floor_blocks("NO", l1=0.1, deadband=0.1) is False

    def test_zero_deadband_is_strict_sign(self) -> None:
        # deadband=0 → any bet against the line is blocked, on-line allowed.
        assert l1_directional_floor_blocks("YES", l1=-0.001, deadband=0.0) is True
        assert l1_directional_floor_blocks("NO", l1=0.001, deadband=0.0) is True
        assert l1_directional_floor_blocks("YES", l1=0.0, deadband=0.0) is False
        assert l1_directional_floor_blocks("NO", l1=0.0, deadband=0.0) is False


class TestUnknownSide:
    def test_empty_side_never_blocks(self) -> None:
        assert l1_directional_floor_blocks("", l1=-0.9, deadband=0.1) is False


class TestConfigWiring:
    """Default off (Bot G/K unchanged); opt-in via filters config."""

    def test_floor_disabled_by_default(self) -> None:
        s = ContrarianEvStrategy("t", {})
        assert s._directional_l1_floor is False

    def test_default_deadband(self) -> None:
        s = ContrarianEvStrategy("t", {})
        assert s._directional_l1_floor_deadband == 0.1

    def test_config_enables_floor_and_deadband(self) -> None:
        cfg = {"filters": {"directional_l1_floor": True,
                           "directional_l1_floor_deadband": 0.05}}
        s = ContrarianEvStrategy("t", cfg)
        assert s._directional_l1_floor is True
        assert s._directional_l1_floor_deadband == 0.05


class TestApplyEntryFiltersIntegration:
    """The floor blocks against-the-line entries only when enabled."""

    def _decision(self, side: str) -> EntryDecision:
        return EntryDecision(should_enter=True, side=side, price=0.46, best_ev=0.01)

    def test_disabled_floor_allows_against_line_yes(self) -> None:
        # Safety: with the floor off (Bot K baseline), an against-line YES
        # is NOT blocked by this filter.
        s = ContrarianEvStrategy("t", {})
        s._oracle_lag_val = -0.3
        assert s._apply_entry_filters(self._decision("YES"), secs_remaining=200.0) == ""

    def test_enabled_floor_blocks_against_line_yes(self) -> None:
        cfg = {"filters": {"directional_l1_floor": True,
                           "directional_l1_floor_deadband": 0.1}}
        s = ContrarianEvStrategy("t", cfg)
        s._oracle_lag_val = -0.3
        assert "l1_against_line" in s._apply_entry_filters(
            self._decision("YES"), secs_remaining=200.0)

    def test_enabled_floor_blocks_against_line_no(self) -> None:
        cfg = {"filters": {"directional_l1_floor": True,
                           "directional_l1_floor_deadband": 0.1}}
        s = ContrarianEvStrategy("t", cfg)
        s._oracle_lag_val = 0.3
        assert "l1_against_line" in s._apply_entry_filters(
            self._decision("NO"), secs_remaining=200.0)

    def test_enabled_floor_allows_with_line_yes(self) -> None:
        cfg = {"filters": {"directional_l1_floor": True,
                           "directional_l1_floor_deadband": 0.1}}
        s = ContrarianEvStrategy("t", cfg)
        s._oracle_lag_val = 0.5
        assert s._apply_entry_filters(self._decision("YES"), secs_remaining=200.0) == ""
