"""Tests for the entry_side_filter entry filter (ADR-0005 / J38).

The K2 live probe trades both sides by default, but needs a config lever to
restrict to one side — used by the adaptive guard (flip to NO-only if live YES
bleeds). `entry_side_filter: both | NO | YES` (default both → every other bot
unchanged). When set, the opposite side's entries are blocked.
"""

import pytest

from strategies.contrarian_ev import ContrarianEvStrategy
from strategy.entry_logic import EntryDecision


def _decision(side: str) -> EntryDecision:
    return EntryDecision(should_enter=True, side=side, price=0.46, best_ev=0.01)


def _strategy(side_filter: str | None) -> ContrarianEvStrategy:
    cfg = {} if side_filter is None else {"filters": {"entry_side_filter": side_filter}}
    return ContrarianEvStrategy("t", cfg)


def test_no_only_blocks_yes_allows_no() -> None:
    s = _strategy("NO")
    assert "side_filter" in s._apply_entry_filters(_decision("YES"), secs_remaining=200.0)
    assert s._apply_entry_filters(_decision("NO"), secs_remaining=200.0) == ""


def test_yes_only_blocks_no_allows_yes() -> None:
    s = _strategy("YES")
    assert "side_filter" in s._apply_entry_filters(_decision("NO"), secs_remaining=200.0)
    assert s._apply_entry_filters(_decision("YES"), secs_remaining=200.0) == ""


def test_both_blocks_neither_side() -> None:
    # Default "both" (and an absent config) never blocks on side — every other
    # bot is unchanged.
    for s in (_strategy("both"), _strategy(None)):
        assert s._apply_entry_filters(_decision("YES"), secs_remaining=200.0) == ""
        assert s._apply_entry_filters(_decision("NO"), secs_remaining=200.0) == ""


def test_config_wiring_default_and_override() -> None:
    assert ContrarianEvStrategy("t", {})._entry_side_filter == "BOTH"
    assert _strategy("NO")._entry_side_filter == "NO"


def test_side_filter_normalizes_case() -> None:
    # A lowercase config value must not silently fail open to both sides.
    s = ContrarianEvStrategy("t", {"filters": {"entry_side_filter": "no"}})
    assert s._entry_side_filter == "NO"
    assert "side_filter" in s._apply_entry_filters(_decision("YES"), secs_remaining=200.0)


def test_side_filter_rejects_unknown_value() -> None:
    # Fail-fast at construction — a typo must not become "trade both sides
    # with real money" on a live-capital probe.
    with pytest.raises(ValueError):
        ContrarianEvStrategy("t", {"filters": {"entry_side_filter": "sell"}})
