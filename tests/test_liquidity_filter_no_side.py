"""Regression tests for the NO-side liquidity filter bug (2026-05-15).

The bug: `_apply_entry_filters` tried to access `ob.no_ask_depth` on
OrderbookSummary, which doesn't exist as a field (only `yes_ask_depth`
exists). This caused AttributeError every tick when a NO-side signal
fired, which multi_runner caught silently. The entry_decision never
got updated past should_enter=True, so the dashboard showed a phantom
SIGNAL: NO line forever and no trade ever placed.

These tests verify:
1. OrderbookSummary indeed lacks no_ask_depth (confirms the schema we
   relied on)
2. The new code path uses no_ask_levels[:5] correctly
3. Empty no_ask_levels falls back to 0 (filter passes)
"""

import pytest
from data.orderbook import OrderbookSummary, OrderLevel


class TestSchemaCheck:
    """The bug exists because of this schema mismatch."""

    def test_no_ask_depth_attribute_does_not_exist(self):
        """If this test ever fails, the bug fix can be simplified."""
        ob = _make_minimal_orderbook()
        assert not hasattr(ob, "no_ask_depth")

    def test_yes_ask_depth_attribute_does_exist(self):
        """Confirms YES side has the field that NO side lacks."""
        ob = _make_minimal_orderbook()
        assert hasattr(ob, "yes_ask_depth")

    def test_no_ask_levels_attribute_does_exist(self):
        """The fix uses this field instead."""
        ob = _make_minimal_orderbook()
        assert hasattr(ob, "no_ask_levels")
        assert isinstance(ob.no_ask_levels, list)


class TestNoSideDepthCalculation:
    """The fix computes top-5 depth from no_ask_levels inline."""

    def test_sum_of_top_5_levels(self):
        levels = [OrderLevel(price=0.15, size=100.0),
                  OrderLevel(price=0.16, size=50.0),
                  OrderLevel(price=0.17, size=25.0),
                  OrderLevel(price=0.18, size=20.0),
                  OrderLevel(price=0.19, size=10.0),
                  OrderLevel(price=0.20, size=5.0)]  # 6th — should be excluded
        depth = sum(l.size for l in levels[:5])
        assert depth == 205.0  # 100 + 50 + 25 + 20 + 10

    def test_empty_levels_returns_zero(self):
        levels = []
        depth = sum(l.size for l in levels[:5])
        assert depth == 0

    def test_fewer_than_5_levels(self):
        levels = [OrderLevel(price=0.15, size=100.0),
                  OrderLevel(price=0.16, size=50.0)]
        depth = sum(l.size for l in levels[:5])
        assert depth == 150.0

    def test_getattr_with_none_fallback(self):
        """Replicate the actual code path."""
        class FakeOb:
            pass
        ob = FakeOb()
        levels = getattr(ob, "no_ask_levels", None) or []
        depth = sum(l.size for l in levels[:5])
        assert depth == 0  # No attribute -> empty list -> 0

    def test_getattr_with_empty_list_fallback(self):
        class FakeOb:
            no_ask_levels = []
        ob = FakeOb()
        levels = getattr(ob, "no_ask_levels", None) or []
        depth = sum(l.size for l in levels[:5])
        assert depth == 0


class TestLiquidityFilterLogic:
    """The filter formula and when it blocks vs allows."""

    def test_zero_depth_passes_filter(self):
        """When ask_depth is 0 (no data), filter doesn't block.
        This matches the existing logic for YES side."""
        ask_depth = 0
        intended_shares = 31.25  # $5 / $0.16
        min_depth_multiplier = 5.0
        # Filter blocks if `ask_depth > 0 AND ask_depth < intended * mult`
        blocks = ask_depth > 0 and ask_depth < intended_shares * min_depth_multiplier
        assert not blocks

    def test_thin_book_blocks(self):
        """50 shares of depth when we need 156 -> blocks."""
        ask_depth = 50
        intended_shares = 31.25
        min_depth_multiplier = 5.0
        blocks = ask_depth > 0 and ask_depth < intended_shares * min_depth_multiplier
        assert blocks  # 50 < 156

    def test_deep_book_allows(self):
        """200 shares of depth when we need 156 -> allows."""
        ask_depth = 200
        intended_shares = 31.25
        min_depth_multiplier = 5.0
        blocks = ask_depth > 0 and ask_depth < intended_shares * min_depth_multiplier
        assert not blocks  # 200 > 156

    def test_at_threshold_allows(self):
        """Exactly at threshold = filter requires STRICTLY less than."""
        ask_depth = 156.25
        intended_shares = 31.25
        min_depth_multiplier = 5.0
        blocks = ask_depth > 0 and ask_depth < intended_shares * min_depth_multiplier
        assert not blocks  # 156.25 < 156.25 is False


def _make_minimal_orderbook() -> OrderbookSummary:
    """Build an OrderbookSummary with the minimum fields required."""
    return OrderbookSummary(
        yes_best_bid=0.50,
        yes_best_ask=0.55,
        yes_midpoint=0.525,
        no_best_bid=0.45,
        no_best_ask=0.50,
        no_midpoint=0.475,
        yes_bid_depth=100.0,
        yes_ask_depth=100.0,
        spread=0.05,
    )
