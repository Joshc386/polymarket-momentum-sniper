"""Tests for Layer 11 Trade Size Signal.

Verifies:
- Large trade bias detection (directional flow from outsized trades)
- Trade count asymmetry (buy/sell count ratio)
- Average size divergence (mean trade size per side)
- Minimum trade count enforcement
- Signal decay when no trades arrive
- Reset clears all state
- Clamping to [-1, 1]
- Bullish/bearish classification of YES/NO × BUY/SELL combinations
"""

import time

import pytest

from signals.trade_size import TradeSizeSignal


def _trade(
    token: str = "YES",
    side: str = "BUY",
    size: float = 50.0,
    price: float = 0.50,
    ts: float | None = None,
) -> tuple[float, str, str, float, float]:
    """Create a trade tuple matching CLOBTradeFlow.recent_trades format."""
    return (ts or time.time(), token, side, price, size)


def _bulk_trades(
    token: str,
    side: str,
    sizes: list[float],
    price: float = 0.50,
) -> list[tuple[float, str, str, float, float]]:
    """Create multiple trades with different sizes."""
    now = time.time()
    return [
        (now + i * 0.01, token, side, price, s)
        for i, s in enumerate(sizes)
    ]


# ── Large trade bias ─────────────────────────────────────────────


class TestLargeTradeBias:
    """Sub-signal 1: directional flow from large trades."""

    def test_large_buys_are_bullish(self) -> None:
        """One large buy among small trades -> positive signal."""
        sig = TradeSizeSignal(
            large_multiplier=3.0, large_floor=10.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [10, 10, 10, 200])
            + _bulk_trades("YES", "SELL", [10, 10, 10, 10])
        )
        val = sig.compute(trades)
        assert val > 0  # Bullish

    def test_large_sells_are_bearish(self) -> None:
        """One large sell among small trades -> negative signal."""
        sig = TradeSizeSignal(
            large_multiplier=3.0, large_floor=10.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [10, 10, 10, 10])
            + _bulk_trades("YES", "SELL", [10, 10, 10, 200])
        )
        val = sig.compute(trades)
        assert val < 0  # Bearish

    def test_no_large_trades_zero_sub_signal(self) -> None:
        """All trades below threshold -> large bias is zero."""
        sig = TradeSizeSignal(
            large_multiplier=3.0, large_floor=100.0, min_trades=3,
        )
        # All trades are 10 shares, floor is 100
        trades = (
            _bulk_trades("YES", "BUY", [10, 10, 10, 10, 10])
            + _bulk_trades("YES", "SELL", [10, 10, 10, 10, 10])
        )
        val = sig.compute(trades)
        # Large bias component is zero, but count and size div may
        # still contribute. With balanced trades, overall should be ~0
        assert abs(val) < 0.01

    def test_floor_prevents_noise(self) -> None:
        """Floor prevents tiny trades from being classified as 'large'."""
        sig = TradeSizeSignal(
            large_multiplier=3.0, large_floor=100.0, min_trades=3,
        )
        # Median = 5, so 3× = 15. But floor = 100, so 20-share trade
        # is NOT large.
        trades = (
            _bulk_trades("YES", "BUY", [5, 5, 5, 20])
            + _bulk_trades("YES", "SELL", [5, 5, 5, 5])
        )
        # The 20-share trade is not "large" (below floor of 100)
        # So large_bias = 0, signal driven only by count/size divergence
        val = sig.compute(trades)
        # Small positive from count/size but not dominated by large_bias
        assert abs(val) < 0.3


# ── Trade count asymmetry ────────────────────────────────────────


class TestCountAsymmetry:
    """Sub-signal 2: ratio of buy to sell trade count."""

    def test_more_buys_is_bullish(self) -> None:
        """8 buys vs 2 sells -> positive count asymmetry."""
        sig = TradeSizeSignal(
            large_floor=1000.0, min_trades=3,  # High floor to suppress large bias
        )
        trades = (
            _bulk_trades("YES", "BUY", [10] * 8)
            + _bulk_trades("YES", "SELL", [10] * 2)
        )
        val = sig.compute(trades)
        assert val > 0

    def test_more_sells_is_bearish(self) -> None:
        """2 buys vs 8 sells -> negative count asymmetry."""
        sig = TradeSizeSignal(
            large_floor=1000.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [10] * 2)
            + _bulk_trades("YES", "SELL", [10] * 8)
        )
        val = sig.compute(trades)
        assert val < 0

    def test_equal_counts_neutral(self) -> None:
        """Same number of buys and sells -> count asymmetry is zero."""
        sig = TradeSizeSignal(
            large_floor=1000.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [10] * 5)
            + _bulk_trades("YES", "SELL", [10] * 5)
        )
        val = sig.compute(trades)
        assert abs(val) < 0.01


# ── Average size divergence ──────────────────────────────────────


class TestSizeDivergence:
    """Sub-signal 3: average trade size per side."""

    def test_larger_buys_is_bullish(self) -> None:
        """Buyers placing bigger clips than sellers -> bullish."""
        sig = TradeSizeSignal(
            large_floor=1000.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [100, 100, 100])  # avg 100
            + _bulk_trades("YES", "SELL", [20, 20, 20])  # avg 20
        )
        val = sig.compute(trades)
        assert val > 0

    def test_larger_sells_is_bearish(self) -> None:
        """Sellers placing bigger clips -> bearish."""
        sig = TradeSizeSignal(
            large_floor=1000.0, min_trades=3,
        )
        trades = (
            _bulk_trades("YES", "BUY", [20, 20, 20])
            + _bulk_trades("YES", "SELL", [100, 100, 100])
        )
        val = sig.compute(trades)
        assert val < 0


# ── YES/NO × BUY/SELL classification ────────────────────────────


class TestTradeClassification:
    """Verify bullish/bearish mapping for all token×side combos."""

    def test_yes_buy_is_bullish(self) -> None:
        sig = TradeSizeSignal(large_floor=1000.0, min_trades=3)
        trades = (
            _bulk_trades("YES", "BUY", [50] * 8)
            + _bulk_trades("YES", "SELL", [50] * 2)
        )
        assert sig.compute(trades) > 0

    def test_yes_sell_is_bearish(self) -> None:
        sig = TradeSizeSignal(large_floor=1000.0, min_trades=3)
        trades = (
            _bulk_trades("YES", "BUY", [50] * 2)
            + _bulk_trades("YES", "SELL", [50] * 8)
        )
        assert sig.compute(trades) < 0

    def test_no_sell_is_bullish(self) -> None:
        """Selling NO = betting on YES = bullish."""
        sig = TradeSizeSignal(large_floor=1000.0, min_trades=3)
        trades = (
            _bulk_trades("NO", "SELL", [50] * 8)
            + _bulk_trades("NO", "BUY", [50] * 2)
        )
        assert sig.compute(trades) > 0

    def test_no_buy_is_bearish(self) -> None:
        """Buying NO = betting against YES = bearish."""
        sig = TradeSizeSignal(large_floor=1000.0, min_trades=3)
        trades = (
            _bulk_trades("NO", "BUY", [50] * 8)
            + _bulk_trades("NO", "SELL", [50] * 2)
        )
        assert sig.compute(trades) < 0


# ── Edge cases and dynamics ──────────────────────────────────────


class TestMinTrades:
    """Signal requires minimum trade count."""

    def test_below_min_returns_zero(self) -> None:
        """Too few trades -> signal is zero."""
        sig = TradeSizeSignal(min_trades=10)
        trades = _bulk_trades("YES", "BUY", [100] * 5)
        val = sig.compute(trades)
        assert val == 0.0

    def test_at_min_fires(self) -> None:
        """Exactly min_trades -> signal fires."""
        sig = TradeSizeSignal(min_trades=5, large_floor=1000.0)
        trades = (
            _bulk_trades("YES", "BUY", [100] * 4)
            + _bulk_trades("YES", "SELL", [10] * 1)
        )
        val = sig.compute(trades)
        assert val > 0

    def test_empty_trades_zero(self) -> None:
        """No trades at all -> zero."""
        sig = TradeSizeSignal()
        assert sig.compute([]) == 0.0


class TestSignalDynamics:
    """Decay, reset, and clamping."""

    def test_signal_decays(self) -> None:
        """Signal decays toward zero when no new trades arrive."""
        sig = TradeSizeSignal(
            min_trades=3, large_floor=1000.0, decay_rate=0.5,
        )
        trades = (
            _bulk_trades("YES", "BUY", [50] * 8)
            + _bulk_trades("YES", "SELL", [50] * 2)
        )
        val = sig.compute(trades)
        assert val > 0

        # Now pass empty trades -> should decay
        for _ in range(5):
            sig.compute([])
        assert abs(sig.last_signal) < abs(val)

    def test_reset_clears_all(self) -> None:
        """Reset zeroes signal and trade count."""
        sig = TradeSizeSignal(min_trades=3, large_floor=1000.0)
        trades = (
            _bulk_trades("YES", "BUY", [50] * 8)
            + _bulk_trades("YES", "SELL", [50] * 2)
        )
        sig.compute(trades)
        assert sig.last_signal != 0.0

        sig.reset()
        assert sig.last_signal == 0.0
        assert sig._last_trade_count == 0

    def test_signal_clamped(self) -> None:
        """Signal stays within [-1, 1] even with extreme inputs."""
        sig = TradeSizeSignal(
            min_trades=3, large_floor=1.0, large_multiplier=1.0,
            large_bias_weight=10.0,  # Extreme weight to test clamping
        )
        trades = _bulk_trades("YES", "BUY", [1000] * 10)
        val = sig.compute(trades)
        assert -1.0 <= val <= 1.0

    def test_zero_size_trades_ignored(self) -> None:
        """Trades with size 0 are filtered out."""
        sig = TradeSizeSignal(min_trades=3, large_floor=1000.0)
        trades = (
            _bulk_trades("YES", "BUY", [0, 0, 0, 50, 50])
            + _bulk_trades("YES", "SELL", [50, 50, 50])
        )
        # Only 5 non-zero trades (2 buy + 3 sell)
        val = sig.compute(trades)
        # 2 buys vs 3 sells -> bearish
        assert val < 0


class TestCombinedSubSignals:
    """Verify sub-signals combine correctly."""

    def test_all_bullish_strong_positive(self) -> None:
        """Large buys + more buys + bigger avg buy -> strong positive."""
        sig = TradeSizeSignal(
            min_trades=3, large_multiplier=2.0, large_floor=10.0,
        )
        trades = (
            _bulk_trades("YES", "BUY", [10, 10, 10, 200, 200])
            + _bulk_trades("YES", "SELL", [10, 10])
        )
        val = sig.compute(trades)
        assert val > 0.3  # Strong bullish from all 3 sub-signals

    def test_conflicting_sub_signals_moderate(self) -> None:
        """Large sells but more buy trades -> weaker signal."""
        sig = TradeSizeSignal(
            min_trades=3, large_multiplier=2.0, large_floor=10.0,
        )
        # Many small buys (count favours bulls)
        # But one large sell (large_bias favours bears)
        trades = (
            _bulk_trades("YES", "BUY", [10] * 8)
            + _bulk_trades("YES", "SELL", [10, 10, 500])
        )
        val = sig.compute(trades)
        # Should be moderate — sub-signals partially cancel.
        # The large sell dominates large_bias and size_div, but count
        # asymmetry pulls the other way. Net result is bearish but
        # not as strong as a fully aligned signal.
        assert abs(val) < 0.8

    def test_balanced_trades_near_zero(self) -> None:
        """Perfectly balanced trades -> signal near zero."""
        sig = TradeSizeSignal(
            min_trades=3, large_floor=1000.0,
        )
        trades = (
            _bulk_trades("YES", "BUY", [50] * 5)
            + _bulk_trades("YES", "SELL", [50] * 5)
        )
        val = sig.compute(trades)
        assert abs(val) < 0.01
