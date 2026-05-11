"""Layer 11: Trade Size Signal — conviction asymmetry from trade sizes.

Analyses individual CLOB trade sizes to detect conviction differences
between buyers and sellers that aggregate flow (L8) misses.

Three sub-signals:
1. Large trade bias: directional flow from unusually large trades only.
   Filters to trades above a relative threshold (median × multiplier),
   then computes net flow. Large trades = higher conviction.
2. Trade count asymmetry: ratio of buy trades to sell trades, ignoring
   size. Many small buys vs few sells (or vice versa) indicates retail
   FOMO or panic vs informed positioning.
3. Average size divergence: compares mean trade size on each side.
   Even when total volume is balanced, larger average buy size vs sell
   size signals that buyers have more conviction per trade.

Data source: individual trade tuples from CLOBTradeFlow.recent_trades,
populated by the Polymarket WebSocket feed. Each tuple is:
    (timestamp, token, side, price, size)

Output is [-1.0, 1.0]:
  Positive = buy-side conviction (bullish)
  Negative = sell-side conviction (bearish)
  Zero = balanced or insufficient data
"""

import logging
from dataclasses import dataclass, field
from statistics import median

logger = logging.getLogger(__name__)


@dataclass
class TradeSizeSignal:
    """Detects conviction asymmetry from trade size distribution.

    Args:
        large_multiplier: A trade is "large" if size > median × this.
            Default 3.0 = trade must be 3× the median to count as large.
        large_floor: Absolute minimum threshold for a "large" trade.
            Prevents noise in very low-volume windows where the median
            might be tiny.
        min_trades: Minimum number of trades in the window before the
            signal fires. Below this, the distribution is too thin to
            draw conclusions.
        large_bias_weight: Weight of the large trade bias sub-signal.
        count_asym_weight: Weight of the trade count asymmetry sub-signal.
        size_div_weight: Weight of the average size divergence sub-signal.
        decay_rate: How quickly the signal decays when no new trades
            arrive. 0.3 = lose 30% per tick. Slower than L10 because
            trade patterns are stickier than orderbook states.
    """

    large_multiplier: float = 3.0
    large_floor: float = 50.0
    min_trades: int = 10
    large_bias_weight: float = 0.40
    count_asym_weight: float = 0.25
    size_div_weight: float = 0.35
    decay_rate: float = 0.3

    # Internal state
    _current_signal: float = 0.0
    _last_trade_count: int = 0

    def reset(self) -> None:
        """Clear all state for a new window."""
        self._current_signal = 0.0
        self._last_trade_count = 0

    def compute(self, recent_trades: list) -> float:
        """Compute trade size conviction signal.

        Args:
            recent_trades: List of trade tuples from CLOBTradeFlow.
                Each tuple: (timestamp, token, side, price, size).
                Token is "YES" or "NO", side is "BUY" or "SELL".

        Returns:
            Signal in [-1.0, 1.0].
            Positive = buy-side conviction (bullish).
            Negative = sell-side conviction (bearish).
        """
        if not recent_trades or len(recent_trades) < self.min_trades:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # Classify each trade as bullish or bearish action
        # YES BUY or NO SELL = bullish
        # YES SELL or NO BUY = bearish
        bull_sizes: list[float] = []
        bear_sizes: list[float] = []

        for _ts, token, side, _price, size in recent_trades:
            if size <= 0:
                continue
            is_bullish = (
                (token == "YES" and side == "BUY")
                or (token == "NO" and side == "SELL")
            )
            if is_bullish:
                bull_sizes.append(size)
            else:
                bear_sizes.append(size)

        total_trades = len(bull_sizes) + len(bear_sizes)
        if total_trades < self.min_trades:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # ── Sub-signal 1: Large trade bias ──
        large_bias = self._compute_large_bias(bull_sizes, bear_sizes)

        # ── Sub-signal 2: Trade count asymmetry ──
        count_asym = self._compute_count_asymmetry(
            len(bull_sizes), len(bear_sizes),
        )

        # ── Sub-signal 3: Average size divergence ──
        size_div = self._compute_size_divergence(bull_sizes, bear_sizes)

        # Blend sub-signals
        raw = (
            self.large_bias_weight * large_bias
            + self.count_asym_weight * count_asym
            + self.size_div_weight * size_div
        )

        self._current_signal = _clamp(raw)
        self._last_trade_count = total_trades

        return self._current_signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._current_signal

    def _compute_large_bias(
        self,
        bull_sizes: list[float],
        bear_sizes: list[float],
    ) -> float:
        """Net flow from large trades only.

        Returns [-1, 1]. Positive = large trades are net bullish.
        """
        all_sizes = bull_sizes + bear_sizes
        if not all_sizes:
            return 0.0

        med = median(all_sizes)
        threshold = max(med * self.large_multiplier, self.large_floor)

        large_bull = sum(s for s in bull_sizes if s >= threshold)
        large_bear = sum(s for s in bear_sizes if s >= threshold)

        total_large = large_bull + large_bear
        if total_large <= 0:
            return 0.0

        return (large_bull - large_bear) / total_large

    @staticmethod
    def _compute_count_asymmetry(
        bull_count: int,
        bear_count: int,
    ) -> float:
        """Ratio of bullish to bearish trade count.

        Returns [-1, 1]. Positive = more buy trades than sell trades.
        """
        total = bull_count + bear_count
        if total <= 0:
            return 0.0
        return (bull_count - bear_count) / total

    @staticmethod
    def _compute_size_divergence(
        bull_sizes: list[float],
        bear_sizes: list[float],
    ) -> float:
        """Average trade size on buy side vs sell side.

        Returns [-1, 1]. Positive = buyers placing larger clips.
        """
        avg_bull = sum(bull_sizes) / len(bull_sizes) if bull_sizes else 0.0
        avg_bear = sum(bear_sizes) / len(bear_sizes) if bear_sizes else 0.0

        total_avg = avg_bull + avg_bear
        if total_avg <= 0:
            return 0.0

        return (avg_bull - avg_bear) / total_avg


def _clamp(x: float) -> float:
    """Clamp to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, x))
