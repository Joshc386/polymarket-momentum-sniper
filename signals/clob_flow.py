"""Layer 8: CLOB Trade Flow Signal — reads actual Polymarket trade volume.

Unlike L4 (orderbook depth/imbalance), this signal tracks EXECUTED trades
on the Polymarket CLOB via WebSocket last_trade_price events. It measures
where real money is flowing — not just resting orders, but actual fills.

Sub-signals:
1. Volume Imbalance: Net buying vs selling across YES/NO tokens
2. Momentum: Is the flow accelerating in one direction?
3. Size Conviction: Are the trades large (informed) or small (noise)?

Output is [-1.0, 1.0]:
  Positive = bullish flow (net buying YES / selling NO)
  Negative = bearish flow (net selling YES / buying NO)
  Zero = no flow data or balanced flow
"""

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CLOBFlowSignal:
    """Layer 8: Polymarket CLOB trade flow analysis.

    Reads CLOBTradeFlow from the DataSnapshot (populated by the
    PolymarketWebSocket from last_trade_price events).

    Args:
        imbalance_weight: Weight for volume imbalance sub-signal.
        momentum_weight: Weight for flow momentum sub-signal.
        conviction_weight: Weight for trade size conviction sub-signal.
        min_trades: Minimum trades in the window to generate a signal.
            Below this threshold, returns 0.0 (not enough data).
        large_trade_threshold: Trades above this size (shares) are
            considered "large" / potentially informed.
    """

    imbalance_weight: float = 0.50
    momentum_weight: float = 0.30
    conviction_weight: float = 0.20
    min_trades: int = 5
    large_trade_threshold: float = 100.0

    # History for momentum calculation
    _imbalance_history: deque = field(default_factory=lambda: deque(maxlen=20))

    def reset(self) -> None:
        """Reset state for a new window."""
        self._imbalance_history.clear()

    def compute(self, trade_flow) -> float:
        """Compute CLOB flow signal from trade flow data.

        Args:
            trade_flow: CLOBTradeFlow from DataSnapshot, or None.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bullish (buying YES),
            negative = bearish (selling YES).
        """
        if trade_flow is None:
            return 0.0

        total_trades = trade_flow.yes_trade_count + trade_flow.no_trade_count
        if total_trades < self.min_trades:
            return 0.0

        # ── Sub-signal 1: Volume Imbalance ──
        # Direct from CLOBTradeFlow — already computed as [-1, 1]
        imbalance = trade_flow.imbalance

        # ── Sub-signal 2: Flow Momentum ──
        # Is the imbalance accelerating? Compare current to recent history.
        self._imbalance_history.append(imbalance)
        momentum = 0.0
        if len(self._imbalance_history) >= 3:
            recent = list(self._imbalance_history)
            # Compare latest third to earliest third
            third = max(1, len(recent) // 3)
            early_avg = sum(recent[:third]) / third
            late_avg = sum(recent[-third:]) / third
            # Momentum: positive if imbalance is becoming more bullish
            momentum = _clamp(late_avg - early_avg, -1.0, 1.0)

        # ── Sub-signal 3: Size Conviction ──
        # Are the bullish trades larger than bearish? Large trades
        # are more likely informed.
        conviction = 0.0
        if trade_flow.total_volume > 0:
            # Average trade size for bullish vs bearish actions
            bullish_vol = trade_flow.yes_buy_volume + trade_flow.no_sell_volume
            bearish_vol = trade_flow.yes_sell_volume + trade_flow.no_buy_volume
            bullish_count = max(1, trade_flow.yes_trade_count)
            bearish_count = max(1, trade_flow.no_trade_count)

            avg_bullish = bullish_vol / bullish_count if bullish_count > 0 else 0
            avg_bearish = bearish_vol / bearish_count if bearish_count > 0 else 0

            total_avg = avg_bullish + avg_bearish
            if total_avg > 0:
                # Positive if bullish trades are larger on average
                conviction = _clamp(
                    (avg_bullish - avg_bearish) / total_avg, -1.0, 1.0
                )

        # ── Combine ──
        signal = (
            self.imbalance_weight * imbalance
            + self.momentum_weight * momentum
            + self.conviction_weight * conviction
        )

        return _clamp(signal, -1.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))
