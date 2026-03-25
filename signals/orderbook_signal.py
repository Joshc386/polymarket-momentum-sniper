"""Layer 4: Polymarket Orderbook Signal — reads depth, imbalance, and flow
from the Polymarket CLOB to detect informed order activity.

Sub-signals (all derived from Polymarket orderbook only):
1. Bid/Ask Imbalance: Volume asymmetry across all price levels
2. Depth Shift (Order Flow): Change in depth between snapshots
3. Size-Weighted Mid Deviation: Divergence between simple and weighted mid
4. Top-of-Book Pressure: Size ratio at best bid vs best ask
5. Book Thickness: Overall liquidity — thin books = higher conviction signals

Combined output is [-1.0, 1.0]:
  Positive = bullish orderbook (buyers dominating)
  Negative = bearish orderbook (sellers dominating)
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class OrderbookSignal:
    """Layer 4: Polymarket CLOB orderbook analysis."""

    # Sub-signal weights
    imbalance_weight: float = 0.30
    flow_weight: float = 0.25
    weighted_mid_weight: float = 0.20
    top_pressure_weight: float = 0.15
    thickness_weight: float = 0.10

    # History for flow analysis
    _imbalance_history: deque = field(default_factory=lambda: deque(maxlen=30))
    _flow_history: deque = field(default_factory=lambda: deque(maxlen=30))

    def compute(
        self,
        yes_bid_total: float,
        yes_ask_total: float,
        no_bid_total: float,
        no_ask_total: float,
        yes_best_bid: float,
        yes_best_ask: float,
        yes_bid_depth_top5: float,
        yes_ask_depth_top5: float,
        size_weighted_mid: float,
        simple_mid: float,
        yes_bid_delta: float,
        yes_ask_delta: float,
        no_bid_delta: float,
        no_ask_delta: float,
        num_bid_levels: int,
        num_ask_levels: int,
    ) -> float:
        """Compute the L4 orderbook signal.

        All inputs come from Polymarket's CLOB orderbook.

        Args:
            yes_bid_total: Total YES bid volume across all levels.
            yes_ask_total: Total YES ask volume across all levels.
            no_bid_total: Total NO bid volume across all levels.
            no_ask_total: Total NO ask volume across all levels.
            yes_best_bid: Best bid price for YES.
            yes_best_ask: Best ask price for YES.
            yes_bid_depth_top5: Total size at top 5 YES bid levels.
            yes_ask_depth_top5: Total size at top 5 YES ask levels.
            size_weighted_mid: Volume-weighted midpoint.
            simple_mid: Simple (best_bid + best_ask) / 2 midpoint.
            yes_bid_delta: Change in YES bid total since last snapshot.
            yes_ask_delta: Change in YES ask total since last snapshot.
            no_bid_delta: Change in NO bid total since last snapshot.
            no_ask_delta: Change in NO ask total since last snapshot.
            num_bid_levels: Number of bid levels in the book.
            num_ask_levels: Number of ask levels in the book.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bullish, negative = bearish.
        """
        # 1. Bid/Ask Imbalance
        imbalance = self._bid_ask_imbalance(
            yes_bid_total, yes_ask_total, no_bid_total, no_ask_total
        )

        # 2. Order Flow (depth changes)
        flow = self._order_flow(
            yes_bid_delta, yes_ask_delta, no_bid_delta, no_ask_delta
        )

        # 3. Size-weighted mid deviation
        mid_dev = self._weighted_mid_deviation(size_weighted_mid, simple_mid)

        # 4. Top-of-book pressure
        top_pressure = self._top_of_book_pressure(
            yes_bid_depth_top5, yes_ask_depth_top5
        )

        # 5. Book thickness confidence scaler
        thickness = self._book_thickness(
            yes_bid_total, yes_ask_total, num_bid_levels, num_ask_levels
        )

        # Track history for smoothing
        self._imbalance_history.append(imbalance)
        self._flow_history.append(flow)

        # Smooth imbalance using recent history (reduces noise)
        if len(self._imbalance_history) >= 3:
            smoothed_imbalance = sum(self._imbalance_history) / len(self._imbalance_history)
        else:
            smoothed_imbalance = imbalance

        # Smooth flow
        if len(self._flow_history) >= 3:
            smoothed_flow = sum(self._flow_history) / len(self._flow_history)
        else:
            smoothed_flow = flow

        # Combine sub-signals
        raw = (
            self.imbalance_weight * smoothed_imbalance
            + self.flow_weight * smoothed_flow
            + self.weighted_mid_weight * mid_dev
            + self.top_pressure_weight * top_pressure
            + self.thickness_weight * thickness  # thickness acts as directional modifier
        )

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, raw))

    def _bid_ask_imbalance(
        self,
        yes_bid: float,
        yes_ask: float,
        no_bid: float,
        no_ask: float,
    ) -> float:
        """Compute volume imbalance across both YES and NO books.

        More YES bids + more NO asks = bullish (people buying YES, selling NO).
        More YES asks + more NO bids = bearish (people selling YES, buying NO).
        """
        # Bullish volume: YES bids (buying UP) + NO asks (selling DOWN protection)
        bullish_vol = yes_bid + no_ask
        # Bearish volume: YES asks (selling UP) + NO bids (buying DOWN protection)
        bearish_vol = yes_ask + no_bid

        total = bullish_vol + bearish_vol
        if total <= 0:
            return 0.0

        imbalance = (bullish_vol - bearish_vol) / total
        # Scale: typical imbalance of ±0.3 maps to ±1.0
        return max(-1.0, min(1.0, imbalance / 0.3))

    def _order_flow(
        self,
        yes_bid_delta: float,
        yes_ask_delta: float,
        no_bid_delta: float,
        no_ask_delta: float,
    ) -> float:
        """Detect order flow from depth changes between snapshots.

        Bullish flow: YES bids increasing, YES asks decreasing
                      (buyers adding, sellers getting lifted)
        Bearish flow: YES asks increasing, YES bids decreasing
                      (sellers adding, buyers getting hit)
        """
        # Bullish signals: YES bid depth increasing, YES ask depth decreasing
        # Also: NO ask increasing (more people willing to sell NO = bullish on UP)
        bullish_flow = yes_bid_delta - yes_ask_delta + no_ask_delta - no_bid_delta

        # Normalize by a reasonable typical delta
        # Typical depth changes might be in the range of 50-500 shares
        normalized = bullish_flow / 200.0  # 200 share delta = full signal

        return max(-1.0, min(1.0, normalized))

    def _weighted_mid_deviation(
        self,
        weighted_mid: float,
        simple_mid: float,
    ) -> float:
        """Compare size-weighted mid to simple mid.

        If weighted mid > simple mid: large bids pulling fair value up = bullish.
        If weighted mid < simple mid: large asks pulling fair value down = bearish.
        """
        if simple_mid <= 0:
            return 0.0

        deviation = weighted_mid - simple_mid
        # Typical deviation is 0-0.03 cents. Scale so 0.02 = full signal
        return max(-1.0, min(1.0, deviation / 0.02))

    def _top_of_book_pressure(
        self,
        bid_depth_top5: float,
        ask_depth_top5: float,
    ) -> float:
        """Pressure at the top of the book (most immediately actionable levels).

        Heavy size at top bids = buyer support = bullish.
        Heavy size at top asks = seller wall = bearish.
        """
        total = bid_depth_top5 + ask_depth_top5
        if total <= 0:
            return 0.0

        ratio = (bid_depth_top5 - ask_depth_top5) / total
        return max(-1.0, min(1.0, ratio / 0.3))

    def _book_thickness(
        self,
        total_bid: float,
        total_ask: float,
        num_bid_levels: int,
        num_ask_levels: int,
    ) -> float:
        """Book thickness as a confidence/directional modifier.

        Thin book on one side = that side can be moved easily.
        If asks are thin but bids are thick: price likely to move UP (bullish).
        If bids are thin but asks are thick: price likely to move DOWN (bearish).

        Returns a directional signal, not just magnitude.
        """
        total = total_bid + total_ask
        if total <= 0:
            return 0.0

        # Directional: thick bids + thin asks = bullish
        thickness_imbalance = (total_bid - total_ask) / total

        # Also consider number of levels
        total_levels = num_bid_levels + num_ask_levels
        if total_levels > 0:
            level_imbalance = (num_bid_levels - num_ask_levels) / total_levels
            # Blend volume and level count imbalance
            return max(-1.0, min(1.0, (thickness_imbalance * 0.7 + level_imbalance * 0.3) / 0.3))

        return max(-1.0, min(1.0, thickness_imbalance / 0.3))

    def reset(self):
        """Reset history (call on window transitions)."""
        self._imbalance_history.clear()
        self._flow_history.clear()
