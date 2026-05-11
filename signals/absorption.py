"""Layer 9b: Absorption Detection Signal — identifies hidden liquidity.

Absorption occurs when aggressive order flow hits one side of the book
but the depth doesn't collapse — a large participant is absorbing the
pressure with hidden or refilling orders.

Detection cross-references two independent data sources:
- CLOB trade flow (L8 data): actual executed trades showing selling/buying pressure
- Orderbook depth deltas (L4 data): tick-to-tick changes in resting depth

When heavy selling pressure coincides with stable or increasing bid depth,
a hidden buyer is accumulating (bullish absorption). The inverse indicates
a hidden seller distributing (bearish absorption).

Output is [-1.0, 1.0]:
  Positive = bid absorption (hidden buyer, bullish)
  Negative = ask absorption (hidden seller, bearish)
  Zero = no absorption detected or insufficient data
"""

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionSignal:
    """Detects absorption events by cross-referencing trade flow and depth.

    Args:
        min_pressure_volume: Minimum aggressive volume (shares) on one side
            to consider it "pressure". Below this, no absorption check.
        pressure_imbalance_threshold: Minimum trade flow imbalance (0-1)
            to qualify as one-sided pressure. 0.3 = 65/35 split.
        resilience_threshold: Depth delta threshold — if the receiving side's
            depth is above this (relative to pressure volume), absorption
            is detected. 0.0 = depth flat or increasing despite pressure.
        smoothing_window: Number of ticks to smooth pressure and resilience
            readings over. Prevents false triggers on single-tick spikes.
        decay_rate: How quickly the signal decays when absorption stops.
            1.0 = instant decay, 0.0 = no decay (latches forever).
            0.3 = signal loses 30% per tick when not reinforced.
        strength_scale: Multiplier for raw absorption strength before clamp.
            Higher = more aggressive signal response.
    """

    min_pressure_volume: float = 50.0
    pressure_imbalance_threshold: float = 0.30
    resilience_threshold: float = 0.0
    smoothing_window: int = 10
    decay_rate: float = 0.3
    strength_scale: float = 2.0

    # Internal state
    _pressure_history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )
    _bid_resilience_history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )
    _ask_resilience_history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )
    _current_signal: float = 0.0

    def reset(self) -> None:
        """Clear state for a new window."""
        self._pressure_history.clear()
        self._bid_resilience_history.clear()
        self._ask_resilience_history.clear()
        self._current_signal = 0.0

    def compute(
        self,
        trade_flow,
        ob_yes_bid_delta: float,
        ob_yes_ask_delta: float,
        ob_no_bid_delta: float,
        ob_no_ask_delta: float,
    ) -> float:
        """Compute absorption signal from trade flow and depth changes.

        Args:
            trade_flow: CLOBTradeFlow from DataSnapshot, or None.
            ob_yes_bid_delta: Change in YES bid depth since last tick.
            ob_yes_ask_delta: Change in YES ask depth since last tick.
            ob_no_bid_delta: Change in NO bid depth since last tick.
            ob_no_ask_delta: Change in NO ask depth since last tick.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bid absorption (bullish),
            negative = ask absorption (bearish).
        """
        if trade_flow is None:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # ── Measure directional pressure from executed trades ──
        # Sell pressure: trades that push price down
        # (selling YES = bearish, buying NO = bearish)
        sell_pressure = trade_flow.yes_sell_volume + trade_flow.no_buy_volume

        # Buy pressure: trades that push price up
        # (buying YES = bullish, selling NO = bullish)
        buy_pressure = trade_flow.yes_buy_volume + trade_flow.no_sell_volume

        total_pressure = sell_pressure + buy_pressure

        # ── Measure depth resilience ──
        # Bid resilience: are bids holding/refilling despite selling?
        # Positive delta = bids are growing (absorbing sells)
        bid_resilience = ob_yes_bid_delta + ob_no_ask_delta  # bullish depth
        ask_resilience = ob_yes_ask_delta + ob_no_bid_delta  # bearish depth

        # Track history for smoothing
        self._pressure_history.append((sell_pressure, buy_pressure))
        self._bid_resilience_history.append(bid_resilience)
        self._ask_resilience_history.append(ask_resilience)

        # Need enough history
        if len(self._pressure_history) < min(self.smoothing_window, 3):
            return 0.0

        # ── Smooth over recent window ──
        window = min(self.smoothing_window, len(self._pressure_history))
        recent_pressure = list(self._pressure_history)[-window:]
        recent_bid_res = list(self._bid_resilience_history)[-window:]
        recent_ask_res = list(self._ask_resilience_history)[-window:]

        avg_sell = sum(sp for sp, _ in recent_pressure) / window
        avg_buy = sum(bp for _, bp in recent_pressure) / window
        avg_total = avg_sell + avg_buy
        avg_bid_res = sum(recent_bid_res) / window
        avg_ask_res = sum(recent_ask_res) / window

        # ── Check for absorption conditions ──
        bid_absorption = 0.0
        ask_absorption = 0.0

        if avg_total >= self.min_pressure_volume:
            # Pressure imbalance: how one-sided is the flow?
            if avg_total > 0:
                sell_frac = avg_sell / avg_total
                buy_frac = avg_buy / avg_total
            else:
                sell_frac = buy_frac = 0.5

            # BID ABSORPTION: heavy selling but bids hold
            if sell_frac >= (0.5 + self.pressure_imbalance_threshold):
                if avg_bid_res >= self.resilience_threshold:
                    # Strength: how much depth is absorbing relative to
                    # the pressure volume? More absorption = stronger signal.
                    pressure_magnitude = sell_frac - 0.5  # 0-0.5 range
                    if avg_sell > 0:
                        resilience_ratio = max(0.0, avg_bid_res / avg_sell)
                    else:
                        resilience_ratio = 0.0
                    # Combine: pressure magnitude * (1 + resilience bonus)
                    bid_absorption = pressure_magnitude * (1.0 + resilience_ratio)

            # ASK ABSORPTION: heavy buying but asks hold
            if buy_frac >= (0.5 + self.pressure_imbalance_threshold):
                if avg_ask_res >= self.resilience_threshold:
                    pressure_magnitude = buy_frac - 0.5
                    if avg_buy > 0:
                        resilience_ratio = max(0.0, avg_ask_res / avg_buy)
                    else:
                        resilience_ratio = 0.0
                    ask_absorption = pressure_magnitude * (1.0 + resilience_ratio)

        # ── Combine into directional signal ──
        raw = (bid_absorption - ask_absorption) * self.strength_scale

        if abs(raw) > 0.01:
            # New absorption detected — update signal
            self._current_signal = _clamp(raw)
        else:
            # No absorption — decay existing signal
            self._current_signal *= (1.0 - self.decay_rate)

        return self._current_signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._current_signal


def _clamp(x: float) -> float:
    """Clamp to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, x))
