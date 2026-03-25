from collections import deque
from dataclasses import dataclass


@dataclass
class MomentumSignal:
    """Layer 2: Analyses 1-minute candle data for directional momentum.

    Sub-signals (weighted average):
    - Rate of Change (ROC): Price change over last 1-3 candles
    - Direction Consistency: Proportion of recent candles closing same direction
    - Volume Spike: Current volume vs rolling average
    - Candle Body Ratio: avg(close-open)/(high-low) — clean vs choppy
    - RSI (5-period, 1-min): Contrarian dampening at extremes
    """

    roc_weight: float = 0.30
    direction_weight: float = 0.25
    volume_weight: float = 0.25
    body_ratio_weight: float = 0.10
    rsi_weight: float = 0.10
    rsi_period: int = 5
    lookback_candles: int = 10

    def compute(self, candles: deque, current_candle=None) -> float:
        """Compute momentum signal in [-1.0, 1.0].

        Args:
            candles: Deque of closed Candle objects (most recent last).
            current_candle: The in-progress candle (may be None).

        Returns:
            Signal between -1.0 (strong bearish) and 1.0 (strong bullish).
        """
        # Build working list: closed candles + current
        working = list(candles)
        if current_candle:
            working.append(current_candle)

        if len(working) < 2:
            return 0.0

        recent = working[-self.lookback_candles:]

        roc = self._rate_of_change(recent)
        direction = self._direction_consistency(recent)
        volume = self._volume_spike(recent)
        body = self._candle_body_ratio(recent)
        rsi = self._rsi_signal(recent)

        raw = (
            self.roc_weight * roc
            + self.direction_weight * direction
            + self.volume_weight * volume
            + self.body_ratio_weight * body
            + self.rsi_weight * rsi
        )

        return max(-1.0, min(1.0, raw))

    def _rate_of_change(self, candles: list) -> float:
        """Price change over last 1-3 candles, normalized."""
        if len(candles) < 2:
            return 0.0

        # Use 1-candle and 3-candle ROC, averaged
        roc_1 = (candles[-1].close - candles[-2].close) / candles[-2].close if candles[-2].close else 0
        roc_3 = 0.0
        if len(candles) >= 4:
            roc_3 = (candles[-1].close - candles[-4].close) / candles[-4].close if candles[-4].close else 0

        avg_roc = (roc_1 + roc_3) / 2 if len(candles) >= 4 else roc_1

        # Normalize: 0.1% move = full signal
        return max(-1.0, min(1.0, avg_roc / 0.001))

    def _direction_consistency(self, candles: list) -> float:
        """Proportion of recent candles closing in the same direction."""
        if len(candles) < 2:
            return 0.0

        ups = sum(1 for c in candles if c.close >= c.open)
        downs = len(candles) - ups
        total = len(candles)

        if ups > downs:
            return (ups / total - 0.5) * 2  # Scale 0.5-1.0 → 0.0-1.0
        else:
            return -((downs / total - 0.5) * 2)

    def _volume_spike(self, candles: list) -> float:
        """Current volume relative to rolling average."""
        if len(candles) < 3:
            return 0.0

        volumes = [c.volume for c in candles if c.volume > 0]
        if not volumes:
            return 0.0

        avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        if avg_vol <= 0:
            return 0.0

        current_vol = volumes[-1]
        ratio = current_vol / avg_vol

        # Determine direction of latest candle
        direction = 1.0 if candles[-1].close >= candles[-1].open else -1.0

        # Volume spike amplifies direction: 2x avg = full signal
        spike = max(-1.0, min(1.0, (ratio - 1.0)))
        return spike * direction

    def _candle_body_ratio(self, candles: list) -> float:
        """Average body-to-range ratio. Clean trends have large bodies."""
        ratios = []
        for c in candles[-5:]:
            body = c.close - c.open
            range_ = c.high - c.low
            if range_ > 0:
                ratios.append(body / range_)

        if not ratios:
            return 0.0

        # Average signed ratio: positive = bullish bodies, negative = bearish
        avg = sum(ratios) / len(ratios)
        return max(-1.0, min(1.0, avg))

    def _rsi_signal(self, candles: list) -> float:
        """RSI-based contrarian dampening. Extreme RSI dampens momentum."""
        period = min(self.rsi_period, len(candles) - 1)
        if period < 2:
            return 0.0

        gains = []
        losses = []
        for i in range(-period, 0):
            change = candles[i].close - candles[i - 1].close
            if change >= 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Contrarian: RSI > 80 dampens bullish, RSI < 20 dampens bearish
        # RSI 50 = neutral (0), RSI 80+ = bearish (-1), RSI 20- = bullish (+1)
        if rsi > 80:
            return -((rsi - 80) / 20)  # 80-100 → 0 to -1
        elif rsi < 20:
            return (20 - rsi) / 20     # 20-0 → 0 to +1
        else:
            # Mid range: slight directional signal
            return (rsi - 50) / 50     # 20-80 → -0.6 to +0.6
