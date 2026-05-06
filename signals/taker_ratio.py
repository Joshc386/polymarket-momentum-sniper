"""Layer 7: Taker Buy Ratio Signal — measures aggressive buying vs selling.

Uses Binance's individual trade stream to compute the ratio of taker-buy
volume to total volume over a rolling window. When aggressive buyers
dominate, price tends to go up; when sellers dominate, price tends to
fall.

Each Binance trade message has a "m" (maker side) flag:
  m=True  -> seller is maker -> trade was a taker BUY
  m=False -> buyer is maker  -> trade was a taker SELL

Output is [-1.0, 1.0]:
  Positive = taker buys dominate (bullish pressure)
  Negative = taker sells dominate (bearish pressure)
  Zero = balanced or insufficient data
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple


class TradeRecord(NamedTuple):
    """Single aggregated trade for the rolling window."""
    timestamp: float   # Unix time
    qty: float         # Trade quantity in BTC
    is_taker_buy: bool # True if taker was the buyer


@dataclass
class TakerRatioSignal:
    """Rolling taker buy/sell ratio signal.

    Accumulates individual Binance trades and computes the volume-weighted
    ratio of buyer-initiated trades to total volume over a rolling window.

    Args:
        window_secs: Rolling window duration in seconds.
        min_trades: Minimum trades needed before producing a signal.
        neutral_band: Buy ratio within 0.5 ± neutral_band produces zero
            signal (avoids noise on balanced flow).
        ema_alpha: Exponential smoothing factor for the ratio. Higher =
            more responsive, lower = smoother.
    """

    window_secs: float = 30.0
    min_trades: int = 20
    neutral_band: float = 0.05
    ema_alpha: float = 0.15

    # Internal state
    _trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    _ema_ratio: float = 0.5
    _last_signal: float = 0.0
    _total_buy_vol: float = 0.0
    _total_sell_vol: float = 0.0

    def reset(self) -> None:
        """Reset between windows."""
        self._trades.clear()
        self._ema_ratio = 0.5
        self._last_signal = 0.0
        self._total_buy_vol = 0.0
        self._total_sell_vol = 0.0

    def on_trade(self, qty: float, is_taker_buy: bool, timestamp: float = 0.0) -> None:
        """Ingest a single trade from the Binance stream.

        Called by BinanceFeed on every trade message. This is a hot path
        (~10-50 trades/sec for BTCUSDT), so kept lightweight.

        Args:
            qty: Trade quantity in BTC.
            is_taker_buy: True if taker was buyer (Binance m=True).
            timestamp: Trade timestamp (unix). Uses time.time() if 0.
        """
        if timestamp <= 0:
            timestamp = time.time()

        self._trades.append(TradeRecord(timestamp, qty, is_taker_buy))

        if is_taker_buy:
            self._total_buy_vol += qty
        else:
            self._total_sell_vol += qty

    def compute(self) -> float:
        """Compute the taker ratio signal from accumulated trades.

        Called once per tick (1/sec) by the strategy's on_tick().
        Prunes expired trades, calculates buy ratio, applies EMA
        smoothing, and maps to [-1, 1].

        Returns:
            Signal in [-1.0, 1.0]. Positive = buying pressure,
            negative = selling pressure.
        """
        now = time.time()
        cutoff = now - self.window_secs

        # Prune expired trades from the front
        while self._trades and self._trades[0].timestamp < cutoff:
            old = self._trades.popleft()
            if old.is_taker_buy:
                self._total_buy_vol -= old.qty
            else:
                self._total_sell_vol -= old.qty

        # Clamp to avoid float drift going negative
        self._total_buy_vol = max(self._total_buy_vol, 0.0)
        self._total_sell_vol = max(self._total_sell_vol, 0.0)

        if len(self._trades) < self.min_trades:
            self._last_signal = 0.0
            return 0.0

        total_vol = self._total_buy_vol + self._total_sell_vol
        if total_vol <= 0:
            self._last_signal = 0.0
            return 0.0

        # Raw buy ratio: 0.0 = all sells, 1.0 = all buys
        raw_ratio = self._total_buy_vol / total_vol

        # EMA smoothing
        self._ema_ratio += self.ema_alpha * (raw_ratio - self._ema_ratio)

        # Map to [-1, 1] with neutral dead zone
        # ratio=0.5 -> signal=0, ratio=1.0 -> signal=1.0, ratio=0.0 -> signal=-1.0
        deviation = self._ema_ratio - 0.5

        if abs(deviation) < self.neutral_band:
            signal = 0.0
        else:
            # Remove the neutral band from the deviation, then scale to [-1, 1]
            if deviation > 0:
                signal = min((deviation - self.neutral_band) / (0.5 - self.neutral_band), 1.0)
            else:
                signal = max((deviation + self.neutral_band) / (0.5 - self.neutral_band), -1.0)

        self._last_signal = signal
        return signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._last_signal

    @property
    def buy_ratio(self) -> float:
        """Current EMA-smoothed buy ratio (0.0-1.0)."""
        return self._ema_ratio

    @property
    def trade_count(self) -> int:
        """Number of trades in the current window."""
        return len(self._trades)
