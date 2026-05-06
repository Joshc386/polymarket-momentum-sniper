"""Binance real-time forced liquidation feed via WebSocket.

Connects to Binance Futures forceOrder stream for BTCUSDT.
Tracks every liquidation event in real-time and maintains rolling
statistics over a configurable window (default 5 minutes).

Data contract (LiquidationStats):
    - long_liq_volume: USD volume of long liquidations in window
    - short_liq_volume: USD volume of short liquidations in window
    - long_liq_count: Number of long liquidation events
    - short_liq_count: Number of short liquidation events
    - net_imbalance: Normalized [-1, 1] — positive = more shorts liquidated (bullish)
    - acceleration: Rate of change in liquidation volume (cascade detection)
    - largest_single: Size of the biggest single liquidation in window
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

from core import fast_json as json

logger = logging.getLogger(__name__)

# Binance Futures WebSocket — forced liquidation orders
FORCE_ORDER_STREAM = "wss://fstream.binance.com/ws/btcusdt@forceOrder"


@dataclass
class LiquidationEvent:
    """Single forced liquidation event."""
    timestamp: float          # Unix seconds
    side: str                 # "LONG" or "SHORT" (which side got liquidated)
    price: float              # Liquidation price
    quantity_btc: float       # Size in BTC
    volume_usd: float         # Size in USD (price * quantity)


@dataclass
class LiquidationStats:
    """Rolling statistics over the observation window."""
    # Volume
    long_liq_volume: float = 0.0     # USD liquidated from longs
    short_liq_volume: float = 0.0    # USD liquidated from shorts
    total_volume: float = 0.0

    # Counts
    long_liq_count: int = 0
    short_liq_count: int = 0
    total_count: int = 0

    # Derived
    net_imbalance: float = 0.0       # [-1, 1] positive = more shorts liq'd (bullish)
    acceleration: float = 0.0         # Volume acceleration (cascade indicator)
    largest_single: float = 0.0       # Biggest single liquidation USD
    avg_size: float = 0.0             # Average liquidation size

    # Recent burst detection
    burst_detected: bool = False      # True if sudden spike in liquidations
    burst_side: str = ""               # Which side is getting liquidated in burst

    is_valid: bool = False

    @property
    def is_significant(self) -> bool:
        """True if there's enough liquidation activity to be meaningful."""
        return self.total_volume > 100_000  # At least $100K in liquidations


class BinanceLiquidationFeed:
    """Real-time BTC forced liquidation stream from Binance Futures.

    Maintains a rolling window of liquidation events and computes
    statistics useful for the 5-minute prediction signal.

    Usage:
        feed = BinanceLiquidationFeed()
        task = asyncio.create_task(feed.start())
        stats = feed.stats  # Always-current LiquidationStats
    """

    def __init__(self, window_seconds: float = 300.0, burst_window: float = 30.0):
        """
        Args:
            window_seconds: Rolling window for main stats (default 5 min).
            burst_window: Short window for burst/cascade detection (default 30s).
        """
        self.window_seconds = window_seconds
        self.burst_window = burst_window

        # Event storage — keep events for the full window
        self._events: deque[LiquidationEvent] = deque()
        self._running = False
        self._connected = False
        self._last_event_time: float = 0.0

        # Pre-computed stats
        self.stats = LiquidationStats()

        # For acceleration calculation — track volume in successive time buckets
        self._bucket_size = 10.0  # 10-second buckets
        self._volume_buckets: deque[float] = deque(maxlen=30)  # 5 min of 10s buckets
        self._current_bucket_start: float = 0.0
        self._current_bucket_volume: float = 0.0

        # Burst detection thresholds
        self._burst_volume_threshold = 500_000  # $500K in burst_window = burst
        self._burst_count_threshold = 10         # 10+ liquidations in burst_window = burst

    async def start(self):
        """Start the WebSocket connection and listen for liquidation events."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    FORCE_ORDER_STREAM,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    self._connected = True
                    logger.info("Binance liquidation stream connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_message(msg)
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                self._connected = False
                if not self._running:
                    break
                logger.warning(f"Binance liq stream disconnected: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                self._connected = False
                if not self._running:
                    break
                logger.error(f"Binance liq stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def stop(self):
        self._running = False
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def reset(self):
        """Clear all events — call at window transitions."""
        self._events.clear()
        self._volume_buckets.clear()
        self._current_bucket_volume = 0.0
        self._current_bucket_start = 0.0
        self.stats = LiquidationStats()

    def _process_message(self, raw_msg: str):
        """Parse a forceOrder WebSocket message and update stats."""
        try:
            data = json.loads(raw_msg)
            order = data.get("o", {})

            # Side: SELL = long position liquidated, BUY = short position liquidated
            raw_side = order.get("S", "")
            if raw_side == "SELL":
                side = "LONG"    # A SELL order means a long was force-closed
            elif raw_side == "BUY":
                side = "SHORT"   # A BUY order means a short was force-closed
            else:
                return

            price = float(order.get("p", 0))
            quantity = float(order.get("q", 0))
            trade_time = float(order.get("T", 0)) / 1000.0  # ms to seconds

            if price <= 0 or quantity <= 0:
                return

            volume_usd = price * quantity

            event = LiquidationEvent(
                timestamp=trade_time,
                side=side,
                price=price,
                quantity_btc=quantity,
                volume_usd=volume_usd,
            )

            self._events.append(event)
            self._last_event_time = trade_time

            # Update volume buckets for acceleration tracking
            self._update_bucket(trade_time, volume_usd)

            # Recompute stats
            self._recompute_stats()

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse liquidation message: {e}")

    def _update_bucket(self, timestamp: float, volume: float):
        """Track volume in time buckets for acceleration calculation."""
        if self._current_bucket_start == 0.0:
            self._current_bucket_start = timestamp
            self._current_bucket_volume = volume
            return

        if timestamp - self._current_bucket_start >= self._bucket_size:
            # Finalize current bucket
            self._volume_buckets.append(self._current_bucket_volume)
            self._current_bucket_start = timestamp
            self._current_bucket_volume = volume
        else:
            self._current_bucket_volume += volume

    def _recompute_stats(self):
        """Recompute all rolling statistics from current events."""
        now = time.time()
        cutoff = now - self.window_seconds
        burst_cutoff = now - self.burst_window

        # Prune old events
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

        if not self._events:
            self.stats = LiquidationStats()
            return

        stats = LiquidationStats(is_valid=True)

        # Burst tracking
        burst_long_vol = 0.0
        burst_short_vol = 0.0
        burst_long_count = 0
        burst_short_count = 0

        for event in self._events:
            # Main window stats
            if event.side == "LONG":
                stats.long_liq_volume += event.volume_usd
                stats.long_liq_count += 1
            else:
                stats.short_liq_volume += event.volume_usd
                stats.short_liq_count += 1

            stats.largest_single = max(stats.largest_single, event.volume_usd)

            # Burst window stats
            if event.timestamp >= burst_cutoff:
                if event.side == "LONG":
                    burst_long_vol += event.volume_usd
                    burst_long_count += 1
                else:
                    burst_short_vol += event.volume_usd
                    burst_short_count += 1

        stats.total_volume = stats.long_liq_volume + stats.short_liq_volume
        stats.total_count = stats.long_liq_count + stats.short_liq_count
        stats.avg_size = stats.total_volume / stats.total_count if stats.total_count > 0 else 0.0

        # Net imbalance: positive = more shorts liquidated (bullish for price)
        if stats.total_volume > 0:
            stats.net_imbalance = (
                (stats.short_liq_volume - stats.long_liq_volume) / stats.total_volume
            )

        # Acceleration: compare recent bucket volume to older buckets
        if len(self._volume_buckets) >= 3:
            recent = list(self._volume_buckets)[-3:]    # Last 30 seconds
            older = list(self._volume_buckets)[:-3]     # Before that
            if older:
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                if older_avg > 0:
                    stats.acceleration = (recent_avg - older_avg) / older_avg
                elif recent_avg > 0:
                    stats.acceleration = 1.0  # Went from nothing to something

        # Burst detection
        burst_total_vol = burst_long_vol + burst_short_vol
        burst_total_count = burst_long_count + burst_short_count

        if (burst_total_vol >= self._burst_volume_threshold or
                burst_total_count >= self._burst_count_threshold):
            stats.burst_detected = True
            if burst_long_vol > burst_short_vol:
                stats.burst_side = "LONG"   # Longs getting liquidated → bearish
            else:
                stats.burst_side = "SHORT"  # Shorts getting liquidated → bullish

        self.stats = stats

    @property
    def status_str(self) -> str:
        """Short status string for dashboard display."""
        if not self._connected:
            return "(disconnected)"
        s = self.stats
        if not s.is_valid:
            return "(no events)"
        burst = " BURST!" if s.burst_detected else ""
        return (
            f"L:{s.long_liq_count}/${s.long_liq_volume/1000:.0f}K "
            f"S:{s.short_liq_count}/${s.short_liq_volume/1000:.0f}K "
            f"Imb:{s.net_imbalance:+.2f}{burst}"
        )
