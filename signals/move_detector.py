"""Move detector — detects significant Binance price moves in real-time.

Used by the Latency Arb strategy (Bot F). Identifies when Binance has moved
significantly but the CLOB odds haven't adjusted proportionally.

The edge is structural: Binance leads, CLOB reprices after a lag. Detecting
the move fast and acting before the CLOB adjusts is the entire strategy.

Gap detection works by tracking BOTH Binance price and CLOB midpoint over
time. When Binance moves X bps but the CLOB midpoint only shifted Y bps
in the same period, the gap is X - Y. This measures actual staleness
rather than comparing to a fixed 0.50 anchor.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectedMove:
    """A significant Binance price move with frozen detection state."""

    direction: str          # "up" or "down"
    move_bps: float         # Binance move magnitude in basis points
    move_usd: float         # Binance move magnitude in USD
    duration_secs: float    # How fast the move happened
    binance_price: float    # Binance price at detection
    clob_gap_bps: float     # Binance move minus CLOB repricing
    clob_shift_bps: float   # How much CLOB moved in the same period
    detected_at: float      # Unix timestamp of detection


class MoveDetector:
    """Detects significant Binance price moves from DataSnapshot ticks.

    Maintains a ring buffer of recent (timestamp, binance_price, clob_midpoint)
    ticks. Each tick, checks if Binance has moved significantly from N seconds
    ago, and whether the CLOB midpoint has moved proportionally.

    The gap = Binance move - CLOB shift. A large gap means the CLOB is stale.

    Args:
        threshold_bps: Minimum Binance move in basis points to trigger.
            Default 3.0 (~$24 at $80K BTC).
        lookback_secs: How far back to compare prices. Default 10s.
        cooldown_secs: Minimum seconds between detections in the same
            direction. Prevents counting one move multiple times.
        min_clob_gap_bps: Minimum gap (Binance move - CLOB shift) required.
            If the CLOB has already repriced, there's no latency to exploit.
    """

    def __init__(
        self,
        threshold_bps: float = 3.0,
        lookback_secs: float = 10.0,
        cooldown_secs: float = 20.0,
        min_clob_gap_bps: float = 1.0,
    ) -> None:
        self._threshold_bps = threshold_bps
        self._lookback_secs = lookback_secs
        self._cooldown_secs = cooldown_secs
        self._min_clob_gap_bps = min_clob_gap_bps

        # Ring buffer: (timestamp, binance_price, clob_midpoint)
        self._buffer: deque[tuple[float, float, float]] = deque(maxlen=120)

        # Last detection timestamps per direction (for cooldown)
        self._last_up_at: float = 0.0
        self._last_down_at: float = 0.0

        # Last detected move (for dashboard display)
        self.last_move: Optional[DetectedMove] = None

    def reset(self) -> None:
        """Reset state for a new window."""
        self._buffer.clear()
        self._last_up_at = 0.0
        self._last_down_at = 0.0
        self.last_move = None

    def update(
        self,
        binance_price: float,
        timestamp: float,
        clob_midpoint: float = 0.0,
    ) -> Optional[DetectedMove]:
        """Feed one tick. Returns DetectedMove if significant move detected.

        Args:
            binance_price: Current Binance BTC price.
            timestamp: Unix timestamp of this tick.
            clob_midpoint: YES midpoint from the CLOB orderbook (0-1).
                Tracked over time to measure how much the CLOB repriced
                relative to the Binance move.

        Returns:
            DetectedMove if a significant move was detected, else None.
        """
        if binance_price <= 0 or clob_midpoint <= 0:
            return None

        self._buffer.append((timestamp, binance_price, clob_midpoint))

        # Need at least lookback_secs of data
        cutoff = timestamp - self._lookback_secs
        old_entries = [
            (ts, px, mid) for ts, px, mid in self._buffer
            if ts <= cutoff and px > 0 and mid > 0
        ]
        if not old_entries:
            return None

        # Compare to the oldest entry in the lookback window
        ref_ts, ref_price, ref_midpoint = old_entries[0]

        # ── Binance move ──
        move_usd = binance_price - ref_price
        move_bps = abs(move_usd) / ref_price * 10_000 if ref_price > 0 else 0.0

        if move_bps < self._threshold_bps:
            return None

        direction = "up" if move_usd > 0 else "down"

        # Cooldown: don't double-detect the same directional move
        if direction == "up" and (timestamp - self._last_up_at) < self._cooldown_secs:
            return None
        if direction == "down" and (timestamp - self._last_down_at) < self._cooldown_secs:
            return None

        # ── CLOB shift over the same period ──
        # How much the CLOB midpoint moved in the direction of the
        # Binance move. Convert midpoint delta to bps-equivalent.
        #
        # YES midpoint shift: (current_mid - old_mid) * 10000
        #   Positive = CLOB shifted toward UP
        #   Negative = CLOB shifted toward DOWN
        midpoint_delta = clob_midpoint - ref_midpoint
        if direction == "up":
            clob_shift_bps = midpoint_delta * 10_000
        else:
            # For down-move, a negative midpoint_delta (shifting down)
            # means the CLOB IS repricing, so we flip the sign
            clob_shift_bps = -midpoint_delta * 10_000

        # If the CLOB moved AGAINST the Binance direction, the CLOB
        # disagrees with the move — don't trade against the CLOB
        if clob_shift_bps < 0:
            return None

        # Gap = how much of the Binance move the CLOB hasn't priced in
        clob_gap_bps = max(0, move_bps - clob_shift_bps)

        if clob_gap_bps < self._min_clob_gap_bps:
            return None

        # Record detection
        if direction == "up":
            self._last_up_at = timestamp
        else:
            self._last_down_at = timestamp

        duration = timestamp - ref_ts

        move = DetectedMove(
            direction=direction,
            move_bps=move_bps,
            move_usd=abs(move_usd),
            duration_secs=duration,
            binance_price=binance_price,
            clob_gap_bps=clob_gap_bps,
            clob_shift_bps=clob_shift_bps,
            detected_at=timestamp,
        )

        self.last_move = move

        logger.info(
            f"[MOVE] Binance {direction} {move_bps:.1f}bps "
            f"(${abs(move_usd):.0f}) in {duration:.1f}s | "
            f"CLOB shift: {clob_shift_bps:.1f}bps, "
            f"gap: {clob_gap_bps:.1f}bps "
            f"(mid: {ref_midpoint:.3f}->{clob_midpoint:.3f})"
        )

        return move
