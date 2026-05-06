"""Latency measurement tool — quantifies the double-latency arb opportunity.

Measures three latency gaps from a single price move event:
  Latency 1: Binance move → Oracle catches up to frozen target price
  Latency 2: Binance move → Polymarket odds adjust (independent of Oracle)
  Latency 3: Oracle update → Polymarket odds adjust (sequential)

The key insight: when Binance spikes, the Oracle and market odds both lag.
The bot can trade in that gap. This tool measures how wide the gap is.

Design principles:
- Freeze the target: when a move is detected, record the Binance price at
  that instant. Track Oracle and odds progress toward THAT price, not the
  continuously-changing Binance price.
- Track independently: Oracle and odds catchup are measured from detection
  time, not sequentially. Market makers may front-run the Oracle.
- Percentage-based thresholds: $15 on $80K BTC is noise. Use basis points.
- Separate sharp vs gradual moves: a $50 spike in 2s has a different
  catchup profile than $50 over 30s.

Usage (from multi_runner):
    logger = LatencyLogger(output_dir="data_runtime")
    # In tick loop:
    logger.log_tick(snapshot, binance_price_time, coinbase_price_time)
    # On shutdown:
    logger.print_summary()
    logger.close()
"""

import csv
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PriceMove:
    """A detected price move on Binance with frozen target tracking."""

    # ── Detection state (frozen at detection time) ──
    detected_at: float              # Unix timestamp when move was detected
    source: str                     # "binance"
    direction: str                  # "up" or "down"
    move_size_usd: float            # Absolute price change
    move_bps: float                 # Move size in basis points
    move_duration_secs: float       # How fast the move happened
    target_price: float             # Binance price at detection (FROZEN TARGET)
    oracle_price_at_detection: float  # Oracle price when move detected
    oracle_gap_at_detection: float  # target_price - oracle_price (signed)
    yes_ask_at_detection: float     # YES best ask at detection
    no_ask_at_detection: float      # NO best ask at detection
    yes_mid_at_detection: float     # YES midpoint at detection
    seconds_remaining: float        # Window time remaining at detection

    # ── Oracle catchup (Latency 1) ──
    oracle_catchup_at: Optional[float] = None
    oracle_lag_secs: Optional[float] = None
    oracle_final_gap_usd: Optional[float] = None
    oracle_catchup_pct: Optional[float] = None  # 1.0 = fully caught up

    # ── Odds catchup (Latency 2 — independent, from detection time) ──
    odds_catchup_at: Optional[float] = None
    odds_lag_from_move_secs: Optional[float] = None  # From Binance move to odds shift
    odds_shift: Optional[float] = None  # Change in relevant token price

    # ── Oracle-to-odds (Latency 3 — sequential) ──
    odds_lag_from_oracle_secs: Optional[float] = None  # Oracle catchup → odds shift

    # ── Resolution ──
    timed_out: bool = False
    peak_oracle_gap_usd: float = 0.0  # Largest gap seen during tracking


@dataclass
class TickRecord:
    """One tick of raw price data."""
    timestamp: float
    binance_price: float
    binance_price_time: float       # Exchange-side timestamp
    coinbase_price: float
    coinbase_price_time: float
    oracle_price: float
    oracle_updated_at: float        # On-chain updatedAt timestamp
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    seconds_remaining: float
    market_slug: str


class LatencyLogger:
    """Logs tick data and measures latency between price sources.

    Detects significant price moves on Binance, then tracks how long
    the Oracle and Polymarket odds take to catch up to the FROZEN
    detection-time price. Both are tracked independently.

    Args:
        output_dir: Directory for CSV output files.
        move_threshold_bps: Minimum move in basis points to trigger
            tracking (default 4bps = 0.04%). At $80K BTC this is ~$32.
        move_window_secs: Time window to detect moves in (default 5s).
        catchup_threshold_pct: Fraction of gap that must close to count
            as "caught up" (default 0.60 = 60%).
        max_tracking_secs: Stop tracking a move after this many seconds
            (default 120s — captures the full window).
        min_gap_usd: Minimum Oracle gap at detection to track. Filters
            out moves where Oracle was already caught up (default $10).
        cooldown_secs: Minimum seconds between detecting moves in the
            same direction (default 15s). Prevents double-counting.
    """

    def __init__(
        self,
        output_dir: str = "data_runtime",
        move_threshold_bps: float = 4.0,
        move_window_secs: float = 5.0,
        catchup_threshold_pct: float = 0.60,
        max_tracking_secs: float = 120.0,
        min_gap_usd: float = 10.0,
        cooldown_secs: float = 15.0,
    ) -> None:
        self._output_dir = output_dir
        self._move_threshold_bps = move_threshold_bps
        self._move_window = move_window_secs
        self._catchup_threshold = catchup_threshold_pct
        self._max_tracking = max_tracking_secs
        self._min_gap_usd = min_gap_usd
        self._cooldown_secs = cooldown_secs

        # Ring buffer of recent ticks (last 120 seconds at ~1/sec)
        self._tick_buffer: deque[TickRecord] = deque(maxlen=180)

        # Active move events being tracked
        self._active_moves: list[PriceMove] = []

        # Completed move events
        self._completed_moves: list[PriceMove] = []

        # Raw tick CSV writer
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._tick_csv_path = os.path.join(output_dir, f"latency_ticks_{ts}.csv")
        self._events_csv_path = os.path.join(output_dir, f"latency_events_{ts}.csv")

        self._tick_file = open(self._tick_csv_path, "w", newline="")
        self._tick_writer = csv.writer(self._tick_file)
        self._tick_writer.writerow([
            "timestamp", "binance_price", "binance_price_time",
            "coinbase_price", "coinbase_price_time",
            "oracle_price", "oracle_updated_at",
            "oracle_age_secs", "binance_oracle_gap_usd", "binance_oracle_gap_bps",
            "yes_ask", "yes_bid", "no_ask", "no_bid",
            "yes_mid", "seconds_remaining", "market_slug",
        ])

        self._events_file = open(self._events_csv_path, "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow([
            "detected_at", "detected_time_utc", "source", "direction",
            "move_size_usd", "move_bps", "move_duration_secs",
            "target_price", "oracle_price_at_detection", "oracle_gap_at_detection",
            "yes_ask_at_detection", "no_ask_at_detection", "yes_mid_at_detection",
            "seconds_remaining",
            "oracle_lag_secs", "oracle_catchup_pct", "oracle_final_gap_usd",
            "peak_oracle_gap_usd",
            "odds_lag_from_move_secs", "odds_shift",
            "odds_lag_from_oracle_secs",
            "timed_out",
        ])

        self._tick_count = 0
        self._start_time = time.time()

        logger.info(
            f"LatencyLogger started — threshold: {move_threshold_bps:.0f}bps, "
            f"window: {move_window_secs:.0f}s, "
            f"ticks: {self._tick_csv_path}"
        )

    def log_tick(
        self,
        snapshot,
        binance_price_time: float = 0.0,
        coinbase_price_time: float = 0.0,
    ) -> None:
        """Log one tick of price data and check for latency events.

        Args:
            snapshot: DataSnapshot from the multi-bot runner.
            binance_price_time: Exchange-side timestamp of Binance price.
            coinbase_price_time: Timestamp of Coinbase price.
        """
        now = snapshot.timestamp or time.time()
        ob = snapshot.orderbook

        tick = TickRecord(
            timestamp=now,
            binance_price=snapshot.binance_price,
            binance_price_time=binance_price_time,
            coinbase_price=snapshot.coinbase_price,
            coinbase_price_time=coinbase_price_time,
            oracle_price=snapshot.oracle_price,
            oracle_updated_at=snapshot.oracle_updated_at,
            yes_ask=ob.yes_best_ask if ob else 0.0,
            yes_bid=ob.yes_best_bid if ob else 0.0,
            no_ask=ob.no_best_ask if ob else 0.0,
            no_bid=ob.no_best_bid if ob else 0.0,
            seconds_remaining=snapshot.seconds_remaining,
            market_slug=snapshot.market_slug,
        )

        self._tick_buffer.append(tick)
        self._tick_count += 1

        # Write raw tick to CSV
        oracle_age = now - snapshot.oracle_updated_at if snapshot.oracle_updated_at > 0 else 0.0
        gap_usd = snapshot.binance_price - snapshot.oracle_price if snapshot.oracle_price > 0 else 0.0
        gap_bps = (gap_usd / snapshot.oracle_price * 10_000) if snapshot.oracle_price > 0 else 0.0
        yes_mid = (tick.yes_ask + tick.yes_bid) / 2.0 if tick.yes_ask > 0 and tick.yes_bid > 0 else 0.0

        self._tick_writer.writerow([
            f"{now:.3f}", f"{snapshot.binance_price:.2f}", f"{binance_price_time:.3f}",
            f"{snapshot.coinbase_price:.2f}", f"{coinbase_price_time:.3f}",
            f"{snapshot.oracle_price:.2f}", f"{snapshot.oracle_updated_at:.3f}",
            f"{oracle_age:.2f}", f"{gap_usd:.2f}", f"{gap_bps:.1f}",
            f"{tick.yes_ask:.3f}", f"{tick.yes_bid:.3f}",
            f"{tick.no_ask:.3f}", f"{tick.no_bid:.3f}",
            f"{yes_mid:.3f}", f"{snapshot.seconds_remaining:.0f}",
            snapshot.market_slug,
        ])

        # Flush every 30 ticks (~30 seconds)
        if self._tick_count % 30 == 0:
            self._tick_file.flush()

        # Detect new moves
        self._detect_moves(tick)

        # Update active move tracking
        self._update_active_moves(tick)

    def _detect_moves(self, tick: TickRecord) -> None:
        """Check if Binance moved significantly in the recent window.

        Uses basis points (not USD) so the threshold scales with BTC price.
        Requires a minimum Oracle gap at detection — if Oracle is already
        caught up, there's no latency to measure.
        """
        if tick.binance_price <= 0 or tick.oracle_price <= 0:
            return

        # Look back move_window seconds for reference price
        cutoff = tick.timestamp - self._move_window
        old_ticks = [t for t in self._tick_buffer if t.timestamp <= cutoff and t.binance_price > 0]
        if not old_ticks:
            return

        # Use the oldest tick in the window as reference
        ref = old_ticks[0]
        move_usd = tick.binance_price - ref.binance_price
        move_bps = abs(move_usd) / ref.binance_price * 10_000 if ref.binance_price > 0 else 0.0

        if move_bps < self._move_threshold_bps:
            return

        direction = "up" if move_usd > 0 else "down"

        # Cooldown: don't double-detect moves in the same direction
        for m in self._active_moves:
            if (tick.timestamp - m.detected_at < self._cooldown_secs
                    and m.direction == direction):
                return
        # Also check recent completed moves
        for m in self._completed_moves[-10:]:
            if (tick.timestamp - m.detected_at < self._cooldown_secs
                    and m.direction == direction):
                return

        # Require minimum Oracle gap — if Oracle is already at Binance,
        # there's no latency to measure
        oracle_gap = tick.binance_price - tick.oracle_price  # Signed
        if abs(oracle_gap) < self._min_gap_usd:
            return

        # For "up" moves, Oracle should be BELOW Binance (positive gap)
        # For "down" moves, Oracle should be ABOVE Binance (negative gap)
        if direction == "up" and oracle_gap < 0:
            return  # Oracle already ahead — not a lag event
        if direction == "down" and oracle_gap > 0:
            return  # Oracle already ahead — not a lag event

        # Calculate odds midpoint at detection
        yes_mid = 0.0
        if tick.yes_ask > 0 and tick.yes_bid > 0:
            yes_mid = (tick.yes_ask + tick.yes_bid) / 2.0

        move_duration = tick.timestamp - ref.timestamp

        move = PriceMove(
            detected_at=tick.timestamp,
            source="binance",
            direction=direction,
            move_size_usd=abs(move_usd),
            move_bps=move_bps,
            move_duration_secs=move_duration,
            target_price=tick.binance_price,         # FROZEN — never changes
            oracle_price_at_detection=tick.oracle_price,
            oracle_gap_at_detection=oracle_gap,
            yes_ask_at_detection=tick.yes_ask,
            no_ask_at_detection=tick.no_ask,
            yes_mid_at_detection=yes_mid,
            seconds_remaining=tick.seconds_remaining,
            peak_oracle_gap_usd=abs(oracle_gap),
        )

        self._active_moves.append(move)
        logger.info(
            f"[LATENCY] Move detected: Binance {direction} "
            f"${move.move_size_usd:.0f} ({move.move_bps:.1f}bps) in "
            f"{move_duration:.1f}s | "
            f"Target: ${move.target_price:,.2f} | "
            f"Oracle gap: ${oracle_gap:+.2f} | "
            f"YES={tick.yes_ask:.3f} NO={tick.no_ask:.3f} | "
            f"{tick.seconds_remaining:.0f}s left"
        )

    def _update_active_moves(self, tick: TickRecord) -> None:
        """Track Oracle and odds catch-up against frozen target prices.

        Oracle catchup: measured against the FROZEN target_price — not the
        live Binance price. If Binance was $80,050 at detection and Oracle
        was $80,000, we track how long until Oracle reaches ~$80,050.

        Odds catchup: measured independently from detection time. Market
        makers may adjust odds before the Oracle updates. We track when
        the YES/NO midpoint shifts in the expected direction by at least
        $0.01 (1 cent).
        """
        still_active = []

        for move in self._active_moves:
            elapsed = tick.timestamp - move.detected_at

            # Track peak gap (informational — shows how wide the gap got)
            current_gap = abs(tick.binance_price - tick.oracle_price)
            if current_gap > move.peak_oracle_gap_usd:
                move.peak_oracle_gap_usd = current_gap

            # ── Timeout ──
            if elapsed > self._max_tracking:
                move.timed_out = True
                # Record whatever state we have
                if move.oracle_lag_secs is None:
                    move.oracle_lag_secs = self._max_tracking
                    gap_remaining = tick.oracle_price - move.oracle_price_at_detection
                    total_gap = move.target_price - move.oracle_price_at_detection
                    if abs(total_gap) > 0.01:
                        move.oracle_catchup_pct = abs(gap_remaining / total_gap)
                    else:
                        move.oracle_catchup_pct = 1.0
                    move.oracle_final_gap_usd = move.target_price - tick.oracle_price
                if move.odds_lag_from_move_secs is None:
                    move.odds_lag_from_move_secs = self._max_tracking
                    move.odds_shift = 0.0
                self._complete_move(move, tick)
                continue

            # ── Oracle catchup (Latency 1) — against FROZEN target ──
            if move.oracle_catchup_at is None:
                # How much has Oracle moved toward the target?
                oracle_movement = tick.oracle_price - move.oracle_price_at_detection
                total_needed = move.target_price - move.oracle_price_at_detection

                if abs(total_needed) > 0.01:
                    if move.direction == "up":
                        # Oracle needs to rise toward target
                        catchup_pct = oracle_movement / total_needed if total_needed > 0 else 0.0
                    else:
                        # Oracle needs to fall toward target (both negative)
                        catchup_pct = oracle_movement / total_needed if total_needed < 0 else 0.0

                    catchup_pct = max(0.0, min(2.0, catchup_pct))  # Cap at 200% (overshoot)

                    if catchup_pct >= self._catchup_threshold:
                        move.oracle_catchup_at = tick.timestamp
                        move.oracle_lag_secs = elapsed
                        move.oracle_catchup_pct = catchup_pct
                        move.oracle_final_gap_usd = move.target_price - tick.oracle_price
                        logger.info(
                            f"[LATENCY] Oracle caught up in {elapsed:.1f}s "
                            f"({catchup_pct:.0%}) | "
                            f"Target: ${move.target_price:,.2f}, "
                            f"Oracle: ${tick.oracle_price:,.2f}"
                        )

            # ── Odds catchup (Latency 2) — independent, from detection ──
            if move.odds_catchup_at is None:
                if tick.yes_ask > 0 and tick.yes_bid > 0:
                    current_mid = (tick.yes_ask + tick.yes_bid) / 2.0

                    if move.yes_mid_at_detection > 0:
                        mid_shift = current_mid - move.yes_mid_at_detection

                        # For up-moves: YES mid should increase (more likely BTC went up)
                        # For down-moves: YES mid should decrease
                        if move.direction == "up" and mid_shift >= 0.01:
                            move.odds_catchup_at = tick.timestamp
                            move.odds_lag_from_move_secs = elapsed
                            move.odds_shift = mid_shift
                        elif move.direction == "down" and mid_shift <= -0.01:
                            move.odds_catchup_at = tick.timestamp
                            move.odds_lag_from_move_secs = elapsed
                            move.odds_shift = mid_shift

                        if move.odds_catchup_at is not None:
                            # Calculate Latency 3 if Oracle already caught up
                            if move.oracle_catchup_at is not None:
                                move.odds_lag_from_oracle_secs = (
                                    move.odds_catchup_at - move.oracle_catchup_at
                                )
                            logger.info(
                                f"[LATENCY] Odds adjusted in {elapsed:.1f}s "
                                f"(mid shift: {move.odds_shift:+.3f}) | "
                                f"Oracle lag was: "
                                f"{move.oracle_lag_secs:.1f}s"
                                if move.oracle_lag_secs is not None
                                else f"[LATENCY] Odds adjusted in {elapsed:.1f}s "
                                     f"(mid shift: {move.odds_shift:+.3f}) | "
                                     f"Oracle: not yet caught up"
                            )

            # ── Both resolved — complete the event ──
            if move.oracle_catchup_at is not None and move.odds_catchup_at is not None:
                # If Oracle resolved but odds didn't get Latency 3 yet
                if move.odds_lag_from_oracle_secs is None:
                    move.odds_lag_from_oracle_secs = (
                        move.odds_catchup_at - move.oracle_catchup_at
                    )
                self._complete_move(move, tick)
                continue

            still_active.append(move)

        self._active_moves = still_active

    def _complete_move(self, move: PriceMove, tick: TickRecord) -> None:
        """Write completed move event to CSV and store."""
        self._completed_moves.append(move)

        detected_utc = datetime.fromtimestamp(
            move.detected_at, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        self._events_writer.writerow([
            f"{move.detected_at:.3f}", detected_utc,
            move.source, move.direction,
            f"{move.move_size_usd:.2f}", f"{move.move_bps:.1f}",
            f"{move.move_duration_secs:.2f}",
            f"{move.target_price:.2f}", f"{move.oracle_price_at_detection:.2f}",
            f"{move.oracle_gap_at_detection:.2f}",
            f"{move.yes_ask_at_detection:.3f}", f"{move.no_ask_at_detection:.3f}",
            f"{move.yes_mid_at_detection:.3f}",
            f"{move.seconds_remaining:.0f}",
            f"{move.oracle_lag_secs:.2f}" if move.oracle_lag_secs is not None else "",
            f"{move.oracle_catchup_pct:.2%}" if move.oracle_catchup_pct is not None else "",
            f"{move.oracle_final_gap_usd:.2f}" if move.oracle_final_gap_usd is not None else "",
            f"{move.peak_oracle_gap_usd:.2f}",
            f"{move.odds_lag_from_move_secs:.2f}" if move.odds_lag_from_move_secs is not None else "",
            f"{move.odds_shift:.3f}" if move.odds_shift is not None else "",
            f"{move.odds_lag_from_oracle_secs:.2f}" if move.odds_lag_from_oracle_secs is not None else "",
            "Y" if move.timed_out else "N",
        ])
        self._events_file.flush()

        status = "TIMEOUT" if move.timed_out else "COMPLETE"
        oracle_str = f"{move.oracle_lag_secs:.1f}s" if move.oracle_lag_secs is not None else "N/A"
        odds_str = f"{move.odds_lag_from_move_secs:.1f}s" if move.odds_lag_from_move_secs is not None else "N/A"
        logger.info(
            f"[LATENCY] [{status}] {move.direction} ${move.move_size_usd:.0f} "
            f"({move.move_bps:.1f}bps) | "
            f"Oracle: {oracle_str}, Odds: {odds_str}, "
            f"Peak gap: ${move.peak_oracle_gap_usd:.0f}"
        )

    def get_stats(self) -> dict:
        """Return summary statistics of measured latencies.

        Separates resolved moves from timeouts. Only resolved moves
        contribute to latency statistics.
        """
        if not self._completed_moves:
            return {
                "total_moves": 0,
                "resolved": 0,
                "timed_out": 0,
                "active": len(self._active_moves),
                "message": "No moves detected yet — need price moves > "
                           f"{self._move_threshold_bps:.0f}bps in "
                           f"{self._move_window:.0f}s with Oracle gap > "
                           f"${self._min_gap_usd:.0f}",
            }

        resolved = [m for m in self._completed_moves if not m.timed_out]
        timed_out = [m for m in self._completed_moves if m.timed_out]

        stats: dict = {
            "total_moves": len(self._completed_moves),
            "resolved": len(resolved),
            "timed_out": len(timed_out),
            "active": len(self._active_moves),
            "timeout_rate_pct": len(timed_out) / len(self._completed_moves) * 100,
            "runtime_mins": (time.time() - self._start_time) / 60.0,
        }

        # ── Latency 1: Binance → Oracle (resolved only) ──
        oracle_lags = [
            m.oracle_lag_secs for m in resolved
            if m.oracle_lag_secs is not None
        ]
        if oracle_lags:
            oracle_lags.sort()
            n = len(oracle_lags)
            stats["oracle_lag"] = {
                "count": n,
                "mean_secs": sum(oracle_lags) / n,
                "median_secs": oracle_lags[n // 2],
                "min_secs": min(oracle_lags),
                "max_secs": max(oracle_lags),
                "p10_secs": oracle_lags[max(0, int(n * 0.1))],
                "p90_secs": oracle_lags[min(n - 1, int(n * 0.9))],
            }

        # ── Latency 2: Binance → Odds (resolved only) ──
        odds_lags = [
            m.odds_lag_from_move_secs for m in resolved
            if m.odds_lag_from_move_secs is not None
        ]
        if odds_lags:
            odds_lags.sort()
            n = len(odds_lags)
            stats["odds_lag_from_move"] = {
                "count": n,
                "mean_secs": sum(odds_lags) / n,
                "median_secs": odds_lags[n // 2],
                "min_secs": min(odds_lags),
                "max_secs": max(odds_lags),
                "p10_secs": odds_lags[max(0, int(n * 0.1))],
                "p90_secs": odds_lags[min(n - 1, int(n * 0.9))],
            }

        # ── Latency 3: Oracle → Odds (resolved, may be negative if odds led) ──
        oracle_to_odds = [
            m.odds_lag_from_oracle_secs for m in resolved
            if m.odds_lag_from_oracle_secs is not None
        ]
        if oracle_to_odds:
            oracle_to_odds.sort()
            n = len(oracle_to_odds)
            negative_count = sum(1 for x in oracle_to_odds if x < 0)
            stats["oracle_to_odds"] = {
                "count": n,
                "mean_secs": sum(oracle_to_odds) / n,
                "median_secs": oracle_to_odds[n // 2],
                "min_secs": min(oracle_to_odds),
                "max_secs": max(oracle_to_odds),
                "odds_led_oracle_count": negative_count,
                "odds_led_oracle_pct": negative_count / n * 100 if n > 0 else 0,
            }

        # ── Move size distribution ──
        move_sizes = [m.move_bps for m in resolved]
        if move_sizes:
            move_sizes.sort()
            n = len(move_sizes)
            stats["move_sizes_bps"] = {
                "mean": sum(move_sizes) / n,
                "median": move_sizes[n // 2],
                "min": min(move_sizes),
                "max": max(move_sizes),
            }

        # ── Peak gaps ──
        peak_gaps = [m.peak_oracle_gap_usd for m in resolved]
        if peak_gaps:
            stats["peak_oracle_gap_usd"] = {
                "mean": sum(peak_gaps) / len(peak_gaps),
                "max": max(peak_gaps),
            }

        return stats

    def print_summary(self) -> None:
        """Print a human-readable summary of latency measurements."""
        stats = self.get_stats()
        runtime = stats.get("runtime_mins", 0)

        print("\n" + "=" * 60)
        print("  LATENCY MEASUREMENT SUMMARY (v2)")
        print("=" * 60)
        print(f"  Runtime: {runtime:.1f} minutes")
        print(f"  Ticks logged: {self._tick_count}")
        print(f"  Moves detected: {stats['total_moves']}")
        print(f"  ├─ Resolved: {stats.get('resolved', 0)}")
        print(f"  ├─ Timed out: {stats.get('timed_out', 0)} "
              f"({stats.get('timeout_rate_pct', 0):.0f}%)")
        print(f"  └─ Still active: {stats.get('active', 0)}")

        if "move_sizes_bps" in stats:
            ms = stats["move_sizes_bps"]
            print(f"\n  ── Move Sizes (resolved) ──")
            print(f"  Mean:   {ms['mean']:.1f}bps")
            print(f"  Median: {ms['median']:.1f}bps")
            print(f"  Range:  {ms['min']:.1f} – {ms['max']:.1f}bps")

        if "oracle_lag" in stats:
            ol = stats["oracle_lag"]
            print(f"\n  ── Latency 1: Binance → Oracle ──")
            print(f"  Samples:  {ol['count']}")
            print(f"  Mean:     {ol['mean_secs']:.2f}s")
            print(f"  Median:   {ol['median_secs']:.2f}s")
            print(f"  P10:      {ol['p10_secs']:.2f}s")
            print(f"  P90:      {ol['p90_secs']:.2f}s")
            print(f"  Range:    {ol['min_secs']:.2f}s – {ol['max_secs']:.2f}s")

        if "odds_lag_from_move" in stats:
            odl = stats["odds_lag_from_move"]
            print(f"\n  ── Latency 2: Binance → Market Odds ──")
            print(f"  Samples:  {odl['count']}")
            print(f"  Mean:     {odl['mean_secs']:.2f}s")
            print(f"  Median:   {odl['median_secs']:.2f}s")
            print(f"  P10:      {odl['p10_secs']:.2f}s")
            print(f"  P90:      {odl['p90_secs']:.2f}s")
            print(f"  Range:    {odl['min_secs']:.2f}s – {odl['max_secs']:.2f}s")

        if "oracle_to_odds" in stats:
            oto = stats["oracle_to_odds"]
            print(f"\n  ── Latency 3: Oracle → Market Odds ──")
            print(f"  Samples:  {oto['count']}")
            print(f"  Mean:     {oto['mean_secs']:+.2f}s")
            print(f"  Median:   {oto['median_secs']:+.2f}s")
            print(f"  Odds LED Oracle: {oto['odds_led_oracle_count']}/"
                  f"{oto['count']} ({oto['odds_led_oracle_pct']:.0f}%)")
            if oto["odds_led_oracle_pct"] > 50:
                print(f"  ⚠ Odds adjust BEFORE Oracle most of the time!")
                print(f"    → Market makers are watching Binance directly")

        if "peak_oracle_gap_usd" in stats:
            pg = stats["peak_oracle_gap_usd"]
            print(f"\n  ── Peak Oracle Gaps ──")
            print(f"  Mean:  ${pg['mean']:.0f}")
            print(f"  Max:   ${pg['max']:.0f}")

        if "oracle_lag" in stats and "odds_lag_from_move" in stats:
            total_mean = stats["odds_lag_from_move"]["mean_secs"]
            print(f"\n  ── Exploitable Window (Binance → Odds) ──")
            print(f"  Mean: {total_mean:.2f}s")
            print(f"  This is how long you have to trade after detecting")
            print(f"  a Binance move before the odds fully adjust.")

        print(f"\n  CSV files:")
        print(f"    Ticks:  {self._tick_csv_path}")
        print(f"    Events: {self._events_csv_path}")
        print("=" * 60 + "\n")

    def close(self) -> None:
        """Flush and close CSV files."""
        try:
            self._tick_file.close()
        except Exception:
            pass
        try:
            self._events_file.close()
        except Exception:
            pass
        logger.info("LatencyLogger closed")
