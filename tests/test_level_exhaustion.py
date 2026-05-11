"""Tests for Layer 10 Level Exhaustion Signal.

Verifies:
Phase 1 — Gap detection:
- Gaps >= 2 cents detected on bid/ask sides
- Gaps < threshold ignored
- Empty books produce no gaps
- YES/NO cross-referencing into support/resistance
- Dust orders filtered, depth limits respected
- worst_gap_asymmetry returns correct values

Phase 2 — Wall depletion + full signal:
- Wall depleting behind a gap produces directional signal
- No signal without a gap (thinning alone is not enough)
- No signal without depletion (gap alone is not enough)
- Depletion below threshold does not fire
- Walls below min_wall_size are ignored
- Signal decays when conditions stop
- Signal suppressed late in window (min_seconds_remaining)
- Reset clears all state
- WallState reporting for dashboard
"""

import pytest
from dataclasses import dataclass

from signals.level_exhaustion import (
    GapInfo,
    LevelExhaustionSignal,
    detect_gaps,
    detect_book_gaps,
    worst_gap_asymmetry,
)


@dataclass
class MockLevel:
    """Minimal mock of OrderLevel."""
    price: float
    size: float


# ── Phase 1: Gap detection ─────────────────────────────────────


class TestBidGapDetection:
    """Gaps in the bid side (support gaps -> bearish if broken)."""

    def test_gap_detected_between_bid_levels(self) -> None:
        bids = [MockLevel(0.50, 100), MockLevel(0.47, 200)]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 1
        assert gaps[0].side == "bid"
        assert gaps[0].gap_cents == pytest.approx(0.03, abs=0.001)
        assert gaps[0].top_price == 0.50
        assert gaps[0].next_price == 0.47

    def test_no_gap_when_levels_adjacent(self) -> None:
        bids = [MockLevel(0.50, 100), MockLevel(0.49, 200)]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 0

    def test_exact_threshold_detected(self) -> None:
        bids = [MockLevel(0.50, 100), MockLevel(0.48, 200)]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 1
        assert gaps[0].gap_cents == pytest.approx(0.02, abs=0.001)

    def test_multiple_gaps(self) -> None:
        bids = [
            MockLevel(0.50, 100),
            MockLevel(0.47, 50),
            MockLevel(0.44, 200),
        ]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 2

    def test_levels_skipped_calculation(self) -> None:
        bids = [MockLevel(0.50, 100), MockLevel(0.47, 200)]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert gaps[0].levels_skipped == 2

    def test_large_gap(self) -> None:
        bids = [MockLevel(0.55, 300), MockLevel(0.50, 100)]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 1
        assert gaps[0].gap_cents == pytest.approx(0.05, abs=0.001)
        assert gaps[0].levels_skipped == 4
        assert gaps[0].top_size == 300
        assert gaps[0].next_size == 100


class TestAskGapDetection:
    """Gaps in the ask side (resistance gaps -> bullish if broken)."""

    def test_gap_detected_between_ask_levels(self) -> None:
        asks = [MockLevel(0.52, 150), MockLevel(0.56, 100)]
        gaps = detect_gaps(bid_levels=[], ask_levels=asks, min_gap_cents=0.02)
        assert len(gaps) == 1
        assert gaps[0].side == "ask"
        assert gaps[0].gap_cents == pytest.approx(0.04, abs=0.001)

    def test_no_gap_when_tight(self) -> None:
        asks = [MockLevel(0.52, 100), MockLevel(0.53, 100)]
        gaps = detect_gaps(bid_levels=[], ask_levels=asks, min_gap_cents=0.02)
        assert len(gaps) == 0


class TestFiltering:
    """Dust orders and depth limits."""

    def test_dust_orders_skipped(self) -> None:
        bids = [
            MockLevel(0.50, 100),
            MockLevel(0.49, 2),
            MockLevel(0.47, 200),
        ]
        gaps = detect_gaps(
            bid_levels=bids, ask_levels=[], min_gap_cents=0.02, min_level_size=5.0
        )
        assert len(gaps) == 1
        assert gaps[0].gap_cents == pytest.approx(0.03, abs=0.001)

    def test_max_depth_limits_scan(self) -> None:
        bids = [
            MockLevel(0.50, 100),
            MockLevel(0.49, 100),
            MockLevel(0.48, 100),
            MockLevel(0.45, 100),
        ]
        gaps = detect_gaps(
            bid_levels=bids, ask_levels=[], min_gap_cents=0.02, max_depth_levels=3
        )
        assert len(gaps) == 0

    def test_custom_min_gap(self) -> None:
        bids = [MockLevel(0.50, 100), MockLevel(0.47, 100)]
        assert len(detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.05)) == 0
        assert len(detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)) == 1


class TestEdgeCases:
    """Empty books, single levels, etc."""

    def test_empty_books(self) -> None:
        assert detect_gaps(bid_levels=[], ask_levels=[]) == []

    def test_single_bid_level(self) -> None:
        assert detect_gaps(bid_levels=[MockLevel(0.50, 100)], ask_levels=[]) == []

    def test_single_ask_level(self) -> None:
        assert detect_gaps(bid_levels=[], ask_levels=[MockLevel(0.52, 100)]) == []

    def test_all_dust(self) -> None:
        bids = [MockLevel(0.50, 1), MockLevel(0.40, 1)]
        assert detect_gaps(bid_levels=bids, ask_levels=[], min_level_size=5.0) == []

    def test_sorted_by_gap_size(self) -> None:
        bids = [
            MockLevel(0.50, 100),
            MockLevel(0.48, 100),
            MockLevel(0.43, 100),
        ]
        gaps = detect_gaps(bid_levels=bids, ask_levels=[], min_gap_cents=0.02)
        assert len(gaps) == 2
        assert gaps[0].gap_cents > gaps[1].gap_cents


class TestBookGaps:
    """Cross-book gap detection combining YES and NO."""

    def test_yes_bid_gap_is_support(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[MockLevel(0.50, 100), MockLevel(0.46, 200)],
            yes_ask_levels=[], no_bid_levels=[], no_ask_levels=[],
        )
        assert len(result["support_gaps"]) == 1
        assert len(result["resistance_gaps"]) == 0

    def test_yes_ask_gap_is_resistance(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[],
            yes_ask_levels=[MockLevel(0.52, 100), MockLevel(0.57, 200)],
            no_bid_levels=[], no_ask_levels=[],
        )
        assert len(result["resistance_gaps"]) == 1
        assert len(result["support_gaps"]) == 0

    def test_no_ask_gap_is_support(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[], yes_ask_levels=[],
            no_bid_levels=[],
            no_ask_levels=[MockLevel(0.48, 100), MockLevel(0.53, 200)],
        )
        assert len(result["support_gaps"]) == 1

    def test_no_bid_gap_is_resistance(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[], yes_ask_levels=[],
            no_bid_levels=[MockLevel(0.50, 100), MockLevel(0.45, 200)],
            no_ask_levels=[],
        )
        assert len(result["resistance_gaps"]) == 1

    def test_combined_gaps_from_both_books(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[MockLevel(0.50, 100), MockLevel(0.46, 200)],
            yes_ask_levels=[MockLevel(0.52, 100), MockLevel(0.57, 200)],
            no_bid_levels=[MockLevel(0.48, 100), MockLevel(0.43, 200)],
            no_ask_levels=[MockLevel(0.50, 100), MockLevel(0.55, 200)],
        )
        assert len(result["support_gaps"]) == 2
        assert len(result["resistance_gaps"]) == 2


class TestWorstGapAsymmetry:

    def test_asymmetric_gaps(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[MockLevel(0.50, 100), MockLevel(0.44, 200)],
            yes_ask_levels=[MockLevel(0.52, 100), MockLevel(0.54, 200)],
            no_bid_levels=[], no_ask_levels=[],
        )
        support_gap, resistance_gap = worst_gap_asymmetry(result)
        assert support_gap == pytest.approx(0.06, abs=0.001)
        assert resistance_gap == pytest.approx(0.02, abs=0.001)
        assert support_gap > resistance_gap

    def test_no_gaps(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[MockLevel(0.50, 100), MockLevel(0.49, 100)],
            yes_ask_levels=[MockLevel(0.51, 100), MockLevel(0.52, 100)],
            no_bid_levels=[], no_ask_levels=[],
        )
        sg, rg = worst_gap_asymmetry(result)
        assert sg == 0.0 and rg == 0.0

    def test_symmetric_gaps(self) -> None:
        result = detect_book_gaps(
            yes_bid_levels=[MockLevel(0.50, 100), MockLevel(0.47, 100)],
            yes_ask_levels=[MockLevel(0.52, 100), MockLevel(0.55, 100)],
            no_bid_levels=[], no_ask_levels=[],
        )
        sg, rg = worst_gap_asymmetry(result)
        assert sg == pytest.approx(rg, abs=0.001)

    def test_empty_books(self) -> None:
        result = detect_book_gaps([], [], [], [])
        sg, rg = worst_gap_asymmetry(result)
        assert sg == 0.0 and rg == 0.0


# ── Phase 2: Full signal with wall depletion ───────────────────


def _make_bid_book(wall_price: float, wall_size: float,
                   next_price: float, next_size: float) -> list[MockLevel]:
    """Create a bid book with a wall guarding a gap."""
    return [MockLevel(wall_price, wall_size), MockLevel(next_price, next_size)]


def _make_ask_book(wall_price: float, wall_size: float,
                   next_price: float, next_size: float) -> list[MockLevel]:
    """Create an ask book with a wall guarding a gap."""
    return [MockLevel(wall_price, wall_size), MockLevel(next_price, next_size)]


def _run_ticks(signal: LevelExhaustionSignal,
               yes_bids: list, yes_asks: list,
               no_bids: list | None = None, no_asks: list | None = None,
               n: int = 5, seconds_remaining: float = 200.0,
               window_volume: float = 0.0) -> float:
    """Run n ticks with the same book state."""
    val = 0.0
    for _ in range(n):
        val = signal.compute(
            yes_bids, yes_asks,
            no_bids or [], no_asks or [],
            seconds_remaining=seconds_remaining,
            window_volume=window_volume,
        )
    return val


class TestBidExhaustion:
    """Support wall depleting behind a gap -> bearish signal."""

    def test_depleting_bid_wall_is_bearish(self) -> None:
        """Wall at 0.50 with gap to 0.46, wall shrinks -> negative signal."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=30, min_gap_cents=0.02,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Build history with wall at full strength
        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )

        # Now wall depletes to 40% of peak (100 -> 40 = 60% depletion)
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 40, 0.46, 200), asks,
            n=5, seconds_remaining=180,
        )
        assert val < 0  # Bearish

    def test_no_signal_without_gap(self) -> None:
        """Wall depleting but no gap behind it -> no signal."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        # Tight book: levels 1 cent apart, no gap
        tight_bids = [MockLevel(0.50, 100), MockLevel(0.49, 200)]
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(tight_bids, asks, [], [], seconds_remaining=200)

        # Wall depletes but no gap
        tight_bids_thin = [MockLevel(0.50, 20), MockLevel(0.49, 200)]
        val = _run_ticks(sig, tight_bids_thin, asks, n=5, seconds_remaining=180)
        assert abs(val) < 0.01

    def test_no_signal_without_depletion(self) -> None:
        """Gap exists but wall is stable -> no signal."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        bids = _make_bid_book(0.50, 100, 0.46, 200)
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Run many ticks with stable wall
        val = _run_ticks(sig, bids, asks, n=15, seconds_remaining=200)
        assert abs(val) < 0.01


class TestAskExhaustion:
    """Resistance wall depleting behind a gap -> bullish signal."""

    def test_depleting_ask_wall_is_bullish(self) -> None:
        """Wall at 0.52 with gap to 0.56, wall shrinks -> positive signal."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=30, min_gap_cents=0.02,
        )
        bids = [MockLevel(0.50, 200), MockLevel(0.49, 200)]

        # Build history with ask wall at full strength
        for _ in range(5):
            sig.compute(
                bids, _make_ask_book(0.52, 100, 0.56, 200), [], [],
                seconds_remaining=200,
            )

        # Ask wall depletes to 30% of peak (100 -> 30 = 70% depletion)
        val = _run_ticks(
            sig,
            bids, _make_ask_book(0.52, 30, 0.56, 200),
            n=5, seconds_remaining=180,
        )
        assert val > 0  # Bullish


class TestDepletionThreshold:
    """Depletion must exceed threshold to trigger."""

    def test_below_threshold_no_signal(self) -> None:
        """Wall loses 20% but threshold is 40% -> no signal."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )

        # Wall drops to 80 (only 20% depletion, below 40% threshold)
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 80, 0.46, 200), asks,
            n=5, seconds_remaining=180,
        )
        assert abs(val) < 0.01

    def test_at_threshold_fires(self) -> None:
        """Wall loses exactly 40% -> signal fires."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )

        # Wall drops to 60 (40% depletion = at threshold)
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 60, 0.46, 200), asks,
            n=5, seconds_remaining=180,
        )
        assert val < 0  # Should fire bearish


class TestMinWallSize:
    """Walls below min_wall_size are noise."""

    def test_tiny_wall_ignored(self) -> None:
        """Wall peak is 20 shares, min_wall_size is 30 -> ignored."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 20, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )

        # Wall depletes 75% but peak was only 20 shares
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 5, 0.46, 200), asks,
            n=5, seconds_remaining=180,
        )
        assert abs(val) < 0.01


class TestSignalDynamics:
    """Decay, timing, and reset behaviour."""

    def test_signal_decays(self) -> None:
        """Signal decays when exhaustion conditions stop."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=30, decay_rate=0.5,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Build up signal
        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )
        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 30, 0.46, 200), asks, [], [],
                seconds_remaining=180,
            )
        peak = abs(sig.last_signal)
        assert peak > 0

        # Remove the gap -> conditions stop -> should decay
        tight_bids = [MockLevel(0.50, 100), MockLevel(0.49, 200)]
        for _ in range(5):
            sig.compute(tight_bids, asks, [], [], seconds_remaining=160)
        assert abs(sig.last_signal) < peak

    def test_suppressed_late_in_window(self) -> None:
        """Signal doesn't fire with < min_seconds_remaining."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=30, min_seconds_remaining=30.0,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Build history
        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )

        # Strong depletion but only 10 seconds remaining
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 20, 0.46, 200), asks,
            n=5, seconds_remaining=10,
        )
        # Should be decaying toward zero, not building new signal
        assert abs(val) < 0.1

    def test_signal_clamped(self) -> None:
        """Signal stays within [-1, 1]."""
        sig = LevelExhaustionSignal(
            lookback_ticks=5, depletion_threshold=0.10,
            min_wall_size_floor=10, strength_scale=100.0,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.40, 200), asks, [], [],
                seconds_remaining=200,
            )
        val = _run_ticks(
            sig,
            _make_bid_book(0.50, 5, 0.40, 200), asks,
            n=5, seconds_remaining=180,
        )
        assert -1.0 <= val <= 1.0

    def test_reset_clears_all(self) -> None:
        """Reset zeroes signal and clears history."""
        sig = LevelExhaustionSignal(lookback_ticks=8, min_wall_size_floor=30)
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
                seconds_remaining=200,
            )
        for _ in range(5):
            sig.compute(
                _make_bid_book(0.50, 30, 0.46, 200), asks, [], [],
                seconds_remaining=180,
            )

        sig.reset()
        assert sig.last_signal == 0.0
        assert sig._tick_count == 0
        assert len(sig._bid_wall_history) == 0
        assert len(sig._ask_wall_history) == 0

    def test_needs_minimum_history(self) -> None:
        """First few ticks return 0 while building history."""
        sig = LevelExhaustionSignal(lookback_ticks=8, min_wall_size_floor=30)
        bids = _make_bid_book(0.50, 100, 0.46, 200)
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        v1 = sig.compute(bids, asks, [], [], seconds_remaining=200)
        v2 = sig.compute(bids, asks, [], [], seconds_remaining=200)
        assert v1 == 0.0
        assert v2 == 0.0


class TestWallStates:
    """Dashboard wall state reporting."""

    def test_get_wall_states(self) -> None:
        """Wall states returned for tracked walls."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, min_wall_size_floor=30, min_gap_cents=0.02,
        )
        bids = _make_bid_book(0.50, 100, 0.46, 200)
        asks = _make_ask_book(0.52, 80, 0.56, 200)

        # Build up history
        for _ in range(5):
            sig.compute(bids, asks, [], [], seconds_remaining=200)

        states = sig.get_wall_states(bids, asks, [], [])
        assert len(states) >= 1  # At least the bid wall

        # Check wall state fields
        for s in states:
            assert s.peak_depth > 0
            assert s.current_depth > 0
            assert 0.0 <= s.depletion_ratio <= 1.0
            assert s.gap_behind >= 0.02

    def test_no_wall_states_without_history(self) -> None:
        """Fresh signal has no wall states."""
        sig = LevelExhaustionSignal(lookback_ticks=8, min_wall_size_floor=30)
        bids = _make_bid_book(0.50, 100, 0.46, 200)
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        states = sig.get_wall_states(bids, asks, [], [])
        assert len(states) == 0


class TestGapWeighting:
    """Bigger gaps produce stronger signals."""

    def test_bigger_gap_stronger_signal(self) -> None:
        """5-cent gap produces stronger signal than 2-cent gap."""
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Small gap: 2 cents
        sig_small = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        for _ in range(5):
            sig_small.compute(
                _make_bid_book(0.50, 100, 0.48, 200), asks, [], [],
                seconds_remaining=200,
            )
        v_small = _run_ticks(
            sig_small,
            _make_bid_book(0.50, 30, 0.48, 200), asks,
            n=5, seconds_remaining=180,
        )

        # Big gap: 8 cents
        sig_big = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40, min_wall_size_floor=30,
        )
        for _ in range(5):
            sig_big.compute(
                _make_bid_book(0.50, 100, 0.42, 200), asks, [], [],
                seconds_remaining=200,
            )
        v_big = _run_ticks(
            sig_big,
            _make_bid_book(0.50, 30, 0.42, 200), asks,
            n=5, seconds_remaining=180,
        )

        # Both bearish, but bigger gap should produce stronger signal
        assert v_small < 0
        assert v_big < 0
        assert abs(v_big) >= abs(v_small)


class TestRelativeWallSize:
    """Wall size threshold scales with window volume."""

    def test_floor_used_when_no_volume(self) -> None:
        """With zero window volume, falls back to min_wall_size_floor."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=30.0, min_wall_fraction=0.02,
        )
        # No volume passed -> effective min = floor = 30
        assert sig.effective_min_wall_size == 30.0

    def test_fraction_dominates_with_high_volume(self) -> None:
        """With high window volume, fraction × volume > floor."""
        sig = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=10.0, min_wall_fraction=0.02,
        )
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Pass window_volume=5000 -> effective min = 0.02 * 5000 = 100
        sig.compute(
            _make_bid_book(0.50, 100, 0.46, 200), asks, [], [],
            seconds_remaining=200, window_volume=5000.0,
        )
        assert sig.effective_min_wall_size == 100.0

    def test_high_volume_suppresses_small_wall(self) -> None:
        """Wall that would fire at low volume gets ignored at high volume.

        Wall peak = 50 shares. At floor=10 this is meaningful.
        At window_volume=5000 (effective min = 100), it's noise.
        """
        asks = [MockLevel(0.52, 200), MockLevel(0.53, 200)]

        # Low volume scenario: wall of 50 is above floor of 10
        sig_low = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=10.0, min_wall_fraction=0.02,
        )
        for _ in range(5):
            sig_low.compute(
                _make_bid_book(0.50, 50, 0.46, 200), asks, [], [],
                seconds_remaining=200, window_volume=100.0,
            )
        v_low = _run_ticks(
            sig_low,
            _make_bid_book(0.50, 15, 0.46, 200), asks,
            n=5, seconds_remaining=180, window_volume=100.0,
        )
        assert v_low < 0  # Should fire — wall is above effective min (10)

        # High volume scenario: same wall of 50 is below effective min (100)
        sig_high = LevelExhaustionSignal(
            lookback_ticks=8, depletion_threshold=0.40,
            min_wall_size_floor=10.0, min_wall_fraction=0.02,
        )
        for _ in range(5):
            sig_high.compute(
                _make_bid_book(0.50, 50, 0.46, 200), asks, [], [],
                seconds_remaining=200, window_volume=5000.0,
            )
        v_high = _run_ticks(
            sig_high,
            _make_bid_book(0.50, 15, 0.46, 200), asks,
            n=5, seconds_remaining=180, window_volume=5000.0,
        )
        assert abs(v_high) < 0.01  # Should NOT fire — wall too small for volume

    def test_reset_clears_window_volume(self) -> None:
        """Reset zeroes window volume along with other state."""
        sig = LevelExhaustionSignal(min_wall_size_floor=10.0, min_wall_fraction=0.02)
        sig.compute(
            [MockLevel(0.50, 100)], [MockLevel(0.52, 100)], [], [],
            seconds_remaining=200, window_volume=5000.0,
        )
        assert sig._window_volume == 5000.0
        sig.reset()
        assert sig._window_volume == 0.0
        assert sig.effective_min_wall_size == 10.0
