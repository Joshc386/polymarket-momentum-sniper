"""Layer 10: Level Exhaustion Signal — detects thin book behind top-of-book.

Level exhaustion occurs when the top-of-book depth is being consumed and
the next price level with meaningful depth is far away. If the current
level breaks, price jumps/drops suddenly across the gap.

Detection uses per-level orderbook depth from both YES and NO books:
- YES bids + NO asks represent the bullish side (support)
- YES asks + NO bids represent the bearish side (resistance)

The signal cross-references two components:
1. Gap detection: distance (in cents) to the next level with meaningful depth
2. Wall depletion: relative decline in depth at the level guarding the gap

The signal fires when BOTH conditions are met:
- A gap >= min_gap_cents exists behind the top-of-book level
- The wall guarding that gap has lost >= depletion_threshold of its recent peak

Designed for 5-minute binary markets at 8Hz tick rate (~2,400 ticks/window).
Uses short lookback windows (8-15 ticks = 1-2 seconds) to react fast
within the limited time horizon.

Output is [-1.0, 1.0]:
  Positive = ask exhaustion (resistance thinning with gap above -> bullish breakout risk)
  Negative = bid exhaustion (support thinning with gap below -> bearish breakdown risk)
  Zero = no exhaustion detected or insufficient data
"""

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────


@dataclass
class GapInfo:
    """Describes a gap found behind the top-of-book level.

    Attributes:
        side: "bid" or "ask" — which side the gap is on.
        top_price: Price of the current top-of-book level.
        top_size: Depth (shares) at the current top-of-book level.
        next_price: Price of the next level with meaningful depth.
        next_size: Depth at the next level.
        gap_cents: Distance in cents between top_price and next_price.
        levels_skipped: Number of empty price slots in the gap.
    """

    side: str
    top_price: float
    top_size: float
    next_price: float
    next_size: float
    gap_cents: float
    levels_skipped: int


@dataclass
class WallState:
    """Tracks the depletion state of a wall guarding a gap.

    Attributes:
        price: Price level being tracked.
        side: "bid" or "ask".
        peak_depth: Highest depth seen at this level in the lookback window.
        current_depth: Current depth at this level.
        depletion_ratio: 1.0 - (current / peak). 0 = full strength, 1 = empty.
        gap_behind: Size of the gap this wall guards (in cents).
    """

    price: float
    side: str
    peak_depth: float
    current_depth: float
    depletion_ratio: float
    gap_behind: float


# ── Main signal class ────────────────────────────────────────────


@dataclass
class LevelExhaustionSignal:
    """Detects level exhaustion by combining gap detection with wall depletion.

    Args:
        min_gap_cents: Minimum gap behind a level to consider it a risk.
            Default 0.02 (2 cents). On a 100-slot binary contract, 2 cents
            means 2 empty price levels between the wall and next depth.
        min_level_size: Minimum shares to count as meaningful depth.
            Below this, orders are dust and ignored.
        min_wall_size_floor: Absolute minimum peak depth (shares) for a
            wall to be tracked. Acts as a floor — even in very low volume
            windows, walls smaller than this are always noise.
        min_wall_fraction: Fraction of window traded volume used as the
            relative minimum wall size. The effective minimum is:
            max(min_wall_size_floor, min_wall_fraction × window_volume).
            Default 0.02 = wall must be at least 2% of window volume.
        max_depth_levels: How many raw levels deep to scan for gaps.
        depletion_threshold: Minimum depletion ratio (0-1) to trigger signal.
            0.4 = wall must have lost 40% of its recent peak depth.
        lookback_ticks: Number of ticks to track wall depth history.
            At 8Hz, 12 ticks = 1.5 seconds. Short enough to react fast
            in a 5-minute window, long enough to filter single-tick noise.
        decay_rate: How quickly the signal decays when exhaustion conditions
            stop being met. 0.5 = lose 50% per tick. Fast decay because
            conditions change rapidly in 5-min windows.
        strength_scale: Multiplier for raw signal before clamping.
        min_seconds_remaining: Don't fire signal when less than this many
            seconds remain in the window — not enough time for the gap
            break to matter.
    """

    min_gap_cents: float = 0.02
    min_level_size: float = 5.0
    min_wall_size_floor: float = 10.0
    min_wall_fraction: float = 0.02
    max_depth_levels: int = 10
    depletion_threshold: float = 0.40
    lookback_ticks: int = 12
    decay_rate: float = 0.5
    strength_scale: float = 2.0
    min_seconds_remaining: float = 30.0

    # Internal state
    _bid_wall_history: dict = field(default_factory=dict)
    _ask_wall_history: dict = field(default_factory=dict)
    _current_signal: float = 0.0
    _tick_count: int = 0
    _window_volume: float = 0.0

    def reset(self) -> None:
        """Clear all state for a new window."""
        self._bid_wall_history.clear()
        self._ask_wall_history.clear()
        self._current_signal = 0.0
        self._tick_count = 0
        self._window_volume = 0.0

    def compute(
        self,
        yes_bid_levels: list,
        yes_ask_levels: list,
        no_bid_levels: list,
        no_ask_levels: list,
        seconds_remaining: float = 300.0,
        window_volume: float = 0.0,
    ) -> float:
        """Compute level exhaustion signal from full orderbook depth.

        Args:
            yes_bid_levels: YES bid OrderLevels, highest price first.
            yes_ask_levels: YES ask OrderLevels, lowest price first.
            no_bid_levels: NO bid OrderLevels, highest price first.
            no_ask_levels: NO ask OrderLevels, lowest price first.
            seconds_remaining: Seconds left in the 5-min window.
            window_volume: Total volume (shares) traded in this window so
                far. Used to compute the relative minimum wall size.
                Pass 0 to fall back to min_wall_size_floor only.

        Returns:
            Signal in [-1.0, 1.0].
            Positive = ask exhaustion (bullish breakout risk).
            Negative = bid exhaustion (bearish breakdown risk).
        """
        self._tick_count += 1
        self._window_volume = window_volume

        # Too late in the window — gap break can't play out
        if seconds_remaining < self.min_seconds_remaining:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # Need minimum history before computing
        if self._tick_count < 3:
            # Still record wall depths for history building
            self._update_wall_histories(
                yes_bid_levels, yes_ask_levels,
                no_bid_levels, no_ask_levels,
            )
            return 0.0

        # ── Step 1: Detect gaps ──
        gap_result = detect_book_gaps(
            yes_bid_levels, yes_ask_levels,
            no_bid_levels, no_ask_levels,
            min_gap_cents=self.min_gap_cents,
            min_level_size=self.min_level_size,
            max_depth_levels=self.max_depth_levels,
        )

        # ── Step 2: Update wall depth histories ──
        self._update_wall_histories(
            yes_bid_levels, yes_ask_levels,
            no_bid_levels, no_ask_levels,
        )

        # ── Step 3: Check depletion at walls guarding gaps ──
        support_exhaustion = self._check_side_exhaustion(
            gap_result["support_gaps"], "bid",
        )
        resistance_exhaustion = self._check_side_exhaustion(
            gap_result["resistance_gaps"], "ask",
        )

        # ── Step 4: Combine into directional signal ──
        # Resistance exhaustion = bullish (positive)
        # Support exhaustion = bearish (negative)
        raw = (resistance_exhaustion - support_exhaustion) * self.strength_scale

        if abs(raw) > 0.01:
            self._current_signal = _clamp(raw)
        else:
            self._current_signal *= (1.0 - self.decay_rate)

        return self._current_signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._current_signal

    def get_wall_states(
        self,
        yes_bid_levels: list,
        yes_ask_levels: list,
        no_bid_levels: list,
        no_ask_levels: list,
    ) -> list[WallState]:
        """Get current wall depletion states for dashboard display.

        Returns a list of WallState objects for all tracked walls that
        currently guard a gap. Useful for debugging and visualisation.
        """
        gap_result = detect_book_gaps(
            yes_bid_levels, yes_ask_levels,
            no_bid_levels, no_ask_levels,
            min_gap_cents=self.min_gap_cents,
            min_level_size=self.min_level_size,
            max_depth_levels=self.max_depth_levels,
        )

        states = []
        for gap in gap_result["support_gaps"]:
            state = self._get_wall_state(gap, "bid")
            if state:
                states.append(state)
        for gap in gap_result["resistance_gaps"]:
            state = self._get_wall_state(gap, "ask")
            if state:
                states.append(state)

        return states

    # ── Internal methods ─────────────────────────────────────────

    def _update_wall_histories(
        self,
        yes_bid_levels: list,
        yes_ask_levels: list,
        no_bid_levels: list,
        no_ask_levels: list,
    ) -> None:
        """Record current depth at each price level for depletion tracking."""
        # Build price -> depth maps for current tick
        bid_depths = _build_depth_map(yes_bid_levels)
        ask_depths = _build_depth_map(yes_ask_levels)

        # Also incorporate NO book (NO asks = support, NO bids = resistance)
        no_ask_depths = _build_depth_map(no_ask_levels)
        no_bid_depths = _build_depth_map(no_bid_levels)

        # Update bid wall history (YES bids + NO asks = support)
        all_support_depths = {**bid_depths}
        for price, size in no_ask_depths.items():
            # NO asks at price X contribute to support at (1-X) on the YES scale
            # But for tracking, we key by the actual price observed
            all_support_depths[price] = all_support_depths.get(price, 0) + size

        for price, size in all_support_depths.items():
            key = round(price, 4)
            if key not in self._bid_wall_history:
                self._bid_wall_history[key] = deque(maxlen=self.lookback_ticks)
            self._bid_wall_history[key].append(size)

        # Update ask wall history (YES asks + NO bids = resistance)
        all_resist_depths = {**ask_depths}
        for price, size in no_bid_depths.items():
            all_resist_depths[price] = all_resist_depths.get(price, 0) + size

        for price, size in all_resist_depths.items():
            key = round(price, 4)
            if key not in self._ask_wall_history:
                self._ask_wall_history[key] = deque(maxlen=self.lookback_ticks)
            self._ask_wall_history[key].append(size)

        # Prune stale entries (prices no longer in the book)
        self._prune_stale_histories()

    def _prune_stale_histories(self) -> None:
        """Remove wall histories for prices that haven't been seen recently."""
        max_age = self.lookback_ticks * 2
        for history_dict in [self._bid_wall_history, self._ask_wall_history]:
            stale_keys = [
                k for k, v in history_dict.items()
                if len(v) > 0 and self._tick_count > max_age
                and len(v) < self.lookback_ticks // 2
            ]
            for k in stale_keys:
                del history_dict[k]

    def _check_side_exhaustion(
        self,
        gaps: list[GapInfo],
        side: str,
    ) -> float:
        """Check if walls guarding gaps on one side are depleting.

        Returns an exhaustion score in [0, 1]. Higher = more exhausted.
        Takes the worst (most depleted) wall guarding a gap.
        """
        if not gaps:
            return 0.0

        worst_exhaustion = 0.0

        for gap in gaps:
            wall_state = self._get_wall_state(gap, side)
            if wall_state is None:
                continue

            if wall_state.depletion_ratio >= self.depletion_threshold:
                # Scale: 0 at threshold, 1 at full depletion
                # Weight by gap size — bigger gap = more impact
                gap_weight = min(gap.gap_cents / 0.05, 1.0)  # Normalise: 5c gap = weight 1.0
                exhaustion = wall_state.depletion_ratio * gap_weight
                worst_exhaustion = max(worst_exhaustion, exhaustion)

        return min(1.0, worst_exhaustion)

    @property
    def effective_min_wall_size(self) -> float:
        """Compute the effective minimum wall size.

        Uses the larger of the absolute floor and a fraction of window
        volume. In low-volume windows, the floor prevents noise from
        tiny walls. In high-volume windows, the fraction scales the
        threshold so only proportionally meaningful walls are tracked.
        """
        return max(
            self.min_wall_size_floor,
            self.min_wall_fraction * self._window_volume,
        )

    def _get_wall_state(self, gap: GapInfo, side: str) -> WallState | None:
        """Get the depletion state of the wall guarding a specific gap."""
        price_key = round(gap.top_price, 4)
        history_dict = self._bid_wall_history if side == "bid" else self._ask_wall_history

        history = history_dict.get(price_key)
        if not history or len(history) < 3:
            return None

        peak = max(history)
        current = history[-1]

        if peak < self.effective_min_wall_size:
            return None

        depletion = 1.0 - (current / peak) if peak > 0 else 0.0

        return WallState(
            price=gap.top_price,
            side=side,
            peak_depth=peak,
            current_depth=current,
            depletion_ratio=max(0.0, depletion),
            gap_behind=gap.gap_cents,
        )


# ── Gap detection functions (Phase 1) ───────────────────────────


def detect_gaps(
    bid_levels: list,
    ask_levels: list,
    min_gap_cents: float = 0.02,
    min_level_size: float = 5.0,
    max_depth_levels: int = 10,
) -> list[GapInfo]:
    """Find gaps in the orderbook near top-of-book.

    Scans the top N levels on each side looking for price gaps >= min_gap_cents.
    A "gap" means the distance between two adjacent levels with meaningful
    depth (>= min_level_size) is wider than min_gap_cents.

    Args:
        bid_levels: Bid levels sorted highest-price-first (as from OrderbookManager).
        ask_levels: Ask levels sorted lowest-price-first (as from OrderbookManager).
        min_gap_cents: Minimum gap size to report. Default 0.02 (2 cents).
        min_level_size: Minimum shares at a level to consider it "meaningful".
            Tiny resting orders (dust) don't count as real depth.
        max_depth_levels: How many levels deep to scan from top-of-book.
            Looking too deep adds noise — gaps far from the market don't matter.

    Returns:
        List of GapInfo objects for each gap found, sorted by gap_cents descending.
        Empty list if no gaps are detected.
    """
    gaps: list[GapInfo] = []

    # ── Scan bid side (highest price first -> looking downward) ──
    if bid_levels:
        meaningful_bids = _filter_meaningful_levels(
            bid_levels, min_level_size, max_depth_levels
        )
        bid_gaps = _find_gaps_in_sequence(
            meaningful_bids, side="bid", min_gap_cents=min_gap_cents
        )
        gaps.extend(bid_gaps)

    # ── Scan ask side (lowest price first -> looking upward) ──
    if ask_levels:
        meaningful_asks = _filter_meaningful_levels(
            ask_levels, min_level_size, max_depth_levels
        )
        ask_gaps = _find_gaps_in_sequence(
            meaningful_asks, side="ask", min_gap_cents=min_gap_cents
        )
        gaps.extend(ask_gaps)

    # Sort by gap size descending (biggest gap first)
    gaps.sort(key=lambda g: g.gap_cents, reverse=True)
    return gaps


def detect_book_gaps(
    yes_bid_levels: list,
    yes_ask_levels: list,
    no_bid_levels: list,
    no_ask_levels: list,
    min_gap_cents: float = 0.02,
    min_level_size: float = 5.0,
    max_depth_levels: int = 10,
) -> dict[str, list[GapInfo]]:
    """Detect gaps across both YES and NO books.

    YES bids + NO asks = bullish side (support).
    YES asks + NO bids = bearish side (resistance).

    Gaps on the bid/support side -> bearish risk (if support breaks, price drops).
    Gaps on the ask/resistance side -> bullish risk (if resistance breaks, price jumps).

    Args:
        yes_bid_levels: YES token bid levels (highest price first).
        yes_ask_levels: YES token ask levels (lowest price first).
        no_bid_levels: NO token bid levels (highest price first).
        no_ask_levels: NO token ask levels (lowest price first).
        min_gap_cents: Minimum gap size to report (default 2 cents).
        min_level_size: Minimum shares to count as real depth.
        max_depth_levels: How deep to scan from top-of-book.

    Returns:
        Dict with keys:
        - "yes_bid_gaps": Gaps in YES bid book (bearish if support breaks)
        - "yes_ask_gaps": Gaps in YES ask book (bullish if resistance breaks)
        - "no_bid_gaps": Gaps in NO bid book (bullish if NO support breaks -> YES up)
        - "no_ask_gaps": Gaps in NO ask book (bearish if NO resistance breaks -> YES down)
        - "support_gaps": Combined YES bid + NO ask gaps (bearish)
        - "resistance_gaps": Combined YES ask + NO bid gaps (bullish)
    """
    kwargs = dict(
        min_gap_cents=min_gap_cents,
        min_level_size=min_level_size,
        max_depth_levels=max_depth_levels,
    )

    yes_bid_gaps = detect_gaps(bid_levels=yes_bid_levels, ask_levels=[], **kwargs)
    yes_ask_gaps = detect_gaps(bid_levels=[], ask_levels=yes_ask_levels, **kwargs)
    no_bid_gaps = detect_gaps(bid_levels=no_bid_levels, ask_levels=[], **kwargs)
    no_ask_gaps = detect_gaps(bid_levels=[], ask_levels=no_ask_levels, **kwargs)

    # Combine: support = YES bids + NO asks (bearish if broken)
    support_gaps = yes_bid_gaps + no_ask_gaps
    support_gaps.sort(key=lambda g: g.gap_cents, reverse=True)

    # Combine: resistance = YES asks + NO bids (bullish if broken)
    resistance_gaps = yes_ask_gaps + no_bid_gaps
    resistance_gaps.sort(key=lambda g: g.gap_cents, reverse=True)

    return {
        "yes_bid_gaps": yes_bid_gaps,
        "yes_ask_gaps": yes_ask_gaps,
        "no_bid_gaps": no_bid_gaps,
        "no_ask_gaps": no_ask_gaps,
        "support_gaps": support_gaps,
        "resistance_gaps": resistance_gaps,
    }


def worst_gap_asymmetry(
    gap_result: dict[str, list[GapInfo]],
) -> tuple[float, float]:
    """Compute the worst gap on each side for signal generation.

    Returns:
        (worst_support_gap_cents, worst_resistance_gap_cents)
        Both are non-negative. A larger support gap means more bearish
        risk; a larger resistance gap means more bullish risk.
    """
    worst_support = 0.0
    if gap_result["support_gaps"]:
        worst_support = gap_result["support_gaps"][0].gap_cents

    worst_resistance = 0.0
    if gap_result["resistance_gaps"]:
        worst_resistance = gap_result["resistance_gaps"][0].gap_cents

    return worst_support, worst_resistance


# ── Internal helpers ─────────────────────────────────────────────


def _clamp(x: float) -> float:
    """Clamp to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, x))


def _build_depth_map(levels: list) -> dict[float, float]:
    """Build a price -> size map from a list of OrderLevels."""
    depth_map: dict[float, float] = {}
    for level in levels:
        key = round(level.price, 4)
        depth_map[key] = depth_map.get(key, 0) + level.size
    return depth_map


def _filter_meaningful_levels(
    levels: list,
    min_size: float,
    max_levels: int,
) -> list:
    """Filter to levels with meaningful depth, capped at max_levels.

    Scans from top-of-book (index 0) outward. Keeps levels with
    size >= min_size. Stops after scanning max_levels raw levels
    (not max_levels meaningful ones — we want to find gaps within
    the top N levels of the book, including empty ones).

    Args:
        levels: OrderLevel list, already sorted by proximity to market.
        min_size: Minimum size to keep.
        max_levels: Maximum raw levels to scan.

    Returns:
        Filtered list of levels with meaningful depth.
    """
    meaningful = []
    for i, level in enumerate(levels):
        if i >= max_levels:
            break
        if level.size >= min_size:
            meaningful.append(level)
    return meaningful


def _find_gaps_in_sequence(
    levels: list,
    side: str,
    min_gap_cents: float,
) -> list[GapInfo]:
    """Find gaps between adjacent meaningful levels.

    Args:
        levels: Meaningful levels sorted by proximity to market.
            For bids: highest price first (scanning downward).
            For asks: lowest price first (scanning upward).
        side: "bid" or "ask".
        min_gap_cents: Minimum gap to report.

    Returns:
        List of GapInfo for each qualifying gap.
    """
    if len(levels) < 2:
        return []

    gaps = []
    for i in range(len(levels) - 1):
        current = levels[i]
        next_level = levels[i + 1]

        gap = abs(current.price - next_level.price)

        if gap >= min_gap_cents:
            # Count how many 1-cent slots are empty
            levels_skipped = max(0, round(gap / 0.01) - 1)

            gaps.append(GapInfo(
                side=side,
                top_price=current.price,
                top_size=current.size,
                next_price=next_level.price,
                next_size=next_level.size,
                gap_cents=round(gap, 4),
                levels_skipped=levels_skipped,
            ))

    return gaps
