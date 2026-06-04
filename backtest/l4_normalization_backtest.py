"""L4 Orderbook Signal — Normalization Constant Backtest.

Replays the L4 orderbook signal computation from raw tick_data.db snapshots
using different normalization constants, measuring predictive accuracy
(does signal direction predict the market outcome?).

Includes:
- Constant sweep across all 5 sub-signal normalization divisors
- Chronological train/test split for overfitting detection
- Permutation test for statistical significance

Usage:
    python -m backtest.l4_normalization_backtest [--db backtest/data/tick_data.db]

Requires tick_data.db with markets and snapshots tables.
"""

import argparse
import logging
import os
import random
import sqlite3
import sys
import time
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tick_data.db")


# ── Normalization config ─────────────────────────────────────────
@dataclass(frozen=True)
class NormConfig:
    """Normalization constants for L4 sub-signals."""

    name: str
    imbalance_div: float = 0.3       # bid/ask imbalance divisor
    flow_div: float = 200.0          # order flow divisor
    mid_dev_div: float = 0.02        # weighted mid deviation divisor
    top_pressure_div: float = 0.3    # top-of-book pressure divisor
    thickness_div: float = 0.3       # book thickness divisor


# Configs to test
CONFIGS: list[NormConfig] = [
    NormConfig("current",   imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("proposed",  imbalance_div=0.5,  flow_div=300,  mid_dev_div=0.005, top_pressure_div=0.5,  thickness_div=0.5),
    # Individual constant sweeps
    NormConfig("imb_0.4",   imbalance_div=0.4,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("imb_0.5",   imbalance_div=0.5,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("imb_0.6",   imbalance_div=0.6,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("flow_150",  imbalance_div=0.3,  flow_div=150,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("flow_300",  imbalance_div=0.3,  flow_div=300,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("flow_500",  imbalance_div=0.3,  flow_div=500,  mid_dev_div=0.02,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("mid_0.005", imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.005, top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("mid_0.008", imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.008, top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("mid_0.01",  imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.01,  top_pressure_div=0.3,  thickness_div=0.3),
    NormConfig("top_0.4",   imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.4,  thickness_div=0.3),
    NormConfig("top_0.5",   imbalance_div=0.3,  flow_div=200,  mid_dev_div=0.02,  top_pressure_div=0.5,  thickness_div=0.3),
]


# ── L4 Signal Replay ─────────────────────────────────────────────
@dataclass
class L4Replayer:
    """Replays the L4 orderbook signal with configurable normalization."""

    config: NormConfig

    # Sub-signal weights (same as production)
    imbalance_weight: float = 0.30
    flow_weight: float = 0.25
    weighted_mid_weight: float = 0.20
    top_pressure_weight: float = 0.15
    thickness_weight: float = 0.10

    # Smoothing history
    _imbalance_history: deque = field(default_factory=lambda: deque(maxlen=30))
    _flow_history: deque = field(default_factory=lambda: deque(maxlen=30))

    def reset(self) -> None:
        """Clear history for a new market window."""
        self._imbalance_history.clear()
        self._flow_history.clear()

    def compute(
        self,
        yes_bid: float,
        yes_ask: float,
        no_bid: float,
        no_ask: float,
        yes_best_bid: float,
        yes_best_ask: float,
        yes_bid_delta: float,
        yes_ask_delta: float,
        no_bid_delta: float,
        no_ask_delta: float,
    ) -> float:
        """Compute L4 signal from a single snapshot.

        Args:
            yes_bid: Total YES bid depth (top 5 levels).
            yes_ask: Total YES ask depth (top 5 levels).
            no_bid: Total NO bid depth (top 5 levels).
            no_ask: Total NO ask depth (top 5 levels).
            yes_best_bid: Best YES bid price.
            yes_best_ask: Best YES ask price.
            yes_bid_delta: Change in YES bid depth since last snapshot.
            yes_ask_delta: Change in YES ask depth since last snapshot.
            no_bid_delta: Change in NO bid depth since last snapshot.
            no_ask_delta: Change in NO ask depth since last snapshot.

        Returns:
            L4 signal in [-1.0, 1.0].
        """
        cfg = self.config

        # 1. Bid/Ask Imbalance
        bullish_vol = yes_bid + no_ask
        bearish_vol = yes_ask + no_bid
        total = bullish_vol + bearish_vol
        if total > 0:
            imbalance = (bullish_vol - bearish_vol) / total
            imbalance = _clamp(imbalance / cfg.imbalance_div)
        else:
            imbalance = 0.0

        # 2. Order Flow
        bullish_flow = yes_bid_delta - yes_ask_delta + no_ask_delta - no_bid_delta
        flow = _clamp(bullish_flow / cfg.flow_div)

        # 3. Weighted Mid Deviation
        if yes_best_bid and yes_best_ask and yes_best_bid > 0 and yes_best_ask > 0:
            simple_mid = (yes_best_bid + yes_best_ask) / 2.0
            depth_total = yes_bid + yes_ask
            if depth_total > 0:
                weighted_mid = (yes_best_bid * yes_ask + yes_best_ask * yes_bid) / depth_total
                deviation = weighted_mid - simple_mid
                mid_dev = _clamp(deviation / cfg.mid_dev_div)
            else:
                mid_dev = 0.0
        else:
            mid_dev = 0.0

        # 4. Top-of-Book Pressure (using same depth as total since we only have top 5)
        top_total = yes_bid + yes_ask
        if top_total > 0:
            top_pressure = _clamp(((yes_bid - yes_ask) / top_total) / cfg.top_pressure_div)
        else:
            top_pressure = 0.0

        # 5. Book Thickness (directional modifier)
        # We don't have individual level counts in tick_data, so use depth ratio only
        thickness_total = yes_bid + yes_ask
        if thickness_total > 0:
            thickness_imbalance = (yes_bid - yes_ask) / thickness_total
            thickness = _clamp(thickness_imbalance / cfg.thickness_div)
        else:
            thickness = 0.0

        # Track history for smoothing
        self._imbalance_history.append(imbalance)
        self._flow_history.append(flow)

        # Smooth (same logic as production)
        if len(self._imbalance_history) >= 3:
            smoothed_imbalance = sum(self._imbalance_history) / len(self._imbalance_history)
        else:
            smoothed_imbalance = imbalance

        if len(self._flow_history) >= 3:
            smoothed_flow = sum(self._flow_history) / len(self._flow_history)
        else:
            smoothed_flow = flow

        # Combine
        raw = (
            self.imbalance_weight * smoothed_imbalance
            + self.flow_weight * smoothed_flow
            + self.weighted_mid_weight * mid_dev
            + self.top_pressure_weight * top_pressure
            + self.thickness_weight * thickness
        )

        return _clamp(raw)


def _clamp(x: float) -> float:
    """Clamp to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, x))


# ── Market result at specific time points ────────────────────────
@dataclass
class MarketResult:
    """L4 signal readings and outcome for one market."""

    market_id: str
    start_time: str
    winner: str  # "Up" or "Down"
    # Signal values at key time points (seconds into window)
    signals_at: dict[int, float] = field(default_factory=dict)
    # {120: 0.35, 180: 0.42, 240: 0.51} = signal at min 2, 3, 4


# Time points to sample (seconds into the 5-min window)
SAMPLE_SECONDS = [120, 150, 180, 210, 240]  # min 2, 2.5, 3, 3.5, 4


def replay_market(
    snapshots: list[tuple],
    winner: str,
    market_id: str,
    start_time: str,
    config: NormConfig,
) -> MarketResult | None:
    """Replay L4 signal for one market with the given normalization config.

    Args:
        snapshots: List of (seconds_into_window, up_bid_depth_5, up_ask_depth_5,
                   down_bid_depth_5, down_ask_depth_5, up_best_bid, up_best_ask) tuples,
                   ordered by time.
        winner: "Up" or "Down".
        market_id: Market identifier.
        start_time: Market start time string.
        config: Normalization constants to use.

    Returns:
        MarketResult with signal readings at sample points, or None if insufficient data.
    """
    if len(snapshots) < 50:  # need enough data for smoothing
        return None

    replayer = L4Replayer(config=config)
    result = MarketResult(
        market_id=market_id,
        start_time=start_time,
        winner=winner,
    )

    # Track which sample points we've captured
    next_sample_idx = 0
    samples_needed = set(SAMPLE_SECONDS)

    prev = None
    for snap in snapshots:
        secs, yb, ya, nb, na, bb, ba = snap

        if secs is None or yb is None or ya is None:
            continue

        # Compute deltas
        if prev is not None:
            yb_delta = yb - prev[1]
            ya_delta = ya - prev[2]
            nb_delta = (nb or 0) - (prev[3] or 0)
            na_delta = (na or 0) - (prev[4] or 0)
        else:
            yb_delta = ya_delta = nb_delta = na_delta = 0.0

        signal = replayer.compute(
            yes_bid=yb or 0,
            yes_ask=ya or 0,
            no_bid=nb or 0,
            no_ask=na or 0,
            yes_best_bid=bb,
            yes_best_ask=ba,
            yes_bid_delta=yb_delta,
            yes_ask_delta=ya_delta,
            no_bid_delta=nb_delta,
            no_ask_delta=na_delta,
        )

        # Record at sample points (take the closest snapshot to each target second)
        for target_sec in list(samples_needed):
            if secs is not None and abs(secs - target_sec) < 1.0:
                result.signals_at[target_sec] = signal
                samples_needed.discard(target_sec)

        prev = snap

    # Need at least the minute 3 and 4 readings
    if 180 not in result.signals_at and 240 not in result.signals_at:
        return None

    return result


# ── Accuracy calculation ─────────────────────────────────────────
def calc_accuracy(
    results: list[MarketResult],
    sample_sec: int,
) -> dict:
    """Calculate predictive accuracy at a specific time point.

    Args:
        results: List of MarketResult objects.
        sample_sec: Which time point to evaluate (e.g., 240 for minute 4).

    Returns:
        Dict with accuracy metrics.
    """
    correct = 0
    incorrect = 0
    neutral = 0  # signal ~0, no prediction
    total = 0

    for r in results:
        if sample_sec not in r.signals_at:
            continue

        signal = r.signals_at[sample_sec]
        total += 1

        if abs(signal) < 0.01:  # too close to zero to call
            neutral += 1
            continue

        predicted_up = signal > 0
        actual_up = r.winner == "Up"

        if predicted_up == actual_up:
            correct += 1
        else:
            incorrect += 1

    decisive = correct + incorrect
    accuracy = correct / decisive if decisive > 0 else 0.0

    return {
        "total": total,
        "decisive": decisive,
        "correct": correct,
        "incorrect": incorrect,
        "neutral": neutral,
        "accuracy": accuracy,
    }


# ── Permutation test ─────────────────────────────────────────────
def permutation_test(
    results: list[MarketResult],
    sample_sec: int,
    n_permutations: int = 1000,
) -> dict:
    """Test whether accuracy is significantly above chance.

    Shuffles market outcomes and recomputes accuracy to build a null
    distribution. Returns p-value = fraction of permuted accuracies >= observed.
    """
    observed = calc_accuracy(results, sample_sec)
    observed_acc = observed["accuracy"]

    if observed["decisive"] < 20:
        return {"observed_accuracy": observed_acc, "p_value": 1.0, "n_permutations": 0}

    count_above = 0

    for _ in range(n_permutations):
        # Shuffle outcomes
        shuffled = []
        winners = [r.winner for r in results]
        random.shuffle(winners)
        for r, w in zip(results, winners):
            shuffled_r = MarketResult(
                market_id=r.market_id,
                start_time=r.start_time,
                winner=w,
                signals_at=r.signals_at,
            )
            shuffled.append(shuffled_r)

        perm_acc = calc_accuracy(shuffled, sample_sec)["accuracy"]
        if perm_acc >= observed_acc:
            count_above += 1

    p_value = count_above / n_permutations

    return {
        "observed_accuracy": observed_acc,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ── Main backtest ────────────────────────────────────────────────
def run_backtest(db_path: str, train_frac: float = 0.7) -> None:
    """Run the full normalization constant backtest.

    Args:
        db_path: Path to tick_data.db.
        train_frac: Fraction of markets for training (chronological split).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Load all markets with outcomes, ordered chronologically
    cur.execute("""
        SELECT market_id, start_time, winner
        FROM markets
        WHERE winner IS NOT NULL
          AND snapshots_fetched > 0
        ORDER BY start_time
    """)
    markets = cur.fetchall()
    logger.info("Loaded %d markets with outcomes", len(markets))

    if not markets:
        logger.error("No markets found in %s", db_path)
        conn.close()
        return

    # Train/test split
    split_idx = int(len(markets) * train_frac)
    train_markets = markets[:split_idx]
    test_markets = markets[split_idx:]
    logger.info(
        "Train: %d markets (%s to %s)",
        len(train_markets), train_markets[0][1][:10], train_markets[-1][1][:10],
    )
    logger.info(
        "Test:  %d markets (%s to %s)",
        len(test_markets), test_markets[0][1][:10], test_markets[-1][1][:10],
    )

    # For each config, replay all markets and collect results
    all_results: dict[str, dict[str, list[MarketResult]]] = {}
    # config_name -> {"train": [...], "test": [...]}

    total_markets = len(markets)
    start_ts = time.time()

    for cfg in CONFIGS:
        logger.info("Replaying with config: %s", cfg.name)
        train_results: list[MarketResult] = []
        test_results: list[MarketResult] = []

        for i, (market_id, start_time, winner) in enumerate(markets):
            # Load snapshots for this market
            cur.execute("""
                SELECT seconds_into_window,
                       up_bid_depth_5, up_ask_depth_5,
                       down_bid_depth_5, down_ask_depth_5,
                       up_best_bid, up_best_ask
                FROM snapshots
                WHERE market_id = ?
                ORDER BY seconds_into_window, rowid
            """, (market_id,))
            snapshots = cur.fetchall()

            result = replay_market(snapshots, winner, market_id, start_time, cfg)
            if result is None:
                continue

            if i < split_idx:
                train_results.append(result)
            else:
                test_results.append(result)

            # Progress
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_ts
                rate = (i + 1) / elapsed
                eta = (total_markets - i - 1) / rate if rate > 0 else 0
                logger.info(
                    "  %s: %d/%d markets (%.0f/sec, ETA %.0fs)",
                    cfg.name, i + 1, total_markets, rate, eta,
                )

        all_results[cfg.name] = {"train": train_results, "test": test_results}
        logger.info(
            "  %s: %d train + %d test results",
            cfg.name, len(train_results), len(test_results),
        )

    conn.close()

    # ── Report ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  L4 NORMALIZATION BACKTEST RESULTS")
    print("=" * 90)

    for sample_sec in SAMPLE_SECONDS:
        mins = sample_sec / 60
        print("\n--- Signal at minute %.1f (t=%ds) ---\n" % (mins, sample_sec))
        print(
            "%-14s | %-6s %-6s %-7s %-6s | %-6s %-6s %-7s %-6s | %s"
            % (
                "Config", "TrN", "TrDec", "TrAcc", "TrSig",
                "TeN", "TeDec", "TeAcc", "TeSig", "Overfit?",
            )
        )
        print("-" * 90)

        for cfg in CONFIGS:
            data = all_results[cfg.name]
            train_acc = calc_accuracy(data["train"], sample_sec)
            test_acc = calc_accuracy(data["test"], sample_sec)

            # Quick significance check (permutation on test set)
            if test_acc["decisive"] >= 20:
                perm = permutation_test(data["test"], sample_sec, n_permutations=500)
                test_sig = "p=%.3f" % perm["p_value"]
            else:
                test_sig = "n/a"

            if train_acc["decisive"] >= 20:
                perm_train = permutation_test(data["train"], sample_sec, n_permutations=500)
                train_sig = "p=%.3f" % perm_train["p_value"]
            else:
                train_sig = "n/a"

            # Overfit flag: train accuracy > test accuracy by > 3pp
            if train_acc["decisive"] >= 20 and test_acc["decisive"] >= 20:
                gap = train_acc["accuracy"] - test_acc["accuracy"]
                if gap > 0.03:
                    overfit = "YES (%.1fpp)" % (gap * 100)
                elif gap > 0.01:
                    overfit = "MILD (%.1fpp)" % (gap * 100)
                else:
                    overfit = "NO"
            else:
                overfit = "n/a"

            print(
                "%-14s | %5d %5d %6.1f%% %-6s | %5d %5d %6.1f%% %-6s | %s"
                % (
                    cfg.name,
                    train_acc["total"], train_acc["decisive"],
                    train_acc["accuracy"] * 100, train_sig,
                    test_acc["total"], test_acc["decisive"],
                    test_acc["accuracy"] * 100, test_sig,
                    overfit,
                )
            )

    # ── Best config summary ───────────────────────────────────────
    print("\n" + "=" * 90)
    print("  BEST CONFIG BY TEST ACCURACY (minute 4)")
    print("=" * 90)

    target_sec = 240
    scored: list[tuple[str, float, float]] = []
    for cfg in CONFIGS:
        data = all_results[cfg.name]
        test_acc = calc_accuracy(data["test"], target_sec)
        train_acc = calc_accuracy(data["train"], target_sec)
        if test_acc["decisive"] >= 20:
            scored.append((cfg.name, test_acc["accuracy"], train_acc["accuracy"]))

    scored.sort(key=lambda x: x[1], reverse=True)
    print("\n%-14s | %8s | %8s | %8s" % ("Config", "Test Acc", "Train Acc", "Gap"))
    print("-" * 50)
    for name, test_a, train_a in scored[:10]:
        gap = train_a - test_a
        print("%-14s | %7.1f%% | %7.1f%% | %+.1fpp" % (name, test_a * 100, train_a * 100, gap * 100))

    # ── Signal distribution summary ───────────────────────────────
    print("\n" + "=" * 90)
    print("  SIGNAL DISTRIBUTION (minute 4)")
    print("=" * 90)

    for cfg_name in ["current", "proposed"]:
        if cfg_name not in all_results:
            continue
        data = all_results[cfg_name]
        all_signals = []
        for r in data["train"] + data["test"]:
            if target_sec in r.signals_at:
                all_signals.append(r.signals_at[target_sec])

        if not all_signals:
            continue

        all_signals.sort()
        n = len(all_signals)
        saturated = sum(1 for s in all_signals if abs(s) >= 0.99)
        near_zero = sum(1 for s in all_signals if abs(s) < 0.05)

        print("\n%s (N=%d):" % (cfg_name, n))
        print("  P5=%.3f  P25=%.3f  P50=%.3f  P75=%.3f  P95=%.3f" % (
            all_signals[int(n * 0.05)],
            all_signals[int(n * 0.25)],
            all_signals[int(n * 0.50)],
            all_signals[int(n * 0.75)],
            all_signals[int(n * 0.95)],
        ))
        print("  Saturated (|s|>=0.99): %d (%.1f%%)" % (saturated, saturated / n * 100))
        print("  Near zero (|s|<0.05): %d (%.1f%%)" % (near_zero, near_zero / n * 100))


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Backtest L4 normalization constants against tick data"
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Path to tick_data.db (default: backtest/data/tick_data.db)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7,
        help="Fraction of markets for training (default: 0.7)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        logger.error("Database not found: %s", args.db)
        sys.exit(1)

    run_backtest(args.db, train_frac=args.train_frac)


if __name__ == "__main__":
    main()
