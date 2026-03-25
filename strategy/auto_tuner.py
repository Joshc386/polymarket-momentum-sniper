"""Phase 7: Auto-tuner — optimizes signal weights and thresholds from historical data.

Analyses past trades and signal logs to recommend parameter adjustments.
Can be run standalone: python -m strategy.auto_tuner --db PATH

Approach:
1. Measures each signal layer's predictive power (correlation with outcomes)
2. Adjusts weights proportional to each layer's information value
3. Tunes edge thresholds based on realized hit rate at different levels
4. Outputs recommended config changes (does NOT auto-apply)
"""

import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerStats:
    """Statistics for a single signal layer."""
    name: str
    avg_when_up: float = 0.0
    avg_when_down: float = 0.0
    separation: float = 0.0       # avg_up - avg_down (higher = more predictive)
    correlation: float = 0.0      # point-biserial approximation
    win_rate_when_positive: float = 0.0
    win_rate_when_negative: float = 0.0
    recommended_weight: float = 0.0


@dataclass
class TimingStats:
    """Performance by entry timing bucket."""
    bucket: str
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    total_pnl: float = 0.0
    hit_rate_above_threshold: float = 0.0  # % of trades above edge threshold that won


@dataclass
class TuningReport:
    """Full auto-tuning recommendation."""
    total_signals: int = 0
    total_trades: int = 0
    layer_stats: list[LayerStats] = field(default_factory=list)
    timing_stats: list[TimingStats] = field(default_factory=list)
    # Recommendations
    rec_oracle_weight_early: float = 0.35
    rec_oracle_weight_late: float = 0.25
    rec_momentum_weight_early: float = 0.40
    rec_momentum_weight_late: float = 0.65
    rec_liquidation_weight_early: float = 0.25
    rec_liquidation_weight_late: float = 0.10
    rec_max_adjustment: float = 0.15
    rec_min_edge: float = 0.02
    rec_max_edge: float = 0.10
    rec_notes: list[str] = field(default_factory=list)


class AutoTuner:
    """Analyses historical data and recommends parameter changes."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def close(self):
        if self.conn:
            self.conn.close()

    def analyse(self, min_trades: int = 20) -> TuningReport:
        """Run full analysis and generate recommendations.

        Args:
            min_trades: Minimum resolved trades required to produce recommendations.
        """
        report = TuningReport()

        # Get resolved trades with signal data
        trades = self.conn.execute("""
            SELECT * FROM trades
            WHERE resolution IS NOT NULL
            ORDER BY timestamp ASC
        """).fetchall()

        report.total_trades = len(trades)

        if len(trades) < min_trades:
            report.rec_notes.append(
                f"Only {len(trades)} trades (need {min_trades}+). "
                "Keeping default parameters. Collect more data."
            )
            return report

        # Get signal log entries matched to resolutions
        signals = self.conn.execute("""
            SELECT s.*, t.resolution
            FROM signal_log s
            INNER JOIN (
                SELECT market_id, resolution
                FROM trades
                WHERE resolution IS NOT NULL
                GROUP BY market_id
            ) t ON s.market_id = t.market_id
        """).fetchall()

        report.total_signals = len(signals)

        # 1. Analyse each signal layer
        report.layer_stats = self._analyse_layers(signals)

        # 2. Analyse timing buckets
        report.timing_stats = self._analyse_timing(trades)

        # 3. Generate weight recommendations
        self._recommend_weights(report)

        # 4. Tune edge thresholds
        self._recommend_edges(trades, report)

        # 5. Tune max_adjustment
        self._recommend_max_adjustment(signals, report)

        return report

    def _analyse_layers(self, signals: list) -> list[LayerStats]:
        """Measure each signal layer's predictive power."""
        layers = {
            "oracle_lag": LayerStats(name="Oracle Lag"),
            "momentum": LayerStats(name="Momentum"),
            "liquidation": LayerStats(name="Liquidation"),
        }

        # Collect values by outcome
        values_up = {k: [] for k in layers}
        values_down = {k: [] for k in layers}

        for row in signals:
            res = row["resolution"]
            is_up = res == "UP"
            target = values_up if is_up else values_down

            olag = row["oracle_lag_signal"]
            mom = row["momentum_signal"]
            liq = row["liquidation_signal"]

            if olag is not None:
                target["oracle_lag"].append(olag)
            if mom is not None:
                target["momentum"].append(mom)
            if liq is not None:
                target["liquidation"].append(liq)

        for key, stats in layers.items():
            up_vals = values_up[key]
            down_vals = values_down[key]

            if up_vals:
                stats.avg_when_up = statistics.mean(up_vals)
            if down_vals:
                stats.avg_when_down = statistics.mean(down_vals)

            stats.separation = stats.avg_when_up - stats.avg_when_down

            # Win rate when signal is positive vs negative
            all_vals = [(v, True) for v in up_vals] + [(v, False) for v in down_vals]
            pos = [is_up for v, is_up in all_vals if v > 0]
            neg = [is_up for v, is_up in all_vals if v < 0]
            stats.win_rate_when_positive = sum(pos) / len(pos) if pos else 0.5
            stats.win_rate_when_negative = sum(neg) / len(neg) if neg else 0.5

            # Simple correlation approximation
            if up_vals and down_vals:
                all_signals = up_vals + down_vals
                all_outcomes = [1.0] * len(up_vals) + [0.0] * len(down_vals)
                if len(all_signals) > 2:
                    stats.correlation = self._simple_correlation(all_signals, all_outcomes)

        return list(layers.values())

    def _analyse_timing(self, trades: list) -> list[TimingStats]:
        """Analyse performance by entry timing."""
        buckets_def = [
            ("5:00-3:00", 300, 180),
            ("3:00-1:30", 180, 90),
            ("1:30-0:30", 90, 30),
            ("0:30-0:05", 30, 5),
        ]

        buckets = {}
        for label, _, _ in buckets_def:
            buckets[label] = TimingStats(bucket=label)

        for row in trades:
            tr = row["time_remaining_secs"] or 0
            label = None
            for bl, high, low in buckets_def:
                if low <= tr < high:
                    label = bl
                    break
            if label is None:
                label = "5:00-3:00" if tr >= 300 else "0:30-0:05"

            b = buckets[label]
            b.trades += 1
            pnl = row["pnl"] or 0
            b.total_pnl += pnl
            if pnl > 0:
                b.wins += 1
            edge = row["edge"]
            if edge is not None:
                b.avg_edge = (b.avg_edge * (b.trades - 1) + edge) / b.trades

        for b in buckets.values():
            b.win_rate = b.wins / b.trades if b.trades > 0 else 0

        return list(buckets.values())

    def _recommend_weights(self, report: TuningReport):
        """Recommend signal weights based on layer predictive power."""
        stats = {s.name: s for s in report.layer_stats}

        # Use separation as weight basis
        separations = {
            "oracle": abs(stats.get("Oracle Lag", LayerStats(name="")).separation),
            "momentum": abs(stats.get("Momentum", LayerStats(name="")).separation),
            "liquidation": abs(stats.get("Liquidation", LayerStats(name="")).separation),
        }

        total_sep = sum(separations.values())
        if total_sep <= 0:
            report.rec_notes.append("No signal separation detected — keeping defaults")
            return

        # Normalized weights
        raw_weights = {k: v / total_sep for k, v in separations.items()}

        # Clamp each weight to [0.10, 0.60] to prevent over-fitting
        for k in raw_weights:
            raw_weights[k] = max(0.10, min(0.60, raw_weights[k]))

        # Re-normalize
        total = sum(raw_weights.values())
        weights = {k: v / total for k, v in raw_weights.items()}

        # Early vs late: momentum gets more weight late, oracle more early
        # Use a 0.8/1.2 shift factor
        report.rec_oracle_weight_early = round(weights["oracle"] * 1.15, 2)
        report.rec_oracle_weight_late = round(weights["oracle"] * 0.85, 2)
        report.rec_momentum_weight_early = round(weights["momentum"] * 0.85, 2)
        report.rec_momentum_weight_late = round(weights["momentum"] * 1.15, 2)
        report.rec_liquidation_weight_early = round(weights["liquidation"] * 1.15, 2)
        report.rec_liquidation_weight_late = round(weights["liquidation"] * 0.85, 2)

        # Set recommended weights on layer stats for display
        for s in report.layer_stats:
            if "Oracle" in s.name:
                s.recommended_weight = weights["oracle"]
            elif "Momentum" in s.name:
                s.recommended_weight = weights["momentum"]
            elif "Liquidation" in s.name:
                s.recommended_weight = weights["liquidation"]

        report.rec_notes.append(
            f"Weight recommendation: Oracle={weights['oracle']:.2f} "
            f"Mom={weights['momentum']:.2f} Liq={weights['liquidation']:.2f}"
        )

    def _recommend_edges(self, trades: list, report: TuningReport):
        """Tune min/max edge thresholds based on realized outcomes."""
        # Group trades by edge level and measure win rate
        edge_bins = {}
        for row in trades:
            edge = row["edge"]
            if edge is None:
                continue
            # Bin to nearest 0.01
            bin_key = round(edge * 100) / 100
            if bin_key not in edge_bins:
                edge_bins[bin_key] = {"total": 0, "wins": 0}
            edge_bins[bin_key]["total"] += 1
            if (row["pnl"] or 0) > 0:
                edge_bins[bin_key]["wins"] += 1

        if not edge_bins:
            return

        # Find the minimum edge where win rate > 52% (need >50% to be profitable after fees)
        profitable_edges = []
        for edge_val in sorted(edge_bins.keys()):
            b = edge_bins[edge_val]
            if b["total"] >= 3:  # Minimum sample
                wr = b["wins"] / b["total"]
                if wr > 0.52:
                    profitable_edges.append(edge_val)

        if profitable_edges:
            report.rec_min_edge = max(0.01, min(profitable_edges) - 0.01)
            report.rec_notes.append(
                f"Min profitable edge observed at {min(profitable_edges):.3f}, "
                f"recommending min_edge={report.rec_min_edge:.3f}"
            )

        # Check if trades at high edge are winning more
        high_edge_trades = [r for r in trades if (r["edge"] or 0) > 0.06]
        if len(high_edge_trades) >= 5:
            wr = sum(1 for r in high_edge_trades if (r["pnl"] or 0) > 0) / len(high_edge_trades)
            if wr > 0.60:
                report.rec_notes.append(
                    f"High-edge trades (>6%) have {wr:.0%} win rate — consider lower max_edge to trade more"
                )
                report.rec_max_edge = 0.08
            elif wr < 0.48:
                report.rec_notes.append(
                    f"High-edge trades (>6%) only {wr:.0%} — signals may be noisy, raise max_edge"
                )
                report.rec_max_edge = 0.12

    def _recommend_max_adjustment(self, signals: list, report: TuningReport):
        """Tune max_adjustment — how far from 50% the model can go."""
        # Check: when model says >57%, does UP actually happen more?
        confident_up = [s for s in signals if (s["estimated_prob_up"] or 0.5) > 0.57]
        confident_down = [s for s in signals if (s["estimated_prob_up"] or 0.5) < 0.43]

        if len(confident_up) >= 10:
            actual_up = sum(1 for s in confident_up if s["resolution"] == "UP") / len(confident_up)
            if actual_up > 0.55:
                report.rec_notes.append(
                    f"Model confident-UP signals are {actual_up:.0%} accurate — "
                    f"consider raising max_adjustment to widen range"
                )
                report.rec_max_adjustment = min(0.20, report.rec_max_adjustment + 0.02)
            elif actual_up < 0.48:
                report.rec_notes.append(
                    f"Model confident-UP signals only {actual_up:.0%} accurate — "
                    f"reduce max_adjustment to be more conservative"
                )
                report.rec_max_adjustment = max(0.10, report.rec_max_adjustment - 0.03)

    @staticmethod
    def _simple_correlation(x: list[float], y: list[float]) -> float:
        """Simple Pearson correlation."""
        n = len(x)
        if n < 3:
            return 0.0
        mx = statistics.mean(x)
        my = statistics.mean(y)
        sx = statistics.stdev(x) if n > 1 else 1.0
        sy = statistics.stdev(y) if n > 1 else 1.0
        if sx == 0 or sy == 0:
            return 0.0
        return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / ((n - 1) * sx * sy)


# ── Terminal Report ───────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_tuning_report(r: TuningReport):
    print()
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  AUTO-TUNING REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"  Signals analysed: {r.total_signals}")
    print(f"  Trades analysed:  {r.total_trades}")
    print()

    # Layer analysis
    print(f"  {BOLD}-- Signal Layer Predictive Power --{RESET}")
    print(f"  {'Layer':<15} {'Sep':>8} {'Corr':>8} {'WR+':>7} {'WR-':>7} {'RecW':>7}")
    print(f"  {'-' * 55}")
    for s in r.layer_stats:
        sc = GREEN if s.separation > 0.01 else RED if s.separation < -0.01 else DIM
        print(
            f"  {s.name:<15} {sc}{s.separation:>+8.4f}{RESET} "
            f"{s.correlation:>+8.4f} "
            f"{s.win_rate_when_positive:>6.0%} "
            f"{s.win_rate_when_negative:>6.0%} "
            f"{s.recommended_weight:>6.0%}"
        )
    print()

    # Timing analysis
    if r.timing_stats:
        print(f"  {BOLD}-- Performance by Entry Timing --{RESET}")
        print(f"  {'Bucket':<14} {'Trades':>7} {'WR':>7} {'AvgEdge':>9} {'P&L':>10}")
        print(f"  {'-' * 50}")
        for t in r.timing_stats:
            if t.trades == 0:
                continue
            pc = GREEN if t.total_pnl > 0 else RED
            print(
                f"  {t.bucket:<14} {t.trades:>7} {t.win_rate:>6.0%} "
                f"{t.avg_edge:>+8.4f} {pc}${t.total_pnl:>+9.2f}{RESET}"
            )
        print()

    # Recommendations
    print(f"  {BOLD}-- Recommended Parameters --{RESET}")
    print(f"  Oracle weight:      early={r.rec_oracle_weight_early:.2f} late={r.rec_oracle_weight_late:.2f}")
    print(f"  Momentum weight:    early={r.rec_momentum_weight_early:.2f} late={r.rec_momentum_weight_late:.2f}")
    print(f"  Liquidation weight: early={r.rec_liquidation_weight_early:.2f} late={r.rec_liquidation_weight_late:.2f}")
    print(f"  max_adjustment:     {r.rec_max_adjustment:.2f}")
    print(f"  min_edge:           {r.rec_min_edge:.3f}")
    print(f"  max_edge:           {r.rec_max_edge:.3f}")
    print()

    if r.rec_notes:
        print(f"  {BOLD}-- Notes --{RESET}")
        for note in r.rec_notes:
            print(f"  {DIM}* {note}{RESET}")
        print()

    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-tune signal parameters")
    parser.add_argument("--db", default="./data_runtime/trades.db")
    parser.add_argument("--min-trades", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    tuner = AutoTuner(args.db)
    tuner.connect()
    report = tuner.analyse(min_trades=args.min_trades)
    print_tuning_report(report)
    tuner.close()


if __name__ == "__main__":
    main()
