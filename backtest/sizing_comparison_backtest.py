"""Sizing Comparison Backtest — flat vs Kelly sizing on Bot G's trades.

Takes Bot G's actual completed trades and rescales each trade's PnL to
different sizing schemes. For binary outcomes, PnL scales linearly with
bet size (win: bet x (1-p)/p; loss: -bet), so we can compute per-trade
PnL ratio and rescale exactly without re-simulating signals.

Schemes tested:
  - Current Kelly (baseline — actual trades as they happened)
  - Flat $1, $1.50, $2, $2.50, $3, $4, $5

Diagnostic question: are Kelly's larger bets correlated with winners?
  - YES -> flat sizing hurts (cuts winners more than losses)
  - NO  -> flat sizing helps (Kelly is overconfident on losers)

Usage:
    python -m backtest.sizing_comparison_backtest
"""

import sqlite3
import statistics
from collections import defaultdict
from typing import NamedTuple

DB_PATH = "data_runtime/bot_g_signal_aligned.db"

# Sizing schemes to test (USD per trade)
FLAT_SIZES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


class Trade(NamedTuple):
    timestamp: str
    side: str
    entry_price: float
    size_usdc: float
    pnl: float
    edge: float
    regime: str
    risk_multiplier: float
    won: bool
    pnl_per_dollar: float  # pnl / size_usdc -- the per-$1 return


def load_trades() -> list[Trade]:
    """Load all completed Bot G trades."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, side, entry_price, size_usdc, pnl, edge,
               regime, risk_size_multiplier
        FROM trades
        WHERE pnl IS NOT NULL
          AND size_usdc > 0
        ORDER BY timestamp
    """)
    trades = []
    for row in cur.fetchall():
        ts, side, entry_price, size, pnl, edge, regime, risk_mult = row
        if size <= 0:
            continue
        ppd = pnl / size
        trades.append(Trade(
            timestamp=ts or "",
            side=side or "",
            entry_price=entry_price or 0.0,
            size_usdc=size,
            pnl=pnl,
            edge=edge or 0.0,
            regime=regime or "unknown",
            risk_multiplier=risk_mult or 1.0,
            won=(pnl > 0),
            pnl_per_dollar=ppd,
        ))
    conn.close()
    return trades


def compute_metrics(pnls: list[float]) -> dict:
    """Compute summary metrics for a PnL series."""
    if not pnls:
        return {}

    total = sum(pnls)
    n = len(pnls)
    mean_pnl = total / n

    # Std and Sharpe
    if n > 1:
        std = statistics.stdev(pnls)
        sharpe = mean_pnl / std if std > 0 else 0.0
    else:
        std = 0.0
        sharpe = 0.0

    # Running cumulative for drawdown
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "total_pnl": total,
        "n": n,
        "mean_pnl": mean_pnl,
        "std": std,
        "sharpe_per_trade": sharpe,
        "max_drawdown": max_dd,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else 0.0,
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
        "biggest_win": max(pnls),
        "biggest_loss": min(pnls),
    }


def main() -> None:
    print("=" * 80)
    print("SIZING COMPARISON BACKTEST")
    print("Rescaling Bot G's actual trades to flat sizing schemes")
    print("=" * 80)

    trades = load_trades()
    print(f"\nLoaded {len(trades):,} completed trades from Bot G")

    if not trades:
        print("No trades found.")
        return

    actual_pnls = [t.pnl for t in trades]
    actual_metrics = compute_metrics(actual_pnls)
    actual_sizes = [t.size_usdc for t in trades]

    print(f"\n--- ACTUAL (Kelly + regime + risk multipliers) ---")
    print(f"  Total PnL:        ${actual_metrics['total_pnl']:+.2f}")
    print(f"  Avg PnL/trade:    ${actual_metrics['mean_pnl']:+.4f}")
    print(f"  Win rate:         {actual_metrics['win_rate']:.1%} ({actual_metrics['wins']:,}/{actual_metrics['n']:,})")
    print(f"  Avg win:          ${actual_metrics['avg_win']:+.3f}")
    print(f"  Avg loss:         ${actual_metrics['avg_loss']:+.3f}")
    print(f"  Biggest win:      ${actual_metrics['biggest_win']:+.2f}")
    print(f"  Biggest loss:     ${actual_metrics['biggest_loss']:+.2f}")
    print(f"  Std/trade:        ${actual_metrics['std']:.3f}")
    print(f"  Sharpe/trade:     {actual_metrics['sharpe_per_trade']:.4f}")
    print(f"  Max drawdown:     ${actual_metrics['max_drawdown']:.2f}")
    print(f"  Avg bet size:     ${sum(actual_sizes)/len(actual_sizes):.3f}")
    print(f"  Min bet size:     ${min(actual_sizes):.2f}")
    print(f"  Max bet size:     ${max(actual_sizes):.2f}")

    # ── Flat sizing tests ─────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("FLAT SIZING SCHEMES")
    print(f"{'=' * 80}")
    print(f"\n{'Size':>8} {'TotalPnL':>10} {'vs Kelly':>10} "
          f"{'AvgPnL':>9} {'WR':>6} {'AvgWin':>8} {'AvgLoss':>8} "
          f"{'Sharpe':>7} {'MaxDD':>8}")
    print("-" * 85)

    # Include current Kelly as reference row
    print(f"{'KELLY':>8} "
          f"${actual_metrics['total_pnl']:>+9.2f} "
          f"{'---':>10} "
          f"${actual_metrics['mean_pnl']:>+8.4f} "
          f"{actual_metrics['win_rate']:>5.1%} "
          f"${actual_metrics['avg_win']:>+7.3f} "
          f"${actual_metrics['avg_loss']:>+7.3f} "
          f"{actual_metrics['sharpe_per_trade']:>7.4f} "
          f"${actual_metrics['max_drawdown']:>7.2f}")
    print()

    flat_results = {}
    for size in FLAT_SIZES:
        flat_pnls = [t.pnl_per_dollar * size for t in trades]
        m = compute_metrics(flat_pnls)
        flat_results[size] = m
        delta = m["total_pnl"] - actual_metrics["total_pnl"]
        print(f"${size:>6.2f} "
              f"${m['total_pnl']:>+9.2f} "
              f"${delta:>+9.2f} "
              f"${m['mean_pnl']:>+8.4f} "
              f"{m['win_rate']:>5.1%} "
              f"${m['avg_win']:>+7.3f} "
              f"${m['avg_loss']:>+7.3f} "
              f"{m['sharpe_per_trade']:>7.4f} "
              f"${m['max_drawdown']:>7.2f}")

    # ── Range $1-$2 (Kelly clamped to that range) ─────────────────────

    print(f"\n{'=' * 80}")
    print("KELLY CLAMPED TO $1-$2 (preserves edge scaling within range)")
    print(f"{'=' * 80}")

    # Recompute Kelly bet within $1-$2 bounds.
    # Take the *unscaled* Kelly intent: actual_size / actual_risk_multiplier
    # gives the "pre-risk" Kelly bet. Then clamp to [1, 2].
    clamped_pnls = []
    n_at_min = 0
    n_at_max = 0
    n_mid = 0
    for t in trades:
        # Reverse out risk multiplier to get the "raw" Kelly intent
        raw_kelly = t.size_usdc / max(t.risk_multiplier, 0.01)
        # Clamp to $1-$2
        new_size = max(1.0, min(2.0, raw_kelly))
        if new_size <= 1.001:
            n_at_min += 1
        elif new_size >= 1.999:
            n_at_max += 1
        else:
            n_mid += 1
        new_pnl = t.pnl_per_dollar * new_size
        clamped_pnls.append(new_pnl)

    clamped_m = compute_metrics(clamped_pnls)
    print(f"\n  Total PnL:        ${clamped_m['total_pnl']:+.2f}")
    print(f"  vs Kelly:         ${clamped_m['total_pnl'] - actual_metrics['total_pnl']:+.2f}")
    print(f"  Avg PnL/trade:    ${clamped_m['mean_pnl']:+.4f}")
    print(f"  Sharpe/trade:     {clamped_m['sharpe_per_trade']:.4f}")
    print(f"  Max drawdown:     ${clamped_m['max_drawdown']:.2f}")
    print(f"\n  Bet distribution:")
    print(f"    At $1 floor:    {n_at_min:,} ({n_at_min/len(trades):.1%})")
    print(f"    In $1-$2 range: {n_mid:,} ({n_mid/len(trades):.1%})")
    print(f"    At $2 ceiling:  {n_at_max:,} ({n_at_max/len(trades):.1%})")

    # ── Diagnostic: is Kelly sizing well-calibrated? ──────────────────

    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC: Is Kelly's sizing well-calibrated?")
    print(f"{'=' * 80}")

    # Group trades by Kelly size buckets, look at win rate and PnL/dollar
    size_buckets = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 99)]

    print(f"\nDoes Kelly bet bigger on winners or losers?\n")
    print(f"  {'Size bucket':<15} {'N':>6} {'WR':>7} {'PnL/$':>8} {'AvgEdge':>9}")
    print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*8} {'-'*9}")
    for lo, hi in size_buckets:
        bucket = [t for t in trades if lo <= t.size_usdc < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t.won) / len(bucket)
        ppd = sum(t.pnl_per_dollar for t in bucket) / len(bucket)
        avg_edge = sum(t.edge for t in bucket) / len(bucket)
        label = f"${lo:.1f}-${hi:.1f}"
        print(f"  {label:<15} {len(bucket):>6,} {wr:>6.1%} "
              f"${ppd:>+7.3f} {avg_edge:>+8.3f}")

    # Correlation: bet size vs PnL per dollar
    sizes = [t.size_usdc for t in trades]
    ppds = [t.pnl_per_dollar for t in trades]
    n = len(trades)
    mean_s = sum(sizes) / n
    mean_p = sum(ppds) / n
    cov = sum((s - mean_s) * (p - mean_p) for s, p in zip(sizes, ppds)) / n
    var_s = sum((s - mean_s) ** 2 for s in sizes) / n
    var_p = sum((p - mean_p) ** 2 for p in ppds) / n
    corr = cov / ((var_s ** 0.5) * (var_p ** 0.5)) if var_s > 0 and var_p > 0 else 0

    print(f"\nCorrelation (bet_size, pnl_per_dollar): {corr:+.4f}")
    if abs(corr) < 0.02:
        print("  -> Effectively zero -- Kelly sizing has NO predictive power over outcomes.")
        print("  -> Flat sizing should produce similar PnL with lower variance.")
    elif corr > 0:
        print("  -> Positive: bigger bets tend to be winners. Kelly adds value.")
        print("  -> Flat sizing would hurt PnL (cuts winners more than losses).")
    else:
        print("  -> Negative: bigger bets tend to be LOSERS. Kelly is miscalibrated!")
        print("  -> Flat sizing would IMPROVE PnL (cuts losers more than winners).")

    # ── Per-regime breakdown ──────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("PER-REGIME BREAKDOWN")
    print(f"{'=' * 80}")

    regimes = defaultdict(list)
    for t in trades:
        regimes[t.regime].append(t)

    print(f"\n  {'Regime':<15} {'N':>6} {'WR':>7} "
          f"{'KellyTotal':>11} {'$1 flat':>9} {'$2 flat':>9} "
          f"{'KellySize':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*11} {'-'*9} {'-'*9} {'-'*10}")
    for regime, rt in sorted(regimes.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for t in rt if t.won) / len(rt)
        kelly_total = sum(t.pnl for t in rt)
        flat_1 = sum(t.pnl_per_dollar * 1.0 for t in rt)
        flat_2 = sum(t.pnl_per_dollar * 2.0 for t in rt)
        avg_size = sum(t.size_usdc for t in rt) / len(rt)
        print(f"  {regime:<15} {len(rt):>6,} {wr:>6.1%} "
              f"${kelly_total:>+10.2f} ${flat_1:>+8.2f} ${flat_2:>+8.2f} "
              f"${avg_size:>9.3f}")

    # ── Winners vs losers analysis ────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("WINNER VS LOSER BET SIZING")
    print(f"{'=' * 80}")
    winners = [t for t in trades if t.won]
    losers = [t for t in trades if not t.won]

    if winners and losers:
        avg_size_w = sum(t.size_usdc for t in winners) / len(winners)
        avg_size_l = sum(t.size_usdc for t in losers) / len(losers)
        avg_edge_w = sum(t.edge for t in winners) / len(winners)
        avg_edge_l = sum(t.edge for t in losers) / len(losers)

        print(f"\n  Winners ({len(winners):,} trades):")
        print(f"    Avg bet size: ${avg_size_w:.3f}")
        print(f"    Avg edge:     {avg_edge_w:+.4f}")
        print(f"  Losers ({len(losers):,} trades):")
        print(f"    Avg bet size: ${avg_size_l:.3f}")
        print(f"    Avg edge:     {avg_edge_l:+.4f}")
        size_diff = avg_size_w - avg_size_l
        print(f"\n  Bet size difference (winners - losers): ${size_diff:+.4f}")
        if abs(size_diff) < 0.05:
            print(f"  -> Essentially identical. Kelly is NOT identifying winners ex-ante.")
        elif size_diff > 0:
            print(f"  -> Kelly bets bigger on winners. Sizing adds value.")
        else:
            print(f"  -> Kelly bets bigger on LOSERS. Sizing actively costs money.")

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    best_flat_size = max(flat_results.keys(), key=lambda s: flat_results[s]["total_pnl"])
    best_flat = flat_results[best_flat_size]
    best_sharpe_size = max(flat_results.keys(),
                            key=lambda s: flat_results[s]["sharpe_per_trade"])
    best_sharpe = flat_results[best_sharpe_size]

    print(f"\n  Current Kelly:           ${actual_metrics['total_pnl']:+.2f} PnL, "
          f"Sharpe {actual_metrics['sharpe_per_trade']:.4f}, "
          f"DD ${actual_metrics['max_drawdown']:.2f}")
    print(f"  Best flat by PnL:        ${best_flat_size:.2f} -> "
          f"${best_flat['total_pnl']:+.2f} "
          f"(${best_flat['total_pnl']-actual_metrics['total_pnl']:+.2f} vs Kelly)")
    print(f"  Best flat by Sharpe:     ${best_sharpe_size:.2f} -> "
          f"Sharpe {best_sharpe['sharpe_per_trade']:.4f}, "
          f"PnL ${best_sharpe['total_pnl']:+.2f}")
    print(f"  Kelly clamped $1-$2:     ${clamped_m['total_pnl']:+.2f} PnL, "
          f"Sharpe {clamped_m['sharpe_per_trade']:.4f}, "
          f"DD ${clamped_m['max_drawdown']:.2f}")


if __name__ == "__main__":
    main()
