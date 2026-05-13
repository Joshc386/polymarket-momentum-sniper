"""High-Edge Loss Investigation — why do Bot G's high-EV trades lose?

The sizing comparison backtest found that Bot G's $4.50+ bet bucket
(820 trades, 34% of all trades, avg EV +0.183) has only 34.8% WR -- well
below break-even of 42%. These are the highest-confidence trades, and
they lose the most often.

This script slices that bucket along multiple dimensions to find the
root cause:
  - Side (YES vs NO)
  - Entry price (long shot vs favourite)
  - Estimated probability bucket
  - Probability divergence (|est_prob - implied|)
  - Regime
  - Time remaining in window
  - Hour of day (UTC)
  - Day of week
  - Which signal layer was loudest
  - Distribution of fee-adjusted EV

Usage:
    python -m backtest.high_edge_investigation
"""

import json
import sqlite3
from collections import defaultdict
from datetime import datetime

DB_PATH = "data_runtime/bot_g_signal_aligned.db"

# Bot G crypto category fee rate (for fee-adjusted EV computation)
FEE_RATE = 0.072


def fmt_pnl(x: float) -> str:
    return f"${x:+.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def load_trades() -> list[dict]:
    """Load all completed Bot G trades into a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, side, entry_price, size_usdc, pnl, edge,
               estimated_prob_up, market_implied_prob,
               oracle_lag_signal, momentum_signal, liquidation_signal,
               orderbook_signal, sentiment_signal,
               regime, time_remaining_secs, combined_signal,
               risk_size_multiplier, signal_weights
        FROM trades
        WHERE pnl IS NOT NULL AND size_usdc > 0
        ORDER BY timestamp
    """)
    cols = [d[0] for d in cur.description]
    trades = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return trades


def enrich(trade: dict) -> dict:
    """Add derived fields to a trade."""
    t = dict(trade)
    t["won"] = t["pnl"] > 0
    t["pnl_per_dollar"] = t["pnl"] / t["size_usdc"] if t["size_usdc"] else 0.0

    # Parse timestamp
    try:
        dt = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
        t["hour_utc"] = dt.hour
        t["dow"] = dt.strftime("%a")
        t["is_weekend"] = dt.weekday() >= 5
    except Exception:
        t["hour_utc"] = -1
        t["dow"] = "?"
        t["is_weekend"] = False

    # Probability divergence (the actual entry-gating value)
    est = t["estimated_prob_up"] or 0.5
    imp = t["market_implied_prob"] or 0.5
    t["prob_edge"] = abs(est - imp)

    # Fee-adjusted EV per share (what entry logic actually evaluates)
    # For YES: est_prob * (1 - price) - (1 - est_prob) * price - fee
    # For NO: (1 - est_prob) * (1 - price) - est_prob * price - fee
    price = t["entry_price"] or 0.0
    fee_per_share = FEE_RATE * price * (1 - price)
    if t["side"] == "YES":
        t["fee_adj_ev"] = est * (1 - price) - (1 - est) * price - fee_per_share
    else:
        t["fee_adj_ev"] = (1 - est) * (1 - price) - est * price - fee_per_share

    # Which signal layer was loudest? (largest absolute value)
    layers = {
        "L1_oracle": t.get("oracle_lag_signal", 0.0) or 0.0,
        "L2_momentum": t.get("momentum_signal", 0.0) or 0.0,
        "L3_liquidation": t.get("liquidation_signal", 0.0) or 0.0,
        "L4_orderbook": t.get("orderbook_signal", 0.0) or 0.0,
        "L5_sentiment": t.get("sentiment_signal", 0.0) or 0.0,
    }
    t["loudest_layer"] = max(layers.items(), key=lambda x: abs(x[1]))[0]
    t["layers"] = layers

    return t


def slice_report(name: str, trades: list[dict], group_key, sort_by_key=False):
    """Generic slicing report by some group key."""
    groups = defaultdict(list)
    for t in trades:
        groups[group_key(t)].append(t)

    print(f"\n--- {name} ---")
    print(f"  {'Bucket':<22} {'N':>6} {'WR':>7} {'PnL':>9} {'AvgEV':>8} {'AvgPnL/$':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*10}")

    items = sorted(groups.items(), key=(lambda x: x[0]) if sort_by_key else (lambda x: -len(x[1])))
    for label, ts in items:
        if not ts:
            continue
        wr = sum(1 for t in ts if t["won"]) / len(ts)
        pnl = sum(t["pnl"] for t in ts)
        avg_ev = sum(t["edge"] for t in ts) / len(ts)
        avg_ppd = sum(t["pnl_per_dollar"] for t in ts) / len(ts)
        print(f"  {str(label):<22} {len(ts):>6,} {fmt_pct(wr):>7} "
              f"{fmt_pnl(pnl):>9} {avg_ev:>+7.4f} {fmt_pnl(avg_ppd):>10}")


def main() -> None:
    print("=" * 80)
    print("HIGH-EDGE LOSS INVESTIGATION")
    print("Why do Bot G's high-EV trades systematically lose?")
    print("=" * 80)

    raw_trades = load_trades()
    trades = [enrich(t) for t in raw_trades]
    print(f"\nLoaded {len(trades):,} completed trades")

    total_pnl = sum(t["pnl"] for t in trades)
    overall_wr = sum(1 for t in trades if t["won"]) / len(trades)
    print(f"Overall: PnL {fmt_pnl(total_pnl)}, WR {fmt_pct(overall_wr)}")

    # ── EV distribution ──────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("EV DISTRIBUTION (the `edge` field = est best_ev per share)")
    print(f"{'=' * 80}")

    ev_buckets = [
        (-99, 0.05, "0-0.05"),
        (0.05, 0.10, "0.05-0.10"),
        (0.10, 0.15, "0.10-0.15"),
        (0.15, 0.20, "0.15-0.20"),
        (0.20, 0.25, "0.20-0.25"),
        (0.25, 0.30, "0.25-0.30"),
        (0.30, 99, "0.30+"),
    ]

    def ev_bucket(t):
        for lo, hi, lbl in ev_buckets:
            if lo <= t["edge"] < hi:
                return lbl
        return "?"

    slice_report("EV bucket (the actual leak we're investigating)",
                 trades, ev_bucket, sort_by_key=False)

    # Focus on high-EV trades only from now on
    high_ev = [t for t in trades if t["edge"] >= 0.15]
    low_ev = [t for t in trades if t["edge"] < 0.15]
    print(f"\nHigh-EV cohort (edge >= 0.15): {len(high_ev):,} trades, "
          f"WR {fmt_pct(sum(1 for t in high_ev if t['won'])/len(high_ev))}, "
          f"PnL {fmt_pnl(sum(t['pnl'] for t in high_ev))}")
    print(f"Low-EV cohort (edge < 0.15):  {len(low_ev):,} trades, "
          f"WR {fmt_pct(sum(1 for t in low_ev if t['won'])/len(low_ev))}, "
          f"PnL {fmt_pnl(sum(t['pnl'] for t in low_ev))}")

    # ── Slice high-EV trades along each dimension ────────────────────

    print(f"\n{'=' * 80}")
    print(f"SLICING THE HIGH-EV COHORT ({len(high_ev):,} trades, edge >= 0.15)")
    print(f"{'=' * 80}")

    # By side
    slice_report("By side (YES vs NO)", high_ev, lambda t: t["side"])

    # By entry price
    def price_bucket(t):
        p = t["entry_price"] or 0
        if p < 0.30: return "1. <0.30"
        if p < 0.40: return "2. 0.30-0.40"
        if p < 0.50: return "3. 0.40-0.50"
        if p < 0.60: return "4. 0.50-0.60"
        if p < 0.70: return "5. 0.60-0.70"
        return "6. 0.70+"

    slice_report("By entry price", high_ev, price_bucket, sort_by_key=True)

    # By probability divergence (the actual entry-gate value)
    def prob_edge_bucket(t):
        pe = t["prob_edge"]
        if pe < 0.05: return "1. <0.05"
        if pe < 0.10: return "2. 0.05-0.10"
        if pe < 0.15: return "3. 0.10-0.15"
        if pe < 0.20: return "4. 0.15-0.20"
        if pe < 0.30: return "5. 0.20-0.30"
        return "6. 0.30+"

    slice_report("By prob_edge (|est_prob - implied|)", high_ev, prob_edge_bucket, sort_by_key=True)

    # By estimated prob bucket
    def est_prob_bucket(t):
        p = t["estimated_prob_up"] or 0.5
        if p < 0.20: return "1. <0.20"
        if p < 0.30: return "2. 0.20-0.30"
        if p < 0.40: return "3. 0.30-0.40"
        if p < 0.50: return "4. 0.40-0.50"
        if p < 0.60: return "5. 0.50-0.60"
        if p < 0.70: return "6. 0.60-0.70"
        if p < 0.80: return "7. 0.70-0.80"
        return "8. 0.80+"

    slice_report("By est_prob_up", high_ev, est_prob_bucket, sort_by_key=True)

    # By regime
    slice_report("By regime", high_ev, lambda t: t["regime"])

    # By time remaining
    def tr_bucket(t):
        tr = t["time_remaining_secs"] or 0
        if tr > 240: return "1. 240-300s"
        if tr > 180: return "2. 180-240s"
        if tr > 120: return "3. 120-180s"
        if tr > 60: return "4. 60-120s"
        return "5. <60s"

    slice_report("By time remaining at entry", high_ev, tr_bucket, sort_by_key=True)

    # By hour UTC
    def hour_bucket(t):
        h = t["hour_utc"]
        return f"{h:02d}:00" if h >= 0 else "?"

    slice_report("By hour (UTC)", high_ev, hour_bucket, sort_by_key=True)

    # By day of week
    slice_report("By day of week", high_ev, lambda t: t["dow"])

    # By loudest signal layer
    slice_report("By loudest signal layer", high_ev, lambda t: t["loudest_layer"])

    # ── Combined slice: side x price (the suspected biggest leak) ────

    print(f"\n{'=' * 80}")
    print(f"CROSS-TAB: side x entry price (high-EV cohort)")
    print(f"{'=' * 80}")

    side_price_groups = defaultdict(list)
    for t in high_ev:
        sp = (t["side"], price_bucket(t))
        side_price_groups[sp].append(t)

    print(f"\n  {'Side':<5} {'Price':<15} {'N':>6} {'WR':>7} {'PnL':>9} {'AvgEV':>8}")
    print(f"  {'-'*5} {'-'*15} {'-'*6} {'-'*7} {'-'*9} {'-'*8}")
    for (side, pb), ts in sorted(side_price_groups.items()):
        wr = sum(1 for t in ts if t["won"]) / len(ts)
        pnl = sum(t["pnl"] for t in ts)
        avg_ev = sum(t["edge"] for t in ts) / len(ts)
        print(f"  {side:<5} {pb:<15} {len(ts):>6,} {fmt_pct(wr):>7} "
              f"{fmt_pnl(pnl):>9} {avg_ev:>+7.4f}")

    # ── Combined slice: regime x side ────────────────────────────────

    print(f"\n{'=' * 80}")
    print(f"CROSS-TAB: regime x side (high-EV cohort)")
    print(f"{'=' * 80}")

    rs_groups = defaultdict(list)
    for t in high_ev:
        rs_groups[(t["regime"], t["side"])].append(t)

    print(f"\n  {'Regime':<15} {'Side':<5} {'N':>6} {'WR':>7} {'PnL':>9} {'AvgEV':>8}")
    print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*7} {'-'*9} {'-'*8}")
    for (rg, sd), ts in sorted(rs_groups.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for t in ts if t["won"]) / len(ts)
        pnl = sum(t["pnl"] for t in ts)
        avg_ev = sum(t["edge"] for t in ts) / len(ts)
        print(f"  {rg:<15} {sd:<5} {len(ts):>6,} {fmt_pct(wr):>7} "
              f"{fmt_pnl(pnl):>9} {avg_ev:>+7.4f}")

    # ── Outlier check: does one tail dominate? ───────────────────────

    print(f"\n{'=' * 80}")
    print(f"OUTLIER ANALYSIS (high-EV cohort)")
    print(f"{'=' * 80}")

    # Top 10 worst losers vs top 10 best winners
    sorted_by_pnl = sorted(high_ev, key=lambda t: t["pnl"])
    print(f"\n  Top 10 LARGEST LOSSES in high-EV cohort:")
    for t in sorted_by_pnl[:10]:
        print(f"    {t['side']:>3} @ ${t['entry_price']:.3f} "
              f"size ${t['size_usdc']:.2f} ev {t['edge']:+.3f} "
              f"est_p {t['estimated_prob_up']:.3f} "
              f"regime {t['regime']:<14} pnl {fmt_pnl(t['pnl'])}")

    print(f"\n  Top 10 LARGEST WINS in high-EV cohort:")
    for t in sorted_by_pnl[-10:][::-1]:
        print(f"    {t['side']:>3} @ ${t['entry_price']:.3f} "
              f"size ${t['size_usdc']:.2f} ev {t['edge']:+.3f} "
              f"est_p {t['estimated_prob_up']:.3f} "
              f"regime {t['regime']:<14} pnl {fmt_pnl(t['pnl'])}")

    # ── Calibration check: estimated vs actual prob ──────────────────

    print(f"\n{'=' * 80}")
    print(f"CALIBRATION: estimated probability vs actual outcome")
    print(f"{'=' * 80}")

    # For YES trades, estimated prob = P(YES wins) directly.
    # For NO trades, estimated prob = P(YES wins), so P(this trade wins) = 1 - est_prob_up.
    cal_buckets = defaultdict(lambda: [0, 0])  # [n, wins]
    for t in high_ev:
        p = t["estimated_prob_up"] or 0.5
        if t["side"] == "NO":
            p = 1 - p
        # Bucket by P(this trade wins) per the model
        if p < 0.50: b = "1. model<0.50"
        elif p < 0.55: b = "2. 0.50-0.55"
        elif p < 0.60: b = "3. 0.55-0.60"
        elif p < 0.65: b = "4. 0.60-0.65"
        elif p < 0.70: b = "5. 0.65-0.70"
        elif p < 0.80: b = "6. 0.70-0.80"
        else: b = "7. 0.80+"
        cal_buckets[b][0] += 1
        if t["won"]:
            cal_buckets[b][1] += 1

    print(f"\n  Model said P(win)=X, actual WR was Y")
    print(f"  If model is calibrated, X should ~= Y in each bucket\n")
    print(f"  {'Model says P(win)':<20} {'N':>6} {'Actual WR':>10} {'Calibration':>13}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*13}")
    for label in sorted(cal_buckets.keys()):
        n, wins = cal_buckets[label]
        actual_wr = wins / n if n else 0
        # Extract midpoint for calibration check
        if "<0.50" in label: model_mid = 0.45
        elif "0.50-0.55" in label: model_mid = 0.525
        elif "0.55-0.60" in label: model_mid = 0.575
        elif "0.60-0.65" in label: model_mid = 0.625
        elif "0.65-0.70" in label: model_mid = 0.675
        elif "0.70-0.80" in label: model_mid = 0.75
        else: model_mid = 0.85
        delta = actual_wr - model_mid
        print(f"  {label:<20} {n:>6,} {fmt_pct(actual_wr):>10} "
              f"{delta:>+12.1%}")

    print(f"\n  -> Negative deltas mean model is OVERCONFIDENT (overestimating P(win))")

    # ── Final summary ────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 80}")
    high_ev_pnl = sum(t["pnl"] for t in high_ev)
    high_ev_wr = sum(1 for t in high_ev if t["won"]) / len(high_ev)
    low_ev_pnl = sum(t["pnl"] for t in low_ev)
    low_ev_wr = sum(1 for t in low_ev if t["won"]) / len(low_ev)
    print(f"\n  Low-EV trades (edge < 0.15):  {len(low_ev):,} | "
          f"WR {fmt_pct(low_ev_wr)} | PnL {fmt_pnl(low_ev_pnl)}")
    print(f"  High-EV trades (edge >= 0.15): {len(high_ev):,} | "
          f"WR {fmt_pct(high_ev_wr)} | PnL {fmt_pnl(high_ev_pnl)}")
    print(f"\n  Total impact of high-EV leak: {fmt_pnl(high_ev_pnl)} on "
          f"{len(high_ev):,} trades = {fmt_pnl(high_ev_pnl/len(high_ev))}/trade")


if __name__ == "__main__":
    main()
