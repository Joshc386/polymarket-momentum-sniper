"""Flow Proxy Investigation — does book depth / spread explain the hour effect?

Hypothesis: Bot G's edge depends on retail flow (sloppy retail orders create
the mispricings Bot G fades). Hours with low retail flow -> Bot G has no
real opponent -> worse outcomes. This investigation tests the hypothesis
using orderbook depth + spread as a proxy for market activity.

Three tests:
  1. Direct: bucket trades by depth/spread, compare WR/PnL across buckets
  2. Mediation: does hour-of-day effect dissolve when we control for depth?
  3. High-EV concentration: are high-EV losers concentrated in low-depth windows?

Available data (per trade in Bot G DB):
  - ob_yes_depth: YES bid depth at entry (5-level sum)
  - ob_ask_depth: YES ask depth at entry (5-level sum)
  - ob_spread: bid-ask spread at entry
  - timestamp: for hour-of-day
  - edge, side, regime, pnl, etc.

Note: orderbook DEPTH is not the same as trade VOLUME, but it correlates
with market activity. Thick book + tight spread = active market.

Usage:
    python -m backtest.flow_proxy_investigation
"""

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime

DB_PATH = "data_runtime/bot_g_signal_aligned.db"


def fmt_pnl(x: float) -> str:
    return f"${x:+.2f}"


def fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def load_trades() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, side, entry_price, size_usdc, pnl, edge,
               estimated_prob_up, regime, time_remaining_secs,
               ob_spread, ob_yes_depth, ob_ask_depth, combined_signal
        FROM trades
        WHERE pnl IS NOT NULL AND size_usdc > 0
        ORDER BY timestamp
    """)
    cols = [d[0] for d in cur.description]
    out = []
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        d["won"] = d["pnl"] > 0
        d["pnl_per_dollar"] = d["pnl"] / d["size_usdc"] if d["size_usdc"] else 0
        d["total_depth"] = (d.get("ob_yes_depth") or 0) + (d.get("ob_ask_depth") or 0)
        d["spread"] = d.get("ob_spread") or 0.0
        try:
            dt = datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00"))
            d["hour_utc"] = dt.hour
            d["dow"] = dt.strftime("%a")
        except Exception:
            d["hour_utc"] = -1
            d["dow"] = "?"
        out.append(d)
    conn.close()
    return out


def summary(label: str, ts: list[dict]) -> str:
    if not ts:
        return f"  {label:<25} (no trades)"
    n = len(ts)
    wins = sum(1 for t in ts if t["won"])
    wr = wins / n
    pnl = sum(t["pnl"] for t in ts)
    avg_pnl = pnl / n
    avg_ev = sum(t["edge"] for t in ts) / n
    return (f"  {label:<25} N={n:>5,}  WR {fmt_pct(wr):>6}  "
            f"PnL {fmt_pnl(pnl):>9}  Avg {fmt_pnl(avg_pnl):>8}  "
            f"AvgEV {avg_ev:+.4f}")


def main() -> None:
    print("=" * 85)
    print("FLOW PROXY INVESTIGATION — does book depth/spread explain hour effect?")
    print("=" * 85)

    trades = load_trades()
    n = len(trades)
    overall_pnl = sum(t["pnl"] for t in trades)
    overall_wr = sum(1 for t in trades if t["won"]) / n
    print(f"\nLoaded {n:,} trades, overall PnL {fmt_pnl(overall_pnl)}, "
          f"WR {fmt_pct(overall_wr)}")

    # ── Distribution of depth and spread ─────────────────────────────

    depths = sorted(t["total_depth"] for t in trades)
    spreads = sorted(t["spread"] for t in trades)

    def pct(arr, p):
        return arr[int(len(arr) * p)] if arr else 0

    print(f"\nDepth (ob_yes + ob_ask) distribution:")
    print(f"  p10 ${pct(depths, 0.10):>8.0f}  p25 ${pct(depths, 0.25):>8.0f}  "
          f"p50 ${pct(depths, 0.50):>8.0f}  p75 ${pct(depths, 0.75):>8.0f}  "
          f"p90 ${pct(depths, 0.90):>8.0f}")

    print(f"\nSpread distribution:")
    print(f"  p10 ${pct(spreads, 0.10):.4f}  p25 ${pct(spreads, 0.25):.4f}  "
          f"p50 ${pct(spreads, 0.50):.4f}  p75 ${pct(spreads, 0.75):.4f}  "
          f"p90 ${pct(spreads, 0.90):.4f}")

    # ── TEST 1A: Bucket by total book depth ──────────────────────────

    print(f"\n{'=' * 85}")
    print("TEST 1A: WR/PnL by BOOK DEPTH bucket (low depth = quiet market?)")
    print(f"{'=' * 85}\n")

    # Use quintiles
    p20 = pct(depths, 0.20)
    p40 = pct(depths, 0.40)
    p60 = pct(depths, 0.60)
    p80 = pct(depths, 0.80)

    depth_buckets = [
        (-1, p20, f"1. <${p20:.0f} (thinnest 20%)"),
        (p20, p40, f"2. ${p20:.0f}-${p40:.0f}"),
        (p40, p60, f"3. ${p40:.0f}-${p60:.0f} (median)"),
        (p60, p80, f"4. ${p60:.0f}-${p80:.0f}"),
        (p80, 1e12, f"5. >${p80:.0f} (thickest 20%)"),
    ]

    for lo, hi, lbl in depth_buckets:
        bucket = [t for t in trades if lo < t["total_depth"] <= hi]
        print(summary(lbl, bucket))

    # ── TEST 1B: Bucket by spread ────────────────────────────────────

    print(f"\n{'=' * 85}")
    print("TEST 1B: WR/PnL by SPREAD bucket (wider spread = nervous MMs?)")
    print(f"{'=' * 85}\n")

    sp20 = pct(spreads, 0.20)
    sp40 = pct(spreads, 0.40)
    sp60 = pct(spreads, 0.60)
    sp80 = pct(spreads, 0.80)

    spread_buckets = [
        (-1, sp20, f"1. <${sp20:.4f} (tightest)"),
        (sp20, sp40, f"2. ${sp20:.4f}-${sp40:.4f}"),
        (sp40, sp60, f"3. ${sp40:.4f}-${sp60:.4f}"),
        (sp60, sp80, f"4. ${sp60:.4f}-${sp80:.4f}"),
        (sp80, 1.0, f"5. >${sp80:.4f} (widest)"),
    ]

    for lo, hi, lbl in spread_buckets:
        bucket = [t for t in trades if lo < t["spread"] <= hi]
        print(summary(lbl, bucket))

    # ── TEST 2: Does hour effect persist controlling for depth? ──────

    print(f"\n{'=' * 85}")
    print("TEST 2: Does HOUR effect persist after controlling for DEPTH?")
    print(f"{'=' * 85}\n")

    # Bad hours from earlier: 09, 10, 11, 12 UTC (UK morning)
    bad_hours = {9, 10, 11, 12}
    good_hours_baseline = set(range(24)) - bad_hours

    print("Hypothesis: if hour effect is mediated by flow (depth), then\n"
          "within the same depth bucket, bad-hour trades should perform similarly\n"
          "to good-hour trades.\n")

    print(f"{'Depth bucket':<28} {'BadHours (9-12)':<25} {'OtherHours':<25}")
    print("-" * 80)

    for lo, hi, lbl in depth_buckets:
        in_bucket = [t for t in trades if lo < t["total_depth"] <= hi]
        bad = [t for t in in_bucket if t["hour_utc"] in bad_hours]
        good = [t for t in in_bucket if t["hour_utc"] not in bad_hours]
        if not bad and not good:
            continue
        bad_str = f"N={len(bad):>4} WR {fmt_pct(sum(1 for t in bad if t['won'])/len(bad) if bad else 0):>5}" if bad else "(no trades)"
        good_str = f"N={len(good):>4} WR {fmt_pct(sum(1 for t in good if t['won'])/len(good) if good else 0):>5}" if good else "(no trades)"
        bad_pnl = f"PnL {fmt_pnl(sum(t['pnl'] for t in bad)):>9}" if bad else ""
        good_pnl = f"PnL {fmt_pnl(sum(t['pnl'] for t in good)):>9}" if good else ""
        # truncate label
        short_lbl = lbl[:28]
        print(f"{short_lbl:<28} {bad_str:>15} {bad_pnl:>15}  {good_str:>15} {good_pnl:>15}")

    # ── TEST 3: High-EV cohort concentration in depth buckets ───────

    print(f"\n{'=' * 85}")
    print("TEST 3: Is the high-EV leak concentrated in low-depth windows?")
    print(f"{'=' * 85}\n")

    high_ev = [t for t in trades if t["edge"] >= 0.15]
    low_ev = [t for t in trades if t["edge"] < 0.15]
    print(f"  High-EV (>=0.15): {len(high_ev):,} trades, "
          f"PnL {fmt_pnl(sum(t['pnl'] for t in high_ev))}, "
          f"WR {fmt_pct(sum(1 for t in high_ev if t['won'])/len(high_ev) if high_ev else 0)}")
    print(f"  Low-EV (<0.15):   {len(low_ev):,} trades, "
          f"PnL {fmt_pnl(sum(t['pnl'] for t in low_ev))}, "
          f"WR {fmt_pct(sum(1 for t in low_ev if t['won'])/len(low_ev) if low_ev else 0)}")

    print(f"\nHigh-EV cohort split by depth:")
    for lo, hi, lbl in depth_buckets:
        bucket = [t for t in high_ev if lo < t["total_depth"] <= hi]
        print(summary(lbl, bucket))

    print(f"\nLow-EV cohort split by depth (for comparison):")
    for lo, hi, lbl in depth_buckets:
        bucket = [t for t in low_ev if lo < t["total_depth"] <= hi]
        print(summary(lbl, bucket))

    # ── TEST 4: Time-of-day × depth heat map ────────────────────────

    print(f"\n{'=' * 85}")
    print("TEST 4: Time-of-day x depth heat map (WR%)")
    print(f"{'=' * 85}\n")

    # 6 time buckets * 3 depth tiers
    time_buckets = [
        (0, 6, "00-06 (overnight)"),
        (6, 12, "06-12 (UK morning)"),
        (12, 16, "12-16 (US morning)"),
        (16, 20, "16-20 (US afternoon)"),
        (20, 24, "20-24 (US evening)"),
    ]
    depth_tiers = [
        (-1, p33 := pct(depths, 0.33), "low depth"),
        (p33, p66 := pct(depths, 0.67), "mid depth"),
        (p66, 1e12, "high depth"),
    ]

    print(f"{'Time bucket':<22} | {'low depth':<22} {'mid depth':<22} {'high depth':<22}")
    print("-" * 95)
    for lo, hi, lbl in time_buckets:
        cells = []
        for dlo, dhi, dlbl in depth_tiers:
            bucket = [t for t in trades
                      if lo <= t["hour_utc"] < hi
                      and dlo < t["total_depth"] <= dhi]
            if bucket:
                wr = sum(1 for t in bucket if t["won"]) / len(bucket)
                pnl = sum(t["pnl"] for t in bucket)
                cells.append(f"N={len(bucket):>3} WR {fmt_pct(wr):>5} {fmt_pnl(pnl):>8}")
            else:
                cells.append("(empty)              ")
        print(f"{lbl:<22} | {cells[0]:<22} {cells[1]:<22} {cells[2]:<22}")

    # ── TEST 5: Correlation analysis ────────────────────────────────

    print(f"\n{'=' * 85}")
    print("TEST 5: Correlation of book depth/spread with PnL per dollar")
    print(f"{'=' * 85}\n")

    def corr(xs, ys):
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            return 0.0
        return sxy / ((sxx ** 0.5) * (syy ** 0.5))

    depths_arr = [t["total_depth"] for t in trades]
    spreads_arr = [t["spread"] for t in trades]
    ppd_arr = [t["pnl_per_dollar"] for t in trades]
    won_arr = [1.0 if t["won"] else 0.0 for t in trades]

    print(f"  Correlation(book_depth, pnl_per_$):  {corr(depths_arr, ppd_arr):+.4f}")
    print(f"  Correlation(book_depth, won):        {corr(depths_arr, won_arr):+.4f}")
    print(f"  Correlation(spread, pnl_per_$):      {corr(spreads_arr, ppd_arr):+.4f}")
    print(f"  Correlation(spread, won):            {corr(spreads_arr, won_arr):+.4f}")
    print("\n  (Values near 0 = no predictive relationship)")

    # ── TEST 6: Filter simulation — skip low-depth trades ───────────

    print(f"\n{'=' * 85}")
    print("TEST 6: Simulated filter — what if we skipped trades below depth threshold?")
    print(f"{'=' * 85}\n")

    thresholds = [pct(depths, 0.10), pct(depths, 0.20), pct(depths, 0.30),
                  pct(depths, 0.40), pct(depths, 0.50)]
    print(f"  {'Threshold':>10} {'Skipped':>8} {'Remaining':>10} "
          f"{'Remaining PnL':>14} {'vs current':>12} {'Remaining WR':>14}")
    for thresh in thresholds:
        skipped = [t for t in trades if t["total_depth"] <= thresh]
        kept = [t for t in trades if t["total_depth"] > thresh]
        if not kept:
            continue
        kept_pnl = sum(t["pnl"] for t in kept)
        kept_wr = sum(1 for t in kept if t["won"]) / len(kept)
        delta = kept_pnl - overall_pnl
        print(f"  ${thresh:>8.0f} {len(skipped):>8,} {len(kept):>10,} "
              f"{fmt_pnl(kept_pnl):>14} {fmt_pnl(delta):>12} "
              f"{fmt_pct(kept_wr):>14}")

    # ── Same for spread (skip high-spread trades) ────────────────────

    print(f"\n  Same idea for SPREAD (skip trades with spread above threshold):")
    spread_thresholds = [pct(spreads, 0.90), pct(spreads, 0.80),
                        pct(spreads, 0.70), pct(spreads, 0.60), pct(spreads, 0.50)]
    print(f"  {'Threshold':>10} {'Skipped':>8} {'Remaining':>10} "
          f"{'Remaining PnL':>14} {'vs current':>12} {'Remaining WR':>14}")
    for thresh in spread_thresholds:
        skipped = [t for t in trades if t["spread"] >= thresh]
        kept = [t for t in trades if t["spread"] < thresh]
        if not kept:
            continue
        kept_pnl = sum(t["pnl"] for t in kept)
        kept_wr = sum(1 for t in kept if t["won"]) / len(kept)
        delta = kept_pnl - overall_pnl
        print(f"  ${thresh:>8.4f} {len(skipped):>8,} {len(kept):>10,} "
              f"{fmt_pnl(kept_pnl):>14} {fmt_pnl(delta):>12} "
              f"{fmt_pct(kept_wr):>14}")

    # ── Stacked filter: depth AND spread ────────────────────────────

    print(f"\n  STACKED: skip if depth < threshold OR spread > threshold:")
    depth_thr = pct(depths, 0.30)
    spread_thr = pct(spreads, 0.70)
    skipped = [t for t in trades if t["total_depth"] <= depth_thr or t["spread"] >= spread_thr]
    kept = [t for t in trades if t["total_depth"] > depth_thr and t["spread"] < spread_thr]
    if kept:
        kept_pnl = sum(t["pnl"] for t in kept)
        kept_wr = sum(1 for t in kept if t["won"]) / len(kept)
        delta = kept_pnl - overall_pnl
        print(f"    depth_threshold = ${depth_thr:.0f} (p30)")
        print(f"    spread_threshold = ${spread_thr:.4f} (p70)")
        print(f"    Skipped: {len(skipped):,}, Kept: {len(kept):,}")
        print(f"    PnL: {fmt_pnl(kept_pnl)}, delta {fmt_pnl(delta)}, WR {fmt_pct(kept_wr)}")

    # ── Verdict ──────────────────────────────────────────────────────

    print(f"\n{'=' * 85}")
    print("VERDICT")
    print(f"{'=' * 85}\n")
    print(f"  Direct depth correlation with PnL:    {corr(depths_arr, ppd_arr):+.4f}")
    print(f"  Direct spread correlation with PnL:   {corr(spreads_arr, ppd_arr):+.4f}")
    print(f"  (Compare to bet_size correlation:     +0.0021 — also effectively zero)")
    print(f"\n  See TEST 2 for whether hour effect is mediated by depth.")
    print(f"  See TEST 6 for filter simulations.")


if __name__ == "__main__":
    main()
