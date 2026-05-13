"""Bot K weight overlay on Bot G's high-EV cohort.

The question: would Bot K's optimised weights have prevented Bot G's
high-EV losers? Or does Bot K still flag the same trades as high-EV
(meaning the issue is structural, not weight-related)?

For each Bot G trade:
  1. Recompute combined_signal using Bot K's optimised time-varying weights
     on the L1-L5 signal values stored in the DB.
  2. Derive est_prob_up_botk and EV_botk under same entry price.
  3. Determine Bot K's preferred side (YES if est_prob_up > 0.5; else NO).
  4. Determine if Bot K would have entered: side's EV >= min_edge.
  5. Compare to Bot G's actual decision.

Outcome categories:
  - SAME_SIDE: Bot K agrees, would have made the same trade. Same outcome.
  - FLIP_SIDE: Bot K disagrees on direction. Outcome flips (win <-> loss).
  - NO_TRADE: Bot K's EV too low. Trade avoided ($0 PnL).

Caveat: additive modifiers (L6/L7/L8) aren't in historical DB. Core L1-L5
weighted blend is the apples-to-apples comparison.

Usage:
    python -m backtest.bot_k_overlay_on_high_ev
"""

import sqlite3
import statistics
from collections import defaultdict
from typing import NamedTuple

DB_PATH = "data_runtime/bot_g_signal_aligned.db"
FEE_RATE = 0.072

# Bot K optimised schedule — copied from signals/combiner.py
# (time_remaining_sec, oracle_w, momentum_w, liquidation_w, orderbook_w, sentiment_w)
WEIGHT_SCHEDULE_BOT_K = [
    (300, 0.15, 0.35, 0.15, 0.20, 0.15),
    (180, 0.11, 0.38, 0.12, 0.24, 0.15),
    (90,  0.08, 0.40, 0.08, 0.29, 0.15),
    (30,  0.05, 0.42, 0.05, 0.33, 0.15),
]

# Bot G entry parameters (from config_multi.yaml)
MIN_EDGE = 0.003   # min EV after fee adjustment
MAX_EDGE = 0.025   # max prob_edge that gates entry (NOT the EV cap)
MIN_CONFIDENCE = 0.02
MAX_ADJUSTMENT = 0.20  # combined_signal clamp


def pick_weights(time_remaining: float) -> tuple:
    """Pick the (oracle, momentum, liq, book, sent) weights for the given time."""
    # Find the bucket whose threshold the time falls under
    for thresh, w1, w2, w3, w4, w5 in WEIGHT_SCHEDULE_BOT_K:
        if time_remaining <= thresh:
            return w1, w2, w3, w4, w5
    # Fallback: use earliest bucket
    return (WEIGHT_SCHEDULE_BOT_K[0][1], WEIGHT_SCHEDULE_BOT_K[0][2],
            WEIGHT_SCHEDULE_BOT_K[0][3], WEIGHT_SCHEDULE_BOT_K[0][4],
            WEIGHT_SCHEDULE_BOT_K[0][5])


def taker_fee(price: float) -> float:
    return FEE_RATE * price * (1.0 - price)


def compute_ev(est_prob_up: float, entry_price: float, side: str) -> float:
    """Compute fee-adjusted EV per share for a given side at given price."""
    fee = taker_fee(entry_price)
    if side == "YES":
        return est_prob_up * (1 - entry_price) - (1 - est_prob_up) * entry_price - fee
    else:
        return (1 - est_prob_up) * (1 - entry_price) - est_prob_up * entry_price - fee


def main() -> None:
    print("=" * 80)
    print("BOT K WEIGHT OVERLAY on Bot G's trades")
    print("Would Bot K's optimised weights have prevented Bot G's high-EV losers?")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, side, entry_price, size_usdc, pnl, edge,
               estimated_prob_up, market_implied_prob,
               oracle_lag_signal, momentum_signal, liquidation_signal,
               orderbook_signal, sentiment_signal, combined_signal,
               regime, time_remaining_secs
        FROM trades
        WHERE pnl IS NOT NULL AND size_usdc > 0
    """)
    cols = [d[0] for d in cur.description]
    raw = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()

    print(f"\nLoaded {len(raw):,} completed Bot G trades")

    # ── Recompute Bot K's view of each trade ─────────────────────────

    overlay = []
    for t in raw:
        # Extract signal values (treat None as 0)
        s1 = t["oracle_lag_signal"] or 0.0
        s2 = t["momentum_signal"] or 0.0
        s3 = t["liquidation_signal"] or 0.0
        s4 = t["orderbook_signal"] or 0.0
        s5 = t["sentiment_signal"] or 0.0
        time_rem = t["time_remaining_secs"] or 180.0

        # Pick Bot K weights for this time bucket
        w1, w2, w3, w4, w5 = pick_weights(time_rem)

        # Compute combined_signal under Bot K weights
        combined_botk = w1*s1 + w2*s2 + w3*s3 + w4*s4 + w5*s5

        # Clamp + apply max_adjustment
        combined_botk = max(-1.0, min(1.0, combined_botk))

        # Derive est_prob_up
        est_prob_botk = 0.5 + combined_botk * 0.5

        # Bot K's preferred side based on its prediction
        if est_prob_botk > 0.5:
            preferred_side = "YES"
        elif est_prob_botk < 0.5:
            preferred_side = "NO"
        else:
            preferred_side = "NONE"

        # Bot K's EV on its preferred side
        if preferred_side != "NONE":
            ev_botk = compute_ev(est_prob_botk, t["entry_price"], preferred_side)
        else:
            ev_botk = 0.0

        # Confidence check
        confidence_botk = abs(est_prob_botk - 0.5) * 2.0

        # Would Bot K have entered?
        would_enter = (
            preferred_side != "NONE"
            and ev_botk >= MIN_EDGE
            and confidence_botk >= MIN_CONFIDENCE
        )

        # Categorise
        if not would_enter:
            category = "NO_TRADE"
            simulated_pnl = 0.0
        elif preferred_side == t["side"]:
            category = "SAME_SIDE"
            simulated_pnl = t["pnl"]  # same outcome
        else:
            category = "FLIP_SIDE"
            # Flip outcome: if Bot G won, the opposite-side bet loses; vice versa.
            # PnL flip math: if Bot G bet $X on side S at price P:
            #   Win:  pnl = num_shares * (1 - P) = (X/P) * (1-P)
            #   Loss: pnl = -X
            # Bot K flips to side ~S at the same time. So Bot G's actual entry_price
            # for the FLIPPED side is (1 - entry_price). And Bot K's bet would be:
            #   If Bot G's actual outcome was a LOSS for Bot G, then the opposite
            #   side WINS. Bot K wins (X / (1 - P)) * P at the cost of $X bet.
            #   If Bot G WON, then the opposite side loses. Bot K loses $X.
            flipped_price = 1.0 - t["entry_price"]
            if t["pnl"] > 0:
                # Bot G won → Bot K loses on opposite side
                simulated_pnl = -t["size_usdc"]
            else:
                # Bot G lost → Bot K wins on opposite side
                # Bot K's pnl: (size / flipped_price) * (1 - flipped_price) - fee
                num_shares = t["size_usdc"] / flipped_price if flipped_price > 0 else 0
                gross_win = num_shares * (1 - flipped_price)
                fee_total = num_shares * taker_fee(flipped_price)
                simulated_pnl = gross_win - fee_total

        overlay.append({
            **t,
            "combined_botk": combined_botk,
            "est_prob_botk": est_prob_botk,
            "preferred_side": preferred_side,
            "ev_botk": ev_botk,
            "would_enter": would_enter,
            "category": category,
            "simulated_pnl": simulated_pnl,
            "won_botk": simulated_pnl > 0,
        })

    # ── Stratify: high-EV cohort vs everything else ──────────────────

    high_ev = [t for t in overlay if t["edge"] >= 0.15]
    low_ev = [t for t in overlay if t["edge"] < 0.15]

    print(f"\n  Bot G high-EV cohort (edge >= 0.15): {len(high_ev):,} trades")
    print(f"  Bot G low-EV cohort  (edge < 0.15):  {len(low_ev):,} trades")

    # ── Bot K classification of Bot G's high-EV trades ──────────────

    print(f"\n{'=' * 80}")
    print("HOW BOT K WOULD HAVE TREATED BOT G'S HIGH-EV LOSERS")
    print(f"{'=' * 80}")

    cats = defaultdict(list)
    for t in high_ev:
        cats[t["category"]].append(t)

    print(f"\n  {'Category':<12} {'N':>6} {'%':>6} {'BotG PnL':>12} {'BotK Sim PnL':>14}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")
    for cat in ["SAME_SIDE", "FLIP_SIDE", "NO_TRADE"]:
        ts = cats.get(cat, [])
        if not ts:
            print(f"  {cat:<12} {0:>6,} {0.0:>5.1%}   {'$+0.00':>12} {'$+0.00':>14}")
            continue
        n = len(ts)
        pct = n / len(high_ev)
        botg_pnl = sum(t["pnl"] for t in ts)
        botk_pnl = sum(t["simulated_pnl"] for t in ts)
        print(f"  {cat:<12} {n:>6,} {pct:>5.1%} "
              f"  ${botg_pnl:>+10.2f} ${botk_pnl:>+12.2f}")

    high_ev_botg_total = sum(t["pnl"] for t in high_ev)
    high_ev_botk_total = sum(t["simulated_pnl"] for t in high_ev)
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")
    print(f"  {'TOTAL':<12} {len(high_ev):>6,} {1.0:>5.1%} "
          f"  ${high_ev_botg_total:>+10.2f} ${high_ev_botk_total:>+12.2f}")

    delta = high_ev_botk_total - high_ev_botg_total
    print(f"\n  Bot K impact on high-EV cohort: ${delta:+.2f}")
    if delta > 0:
        print(f"  -> Bot K's weights would have IMPROVED PnL by ${delta:.2f}")
    else:
        print(f"  -> Bot K's weights would have made it WORSE by ${-delta:.2f}")

    # ── Where does the saving come from? ────────────────────────────

    print(f"\n  Breakdown of saving from each category:")
    for cat in ["SAME_SIDE", "FLIP_SIDE", "NO_TRADE"]:
        ts = cats.get(cat, [])
        if not ts:
            continue
        botg = sum(t["pnl"] for t in ts)
        botk = sum(t["simulated_pnl"] for t in ts)
        diff = botk - botg
        wr_g = sum(1 for t in ts if t["pnl"] > 0) / len(ts)
        wr_k = sum(1 for t in ts if t["simulated_pnl"] > 0) / len(ts)
        print(f"    {cat}: N={len(ts):,} | "
              f"Bot G WR {wr_g:.1%} -> Bot K WR {wr_k:.1%} | "
              f"diff ${diff:+.2f}")

    # ── Now look at Bot G's GOOD trades — does Bot K break those? ────

    print(f"\n{'=' * 80}")
    print("DOES BOT K BREAK BOT G'S WINNING LOW-EV TRADES?")
    print(f"{'=' * 80}")

    cats_low = defaultdict(list)
    for t in low_ev:
        cats_low[t["category"]].append(t)

    print(f"\n  {'Category':<12} {'N':>6} {'%':>6} {'BotG PnL':>12} {'BotK Sim PnL':>14}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")
    for cat in ["SAME_SIDE", "FLIP_SIDE", "NO_TRADE"]:
        ts = cats_low.get(cat, [])
        if not ts:
            continue
        n = len(ts)
        pct = n / len(low_ev)
        botg_pnl = sum(t["pnl"] for t in ts)
        botk_pnl = sum(t["simulated_pnl"] for t in ts)
        print(f"  {cat:<12} {n:>6,} {pct:>5.1%} "
              f"  ${botg_pnl:>+10.2f} ${botk_pnl:>+12.2f}")

    low_ev_botg = sum(t["pnl"] for t in low_ev)
    low_ev_botk = sum(t["simulated_pnl"] for t in low_ev)
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")
    print(f"  {'TOTAL':<12} {len(low_ev):>6,} {1.0:>5.1%} "
          f"  ${low_ev_botg:>+10.2f} ${low_ev_botk:>+12.2f}")
    print(f"\n  Bot K impact on low-EV cohort: ${low_ev_botk - low_ev_botg:+.2f}")

    # ── Full portfolio comparison ────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("FULL PORTFOLIO: Bot G actual vs Bot K weights overlay")
    print(f"{'=' * 80}")

    total_botg = sum(t["pnl"] for t in overlay)
    total_botk = sum(t["simulated_pnl"] for t in overlay)

    n_same = sum(1 for t in overlay if t["category"] == "SAME_SIDE")
    n_flip = sum(1 for t in overlay if t["category"] == "FLIP_SIDE")
    n_skip = sum(1 for t in overlay if t["category"] == "NO_TRADE")

    print(f"\n  Bot G actual:               ${total_botg:+.2f} ({len(overlay):,} trades)")
    print(f"  Bot K weight overlay:       ${total_botk:+.2f}")
    print(f"  Difference:                 ${total_botk - total_botg:+.2f}")
    print(f"\n  Category breakdown:")
    print(f"    SAME_SIDE (same trade):   {n_same:,} ({n_same/len(overlay):.1%})")
    print(f"    FLIP_SIDE (opposite):     {n_flip:,} ({n_flip/len(overlay):.1%})")
    print(f"    NO_TRADE (Bot K skips):   {n_skip:,} ({n_skip/len(overlay):.1%})")

    # ── New EV distribution under Bot K ──────────────────────────────

    print(f"\n{'=' * 80}")
    print("NEW EV DISTRIBUTION UNDER BOT K WEIGHTS")
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

    print(f"\n  Restricted to trades Bot K would have entered (would_enter=True):")
    print(f"  {'Bot K EV bucket':<18} {'N':>6} {'WR':>7} {'Sim PnL':>10}")
    print(f"  {'-'*18} {'-'*6} {'-'*7} {'-'*10}")
    entered = [t for t in overlay if t["would_enter"]]

    for lo, hi, lbl in ev_buckets:
        bucket = [t for t in entered if lo <= t["ev_botk"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["won_botk"]) / len(bucket)
        pnl = sum(t["simulated_pnl"] for t in bucket)
        print(f"  {lbl:<18} {len(bucket):>6,} {wr:>6.1%} ${pnl:>+8.2f}")

    print(f"\n  Compare to Bot G's actual EV distribution:")
    print(f"  {'Bot G EV bucket':<18} {'N':>6} {'WR':>7} {'PnL':>10}")
    print(f"  {'-'*18} {'-'*6} {'-'*7} {'-'*10}")
    for lo, hi, lbl in ev_buckets:
        bucket = [t for t in overlay if lo <= t["edge"] < hi]
        if not bucket:
            continue
        wr = sum(1 for t in bucket if t["pnl"] > 0) / len(bucket)
        pnl = sum(t["pnl"] for t in bucket)
        print(f"  {lbl:<18} {len(bucket):>6,} {wr:>6.1%} ${pnl:>+8.2f}")

    # ── Top 10 specific saves ────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("BIGGEST SAVES (high-EV trades Bot K avoided or flipped correctly)")
    print(f"{'=' * 80}")

    saves = sorted(
        [t for t in high_ev if t["simulated_pnl"] > t["pnl"]],
        key=lambda t: -(t["simulated_pnl"] - t["pnl"])
    )[:10]
    print(f"\n  {'Side':<5} {'Price':>7} {'BotG PnL':>10} {'BotK PnL':>10} "
          f"{'Cat':<10} {'Regime':<15}")
    for t in saves:
        print(f"  {t['side']:<5} ${t['entry_price']:>6.3f} "
              f"${t['pnl']:>+9.2f} ${t['simulated_pnl']:>+9.2f} "
              f"{t['category']:<10} {t['regime']:<15}")

    # ── Top 10 specific damages ──────────────────────────────────────

    print(f"\n  BIGGEST DAMAGES (low-EV winners Bot K would have skipped or flipped wrong)")
    damages = sorted(
        [t for t in low_ev if t["simulated_pnl"] < t["pnl"]],
        key=lambda t: t["simulated_pnl"] - t["pnl"]
    )[:10]
    print(f"\n  {'Side':<5} {'Price':>7} {'BotG PnL':>10} {'BotK PnL':>10} "
          f"{'Cat':<10} {'Regime':<15}")
    for t in damages:
        print(f"  {t['side']:<5} ${t['entry_price']:>6.3f} "
              f"${t['pnl']:>+9.2f} ${t['simulated_pnl']:>+9.2f} "
              f"{t['category']:<10} {t['regime']:<15}")

    # ── Verdict ──────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    print(f"\n  Bot G total PnL:        ${total_botg:+.2f}")
    print(f"  Bot K overlay PnL:      ${total_botk:+.2f}")
    print(f"  Net improvement:        ${total_botk - total_botg:+.2f}")
    print(f"\n  On high-EV (>=0.15) cohort:")
    print(f"    Bot G: ${high_ev_botg_total:+.2f} ({len(high_ev):,} trades)")
    print(f"    Bot K: ${high_ev_botk_total:+.2f} (after re-categorisation)")
    print(f"    Saving: ${high_ev_botk_total - high_ev_botg_total:+.2f}")

    # Does Bot K still have the high-EV leak under its own weighting?
    botk_high_ev_entered = [t for t in entered if t["ev_botk"] >= 0.15]
    if botk_high_ev_entered:
        wr = sum(1 for t in botk_high_ev_entered if t["won_botk"]) / len(botk_high_ev_entered)
        pnl = sum(t["simulated_pnl"] for t in botk_high_ev_entered)
        print(f"\n  Bot K's OWN high-EV (>=0.15) trades: {len(botk_high_ev_entered):,}")
        print(f"    WR: {wr:.1%}")
        print(f"    PnL: ${pnl:+.2f}")
        if wr < 0.42:
            print(f"    -> Bot K STILL has the high-EV leak under its own weighting")
        else:
            print(f"    -> Bot K's high-EV trades are profitable (WR above break-even)")


if __name__ == "__main__":
    main()
