"""
L9 SM Confirmation Overlay Backtest

Takes Bot G's actual historical trades and overlays the L9 SM confirmation
signal to answer: "What would have happened if L9 was active?"

For each trade:
1. Match to Q24b SM flow data via window_ts
2. Apply L9 logic (check_sm_confirmation equivalent)
3. On EXIT: replace PnL with early exit PnL (exit at market price at min 3/4)
4. On HOLD/IGNORE: keep original PnL

Outputs comparison: original PnL vs L9-adjusted PnL
"""

import sqlite3
import csv
import io
import os
import sys
import httpx
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────
EXECUTION_ID = "01KR1YN403Y42BS0N2VTAXGT38"  # Q24b latest execution
BOT_G_DB_1 = Path("data_runtime/bot_g_signal_aligned.db")
BOT_G_DB_2 = Path("data_runtime/trades.db")

# L9 parameters (matching Bot K config)
SM_AGREEMENT_THRESHOLD = 0.60   # 60% dollar-weighted agreement
SM_MIN_VOLUME = 50.0            # minimum $50 SM volume
SM_MIN_WALLETS = 0              # not available in Q24b, skip
SM_PRICE_FLOOR = 0.35           # actionable price zone floor (YES price)
SM_PRICE_CEILING = 0.65         # actionable price zone ceiling (YES price)
SM_CHECK_MINUTES = [4]           # only check at min 4 (min 3 is premature)
SM_EXIT_SIDES = ["YES"]          # only exit YES positions (NO side carries profit)


@dataclass
class Trade:
    """A Bot G historical trade."""
    market_slug: str
    window_ts: int
    timestamp: str
    side: str           # YES or NO
    entry_price: float
    pnl: float
    stake: float
    edge: float
    regime: str
    resolution: str     # UP or DOWN


@dataclass
class SMFlow:
    """SM flow data from Q24b for a single market window."""
    window_ts: int
    resolution: str
    sm_dir_3: str
    sm_strength_3: float
    sm_volume_3: float
    sm_dir_4: str
    sm_strength_4: float
    sm_volume_4: float
    mkt_up_price_3: float | None
    mkt_up_price_4: float | None


def load_bot_g_trades() -> list[Trade]:
    """Load all Bot G trades from both databases."""
    trades = []
    for db_path in [BOT_G_DB_1, BOT_G_DB_2]:
        if not db_path.exists():
            print(f"WARNING: {db_path} not found, skipping")
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT market_slug, timestamp, side, entry_price, pnl,
                   size_usdc, edge, regime, resolution
            FROM trades
            ORDER BY timestamp
        """)
        for row in cursor:
            slug = row["market_slug"]
            # Extract window_ts from slug: btc-updown-5m-1776164700
            parts = slug.rsplit("-", 1)
            try:
                wts = int(parts[-1])
            except (ValueError, IndexError):
                continue

            trades.append(Trade(
                market_slug=slug,
                window_ts=wts,
                timestamp=row["timestamp"],
                side=row["side"],
                entry_price=row["entry_price"],
                pnl=row["pnl"] if row["pnl"] is not None else 0.0,
                stake=row["size_usdc"],
                edge=row["edge"] if row["edge"] else 0.0,
                regime=row["regime"] or "unknown",
                resolution=row["resolution"] or "unknown",
            ))
        conn.close()

    # Sort by timestamp, deduplicate by (market_slug, side, timestamp)
    seen = set()
    unique_trades = []
    for t in sorted(trades, key=lambda x: x.timestamp):
        key = (t.market_slug, t.side, t.timestamp)
        if key not in seen:
            seen.add(key)
            unique_trades.append(t)

    return unique_trades


def download_q24b_results(api_key: str) -> dict[int, SMFlow]:
    """Download all Q24b results from Dune API as CSV and parse into dict."""
    url = f"https://api.dune.com/api/v1/execution/{EXECUTION_ID}/results/csv"
    headers = {"X-Dune-API-Key": api_key}

    print(f"Downloading Q24b results from Dune API...")
    with httpx.Client(timeout=120) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()

    text = resp.text
    reader = csv.DictReader(io.StringIO(text))
    flows: dict[int, SMFlow] = {}

    def safe_float(val: str, default: float = 0.0) -> float:
        """Parse float from CSV, handling <nil>, empty, and sci notation."""
        if not val or val == "<nil>":
            return default
        return float(val)

    def safe_float_or_none(val: str) -> float | None:
        """Parse float or return None for missing values."""
        if not val or val == "<nil>":
            return None
        return float(val)

    for row in reader:
        raw_wts = int(float(row["window_ts"]))
        # Snap to nearest 300-second (5-min) boundary to match Bot G's slug timestamps
        # Q24b uses exact resolution_time - 300s, which is ~20-30s off from the
        # clean 5-minute marks used in market slugs
        wts = round(raw_wts / 300) * 300
        flows[wts] = SMFlow(
            window_ts=wts,
            resolution=row["resolution"],
            sm_dir_3=row["sm_dir_3"],
            sm_strength_3=safe_float(row["sm_strength_3"]),
            sm_volume_3=safe_float(row["sm_volume_3"]),
            sm_dir_4=row["sm_dir_4"],
            sm_strength_4=safe_float(row["sm_strength_4"]),
            sm_volume_4=safe_float(row["sm_volume_4"]),
            mkt_up_price_3=safe_float_or_none(row["mkt_up_price_3"]),
            mkt_up_price_4=safe_float_or_none(row["mkt_up_price_4"]),
        )

    print(f"  Downloaded {len(flows)} market windows from Q24b")
    return flows


def apply_l9_decision(
    trade: Trade,
    flow: SMFlow,
    check_minute: int,
) -> tuple[str, float | None]:
    """
    Apply L9 SM confirmation logic for a single trade at a given minute.

    Returns:
        (decision, exit_price) where decision is HOLD/EXIT/IGNORE
        exit_price is only set for EXIT decisions
    """
    # Get SM data for this minute
    if check_minute == 3:
        sm_dir = flow.sm_dir_3
        sm_strength = flow.sm_strength_3
        sm_volume = flow.sm_volume_3
        mkt_up_price = flow.mkt_up_price_3
    elif check_minute == 4:
        sm_dir = flow.sm_dir_4
        sm_strength = flow.sm_strength_4
        sm_volume = flow.sm_volume_4
        mkt_up_price = flow.mkt_up_price_4
    else:
        return "IGNORE", None

    # No SM activity → IGNORE
    if sm_dir == "Tied" or sm_volume < SM_MIN_VOLUME:
        return "IGNORE", None

    # No market price available → IGNORE
    if mkt_up_price is None:
        return "IGNORE", None

    # Determine position-relative market price
    # For YES position: market price = mkt_up_price (price of UP/YES token)
    # For NO position: market price = 1 - mkt_up_price
    if trade.side == "YES":
        position_price = mkt_up_price
    else:
        position_price = 1.0 - mkt_up_price

    # Price zone check — only actionable in SM_PRICE_FLOOR to SM_PRICE_CEILING
    if position_price < SM_PRICE_FLOOR or position_price > SM_PRICE_CEILING:
        return "IGNORE", None

    # SM agreement check
    # For YES position: SM should lean Up for agreement
    # For NO position: SM should lean Down for agreement
    sm_agreement_pct = sm_strength / 100.0  # convert from percentage

    if trade.side == "YES":
        sm_agrees = (sm_dir == "Up" and sm_agreement_pct >= SM_AGREEMENT_THRESHOLD)
        sm_disagrees = (sm_dir == "Down" and sm_agreement_pct >= SM_AGREEMENT_THRESHOLD)
    else:  # NO
        sm_agrees = (sm_dir == "Down" and sm_agreement_pct >= SM_AGREEMENT_THRESHOLD)
        sm_disagrees = (sm_dir == "Up" and sm_agreement_pct >= SM_AGREEMENT_THRESHOLD)

    if sm_disagrees:
        return "EXIT", position_price
    elif sm_agrees:
        return "HOLD", None
    else:
        # SM has a lean but below threshold
        return "IGNORE", None


def calculate_early_exit_pnl(trade: Trade, exit_price: float) -> float:
    """
    Calculate PnL if we exit at exit_price instead of holding to resolution.

    Early exit means selling our position at current market price.
    PnL = (exit_price - entry_price) * num_shares
    where num_shares = stake / entry_price
    """
    if trade.entry_price <= 0:
        return 0.0

    num_shares = trade.stake / trade.entry_price
    early_pnl = (exit_price - trade.entry_price) * num_shares
    return round(early_pnl, 4)


def run_backtest(trades: list[Trade], flows: dict[int, SMFlow]) -> None:
    """Run the L9 overlay backtest and print results."""

    # Tracking
    original_pnl = 0.0
    adjusted_pnl = 0.0
    l9_decisions = {"HOLD": 0, "EXIT": 0, "IGNORE": 0, "NO_DATA": 0}
    exit_outcomes = {"saved_loss": 0, "reduced_win": 0, "improved": 0, "worsened": 0}
    exit_pnl_impact = 0.0
    matched = 0
    unmatched = 0

    # Per-trade detail for analysis
    trade_details = []

    # Running drawdown tracking
    orig_cumulative = 0.0
    adj_cumulative = 0.0
    orig_peak = 0.0
    adj_peak = 0.0
    orig_max_dd = 0.0
    adj_max_dd = 0.0

    # Daily tracking
    daily_orig: dict[str, float] = {}
    daily_adj: dict[str, float] = {}

    for trade in trades:
        original_pnl += trade.pnl
        orig_cumulative += trade.pnl
        orig_peak = max(orig_peak, orig_cumulative)
        orig_dd = orig_peak - orig_cumulative
        orig_max_dd = max(orig_max_dd, orig_dd)

        date_key = trade.timestamp[:10]

        flow = flows.get(trade.window_ts)
        if flow is None:
            # No SM data for this window — keep original
            l9_decisions["NO_DATA"] += 1
            adjusted_pnl += trade.pnl
            adj_cumulative += trade.pnl
            adj_peak = max(adj_peak, adj_cumulative)
            adj_dd = adj_peak - adj_cumulative
            adj_max_dd = max(adj_max_dd, adj_dd)
            daily_orig[date_key] = daily_orig.get(date_key, 0) + trade.pnl
            daily_adj[date_key] = daily_adj.get(date_key, 0) + trade.pnl
            unmatched += 1
            continue

        matched += 1

        # Apply L9 at each check minute (stop on first EXIT)
        final_decision = "IGNORE"
        exit_price = None
        check_min_used = None

        # Side filter: only allow exits on configured sides
        side_eligible = trade.side in SM_EXIT_SIDES

        for check_min in SM_CHECK_MINUTES:
            decision, ep = apply_l9_decision(trade, flow, check_min)
            if decision == "EXIT":
                if side_eligible:
                    final_decision = "EXIT"
                    exit_price = ep
                    check_min_used = check_min
                else:
                    final_decision = "IGNORE"  # SM disagrees but side filtered out
                break
            elif decision == "HOLD":
                final_decision = "HOLD"
                check_min_used = check_min
                break
            # IGNORE → try next minute

        l9_decisions[final_decision] += 1

        if final_decision == "EXIT" and exit_price is not None:
            early_pnl = calculate_early_exit_pnl(trade, exit_price)
            adjusted_pnl += early_pnl
            adj_cumulative += early_pnl

            impact = early_pnl - trade.pnl
            exit_pnl_impact += impact

            # Classify the exit outcome
            if trade.pnl < 0 and early_pnl > trade.pnl:
                exit_outcomes["saved_loss"] += 1
            elif trade.pnl > 0 and early_pnl < trade.pnl:
                exit_outcomes["reduced_win"] += 1
            elif early_pnl > trade.pnl:
                exit_outcomes["improved"] += 1
            else:
                exit_outcomes["worsened"] += 1

            trade_details.append({
                "slug": trade.market_slug,
                "side": trade.side,
                "entry": trade.entry_price,
                "exit_price": exit_price,
                "orig_pnl": trade.pnl,
                "new_pnl": early_pnl,
                "impact": impact,
                "minute": check_min_used,
                "resolution": trade.resolution,
            })

            daily_orig[date_key] = daily_orig.get(date_key, 0) + trade.pnl
            daily_adj[date_key] = daily_adj.get(date_key, 0) + early_pnl
        else:
            # HOLD or IGNORE — keep original PnL
            adjusted_pnl += trade.pnl
            adj_cumulative += trade.pnl
            daily_orig[date_key] = daily_orig.get(date_key, 0) + trade.pnl
            daily_adj[date_key] = daily_adj.get(date_key, 0) + trade.pnl

        adj_peak = max(adj_peak, adj_cumulative)
        adj_dd = adj_peak - adj_cumulative
        adj_max_dd = max(adj_max_dd, adj_dd)

    # ── Results ───────────────────────────────────────────────────
    total = len(trades)
    print("\n" + "=" * 70)
    print("  L9 SM CONFIRMATION OVERLAY BACKTEST - BOT G HISTORICAL TRADES")
    print("=" * 70)

    print("\nCOVERAGE:")
    print(f"  Total trades:          {total}")
    print(f"  Matched to Q24b:       {matched} ({100*matched/total:.1f}%)")
    print(f"  Unmatched (no SM):     {unmatched} ({100*unmatched/total:.1f}%)")

    print("\nL9 DECISIONS:")
    for dec, count in sorted(l9_decisions.items()):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {dec:12s}: {count:5d}  ({pct:.1f}%)")

    print("\nPNL COMPARISON:")
    print(f"  Original PnL:          ${original_pnl:>10.2f}")
    print(f"  L9-Adjusted PnL:       ${adjusted_pnl:>10.2f}")
    delta = adjusted_pnl - original_pnl
    delta_pct = 100 * delta / abs(original_pnl) if original_pnl != 0 else 0
    print(f"  Delta:                 ${delta:>10.2f}  ({delta_pct:+.1f}%)")

    print("\nDRAWDOWN:")
    print(f"  Original max DD:       ${orig_max_dd:>10.2f}")
    print(f"  L9-Adjusted max DD:    ${adj_max_dd:>10.2f}")
    dd_delta = adj_max_dd - orig_max_dd
    print(f"  DD improvement:        ${dd_delta:>10.2f}")

    exit_count = l9_decisions["EXIT"]
    if exit_count > 0:
        print(f"\nEXIT ANALYSIS ({exit_count} exits):")
        saved = exit_outcomes["saved_loss"]
        reduced = exit_outcomes["reduced_win"]
        print(f"  Saved losses:          {saved:5d}  ({100*saved/exit_count:.1f}%)")
        print(f"  Reduced wins:          {reduced:5d}  ({100*reduced/exit_count:.1f}%)")
        print(f"  Other improved:        {exit_outcomes['improved']:5d}")
        print(f"  Other worsened:        {exit_outcomes['worsened']:5d}")
        print(f"  Total exit PnL impact: ${exit_pnl_impact:>10.2f}")
        avg_impact = exit_pnl_impact / exit_count
        print(f"  Avg impact per exit:   ${avg_impact:>10.4f}")

        # Best and worst exits
        if trade_details:
            best = max(trade_details, key=lambda x: x["impact"])
            worst = min(trade_details, key=lambda x: x["impact"])
            print(f"\n  Best exit:  {best['slug']}")
            b_side, b_entry, b_exit = best["side"], best["entry"], best["exit_price"]
            b_orig, b_new, b_imp = best["orig_pnl"], best["new_pnl"], best["impact"]
            print(f"    {b_side} @ {b_entry:.4f} -> exit @ {b_exit:.4f}")
            print(f"    Original PnL: ${b_orig:.2f} -> New PnL: ${b_new:.2f} (impact: ${b_imp:+.2f})")
            print(f"\n  Worst exit: {worst['slug']}")
            w_side, w_entry, w_exit = worst["side"], worst["entry"], worst["exit_price"]
            w_orig, w_new, w_imp = worst["orig_pnl"], worst["new_pnl"], worst["impact"]
            print(f"    {w_side} @ {w_entry:.4f} -> exit @ {w_exit:.4f}")
            print(f"    Original PnL: ${w_orig:.2f} -> New PnL: ${w_new:.2f} (impact: ${w_imp:+.2f})")

    # Daily comparison
    print("\nDAILY PNL COMPARISON:")
    sep = "-" * 12
    print(f"  {'Date':<12} {'Orig PnL':>10} {'L9 PnL':>10} {'Delta':>10}")
    print(f"  {sep} {sep[:10]} {sep[:10]} {sep[:10]}")
    all_dates = sorted(set(list(daily_orig.keys()) + list(daily_adj.keys())))
    for d in all_dates:
        o = daily_orig.get(d, 0)
        a = daily_adj.get(d, 0)
        diff = a - o
        marker = " ***" if abs(diff) > 10 else ""
        print(f"  {d:<12} ${o:>9.2f} ${a:>9.2f} ${diff:>9.2f}{marker}")

    # Side breakdown of exits
    if trade_details:
        print("\nEXIT BREAKDOWN BY SIDE:")
        for s in ["YES", "NO"]:
            side_exits = [t for t in trade_details if t["side"] == s]
            if side_exits:
                total_impact = sum(t["impact"] for t in side_exits)
                avg = total_impact / len(side_exits)
                print(f"  {s}: {len(side_exits)} exits, total impact ${total_impact:+.2f}, avg ${avg:+.4f}")

        print("\nEXIT BREAKDOWN BY MINUTE:")
        for m in [3, 4]:
            min_exits = [t for t in trade_details if t["minute"] == m]
            if min_exits:
                total_impact = sum(t["impact"] for t in min_exits)
                avg = total_impact / len(min_exits)
                print(f"  Min {m}: {len(min_exits)} exits, total impact ${total_impact:+.2f}, avg ${avg:+.4f}")

    print("\n" + "=" * 70)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DUNE_API_KEY")
    if not api_key:
        print("ERROR: DUNE_API_KEY not found in .env")
        sys.exit(1)

    os.chdir(Path(__file__).parent.parent)

    print("Loading Bot G trades...")
    trades = load_bot_g_trades()
    print(f"  Loaded {len(trades)} trades")
    print(f"  Date range: {trades[0].timestamp[:10]} to {trades[-1].timestamp[:10]}")

    flows = download_q24b_results(api_key)

    # Check match rate
    trade_timestamps = {t.window_ts for t in trades}
    flow_timestamps = set(flows.keys())
    overlap = trade_timestamps & flow_timestamps
    print(f"  Bot G window_ts values: {len(trade_timestamps)}")
    print(f"  Q24b window_ts values:  {len(flow_timestamps)}")
    print(f"  Overlap:                {len(overlap)} ({100*len(overlap)/len(trade_timestamps):.1f}%)")

    run_backtest(trades, flows)


if __name__ == "__main__":
    main()
