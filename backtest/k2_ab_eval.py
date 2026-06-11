"""Bot K2 vs Bot K A/B evaluation (J14 directional L1 floor verdict).

Read-only. Compares the two paper ledgers over the A/B window (K2 go-live
2026-05-29 onward) on post-J28-corrected data.

Filters (per handoff/house rules):
- Resolved trades only (pnl IS NOT NULL).
- Trades with oracle_price_at_open = 0 excluded everywhere (L1 was dead,
  floor inactive -> uninformative for the A/B).
- EARLY_EXIT trades: included in PnL sums (exit-priced is real PnL),
  excluded from resolution win-rate stats.

Floor classification (validated: K2 ledger contains 0 such trades):
against-line = (side=YES and L1 < -0.1) or (side=NO and L1 > +0.1),
where L1 = oracle_lag_signal at entry (L1 > 0 bullish).

Usage: python -m backtest.k2_ab_eval
"""

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

AB_START = "2026-05-29"
DEADBAND = 0.1

K_DB = "data_runtime/bot_k_sm_confirmation.db"
K2_DB = "data_runtime/bot_k2_l1_floor.db"


@dataclass
class Trade:
    slug: str
    day: str
    side: str
    l1: float
    pnl: float
    size: float
    early_exit: bool
    won: bool | None  # None for EARLY_EXIT


def load(db: str) -> list[Trade]:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        "SELECT market_slug, substr(timestamp,1,10), side, oracle_lag_signal, pnl, size_usdc, "
        "COALESCE(resolution,'') FROM trades "
        "WHERE pnl IS NOT NULL AND timestamp >= ? AND oracle_price_at_open != 0 "
        "ORDER BY timestamp",
        (AB_START,),
    )
    out = []
    for slug, day, side, l1, pnl, size, res in cur.fetchall():
        early = res.startswith("EARLY_EXIT")
        out.append(Trade(slug, day, side, l1 or 0.0, pnl, size, early, None if early else pnl > 0))
    con.close()
    return out


def against_line(t: Trade) -> bool:
    return (t.side == "YES" and t.l1 < -DEADBAND) or (t.side == "NO" and t.l1 > DEADBAND)


def summarise(label: str, trades: list[Trade]) -> None:
    pnl = sum(t.pnl for t in trades)
    resolved = [t for t in trades if t.won is not None]
    wr = 100 * sum(t.won for t in resolved) / len(resolved) if resolved else 0
    print(f"{label}: n={len(trades)} pnl={pnl:+,.2f} (${pnl/len(trades):+.3f}/trade) "
          f"WR={wr:.1f}% (n_resolved={len(resolved)})")


def daily_pnl(trades: list[Trade]) -> dict[str, float]:
    d: dict[str, float] = defaultdict(float)
    for t in trades:
        d[t.day] += t.pnl
    return d


def paired_t(diffs: list[float]) -> tuple[float, float]:
    """Mean and t-stat of paired daily differences."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    t = mean / math.sqrt(var / n) if var > 0 else 0.0
    return mean, t


def main() -> None:
    k = load(K_DB)
    k2 = load(K2_DB)

    # Sanity: floor means K2 must have ~0 against-line trades
    k2_against = [t for t in k2 if against_line(t)]
    assert len(k2_against) == 0, f"classification rule broken: {len(k2_against)}"

    print(f"=== A/B window {AB_START} -> now, zero-open windows excluded ===\n")

    print("-- Headline ledgers --")
    summarise("Bot K ", k)
    summarise("Bot K2", k2)

    print("\n-- By side --")
    for side in ("YES", "NO"):
        summarise(f"Bot K  {side}", [t for t in k if t.side == side])
        summarise(f"Bot K2 {side}", [t for t in k2 if t.side == side])

    print("\n-- The floor's target population (on K's own ledger) --")
    k_against = [t for t in k if against_line(t)]
    k_with = [t for t in k if not against_line(t)]
    summarise("K against-line (floor would block)", k_against)
    for side in ("YES", "NO"):
        summarise(f"  of which {side}", [t for t in k_against if t.side == side])
    summarise("K with-line/deadband (floor keeps) ", k_with)
    print(f"K counterfactual (K minus against-line): "
          f"{sum(t.pnl for t in k_with):+,.2f} vs actual {sum(t.pnl for t in k):+,.2f}")

    print("\n-- Paired windows (same market_slug) --")
    k_by_slug = {t.slug: t for t in k}
    k2_by_slug = {t.slug: t for t in k2}
    both = sorted(set(k_by_slug) & set(k2_by_slug))
    same_side = [s for s in both if k_by_slug[s].side == k2_by_slug[s].side]
    diff_side = [s for s in both if k_by_slug[s].side != k2_by_slug[s].side]
    print(f"paired={len(both)} same-side={len(same_side)} diff-side={len(diff_side)}")
    if diff_side:
        dk = sum(k_by_slug[s].pnl for s in diff_side)
        dk2 = sum(k2_by_slug[s].pnl for s in diff_side)
        print(f"diff-side windows: K {dk:+,.2f} vs K2 {dk2:+,.2f}")
    ps_k = sum(k_by_slug[s].pnl for s in same_side)
    ps_k2 = sum(k2_by_slug[s].pnl for s in same_side)
    print(f"same-side windows: K {ps_k:+,.2f} vs K2 {ps_k2:+,.2f} (timing/fill noise)")

    k_only = sorted(set(k_by_slug) - set(k2_by_slug))
    k2_only = sorted(set(k2_by_slug) - set(k_by_slug))
    k_only_against = [s for s in k_only if against_line(k_by_slug[s])]
    k_only_other = [s for s in k_only if not against_line(k_by_slug[s])]
    print(f"\nK-only windows: {len(k_only)} "
          f"(floor-explained: {len(k_only_against)}, other/uptime: {len(k_only_other)})")
    print(f"  floor-explained PnL (K earned, K2 avoided): "
          f"{sum(k_by_slug[s].pnl for s in k_only_against):+,.2f}")
    print(f"  other PnL: {sum(k_by_slug[s].pnl for s in k_only_other):+,.2f}")
    print(f"K2-only windows: {len(k2_only)} "
          f"PnL {sum(k2_by_slug[s].pnl for s in k2_only):+,.2f} (uptime/fill noise)")

    print("\n-- Daily paired difference (K2 - K), all trades --")
    dk, dk2 = daily_pnl(k), daily_pnl(k2)
    days = sorted(set(dk) & set(dk2))
    diffs = [dk2[d] - dk[d] for d in days]
    for d, x in zip(days, diffs):
        print(f"  {d}  K {dk[d]:+8.2f}  K2 {dk2[d]:+8.2f}  diff {x:+8.2f}")
    mean, t = paired_t(diffs)
    print(f"days={len(days)} mean daily diff={mean:+.2f} paired t={t:.2f}")

    print("\nNOTE: small day sample; maker-fill study context: K2 ~10% ROI vs "
          "K ~19-21% under realistic fills (J26). EARLY_EXIT in PnL, not WR.")


if __name__ == "__main__":
    main()
