"""Repair fabricated resolutions from the J28 incident (2026-06-10/11).

Between 2026-06-10 and the fix, detect_resolution's price-comparison fallback
recorded wrong outcomes (with a zero window-open it degenerated to always-UP).
This tool re-fetches the TRUE resolution of every affected window from
Polymarket's Gamma API (authoritative on-chain outcome — these markets are
long settled) and corrects `resolution` + `pnl` in the bot DBs.

Safety:
  - DRY-RUN by default: prints the full correction plan, writes nothing.
  - --apply performs the UPDATEs, but REFUSES while the bot heartbeat is
    fresh (the paper engines derive bankroll from the trades table at
    startup, so corrections must land while the bots are stopped).
  - Self-validation: before trusting the fee formula, it recomputes PnL for
    every correctly-resolved trade in the window and reports the match rate.
    If that is not ~100%, do not --apply — the formula is wrong for the era.

Usage:
    python -m tools.repair_resolutions            # dry-run
    python -m tools.repair_resolutions --apply    # write (bots must be stopped)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "data_runtime"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

BOTS = ["bot_k_sm_confirmation", "bot_k2_l1_floor"]
INCIDENT_START = "2026-06-10"
FEE_RATE = 0.07  # canonical Polymarket taker fee (J17): fee/share = 0.07*p*(1-p)


def expected_pnl(won: bool, entry: float, shares: float) -> float:
    """Post-J17 paper PnL: fee charged at entry on every trade.

    win:  shares*(1-entry) - fee      loss: -shares*entry - fee
    """
    fee = FEE_RATE * entry * (1.0 - entry) * shares
    if won:
        return shares * (1.0 - entry) - fee
    return -shares * entry - fee


def fetch_true_resolution(client: httpx.Client, slug: str) -> str:
    """Authoritative outcome from the Gamma API (same logic as the live
    fetch_resolution, sync). Returns 'UP'/'DOWN'/'UNKNOWN'."""
    try:
        resp = client.get(f"{GAMMA_API_BASE}/events", params={"slug": slug},
                          timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return "UNKNOWN"
        event = data[0] if isinstance(data, list) else data
        markets = event.get("markets", [])
        if not markets:
            return "UNKNOWN"
        op = markets[0].get("outcomePrices")
        if isinstance(op, str):
            op = json.loads(op)
        if op and len(op) >= 2:
            if float(op[0]) > 0.9:
                return "UP"
            if float(op[1]) > 0.9:
                return "DOWN"
    except Exception as e:
        print(f"    fetch error {slug}: {e}")
    return "UNKNOWN"


def heartbeat_fresh(max_age_s: float = 30.0) -> bool:
    hb = RUNTIME / "heartbeat.json"
    if not hb.exists():
        return False
    try:
        ts = json.loads(hb.read_text()).get("ts", 0)
        return (time.time() - ts) < max_age_s
    except Exception:
        return False


def repair_bot(bot: str, apply: bool, cache: dict[str, str]) -> None:
    db = RUNTIME / f"{bot}.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # EARLY_EXIT trades are excluded: they were closed before resolution at an
    # exit price, so their recorded PnL is correct regardless of the window's
    # final outcome (the resolution column intentionally holds the exit tag).
    rows = conn.execute(
        "SELECT id, timestamp, market_slug, side, entry_price, num_shares, "
        "resolution, pnl FROM trades WHERE timestamp >= ? "
        "AND market_slug IS NOT NULL "
        "AND (resolution IS NULL OR resolution IN ('UP', 'DOWN')) "
        "ORDER BY timestamp",
        (INCIDENT_START,),
    ).fetchall()
    print(f"\n=== {bot}: {len(rows)} trades since {INCIDENT_START}")

    with httpx.Client() as client:
        # fetch true resolutions (cached across bots — same windows)
        slugs = sorted({r["market_slug"] for r in rows})
        for s in slugs:
            if s not in cache:
                cache[s] = fetch_true_resolution(client, s)
                time.sleep(0.15)  # polite rate limit
    unknown = sum(1 for s in slugs if cache[s] == "UNKNOWN")
    print(f"  windows: {len(slugs)}  (true resolution unknown: {unknown})")

    # self-validation of the fee formula on already-correct trades
    ok = chk = 0
    for r in rows:
        true_res = cache[r["market_slug"]]
        if true_res == "UNKNOWN" or r["resolution"] != true_res or r["pnl"] is None:
            continue
        won = (r["side"] == "YES") == (true_res == "UP")
        chk += 1
        if abs(expected_pnl(won, r["entry_price"], r["num_shares"]) - r["pnl"]) < 0.01:
            ok += 1
    print(f"  fee-formula self-check on clean trades: {ok}/{chk} match"
          + ("  <- OK" if chk and ok / chk > 0.99 else "  <- MISMATCH, DO NOT APPLY"))

    # build correction plan
    fixes = []
    for r in rows:
        true_res = cache[r["market_slug"]]
        if true_res == "UNKNOWN":
            continue
        needs_res = r["resolution"] != true_res
        won = (r["side"] == "YES") == (true_res == "UP")
        new_pnl = expected_pnl(won, r["entry_price"], r["num_shares"])
        needs_pnl = r["pnl"] is None or abs((r["pnl"] or 0) - new_pnl) > 0.01
        if needs_res or needs_pnl:
            fixes.append((r["id"], r["timestamp"][:19], r["market_slug"][-10:],
                          r["side"], r["resolution"], true_res,
                          r["pnl"], round(new_pnl, 4)))

    old_total = sum((f[6] or 0.0) for f in fixes)
    new_total = sum(f[7] for f in fixes)
    print(f"  corrections needed: {len(fixes)}  "
          f"(PnL on these trades: {old_total:+.1f} -> {new_total:+.1f}, "
          f"delta {new_total - old_total:+.1f})")
    for f in fixes[:8]:
        print(f"    id={f[0]} {f[1]} ..{f[2]} {f[3]}: "
              f"{f[4]}->{f[5]}  pnl {f[6]}->{f[7]}")
    if len(fixes) > 8:
        print(f"    ... and {len(fixes) - 8} more")

    if apply and fixes:
        for f in fixes:
            conn.execute("UPDATE trades SET resolution = ?, pnl = ? WHERE id = ?",
                         (f[5], f[7], f[0]))
        conn.commit()
        print(f"  APPLIED {len(fixes)} corrections to {db.name}")
    conn.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Repair J28 fabricated resolutions")
    p.add_argument("--apply", action="store_true",
                   help="write corrections (default: dry-run)")
    args = p.parse_args()

    if args.apply and heartbeat_fresh():
        print("REFUSING --apply: bot heartbeat is fresh (multi_runner appears "
              "to be running). Stop the bot first — paper bankroll is derived "
              "from the trades table at startup.")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"J28 resolution repair — {mode} — trades since {INCIDENT_START}")
    cache: dict[str, str] = {}
    for bot in BOTS:
        repair_bot(bot, args.apply, cache)
    if not args.apply:
        print("\nDry-run complete. Re-run with --apply (bot stopped) to write.")


if __name__ == "__main__":
    main()
