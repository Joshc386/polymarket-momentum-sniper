"""Read-only watch: is Polymarket's OWN auto-redeem good enough?

Background (2026-06-15): Bot K2 went live; the worry was that resolved wins sit
as ``redeemable: true`` CTF positions and drain spendable USDC, false-halting the
breakers (which read wallet USDC only). A live snapshot found NO accumulated
redeemable positions despite a confirmed win + 0 MATIC on our side — i.e.
Polymarket appears to relay/pay the redeem itself. This tool watches that claim
over a long period before we decide whether to build our own auto-redeem.

Success criterion we're testing: over the watch window, no resolved winning ever
lingers as ``redeemable`` long enough for wallet USDC to starve the breakers
(never co-occurs with a HALT). If that holds → drop the auto-redeem build.

The core signal is auth-free: Data API ``/positions``, flagging rows that are
``redeemable: true`` AND have ``currentValue > 0`` — i.e. resolved WINNINGS not
yet swept back to USDC (a resolved loser is redeemable but worth 0 and locks up
nothing). No wallet auth, no VPN. Optionally (``--usdc``) it also reads the
authoritative CLOB ``COLLATERAL`` balance via ``PolymarketClient.get_balance``;
that path hits ``/auth/api-key`` so it is OFF by default to keep the per-minute
scheduled snapshot from poking the live account. (The proxy holds collateral via
Polymarket's exchange accounting, not a plain ERC-20 balance, so an on-chain
``balanceOf`` reads 0 — only the CLOB number is correct.)

Never signs or sends anything.

Success criterion we're testing: over the watch window, no resolved WINNING ever
lingers ``redeemable`` long enough to starve the breakers (never co-occurs with a
HALT). If that holds → drop the auto-redeem build.

Modes:
  snapshot  one reading appended to data_runtime/redeem_watch.csv (schedulable)
  watch     loop snapshot every --interval seconds (manual long-run)
  report    summarise the CSV — longest lingering-winnings streak, peak value,
            the danger combo (redeemable winnings while HALT present)

Run:  .venv\\Scripts\\python.exe -m tools.redeem_watch {snapshot|watch|report}
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from core.config import Config

DATA_API = "https://data-api.polymarket.com"
# Winnings below this (USDC) are dust/rounding, not a real false-halt risk.
DUST = 0.01

CSV_PATH = Path("data_runtime/redeem_watch.csv")
HALT_FILE = Path("data_runtime/HALT")
FIELDS = [
    "ts_utc", "usdc", "n_positions", "n_redeemable", "redeemable_value",
    "halt", "redeemable_titles",
]


def _positions(http: httpx.Client, funder: str) -> list[dict]:
    """Data API current positions for ``funder`` (empty list on any failure)."""
    try:
        r = http.get(f"{DATA_API}/positions", params={"user": funder}, timeout=10.0)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def _clob_usdc(cfg: Config) -> float | None:
    """Authoritative CLOB COLLATERAL balance (the number the breakers read).

    Best-effort: instantiates the client and connects; returns None on any
    failure so a snapshot never crashes on an auth hiccup. Hits /auth/api-key,
    so callers gate this behind --usdc rather than running it every minute.
    """
    try:
        from core.polymarket_client import PolymarketClient

        async def _go() -> float:
            c = PolymarketClient(cfg)
            await c.connect()
            return await c.get_balance()

        return round(asyncio.run(_go()), 4)
    except Exception:
        return None


def take_snapshot(cfg: Config, *, with_usdc: bool = False) -> dict:
    """One read of redeemable positions + HALT state (+ USDC if requested)."""
    funder = (cfg.polymarket_funder_address or "").strip()
    with httpx.Client() as http:
        positions = _positions(http, funder) if funder else []

    # Resolved WINNINGS = redeemable AND still worth money; losers (value 0)
    # are redeemable too but lock up no USDC, so they don't count.
    winnings = [p for p in positions
                if p.get("redeemable") and float(p.get("currentValue") or 0) > DUST]
    usdc = _clob_usdc(cfg) if with_usdc else None
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usdc": usdc if usdc is not None else "",
        "n_positions": len(positions),
        "n_redeemable": len(winnings),
        "redeemable_value": round(
            sum(float(p.get("currentValue") or 0) for p in winnings), 4),
        "halt": int(HALT_FILE.exists()),
        "redeemable_titles": "; ".join(
            str(p.get("title", ""))[:40] for p in winnings),
    }


def _append(row: dict) -> None:
    """Append a snapshot row, writing the header if the CSV is new."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def summarize(rows: list[dict]) -> dict:
    """Aggregate watch rows into the go/no-go signals.

    Pure function over CSV-parsed dict rows (string values, as ``csv`` yields)
    so it is unit-testable without a wallet. Reports:
      - longest unbroken run of ``redeemable_value > 0`` (count of rows + minutes
        spanned, using the row timestamps) — the headline "lingering winnings"
        metric,
      - peak redeemable count/value, minimum observed USDC,
      - ``danger_rows``: readings where redeemable winnings co-occurred with a
        HALT (the failure we're watching for).
    """
    if not rows:
        return {"rows": 0}

    def _i(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    best_len = best_min = 0
    cur_len = 0
    cur_start: str | None = None
    danger = 0
    usdc_vals: list[float] = []

    for r in rows:
        val = _f(r.get("redeemable_value")) or 0.0
        if _i(r.get("halt")) and val > DUST:
            danger += 1
        u = _f(r.get("usdc"))
        if u is not None:
            usdc_vals.append(u)

        if val > DUST:
            if cur_len == 0:
                cur_start = r.get("ts_utc")
            cur_len += 1
            span = _span_minutes(cur_start, r.get("ts_utc"))
            if cur_len > best_len:
                best_len, best_min = cur_len, span
        else:
            cur_len = 0
            cur_start = None

    return {
        "rows": len(rows),
        "first": rows[0].get("ts_utc"),
        "last": rows[-1].get("ts_utc"),
        "longest_redeemable_streak_rows": best_len,
        "longest_redeemable_streak_min": round(best_min, 1),
        "peak_n_redeemable": max((_i(r.get("n_redeemable")) for r in rows), default=0),
        "peak_redeemable_value": max(
            (_f(r.get("redeemable_value")) or 0 for r in rows), default=0),
        "min_usdc": round(min(usdc_vals), 4) if usdc_vals else None,
        "danger_rows": danger,
    }


def _span_minutes(start_iso: str | None, end_iso: str | None) -> float:
    """Minutes between two ISO timestamps; 0 if either is unparseable."""
    try:
        a = datetime.fromisoformat(start_iso)
        b = datetime.fromisoformat(end_iso)
        return (b - a).total_seconds() / 60.0
    except (TypeError, ValueError):
        return 0.0


def _read_rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _print_snapshot(row: dict) -> None:
    flag = "  <-- redeemable winnings sitting unredeemed" if row["n_redeemable"] else ""
    halt = "  HALT" if row["halt"] else ""
    print(f"{row['ts_utc']}  usdc={row['usdc']}  pos={row['n_positions']}  "
          f"redeemable={row['n_redeemable']} (${row['redeemable_value']}){halt}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["snapshot", "watch", "report"])
    ap.add_argument("--interval", type=float, default=60.0,
                    help="watch mode: seconds between snapshots (default 60)")
    ap.add_argument("--usdc", action="store_true",
                    help="also read the authoritative CLOB COLLATERAL balance "
                         "(hits /auth/api-key — off by default)")
    args = ap.parse_args()

    if args.mode == "report":
        s = summarize(_read_rows())
        if not s.get("rows"):
            print("No data yet — run snapshot/watch first.")
            return
        print("=== redeem_watch report ===")
        for k, v in s.items():
            print(f"  {k:32s}: {v}")
        if s["danger_rows"]:
            print("\n  ** danger: redeemable winnings co-occurred with HALT — "
                  "auto-redeem is NOT keeping up. **")
        elif s["longest_redeemable_streak_rows"] <= 1:
            print("\n  Polymarket auto-redeem looks sufficient so far "
                  "(no lingering redeemable winnings).")
        return

    cfg = Config.load()
    if args.mode == "snapshot":
        row = take_snapshot(cfg, with_usdc=args.usdc)
        _append(row)
        _print_snapshot(row)
        return

    # watch
    print(f"Watching every {args.interval:.0f}s — Ctrl-C to stop. "
          f"Appending to {CSV_PATH}")
    try:
        while True:
            row = take_snapshot(cfg, with_usdc=args.usdc)
            _append(row)
            _print_snapshot(row)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
