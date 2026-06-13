"""One-time live verification of the kill-switch CLOB-V2 calls, at 0 positions.

The §7.4 dry-run exercised the kill path via a direct HALT write; the actual
``cancel_all_orders`` / ``get_positions`` calls against the real account under
CLOB V2 have never run (see vault to-do / J34 pre-flip gap). This script closes
that gap WITHOUT firing anything dangerous:

  * asserts the account is FLAT first and ABORTS if not (never proceeds with an
    open position — this is a verification, not a flatten);
  * confirms ``get_positions()`` (Data API discovery) returns cleanly;
  * confirms ``get_open_orders()`` reads under V2;
  * fires ``cancel_all_orders()`` — a safe no-op at 0 resting orders, but it
    exercises the real V2 cancel POST shape (THIS step needs the VPN connected);
  * NEVER writes HALT and NEVER places a SELL.

It does not exercise the flatten SELL (that needs a real position; it first runs
live on the probe's first early-exit, and uses the same ``place_order`` the
smoke test already verified for BUY).

Run with the VPN connected:  ``.venv\\Scripts\\python.exe -m tools.verify_kill_switch_live``
Exit code 0 = all checks passed; non-zero = something needs attention.
"""

from __future__ import annotations

import asyncio
import sys

from core.config import Config
from core.kill_switch_io import halt_active
from core.polymarket_client import PolymarketClient


async def _main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}")

    print("Kill-switch live verification (0-position) — CLOB V2\n")

    # Safety: refuse to run if HALT is already set (would mask state / confuse).
    if halt_active():
        print("ABORT: HALT flag is already present (data_runtime/HALT). Clear it "
              "first if intended, then re-run.")
        return 2

    config = Config.load()
    poly = PolymarketClient(config)
    await poly.connect()

    # 1. Auth.
    check("CLOB authenticated", poly.is_authenticated,
          "signer derived from .env" if poly.is_authenticated else
          "set POLYMARKET_PRIVATE_KEY/FUNDER_ADDRESS in .env")
    if not poly.is_authenticated:
        print("\nCannot continue without auth.")
        return 1

    # 2. get_positions() — the kill switch's primary discovery (Data API, no VPN).
    positions = await poly.get_positions()
    flat = len(positions) == 0
    check("get_positions() returns cleanly", isinstance(positions, list),
          f"{len(positions)} open position(s)")
    check("account is FLAT (precondition)", flat,
          "0 positions" if flat else f"OPEN: {positions} — re-run only when flat")
    if not flat:
        print("\nABORT: account is not flat. This script will not touch open "
              "positions. Re-run once flat (or let the 5-min markets resolve).")
        return 3

    # 3. get_open_orders() — V2 read shape.
    open_orders = await poly.get_open_orders()
    check("get_open_orders() reads under V2", isinstance(open_orders, list),
          f"{len(open_orders)} resting order(s)")

    # 4. cancel_all_orders() — the V2 cancel POST (NEEDS VPN). Safe no-op at 0 orders.
    print("\n  > Firing cancel_all_orders() (needs VPN; no-op at 0 orders)...")
    cancelled = await poly.cancel_all_orders()
    check("cancel_all_orders() POST succeeded under V2", cancelled,
          "ok" if cancelled else "returned False — VPN connected? check logs above")

    passed = all(ok for _, ok, _ in results)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'} "
          f"({sum(ok for _, ok, _ in results)}/{len(results)})")
    if passed:
        print("Kill-switch V2 cancel/discovery path verified live at 0 positions. "
              "Flatten SELL remains first-exercised on the probe's first early-exit.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
