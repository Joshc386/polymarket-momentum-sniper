"""Emergency kill action for the trading bot (ADR-0002, KILL_SWITCH_PLAN §3).

Runs as a SEPARATE OS process (manual: ``python -m tools.kill_switch``) or is
called in-process by the watchdog (step 4) via :func:`run_kill` — same code
path either way. It talks to Polymarket with its OWN ``PolymarketClient``, never
reaching into the (possibly hung) trading process's memory.

Order of operations is **halt-first** so a bot that recovers mid-kill sees HALT
immediately (the bot's per-order guard then blocks it):

  1. write the sticky HALT flag
  2. cancel all resting orders
  3. discover open positions (account ground truth via the Data API)
  4. flatten each, subject to the >60s-to-resolution guard
  5. re-verify flat; log LOUDLY if anything remains (5-min auto-resolution is
     the backstop)

Flatten is an aggressive marketable-limit SELL that ACCEPTS PARTIAL FILLS and
retries (re-fetch book, cross the current best bid) up to ``MAX_RETRIES`` or a
price floor. Not FOK. There is no confirmation prompt — an emergency tool must
act instantly.

Every action is appended to ``data_runtime/kill_switch.log`` (JSONL) — the kill
switch's own authoritative record, independent of the bot's trades DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from core.kill_switch_io import (
    HALT_PATH,
    HEARTBEAT_PATH,
    read_heartbeat,
    write_halt,
)

logger = logging.getLogger(__name__)

KILL_LOG_PATH = Path("data_runtime") / "kill_switch.log"

FLATTEN_GUARD_SECS = 60.0   # closer than this to resolution -> let it resolve
MAX_RETRIES = 4             # aggressive-sell attempts per position
PRICE_FLOOR = 0.01          # never sell below this
TICK = 0.01                 # cross the bid by one tick to guarantee a marketable price
DUST = 1.0                  # shares at/below this are treated as flat (rounding/dust)
FILL_WAIT_SECS = 1.0        # pause between sell submit and re-querying remaining size


def _log_event(event: dict, *, path: Path = KILL_LOG_PATH) -> None:
    """Append one JSONL event to the kill log. Best-effort — never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:  # pragma: no cover - logging must not break a kill
        logger.error(f"kill log write failed: {e}")


def _best_bid(book: object) -> float | None:
    """Extract the highest bid price from an orderbook.

    Handles both the raw CLOB REST dict (``{"bids": [{"price","size"}]}``) and a
    py-clob-client ``OrderBookSummary`` object (``.bids`` of objects with
    ``.price``). Returns ``None`` if there is no bid.
    """
    if book is None:
        return None
    bids = book.get("bids") if isinstance(book, dict) else getattr(book, "bids", None)
    if not bids:
        return None
    prices: list[float] = []
    for b in bids:
        raw = b.get("price") if isinstance(b, dict) else getattr(b, "price", None)
        try:
            prices.append(float(raw))
        except (TypeError, ValueError):
            continue
    return max(prices) if prices else None


def _remaining_size(positions: list[dict], token_id: str) -> float:
    """Size still held for ``token_id`` across a positions snapshot."""
    return sum(
        float(p.get("size", 0) or 0)
        for p in positions
        if str(p.get("token_id", "")) == token_id
    )


async def _flatten(poly, token_id: str, size: float) -> float:
    """Aggressively sell ``size`` shares of ``token_id``. Returns shares remaining.

    Re-fetches the book each attempt and crosses the current best bid; accepts
    partial fills; re-queries the account for the remaining size; retries up to
    ``MAX_RETRIES`` or until flat / no bid / price floor.
    """
    remaining = size
    for attempt in range(1, MAX_RETRIES + 1):
        if remaining <= DUST:
            break
        bid = _best_bid(poly.get_orderbook(token_id))
        if bid is None or bid <= PRICE_FLOOR:
            _log_event({"ts": time.time(), "action": "flatten_no_bid",
                        "token_id": token_id, "remaining": remaining, "attempt": attempt})
            break
        price = round(max(PRICE_FLOOR, bid - TICK), 2)
        _log_event({"ts": time.time(), "action": "flatten_sell", "token_id": token_id,
                    "price": price, "size": round(remaining, 2), "attempt": attempt})
        await poly.place_order(
            token_id=token_id, side="SELL", price=price, size=round(remaining, 2),
            order_type="GTC",
        )
        await asyncio.sleep(FILL_WAIT_SECS)
        remaining = _remaining_size(await poly.get_positions(), token_id)
    return remaining


async def run_kill(
    trigger: str,
    reason: str,
    poly,
    *,
    now: float | None = None,
    halt_path: Path = HALT_PATH,
    heartbeat_path: Path = HEARTBEAT_PATH,
) -> dict:
    """Execute the kill action. Returns a summary dict (also JSONL-logged).

    ``trigger`` is ``"manual"`` or ``"watchdog"``; ``poly`` is an authenticated
    PolymarketClient (or a mock in tests).
    """
    now = time.time() if now is None else now
    summary: dict = {
        "trigger": trigger, "reason": reason, "ts": now,
        "cancelled": False, "sold": [], "skipped": [], "unflattened": [],
        "flat": True,
    }

    # 1. HALT first.
    write_halt(source=trigger, reason=reason, ts=now, path=halt_path)
    _log_event({"ts": now, "action": "halt", "trigger": trigger, "reason": reason})

    # 2. Cancel all resting orders.
    try:
        summary["cancelled"] = bool(await poly.cancel_all_orders())
    except Exception as e:
        logger.error(f"cancel_all_orders failed: {e}")
    _log_event({"ts": time.time(), "action": "cancel_all", "ok": summary["cancelled"]})

    # 3. Discover positions (account ground truth). CTF on-chain fallback is step 5.
    try:
        positions = await poly.get_positions()
    except Exception as e:
        logger.error(f"get_positions failed during kill: {e}")
        positions = []
    positions = [p for p in positions if float(p.get("size", 0) or 0) > DUST]
    _log_event({"ts": time.time(), "action": "discover", "n": len(positions)})

    # window_end_ts is shared by the tokens in the active heartbeat window.
    hb = read_heartbeat(path=heartbeat_path) or {}
    hb_tokens = set(hb.get("token_ids") or [])
    window_end_ts = hb.get("window_end_ts")

    # 4. Flatten each, subject to the >60s guard.
    for p in positions:
        token = str(p.get("token_id", ""))
        size = float(p.get("size", 0) or 0)
        known = token in hb_tokens and isinstance(window_end_ts, (int, float))
        if known:
            time_left = window_end_ts - now
            if time_left <= FLATTEN_GUARD_SECS:
                # Too close to resolution — cancel+halt already done; let it resolve.
                summary["skipped"].append({"token_id": token, "size": size,
                                           "time_left": time_left})
                _log_event({"ts": time.time(), "action": "skip_guard", "token_id": token,
                            "time_left": time_left})
                continue
        # known & >60s, OR unknown timing -> bias to sell.
        remaining = await _flatten(poly, token, size)
        if remaining <= DUST:
            summary["sold"].append({"token_id": token, "size": size})
        else:
            summary["unflattened"].append({"token_id": token, "remaining": remaining})

    # 5. Re-verify flat with an INDEPENDENT re-query (don't just trust the
    #    in-loop remaining). Only positions we acted on count — guard-skipped
    #    ones are intentionally left to resolve.
    try:
        final = await poly.get_positions()
    except Exception as e:
        logger.error(f"final get_positions failed: {e}")
        final = []
    acted = {s["token_id"] for s in summary["sold"]} | {u["token_id"] for u in summary["unflattened"]}
    still_open = [
        {"token_id": str(p.get("token_id", "")), "remaining": float(p.get("size", 0) or 0)}
        for p in final
        if float(p.get("size", 0) or 0) > DUST and str(p.get("token_id", "")) in acted
    ]
    if still_open:
        summary["flat"] = False
        summary["unflattened"] = still_open
        logger.critical(
            "KILL SWITCH: %d position(s) NOT flattened after %d retries: %s. "
            "Relying on 5-minute auto-resolution backstop.",
            len(still_open), MAX_RETRIES, still_open,
        )
    _log_event({"ts": time.time(), "action": "complete", "flat": summary["flat"],
                "sold": summary["sold"], "skipped": summary["skipped"],
                "unflattened": summary["unflattened"]})
    return summary


async def _main(trigger: str) -> None:
    """Build a real client and fire the kill action (manual invocation)."""
    from core.config import Config
    from core.polymarket_client import PolymarketClient

    config = Config.load()
    poly = PolymarketClient(config)
    await poly.connect()
    summary = await run_kill(trigger=trigger, reason="manual kill switch invocation", poly=poly)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _trigger = sys.argv[1] if len(sys.argv) > 1 else "manual"
    asyncio.run(_main(_trigger))
