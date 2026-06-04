"""End-to-end kill-switch chain (step-7 paper dry-run gate, automatable core).

Wires the REAL watchdog + REAL kill action together: a stale heartbeat must make
the watchdog fire ``run_kill``, which must write the HALT flag that the bot's
loop-top guard then sees. Only the Polymarket client is mocked — paper mode has
no live account, so cancel/flatten are no-ops (cancel returns False, no
positions). The full manual runbook (a real ``multi_runner`` process stopping +
exiting) is in docs/KILL_SWITCH_PLAN.md §7.4.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from core import kill_switch_io as ksio
from tools.kill_switch import run_kill
from tools.watchdog import Watchdog


def _paper_poly():
    """A Polymarket client as it behaves in paper / unauthenticated mode."""
    poly = MagicMock()
    poly.cancel_all_orders = AsyncMock(return_value=False)  # not authenticated
    poly.get_positions = AsyncMock(return_value=[])         # no live positions
    poly.get_orderbook = MagicMock(return_value={"bids": [], "asks": []})
    poly.place_order = AsyncMock()
    return poly


def test_stale_heartbeat_fires_real_kill_and_writes_halt(tmp_path):
    halt = tmp_path / "HALT"
    hb = tmp_path / "heartbeat.json"

    # Bot wrote one heartbeat at t=100, then hung.
    ksio.write_heartbeat(window_end_ts=2000.0, token_ids=["0xA"], ts=100.0, path=hb)
    assert ksio.halt_active(path=halt) is False  # not halted yet

    poly = _paper_poly()
    captured = {}

    async def on_fire():
        captured["summary"] = await run_kill(
            trigger="watchdog",
            reason="heartbeat stale (integration test)",
            poly=poly, now=200.0, halt_path=halt, heartbeat_path=hb,
        )

    wd = Watchdog(
        on_fire=on_fire, heartbeat_path=hb, poll_secs=0.0,
        staleness_secs=20.0, stale_checks_to_fire=3,
    )
    # Deterministic clock: arm at 100, then three stale polls -> fire.
    times = iter([100.0, 125.0, 127.0, 129.0, 131.0])
    wd.clock = lambda: next(times)

    asyncio.run(wd.run(max_polls=5))

    # The whole chain happened with real components:
    assert wd.fired is True                              # watchdog fired
    poly.cancel_all_orders.assert_awaited_once()         # kill action ran (cancel)
    assert ksio.halt_active(path=halt) is True           # HALT now written
    assert captured["summary"]["flat"] is True           # nothing to flatten in paper
    assert captured["summary"]["trigger"] == "watchdog"
    # The bot's loop-top guard (multi_runner) checks exactly this -> it would stop.
    assert ksio.halt_active(path=halt) is True


def test_fresh_heartbeat_never_fires(tmp_path):
    """Negative control: a bot that keeps beating is never killed."""
    halt = tmp_path / "HALT"
    hb = tmp_path / "heartbeat.json"
    poly = _paper_poly()

    async def on_fire():  # pragma: no cover - must never run
        await run_kill(trigger="watchdog", reason="x", poly=poly,
                       halt_path=halt, heartbeat_path=hb)

    wd = Watchdog(on_fire=on_fire, heartbeat_path=hb, poll_secs=0.0,
                  staleness_secs=20.0, stale_checks_to_fire=3)

    # Clock advances, but the heartbeat is rewritten fresh each poll.
    t = {"now": 100.0}

    def clock():
        t["now"] += 5.0
        ksio.write_heartbeat(window_end_ts=t["now"] + 300, token_ids=["0xA"],
                             ts=t["now"], path=hb)
        return t["now"]

    wd.clock = clock
    asyncio.run(wd.run(max_polls=10))

    assert wd.fired is False
    assert ksio.halt_active(path=halt) is False
    poly.cancel_all_orders.assert_not_awaited()
