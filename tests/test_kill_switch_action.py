"""Tests for tools.kill_switch — the kill ACTION (ADR-0002, KILL_SWITCH_PLAN §3).

Order of operations (halt-first): write HALT -> cancel all orders -> discover
positions (account ground truth) -> flatten each subject to the >60s guard ->
re-verify flat. All via an injected PolymarketClient so we never touch a live
account. Flatten is an aggressive marketable SELL that accepts partial fills and
retries up to N.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core import kill_switch_io as ksio
from tools import kill_switch as ks


@pytest.fixture(autouse=True)
def _no_fill_wait(monkeypatch):
    """Zero the inter-retry sleep so retry tests run instantly."""
    monkeypatch.setattr(ks, "FILL_WAIT_SECS", 0)


YES = "0xYES"
NO = "0xNO"
ORPHAN = "0xORPHAN"


def _book(best_bid: float):
    """Raw CLOB book dict (bids highest-first not assumed; code must sort)."""
    return {
        "bids": [{"price": str(best_bid - 0.05), "size": "50"},
                 {"price": str(best_bid), "size": "100"}],
        "asks": [{"price": str(best_bid + 0.02), "size": "100"}],
    }


def _poly(positions_sequence, *, best_bid=0.60):
    """Mock PolymarketClient.

    positions_sequence: list of return values for successive get_positions()
    calls (lets a test model size shrinking as fills land).
    """
    poly = MagicMock()
    poly.cancel_all_orders = AsyncMock(return_value=True)
    poly.get_positions = AsyncMock(side_effect=list(positions_sequence))
    poly.get_orderbook = MagicMock(return_value=_book(best_bid))
    poly.place_order = AsyncMock(return_value={"orderID": "x"})
    return poly


def _pos(token, size, cond="0xc"):
    return {"token_id": token, "size": size, "condition_id": cond}


def _heartbeat(tmp_path, *, window_end_ts, token_ids, ts=1000.0):
    p = tmp_path / "heartbeat.json"
    ksio.write_heartbeat(window_end_ts=window_end_ts, token_ids=token_ids, ts=ts, path=p)
    return p


# --- HALT-first ordering --------------------------------------------------

def test_halt_written_before_cancel(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])

    saw_halt_at_cancel = {}

    async def cancel_side_effect():
        saw_halt_at_cancel["halted"] = ksio.halt_active(path=halt)
        return True

    poly = _poly([[], []])  # no positions
    poly.cancel_all_orders = AsyncMock(side_effect=cancel_side_effect)

    asyncio.run(ks.run_kill(
        trigger="manual", reason="test", poly=poly,
        now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log",
    ))
    assert ksio.halt_active(path=halt) is True
    assert saw_halt_at_cancel["halted"] is True  # HALT existed before cancel ran


def test_cancel_all_orders_called(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    poly = _poly([[], []])
    asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                            now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    poly.cancel_all_orders.assert_awaited_once()


# --- flatten guard --------------------------------------------------------

def test_flattens_position_when_more_than_60s_to_resolution(tmp_path):
    halt = tmp_path / "HALT"
    # window ends at 2000, now 1500 -> 500s left -> SELL
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    poly = _poly([[_pos(YES, 100.0)], []])  # discover 100 shares, then flat
    asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                            now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    poly.place_order.assert_awaited()
    args, kwargs = poly.place_order.call_args
    assert (kwargs.get("side") or args[1]) == "SELL"


def test_skips_flatten_within_60s_guard(tmp_path):
    halt = tmp_path / "HALT"
    # window ends at 2000, now 1950 -> 50s left -> SKIP (let it resolve)
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    poly = _poly([[_pos(YES, 100.0)]])
    summary = asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                                      now=1950.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    poly.place_order.assert_not_awaited()
    assert summary["skipped"] and summary["skipped"][0]["token_id"] == YES


def test_unknown_token_biases_to_sell(tmp_path):
    halt = tmp_path / "HALT"
    # heartbeat only knows YES/NO; ORPHAN is not in it -> bias to sell.
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    poly = _poly([[_pos(ORPHAN, 40.0)], []])
    asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                            now=1950.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    # even though "now" is inside the guard window for known tokens, an unknown
    # token has no known timing -> it must still be sold.
    poly.place_order.assert_awaited()


# --- bounded retry / partial fills ---------------------------------------

def test_retry_continues_on_partial_fills_until_flat(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    # discover 100; after 1st sell 60 remain; after 2nd flat. Final re-verify flat.
    poly = _poly([
        [_pos(YES, 100.0)],   # discovery
        [_pos(YES, 60.0)],    # remaining after attempt 1
        [],                   # flat after attempt 2
        [],                   # final re-verify
    ])
    summary = asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                                      now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    assert poly.place_order.await_count == 2
    assert summary["flat"] is True


def test_retry_bounded_and_loud_when_not_flat(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    # position never shrinks -> retries hit the cap; re-verify still shows size.
    seq = [[_pos(YES, 100.0)]] + [[_pos(YES, 100.0)]] * 10
    poly = _poly(seq)
    summary = asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                                      now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    assert poly.place_order.await_count == ks.MAX_RETRIES
    assert summary["flat"] is False
    assert summary["unflattened"] and summary["unflattened"][0]["token_id"] == YES


# --- CTF fallback merge (step 5) -----------------------------------------

def test_ctf_fallback_surfaces_position_data_api_missed(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    # Data API returns nothing (e.g. lagging/outage); CTF finds YES on-chain.
    poly = _poly([[], []])

    async def ctf_discover(token_ids):
        assert token_ids == [YES, NO]  # scoped to heartbeat tokens
        return [_pos(YES, 25.0)]

    summary = asyncio.run(ks.run_kill(
        trigger="watchdog", reason="t", poly=poly, now=1500.0,
        ctf_discover=ctf_discover, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log",
    ))
    poly.place_order.assert_awaited()  # the CTF-discovered position got flattened
    assert summary["sold"] and summary["sold"][0]["token_id"] == YES


def test_merge_keeps_larger_size(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    # Data API says 30; on-chain says 50 -> flatten the larger (bias to flatten).
    poly = _poly([[_pos(YES, 30.0)], []])

    async def ctf_discover(token_ids):
        return [_pos(YES, 50.0)]

    asyncio.run(ks.run_kill(
        trigger="watchdog", reason="t", poly=poly, now=1500.0,
        ctf_discover=ctf_discover, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log",
    ))
    _, kwargs = poly.place_order.call_args
    assert kwargs["size"] == 50.0  # sold the larger of the two sources


# --- no positions ---------------------------------------------------------

def test_no_positions_is_clean(tmp_path):
    halt = tmp_path / "HALT"
    hb = _heartbeat(tmp_path, window_end_ts=2000.0, token_ids=[YES, NO])
    poly = _poly([[], []])
    summary = asyncio.run(ks.run_kill(trigger="manual", reason="t", poly=poly,
                                      now=1500.0, halt_path=halt, heartbeat_path=hb, log_path=halt.parent / "kill.log"))
    poly.place_order.assert_not_awaited()
    assert summary["flat"] is True
