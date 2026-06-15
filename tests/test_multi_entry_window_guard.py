"""Regression tests for the live multiple-entries-per-window bug (2026-06-15).

Bot K2 went live and placed MULTIPLE orders per 5-minute window on real
capital: same-direction stacking (4 Down buys in one window) and opposite-
side positions (Up AND Down in one window).

Root cause (two failure modes, one shared defect): the strategy tracks its
position with an in-memory flag set only from the synchronous fill-return of
ONE GTC round, and never reconciles against on-chain state.

  (1) Cancel/fill race -- ``_execute_gtc`` read the cancelled order's
      ``size_matched`` exactly ONCE. A fill that settles as the cancel lands
      (a beat after that single read) was missed -> the round returned None
      -> the per-tick re-post loop fired again -> an untracked on-chain
      position stacked under a new order.

  (2) Mid-window side flip -- when the signal flipped between re-post rounds
      the bot posted the opposite side, producing Up AND Down in one window.

Fix (Design A, approved 2026-06-15):
  - Change 1 (core/execution.py): after cancel, poll the order to a TERMINAL
    state and read the FINAL size_matched before declaring a miss. A "miss"
    now means CONFIRMED zero fill.
  - Change 2 (strategies/contrarian_ev.py): lock the side to the first
    placement of the window; block any later decision on the other side.

Invariant restored: at most one net position per 5-minute window, with the
re-post loop preserved for confirmed zero-fill rounds only.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import core.execution_rounds as execution_rounds
from core.execution import LiveExecutionEngine
from strategies.contrarian_ev import ContrarianEvStrategy
from strategy.entry_logic import EntryDecision


def _engine(monkeypatch, tmp_path, **kwargs):
    """LiveExecutionEngine with a mocked CLOB client and a temp rounds sink."""
    monkeypatch.setattr(
        execution_rounds, "ROUNDS_LOG_PATH", tmp_path / "execution_rounds.log"
    )
    poly = MagicMock()
    poly.place_order = AsyncMock()
    poly.get_order_status = AsyncMock()
    poly.cancel_order = AsyncMock()
    poly.get_orderbook = MagicMock(return_value=None)
    eng = LiveExecutionEngine(
        db=MagicMock(), poly_client=poly, bot_id="bot_test", **kwargs
    )
    return eng, poly


# ── Change 1: cancel/fill race ────────────────────────────────────────────


def test_fill_settling_at_cancel_is_captured_not_missed(monkeypatch, tmp_path):
    """The fill lands as the cancel does. The first post-cancel status read is
    stale (still live, nothing matched); the fill is visible on the next read.
    The round must return that fill -- declaring a miss here is what stacked an
    untracked position in the live bug."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-race"}
    poly.get_order_status.side_effect = [
        {"status": "LIVE", "size_matched": 0},                       # stale read
        {"status": "CANCELED", "price": 0.52, "size_matched": 3.0},  # settled
    ]

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result == (0.52, 3.0, "ord-race")
    poly.cancel_order.assert_awaited_once_with("ord-race")


def test_unavailable_status_after_cancel_stays_a_miss(monkeypatch, tmp_path):
    """If the order status can't be read after cancel (no usable dict), we
    cannot confirm a fill -> the round is a miss (the window-end / kill-switch
    reconcile is the deeper backstop). Keeps the no-status path fast."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-none"}
    poly.get_order_status.return_value = None

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None


def test_terminal_zero_match_stays_a_miss(monkeypatch, tmp_path):
    """A terminal status with nothing matched is a CONFIRMED miss -- no further
    polling, no phantom fill."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-zero"}
    poly.get_order_status.return_value = {"status": "CANCELED", "size_matched": 0}

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None


# ── Change 2: per-window side lock ────────────────────────────────────────


def _decision(side: str) -> EntryDecision:
    return EntryDecision(should_enter=True, side=side, price=0.46, best_ev=0.01)


def test_first_entry_is_never_self_blocked():
    """Before any placement this window the side is unlocked -> either side is
    allowed through the filter."""
    s = ContrarianEvStrategy("t", {})
    assert s._window_side is None
    assert s._apply_entry_filters(_decision("YES"), secs_remaining=200.0) == ""
    assert s._apply_entry_filters(_decision("NO"), secs_remaining=200.0) == ""


def test_opposite_side_blocked_once_window_side_locked():
    """After the window commits to YES, a flipped NO decision is blocked --
    this is the opposite-side-in-one-window failure mode."""
    s = ContrarianEvStrategy("t", {})
    s._window_side = "YES"
    assert "side_locked" in s._apply_entry_filters(
        _decision("NO"), secs_remaining=200.0
    )


def test_same_side_still_allowed_when_locked():
    """The re-post loop is preserved: a same-side decision after a confirmed
    zero-fill round is NOT blocked by the side lock."""
    s = ContrarianEvStrategy("t", {})
    s._window_side = "YES"
    assert s._apply_entry_filters(_decision("YES"), secs_remaining=200.0) == ""


def test_new_window_clears_the_side_lock():
    """A fresh window must be free to pick either side."""
    s = ContrarianEvStrategy("t", {})
    s._window_side = "NO"
    snapshot = MagicMock()
    s.on_new_window(snapshot)
    assert s._window_side is None


def _bridge_skeleton(executor) -> ContrarianEvStrategy:
    """Bare strategy with only the entry-bridge + lock state set."""
    s = ContrarianEvStrategy.__new__(ContrarianEvStrategy)
    s.name = "bot_test"
    s._executor = executor
    s._entry_in_flight = False
    s._entry_task = None
    s._has_position = False
    s._last_trade_msg = ""
    s._window_side = None
    return s


class _FakeLiveExecutor:
    is_paper = False

    async def execute_trade(self, **kwargs):
        await asyncio.sleep(0)
        return None


def test_first_submission_locks_the_window_side():
    """The lock is armed by the first actual placement, and a later round on a
    different side does not move it."""
    s = _bridge_skeleton(_FakeLiveExecutor())

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08)
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._window_side == "YES"

    # A later round (in-flight cleared) on the other side must not relock.
    async def scenario2():
        s._submit_entry(side="NO", price=0.48, size_usdc=2.08)
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario2())
    assert s._window_side == "YES"
