"""Tests for the sync-strategy -> async-live-engine bridge (ADR-0003 p3).

The runner ticks bots synchronously; the live engine's execute_trade
awaits a GTC round for up to 10s. Live entries therefore run as
fire-and-forget asyncio tasks guarded by an entry-in-flight flag: the
strategy evaluates no new entries while a round is posting, a miss clears
the flag (advancing the re-post loop), and a fill books the position via
the done-callback. Paper dispatch stays synchronous and unchanged.
"""

import asyncio
import logging
from unittest.mock import MagicMock

from strategies.contrarian_ev import ContrarianEvStrategy


def _skeleton(executor) -> ContrarianEvStrategy:
    """Bare strategy instance with only the bridge-relevant state."""
    s = ContrarianEvStrategy.__new__(ContrarianEvStrategy)
    s.name = "bot_test"
    s._executor = executor
    s._entry_in_flight = False
    s._exit_in_flight = False
    s._entry_task = None
    s._has_position = False
    s._window_side = None
    s._entry_blocked_unknown = False
    s._current_window_slug = ""
    s._entry_task_window = ""
    s._last_trade_msg = ""
    return s


class _FakeLiveExecutor:
    """Async executor double: is_paper False, scripted execute_trade."""

    is_paper = False

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    async def execute_trade(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)  # yield once, like a real GTC round
        if self._error:
            raise self._error
        return self._result

    async def close_position_early(self, exit_price, reason=""):
        self.calls += 1
        await asyncio.sleep(0)
        if self._error:
            raise self._error
        return self._result


def _fake_trade(side="YES", price=0.52, size=2.08):
    t = MagicMock()
    t.side = side
    t.entry_price = price
    t.size_usdc = size
    t.pnl = 0.5
    return t


def test_live_entry_runs_as_task_and_books_on_fill():
    """Tracer: live entry returns immediately, flag up while posting,
    fill books the position and clears the flag."""
    trade = _fake_trade()
    s = _skeleton(_FakeLiveExecutor(result=trade))

    async def scenario():
        out = s._submit_entry(side="YES", price=0.52, size_usdc=2.08)
        assert out is None                  # returned before the round ended
        assert s._entry_in_flight is True
        assert s._has_position is False
        await s._entry_task
        await asyncio.sleep(0)              # let the done-callback run
        return True

    assert asyncio.run(scenario())
    assert s._entry_in_flight is False
    assert s._has_position is True
    assert "[LIVE] YES" in s._last_trade_msg


def test_live_entry_miss_clears_flag_without_position():
    """A miss (None) advances the re-post loop: flag down, no position."""
    s = _skeleton(_FakeLiveExecutor(result=None))

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08)
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._entry_in_flight is False
    assert s._has_position is False


def test_live_entry_exception_clears_flag_and_never_raises():
    """A dying round must not kill the runner loop or wedge the flag."""
    s = _skeleton(_FakeLiveExecutor(error=RuntimeError("CLOB exploded")))

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08)
        await asyncio.gather(s._entry_task, return_exceptions=True)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._entry_in_flight is False
    assert s._has_position is False


def test_entry_refused_while_one_is_in_flight():
    """The in-flight guard is part of the submit unit itself."""
    executor = _FakeLiveExecutor(result=_fake_trade())
    s = _skeleton(executor)
    s._entry_in_flight = True

    async def scenario():
        return s._submit_entry(side="YES", price=0.52, size_usdc=2.08)

    assert asyncio.run(scenario()) is None
    assert executor.calls == 0              # never reached the engine


def test_paper_entry_stays_synchronous():
    """Paper dispatch is unchanged: direct call, result returned, no task."""
    paper = MagicMock()
    paper.is_paper = True
    sentinel = _fake_trade()
    paper.execute_trade = MagicMock(return_value=sentinel)
    s = _skeleton(paper)

    out = s._submit_entry(side="YES", price=0.52, size_usdc=2.08)

    assert out is sentinel
    assert s._entry_in_flight is False
    assert s._entry_task is None
    paper.execute_trade.assert_called_once()


def test_live_exit_runs_as_task_and_invokes_booking_callback():
    """Live exit: task + exit-in-flight guard; booking callback fires on
    an actual fill; second submit while in flight is refused."""
    exit_trade = _fake_trade()
    executor = _FakeLiveExecutor(result=exit_trade)
    s = _skeleton(executor)
    booked = []

    async def scenario():
        s._submit_exit(0.55, "BTC_DISTANCE_STOP_$95", booked.append)
        assert s._exit_in_flight is True
        s._submit_exit(0.55, "BTC_DISTANCE_STOP_$95", booked.append)  # refused
        await s._exit_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert booked == [exit_trade]
    assert executor.calls == 1
    assert s._exit_in_flight is False


def test_paper_exit_stays_synchronous():
    paper = MagicMock()
    paper.is_paper = True
    sentinel = _fake_trade()
    paper.close_position_early = MagicMock(return_value=sentinel)
    s = _skeleton(paper)
    booked = []

    s._submit_exit(0.55, "stop", booked.append)

    assert booked == [sentinel]
    paper.close_position_early.assert_called_once_with(
        exit_price=0.55, reason="stop"
    )
