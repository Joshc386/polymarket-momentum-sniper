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
    """A terminal status that stays at zero through the poll budget is a
    CONFIRMED miss -- no phantom fill, and not flagged unconfirmed."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-zero"}
    poly.get_order_status.return_value = {"status": "CANCELED", "size_matched": 0}

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None
    assert eng.entry_unconfirmed is False


def test_malformed_size_matched_after_cancel_does_not_raise(monkeypatch, tmp_path):
    """V2 returns size_matched as a string; an empty/garbage value must not
    raise out of the round (an unhandled error drops a real fill -> re-post)."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-bad"}
    poly.get_order_status.return_value = {"status": "CANCELED", "size_matched": "abc"}

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None  # empty string -> treated as zero, not an exception


def test_string_typed_fill_after_cancel_is_parsed(monkeypatch, tmp_path):
    """A real fill reported with string-typed fields still books."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-str"}
    poly.get_order_status.return_value = {
        "status": "CANCELED", "price": "0.50", "size_matched": "3",
    }

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result == (0.50, 3.0, "ord-str")


def test_transient_none_then_fill_is_captured(monkeypatch, tmp_path):
    """A status read that hiccups (None) on the first poll then shows the fill
    must NOT be abandoned as a miss -- retry across the budget."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-hiccup"}
    poly.get_order_status.side_effect = [
        None,                                                        # hiccup
        {"status": "CANCELED", "price": 0.52, "size_matched": 4.0},  # fill
    ]

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result == (0.52, 4.0, "ord-hiccup")
    assert eng.entry_unconfirmed is False


def test_persistently_unreadable_status_flags_unconfirmed(monkeypatch, tmp_path):
    """If the order's fate can't be read after cancel (every poll None), we
    cannot rule out a fill -> fail safe: entry_unconfirmed is set so the
    strategy blocks further entries this window (no re-post into a stack)."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-dark"}
    poly.get_order_status.return_value = None

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None
    assert eng.entry_unconfirmed is True


def test_terminal_zero_then_late_fill_is_captured(monkeypatch, tmp_path):
    """A terminal status can race ahead of size_matched: CANCELED with 0 on the
    first read, the match surfacing on the next. Must not break on the first
    terminal-zero read."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-late"}
    poly.get_order_status.side_effect = [
        {"status": "CANCELED", "size_matched": 0},                   # terminal, stale
        {"status": "CANCELED", "price": 0.52, "size_matched": 2.0},  # match surfaced
    ]

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result == (0.52, 2.0, "ord-late")


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
    s._entry_blocked_unknown = False
    s._current_window_slug = ""
    s._entry_task_window = ""
    return s


class _FakeLiveExecutor:
    is_paper = False

    def __init__(self, result=None, entry_unconfirmed=False):
        self._result = result
        self.entry_unconfirmed = entry_unconfirmed

    async def execute_trade(self, **kwargs):
        await asyncio.sleep(0)
        return self._result


def _fake_trade(side="YES"):
    t = MagicMock()
    t.side = side
    t.entry_price = 0.52
    t.size_usdc = 2.08
    return t


def test_unconfirmed_round_blocks_further_entries_this_window():
    """A round that came back unconfirmed (engine.entry_unconfirmed) must set
    the strategy's window-level entry block (fail-safe, no re-post)."""
    s = _bridge_skeleton(_FakeLiveExecutor(result=None, entry_unconfirmed=True))

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08, market_slug="w1")
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._entry_blocked_unknown is True


def test_entry_unconfirmed_filter_blocks_all_entries():
    """Once the window is flagged unconfirmed, the entry filter blocks any
    side (not just the opposite one)."""
    s = ContrarianEvStrategy("t", {})
    s._entry_blocked_unknown = True
    assert "entry_unconfirmed" in s._apply_entry_filters(
        _decision("YES"), secs_remaining=200.0
    )
    assert "entry_unconfirmed" in s._apply_entry_filters(
        _decision("NO"), secs_remaining=200.0
    )


def test_new_window_clears_the_unconfirmed_block():
    s = ContrarianEvStrategy("t", {})
    s._entry_blocked_unknown = True
    s.on_new_window(MagicMock())
    assert s._entry_blocked_unknown is False


def test_fill_for_a_prior_window_is_not_booked_into_the_current_window():
    """A round launched in window w1 that only drains during w2 must NOT be
    booked into w2's state (cross-window straddle), and must block w2 entries
    so we don't stack while the straddle is unresolved."""
    s = _bridge_skeleton(_FakeLiveExecutor(result=_fake_trade("YES")))
    s._current_window_slug = "w2"

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08, market_slug="w1")
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._has_position is False
    assert s._entry_blocked_unknown is True


def test_fill_for_the_current_window_books_normally():
    s = _bridge_skeleton(_FakeLiveExecutor(result=_fake_trade("YES")))
    s._current_window_slug = "w1"

    async def scenario():
        s._submit_entry(side="YES", price=0.52, size_usdc=2.08, market_slug="w1")
        await s._entry_task
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert s._has_position is True
    assert s._entry_blocked_unknown is False


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
