"""Tests for per-round GTC fill telemetry (decisions_BTC J27).

Every ``_execute_gtc`` call is one posting round of the live re-post loop.
Each round must leave one JSONL record in the ``execution_rounds`` sink so
chase EV and the chase-at-floor hybrid are computable offline from REAL
fills. Telemetry is an observer: it must never alter or break order flow.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import core.execution_rounds as execution_rounds
from core.execution import LiveExecutionEngine


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


def _read_rounds(tmp_path):
    path = tmp_path / "execution_rounds.log"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_filled_round_logs_one_complete_record(monkeypatch, tmp_path):
    """A GTC round that fills leaves one record with the fill outcome."""
    eng, poly = _engine(monkeypatch, tmp_path)
    poly.place_order.return_value = {"orderID": "ord-1"}
    poly.get_order_status.return_value = {
        "status": "MATCHED", "price": 0.52, "size_matched": 4.0,
    }

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result == (0.52, 4.0, "ord-1")
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    rec = rounds[0]
    assert rec["bot_id"] == "bot_test"
    assert rec["token_id"] == "0xTOK"
    assert rec["round"] == 1
    assert rec["posted_bid"] == 0.52
    assert rec["spread"] == 0.02
    assert rec["time_remaining_at_post"] == 180.0
    assert rec["outcome"] == "filled"
    assert rec["fill_price"] == 0.52
    assert rec["fill_shares"] == 4.0
    assert rec["order_id"] == "ord-1"
    assert rec["post_ts"] > 0


def test_timeout_round_records_best_ask_and_still_cancels(monkeypatch, tmp_path):
    """A round that times out logs the prevailing best ask (the would-be
    chase price) and the order is still cancelled."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-2"}
    poly.get_orderbook.return_value = {
        "asks": [{"price": "0.55", "size": "10"}, {"price": "0.56", "size": "5"}],
        "bids": [{"price": "0.52", "size": "8"}],
    }

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.03)
    )

    assert result is None
    poly.cancel_order.assert_awaited_once_with("ord-2")
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    rec = rounds[0]
    assert rec["outcome"] == "timeout"
    assert rec["best_ask_at_timeout"] == 0.55
    assert rec["fill_price"] is None


def test_timeout_at_entry_floor_tagged_as_floor_stop(monkeypatch, tmp_path):
    """When the window's 60s entry floor is reached by the time the round
    times out, the re-post loop is over -- tag it so the chase-at-floor
    hybrid is computable offline."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-3"}

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 55.0, spread=0.03)
    )

    assert result is None
    rounds = _read_rounds(tmp_path)
    assert rounds[0]["outcome"] == "timeout_floor"


def test_round_number_increments_per_repost_and_resets_per_token(
    monkeypatch, tmp_path
):
    """The re-post loop on one token counts rounds 1, 2, ...; a new token
    (new window/market) starts again at round 1."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-n"}

    asyncio.run(eng._execute_gtc("0xTOK_A", 0.52, 2.08, 180.0, spread=0.02))
    asyncio.run(eng._execute_gtc("0xTOK_A", 0.53, 2.08, 168.0, spread=0.02))
    asyncio.run(eng._execute_gtc("0xTOK_B", 0.48, 2.08, 290.0, spread=0.02))

    rounds = _read_rounds(tmp_path)
    assert [(r["token_id"], r["round"]) for r in rounds] == [
        ("0xTOK_A", 1), ("0xTOK_A", 2), ("0xTOK_B", 1),
    ]


def test_failed_post_still_leaves_a_record(monkeypatch, tmp_path):
    """A rejected/failed order post is a consumed round -- it must appear
    in the sink or offline fill-rate analysis over-counts fills."""
    eng, poly = _engine(monkeypatch, tmp_path)
    poly.place_order.return_value = None

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
    )

    assert result is None
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    assert rounds[0]["outcome"] == "post_failed"
    assert rounds[0]["round"] == 1


def test_exchange_cancelled_order_recorded(monkeypatch, tmp_path):
    """An order the exchange cancels/expires (not our timeout) is its own
    outcome -- distinct from timeout so chase analysis can exclude it."""
    eng, poly = _engine(monkeypatch, tmp_path)
    poly.place_order.return_value = {"orderID": "ord-c"}
    poly.get_order_status.return_value = {"status": "CANCELED"}

    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(
            eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
        )

    assert result is None
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    assert rounds[0]["outcome"] == "cancelled"


def test_telemetry_failures_never_break_order_flow(monkeypatch, tmp_path):
    """Sink unwritable + orderbook fetch raising: the round still completes
    normally (observer pattern -- telemetry must never cost a fill)."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    # Unwritable sink: a *file* where the parent dir should be.
    blocker = tmp_path / "blocked"
    blocker.write_text("")
    monkeypatch.setattr(
        execution_rounds, "ROUNDS_LOG_PATH", blocker / "execution_rounds.log"
    )
    poly.place_order.return_value = {"orderID": "ord-x"}
    poly.get_orderbook.side_effect = RuntimeError("book fetch died")

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
    )

    assert result is None          # normal timeout outcome, no exception
    poly.cancel_order.assert_awaited_once()


def test_best_ask_handles_object_books_and_garbage():
    """best_ask: py-clob-client object books work; garbage returns None."""
    class Level:
        def __init__(self, price):
            self.price = price

    class Book:
        asks = [Level("0.61"), Level("0.60")]

    assert execution_rounds.best_ask(Book()) == 0.60
    assert execution_rounds.best_ask(None) is None
    assert execution_rounds.best_ask({"asks": []}) is None
    assert execution_rounds.best_ask({"no_asks_key": 1}) is None
    assert execution_rounds.best_ask(object()) is None


def test_partial_fill_before_timeout_becomes_a_real_trade(monkeypatch, tmp_path):
    """Orphaned-shares hole (ADR-0003): shares matched just before the
    timeout-cancel are real money in the wallet -- the round must return
    them as a (partial) fill, not report a miss."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-p"}
    # Final state query after the cancel: 1.5 of 4 shares were matched.
    poly.get_order_status.return_value = {
        "status": "CANCELED", "price": 0.52, "size_matched": 1.5,
    }

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
    )

    assert result == (0.52, 1.5, "ord-p")
    poly.cancel_order.assert_awaited_once_with("ord-p")
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    assert rounds[0]["outcome"] == "filled"
    assert rounds[0]["fill_shares"] == 1.5


def test_timeout_with_zero_matched_stays_a_miss(monkeypatch, tmp_path):
    """The final-state query finding nothing matched keeps the round a
    normal timeout miss."""
    eng, poly = _engine(monkeypatch, tmp_path, gtc_timeout_sec=0)
    poly.place_order.return_value = {"orderID": "ord-z"}
    poly.get_order_status.return_value = {
        "status": "CANCELED", "size_matched": 0,
    }

    result = asyncio.run(
        eng._execute_gtc("0xTOK", 0.52, 2.08, 180.0, spread=0.02)
    )

    assert result is None
    assert _read_rounds(tmp_path)[0]["outcome"] == "timeout"


def test_execute_trade_wires_spread_into_round_record(monkeypatch, tmp_path):
    """Entering via the public interface: the strategy's at-entry ob_spread
    lands in the round record (no extra REST call at post time)."""
    eng, poly = _engine(monkeypatch, tmp_path)
    poly.place_order.return_value = {"orderID": "ord-w"}
    poly.get_order_status.return_value = {
        "status": "MATCHED", "price": 0.52, "size_matched": 4.0,
    }

    kwargs = dict(
        side="YES", price=0.52, size_usdc=2.08, market_id="m", market_slug="s",
        oracle_lag_signal=0.0, momentum_signal=0.0, liquidation_signal=0.0,
        combined_signal=0.0, estimated_prob_up=0.5, market_implied_prob=0.5,
        edge=0.05, time_remaining_secs=120.0, btc_price=70000.0,
        oracle_price=70000.0, oracle_open_price=70000.0,
        yes_token_id="0xYES", no_token_id="0xNO", ob_spread=0.025,
    )
    with patch("core.execution.halt_active", return_value=False), \
         patch("core.execution.asyncio.sleep", new=AsyncMock()), \
         patch("core.execution._log_to_db", return_value=1):
        trade = asyncio.run(eng.execute_trade(**kwargs))

    assert trade is not None
    rounds = _read_rounds(tmp_path)
    assert len(rounds) == 1
    assert rounds[0]["spread"] == 0.025
    assert rounds[0]["token_id"] == "0xYES"
