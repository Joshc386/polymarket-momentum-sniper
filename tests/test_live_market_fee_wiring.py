"""Runner wiring for the per-window live fee (CLOB V2, 2026-06-13).

_apply_live_market_fee fetches the market's live taker fee once per window and
applies it to LIVE executors only (recorded PnL — never the EV gate). It is a
no-op when every bot is paper (so paper-only runs make no fee call) and
best-effort on fetch failure.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from multi_runner import _apply_live_market_fee


class _Bot:
    def __init__(self, executor):
        self._executor = executor


def _live_exec():
    e = MagicMock()
    e.is_paper = False
    return e


def _paper_exec():
    e = MagicMock()
    e.is_paper = True
    return e


def test_applies_fee_to_live_executors_only():
    live, paper = _live_exec(), _paper_exec()
    poly = MagicMock()
    poly.get_market_fee = AsyncMock(return_value=(0.10, 1.0))
    asyncio.run(_apply_live_market_fee([_Bot(live), _Bot(paper)], poly, "0xTOK"))
    live.set_market_fee.assert_called_once_with(0.10, 1.0)
    paper.set_market_fee.assert_not_called()
    poly.get_market_fee.assert_awaited_once_with("0xTOK")


def test_noop_when_all_paper():
    paper = _paper_exec()
    poly = MagicMock()
    poly.get_market_fee = AsyncMock(return_value=(0.10, 1.0))
    asyncio.run(_apply_live_market_fee([_Bot(paper)], poly, "0xTOK"))
    poly.get_market_fee.assert_not_awaited()
    paper.set_market_fee.assert_not_called()


def test_noop_when_fee_fetch_fails():
    live = _live_exec()
    poly = MagicMock()
    poly.get_market_fee = AsyncMock(return_value=None)
    asyncio.run(_apply_live_market_fee([_Bot(live)], poly, "0xTOK"))
    live.set_market_fee.assert_not_called()
