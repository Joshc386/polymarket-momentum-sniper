"""FOK fill-detection against the verified CLOB V2 response shapes (2026-06-13).

Live smoke confirmed:
  - the POST (place_order) body is {errorMsg, orderID, takingAmount,
    makingAmount, status, success} -- it has NO size_matched/price.
  - the order record (get_order_status) carries status, size_matched, price.

So a filled FOK must read its fill price/size from the order record, not the
POST body (which would otherwise fall back to the limit price). _execute_gtc
already reads the order record, so it needs no change.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution import LiveExecutionEngine
from logging_db.database import Database


def _engine() -> LiveExecutionEngine:
    return LiveExecutionEngine(db=Database(":memory:"), poly_client=MagicMock())


def test_fok_reads_fill_from_order_record_not_post_body():
    eng = _engine()
    # POST shape on a fill: no size_matched/price (only making/takingAmount).
    eng.poly.place_order = AsyncMock(return_value={
        "orderID": "0xF", "status": "matched", "success": True,
        "makingAmount": "5", "takingAmount": "2.2",
    })
    # The order record is the source of truth for the actual fill.
    eng.poly.get_order_status = AsyncMock(return_value={
        "id": "0xF", "status": "MATCHED", "size_matched": "5", "price": "0.44",
    })
    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(eng._execute_fok("0xTOK", price=0.43, size_usdc=2.20))
    assert result is not None
    fill_px, fill_sz, oid = result
    assert oid == "0xF"
    assert fill_px == pytest.approx(0.44)   # real fill price, NOT fok_price
    assert fill_sz == pytest.approx(5.0)


def test_fok_unfilled_returns_none():
    eng = _engine()
    eng.poly.place_order = AsyncMock(return_value={
        "orderID": "0xF", "status": "live", "success": True,
    })
    eng.poly.get_order_status = AsyncMock(return_value={
        "id": "0xF", "status": "CANCELED", "size_matched": "0", "price": "0.45",
    })
    with patch("core.execution.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(eng._execute_fok("0xTOK", price=0.43, size_usdc=2.20))
    assert result is None
