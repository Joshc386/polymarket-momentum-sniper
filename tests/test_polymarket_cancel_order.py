"""Regression test for PolymarketClient.cancel_order (CLOB V2, 2026-06-13).

V1 cancelled with ``client.cancel(order_id)`` (a bare string). V2 removed
``cancel`` — cancellation now takes an ``OrderPayload`` object:
``client.cancel_order(OrderPayload(orderID=order_id))``. This pins that
contract so the kill-switch / GTC-timeout cancel path can't silently regress
to the dead v1 method.
"""

import asyncio
from unittest.mock import MagicMock, patch

from core.polymarket_client import PolymarketClient


class _FakePayload:
    """Stand-in for py_clob_client_v2.OrderPayload."""

    def __init__(self, orderID):
        self.orderID = orderID


class _FakeClob:
    def __init__(self):
        self.cancelled = None

    def cancel_order(self, payload):
        self.cancelled = payload
        return {"canceled": ["0xORDER"], "not_canceled": {}}


def test_cancel_order_wraps_id_in_orderpayload():
    fake = _FakeClob()
    c = PolymarketClient(MagicMock())
    c.client = fake
    c._authenticated = True
    with patch("core.polymarket_client.OrderPayload", _FakePayload):
        ok = asyncio.run(c.cancel_order("0xORDER"))
    assert ok is True
    assert isinstance(fake.cancelled, _FakePayload)
    assert fake.cancelled.orderID == "0xORDER"


def test_cancel_order_returns_false_when_unauthenticated():
    c = PolymarketClient(MagicMock())
    c._authenticated = False
    assert asyncio.run(c.cancel_order("0xORDER")) is False
