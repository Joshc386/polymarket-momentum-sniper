"""Regression tests for PolymarketClient.place_order (2026-06-12).

The live smoke test (tools/live_smoke_test.py --place-real-order) surfaced
two real bugs the mocked suite couldn't:
  1. create_order was passed a plain dict; py-clob-client requires an
     OrderArgs OBJECT -> "'dict' object has no attribute 'token_id'".
  2. post_order was called with a non-existent order_type= kwarg; the real
     signature takes the OrderType enum positionally (orderType=).
Both would have failed on the first live order. These tests pin the call
contract against a fake client that behaves like the real one.
"""

import asyncio
from unittest.mock import MagicMock, patch

from core.polymarket_client import PolymarketClient


class _FakeOrderArgs:
    """Stand-in for py_clob_client.clob_types.OrderArgs (an object)."""

    def __init__(self, token_id, price, size, side, **kw):
        self.token_id = token_id
        self.price = price
        self.size = size
        self.side = side


class _FakeOrderType:
    GTC = "GTC"
    FOK = "FOK"
    GTD = "GTD"


class _FakeClob:
    """Behaves like the real client: create_order needs an OBJECT with
    .token_id (a dict raises), post_order takes orderType positionally."""

    def __init__(self):
        self.created = None
        self.posted_type = None

    def create_order(self, order_args, options=None):
        _ = order_args.token_id  # a dict would AttributeError here (the bug)
        self.created = order_args
        return {"signed": True}

    def post_order(self, order, orderType="GTC", post_only=False):
        self.posted_type = orderType
        return {"orderID": "0xORDER", "status": "live"}


def _client(fake):
    c = PolymarketClient(MagicMock())
    c.client = fake
    c._authenticated = True
    return c


def test_place_order_builds_orderargs_object_and_returns_id():
    fake = _FakeClob()
    c = _client(fake)
    with patch("core.polymarket_client.OrderArgs", _FakeOrderArgs), \
         patch("core.polymarket_client.OrderType", _FakeOrderType):
        resp = asyncio.run(c.place_order(
            token_id="0xTOK", side="BUY", price=0.52, size=5.0, order_type="GTC"
        ))
    assert resp["orderID"] == "0xORDER"
    assert fake.created.token_id == "0xTOK"   # an OrderArgs object, not a dict
    assert fake.created.price == 0.52
    assert fake.created.size == 5.0
    assert fake.posted_type == "GTC"          # OrderType passed positionally


def test_place_order_maps_fok_to_order_type_enum():
    fake = _FakeClob()
    c = _client(fake)
    with patch("core.polymarket_client.OrderArgs", _FakeOrderArgs), \
         patch("core.polymarket_client.OrderType", _FakeOrderType):
        asyncio.run(c.place_order(
            token_id="0xTOK", side="SELL", price=0.40, size=5.0, order_type="FOK"
        ))
    assert fake.posted_type == "FOK"


def test_place_order_returns_none_when_unauthenticated():
    c = PolymarketClient(MagicMock())
    c._authenticated = False
    assert asyncio.run(c.place_order(
        token_id="0xTOK", side="BUY", price=0.5, size=5.0
    )) is None
