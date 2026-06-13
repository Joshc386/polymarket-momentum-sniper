"""Regression tests for PolymarketClient.place_order.

Originally written 2026-06-12 to pin three live-only bugs in the
py-clob-client (v1) call path. Migrated 2026-06-13 to the py-clob-client-v2
(CLOB V2) contract: orders are now placed with a SINGLE call,
``create_and_post_order(order_args, options, order_type)``, where:
  1. order_args must be an OrderArgs OBJECT with ``.token_id`` (a dict raises
     "'dict' object has no attribute token_id").
  2. order_type is passed as the plain string "GTC"/"FOK"/"GTD"/"FAK" — v2's
     own OrderType members ARE these strings, so the value is JSON
     serializable in the request body (the v1 local-enum shadow bug).
  3. side is mapped to the v2 ``Side`` enum (BUY=0/SELL=1).

The fakes stand in for the real SDK so these tests run without it installed.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

from core.polymarket_client import PolymarketClient


class _FakeOrderArgs:
    """Stand-in for py_clob_client_v2.OrderArgs (an object, not a dict)."""

    def __init__(self, token_id, price, size, side, **kw):
        self.token_id = token_id
        self.price = price
        self.size = size
        self.side = side


class _FakeSide:
    """Stand-in for py_clob_client_v2.Side (IntEnum BUY=0/SELL=1)."""

    BUY = "BUY"
    SELL = "SELL"


class _FakeClob:
    """Behaves like the v2 client: create_and_post_order needs an OBJECT with
    .token_id (a dict raises) and serializes the order-type string into the
    request body (a non-string-serializable value raises, as the v1 enum did
    live)."""

    def __init__(self):
        self.created = None
        self.posted_type = None
        self.options = None

    def create_and_post_order(
        self, order_args, options=None, order_type="GTC",
        post_only=False, defer_exec=False,
    ):
        _ = order_args.token_id  # a dict would AttributeError here (bug 1)
        json.dumps({"orderType": order_type})  # bug 3: enum is not serializable
        self.created = order_args
        self.options = options
        self.posted_type = order_type
        return {"orderID": "0xORDER", "status": "live"}


def _client(fake):
    c = PolymarketClient(MagicMock())
    c.client = fake
    c._authenticated = True
    return c


def test_place_order_passes_serializable_string_order_type_and_returns_id():
    fake = _FakeClob()
    c = _client(fake)
    with patch("core.polymarket_client.OrderArgs", _FakeOrderArgs), \
         patch("core.polymarket_client.Side", _FakeSide):
        resp = asyncio.run(c.place_order(
            token_id="0xTOK", side="BUY", price=0.52, size=5.0, order_type="GTC"
        ))
    assert resp["orderID"] == "0xORDER"
    assert fake.created.token_id == "0xTOK"   # an OrderArgs object, not a dict
    assert fake.created.price == 0.52
    assert fake.created.size == 5.0
    assert fake.created.side == _FakeSide.BUY  # mapped to the v2 Side enum
    assert fake.posted_type == "GTC"           # the STRING, JSON-serializable
    assert isinstance(fake.posted_type, str)


def test_place_order_maps_sell_side_and_passes_fok_string():
    fake = _FakeClob()
    c = _client(fake)
    with patch("core.polymarket_client.OrderArgs", _FakeOrderArgs), \
         patch("core.polymarket_client.Side", _FakeSide):
        asyncio.run(c.place_order(
            token_id="0xTOK", side="SELL", price=0.40, size=5.0, order_type="FOK"
        ))
    assert fake.created.side == _FakeSide.SELL
    assert fake.posted_type == "FOK"
    assert isinstance(fake.posted_type, str)


def test_place_order_returns_none_when_unauthenticated():
    c = PolymarketClient(MagicMock())
    c._authenticated = False
    assert asyncio.run(c.place_order(
        token_id="0xTOK", side="BUY", price=0.5, size=5.0
    )) is None
