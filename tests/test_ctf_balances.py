"""Tests for core.ctf_balances — on-chain ERC-1155 position discovery fallback.

When the off-chain Data API is down or lagging, the kill switch falls back to
the truth-of-record: the funder/proxy wallet's ERC-1155 balances on the
Polymarket CTF contract, read via raw Polygon JSON-RPC ``eth_call`` to
``balanceOf(account, id)`` (no web3 dependency). Balances are 1e6-scaled
(USDC-denominated shares).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.ctf_balances import (
    BALANCE_OF_SELECTOR,
    CTF_CONTRACT,
    get_ctf_balances,
)

ACCOUNT = "0xabc0000000000000000000000000000000000def"
YES = "100874332490202003342409653443210154663931011393064058052448521209336383955571"
NO = "5044658213"


def _hex_balance(shares: float) -> str:
    """A 32-byte hex eth_call result for a balance of `shares` (1e6-scaled)."""
    return "0x" + format(int(round(shares * 10**6)), "064x")


def _rpc(results_by_id=None, *, error_ids=None, raise_all=False):
    """Mock httpx.AsyncClient whose .post returns canned eth_call results.

    results_by_id: maps the JSON-RPC `id` (== index of token in token_ids, 1-based)
    to a balance in shares.
    """
    results_by_id = results_by_id or {}
    error_ids = error_ids or set()

    async def post(url, json=None):
        resp = MagicMock()
        rid = json.get("id")
        if raise_all:
            raise RuntimeError("rpc down")
        if rid in error_ids:
            resp.json.return_value = {"jsonrpc": "2.0", "id": rid,
                                      "error": {"code": -32000, "message": "boom"}}
        else:
            shares = results_by_id.get(rid, 0.0)
            resp.json.return_value = {"jsonrpc": "2.0", "id": rid,
                                      "result": _hex_balance(shares)}
        return resp

    http = MagicMock()
    http.post = AsyncMock(side_effect=post)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=http)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=cm)
    factory._http = http
    return factory


def _run(token_ids, factory, **kw):
    with patch("core.ctf_balances.httpx.AsyncClient", factory):
        return asyncio.run(get_ctf_balances(ACCOUNT, token_ids, rpcs=["https://rpc.test"], **kw))


# --- happy path -----------------------------------------------------------

def test_returns_positive_balances_scaled_by_decimals():
    factory = _rpc({1: 88.4884, 2: 12.0})
    out = _run([YES, NO], factory)
    assert out == [
        {"token_id": YES, "size": 88.4884, "condition_id": ""},
        {"token_id": NO, "size": 12.0, "condition_id": ""},
    ]


def test_zero_balances_filtered_out():
    factory = _rpc({1: 0.0, 2: 5.0})
    out = _run([YES, NO], factory)
    assert out == [{"token_id": NO, "size": 5.0, "condition_id": ""}]


def test_encodes_balance_of_call_correctly():
    factory = _rpc({1: 1.0})
    _run([YES], factory)
    _, kwargs = factory._http.post.call_args
    payload = kwargs["json"]
    assert payload["method"] == "eth_call"
    call = payload["params"][0]
    assert call["to"].lower() == CTF_CONTRACT.lower()
    data = call["data"]
    # selector + 32-byte account + 32-byte token id
    assert data.startswith(BALANCE_OF_SELECTOR)
    assert len(data) == 2 + 8 + 64 + 64
    assert ACCOUNT.lower()[2:] in data.lower()
    assert format(int(YES), "064x") in data.lower()


# --- fail-safe ------------------------------------------------------------

def test_empty_token_ids_returns_empty_without_rpc():
    factory = _rpc({})
    out = _run([], factory)
    assert out == []
    factory.assert_not_called()


def test_all_rpcs_failing_returns_empty():
    factory = _rpc(raise_all=True)
    out = _run([YES], factory)
    assert out == []


def test_rpc_error_for_one_token_skips_it():
    factory = _rpc({2: 7.0}, error_ids={1})
    out = _run([YES, NO], factory)
    assert out == [{"token_id": NO, "size": 7.0, "condition_id": ""}]
