"""On-chain ERC-1155 balance reader — the kill switch's position-discovery fallback.

Primary discovery is the off-chain Data API (``PolymarketClient.get_positions``).
When that is down or lagging, the truth-of-record is the funder/proxy wallet's
ERC-1155 balances on the Polymarket Conditional Tokens Framework (CTF) contract.
We read them with a raw Polygon JSON-RPC ``eth_call`` to ``balanceOf(account, id)``
— same httpx + fallback-RPC pattern as ``data.sm_trade_monitor`` /
``data.chainlink_oracle``, no ``web3`` dependency.

Scoped to a known set of ``token_ids`` (in a kill, the active window's tokens
from ``heartbeat.json``) — ERC-1155 has no "list all my ids" call, so discovery
here is per-token by design.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Polymarket CTF contract on Polygon (mirrors data.sm_trade_monitor).
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
# CTF token amounts are 1e6-scaled (USDC-denominated shares).
TOKEN_DECIMALS = 6
# ERC-1155 balanceOf(address,uint256) function selector (keccak256[:4]).
BALANCE_OF_SELECTOR = "0x00fdd58e"
# Free Polygon RPCs — same list as chainlink_oracle.py / sm_trade_monitor.py.
DEFAULT_RPCS = [
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
]


def _encode_balance_of(account: str, token_id: str) -> str:
    """ABI-encode balanceOf(account, token_id) calldata."""
    acct = account.lower().replace("0x", "").zfill(64)
    tid = format(int(token_id), "064x")
    return BALANCE_OF_SELECTOR + acct + tid


def _resolve_rpcs(rpcs: list[str] | None) -> list[str]:
    if rpcs:
        return list(rpcs)
    env = os.environ.get("POLYGON_RPC_URL", "").strip()
    return ([env] + DEFAULT_RPCS) if env else list(DEFAULT_RPCS)


async def get_ctf_balances(
    account: str,
    token_ids: list[str],
    *,
    rpcs: list[str] | None = None,
) -> list[dict]:
    """Read on-chain ERC-1155 balances for ``token_ids`` held by ``account``.

    Returns a list of ``{"token_id": str, "size": float, "condition_id": ""}``
    for tokens with a positive balance (shares, de-scaled by ``TOKEN_DECIMALS``).
    Fail-safe: any token whose RPC calls all fail is skipped; an empty
    ``token_ids`` or missing ``account`` returns ``[]`` without any RPC.
    """
    if not account or not token_ids:
        return []

    endpoints = _resolve_rpcs(rpcs)
    out: list[dict] = []

    async with httpx.AsyncClient(timeout=5.0) as http:
        for i, token_id in enumerate(token_ids, start=1):
            try:
                data = _encode_balance_of(account, token_id)
            except (TypeError, ValueError):
                logger.warning("ctf_balances: bad token id %r — skipping", token_id)
                continue
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": CTF_CONTRACT, "data": data}, "latest"],
                "id": i,
            }
            result = await _rpc_call(http, endpoints, payload)
            if result is None:
                continue
            try:
                shares = int(result, 16) / (10 ** TOKEN_DECIMALS)
            except (TypeError, ValueError):
                logger.warning("ctf_balances: unparseable result for %s", token_id)
                continue
            if shares > 0:
                out.append({"token_id": token_id, "size": shares, "condition_id": ""})
    return out


async def _rpc_call(http: httpx.AsyncClient, endpoints: list[str], payload: dict) -> str | None:
    """eth_call across fallback RPCs; returns the hex result string or None."""
    for url in endpoints:
        try:
            resp = await http.post(url, json=payload)
            body = resp.json()
            if "error" in body:
                logger.debug("ctf_balances RPC error via %s: %s", url, body["error"])
                continue
            return body.get("result")
        except Exception as e:
            logger.debug("ctf_balances RPC failed via %s: %s", url, e)
            continue
    logger.warning("ctf_balances: all RPC endpoints failed for id=%s", payload.get("id"))
    return None
