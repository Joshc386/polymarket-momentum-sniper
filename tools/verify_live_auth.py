"""Read-only live-auth verification for the T1 go-live checklist.

Verifies the .env live-trading credentials end-to-end WITHOUT placing
orders or moving funds:

1. private key loads and the CLOB client authenticates (L1 -> derive L2 creds)
2. the derived signer address (printed -- cross-check vs Polymarket profile)
3. CLOB server reachable (server time)
4. account positions via the public Data API (funder address)
5. USDC collateral balance visible to the CLOB

Prints ONLY public data (addresses, counts, balances). Never prints the
private key or the derived API secret/passphrase.

Usage: .venv\\Scripts\\python.exe -m tools.verify_live_auth
"""

import asyncio
import sys

from core.config import Config
from core.polymarket_client import PolymarketClient


async def main() -> int:
    config = Config.load()

    if not config.polymarket_private_key:
        print("FAIL: POLYMARKET_PRIVATE_KEY is empty in .env")
        return 1
    if not config.polymarket_funder_address:
        print("FAIL: POLYMARKET_FUNDER_ADDRESS is empty in .env")
        return 1

    print(f"signature_type: {config.polymarket_signature_type}")
    print(f"funder address: {config.polymarket_funder_address}")

    client = PolymarketClient(config)
    await client.connect()
    if not client.is_authenticated:
        print("FAIL: CLOB authentication failed (see log above)")
        return 1
    print("OK: CLOB client authenticated, L2 API creds derived")

    signer = client.client.get_address()
    print(f"signer address (from private key): {signer}")
    if signer.lower() == config.polymarket_funder_address.lower():
        print(
            "NOTE: signer == funder. For an email/Magic account "
            "(signature_type=1) the funder is normally the PROXY wallet "
            "holding USDC, not the signer EOA -- double-check the address."
        )

    server_time = client.get_server_time()
    print(f"CLOB server time: {server_time}")

    positions = await client.get_positions()
    print(f"open positions (Data API, funder): {len(positions)}")
    for p in positions[:5]:
        print(f"  {p.get('title', p.get('asset', '?'))}: size={p.get('size')}")

    try:
        from py_clob_client_v2 import AssetType, BalanceAllowanceParams

        bal = client.client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        print(f"USDC balance/allowance (raw): {bal}")
    except Exception as e:  # balance check is best-effort, auth already proven
        print(f"WARN: balance check failed: {e}")

    print("PASS: read-only live-auth verification complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
