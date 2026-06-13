"""Manual live smoke test for the T1 go-live gate (ADR-0003).

Validates the live WRITE paths against the REAL Polymarket CLOB — the one
thing the mocked unit suite cannot do. It proves the response shapes the
execution code assumes (order status fields, fill-status strings, orderbook
shape) actually match what py-clob-client returns, and that place/status/
cancel + the wallet balance read work end to end.

SAFE BY DEFAULT. With no flag it is READ-ONLY: auth, wallet balance, market
discovery, and an orderbook dump. To exercise the write path you must pass
``--place-real-order``; it then posts the minimum-size BUY (5 shares — the
market's min_order_size) far below market so it rests and will not fill,
dumps the raw place/status responses, and cancels it. Worst-case exposure
if it somehow filled: price x 5 shares (e.g. $0.10 at price 0.02).

Usage:
  .venv\\Scripts\\python.exe -m tools.live_smoke_test                 # read-only
  .venv\\Scripts\\python.exe -m tools.live_smoke_test --place-real-order
  .venv\\Scripts\\python.exe -m tools.live_smoke_test --place-real-order --price 0.02

Run it, then paste the output back so the response shapes can be checked
against core/execution.py's assumptions BEFORE the unattended probe.
"""

import argparse
import asyncio
import sys

from core.config import Config
from core.market_discovery import MarketDiscovery
from core.polymarket_client import PolymarketClient
from core.execution_rounds import best_ask, best_bid


def dump(label: str, obj: object) -> None:
    """Print a response's TYPE and shape — the point is to learn the real
    structure, so never assume it's a dict."""
    print(f"\n--- {label} ---")
    print(f"  type: {type(obj).__name__}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            shown = v if not isinstance(v, (dict, list)) else f"<{type(v).__name__} len={len(v)}>"
            print(f"    {k!r}: {shown!r}")
    elif obj is None:
        print("    <None>")
    else:
        attrs = getattr(obj, "__dict__", None)
        if attrs:
            for k, v in attrs.items():
                print(f"    .{k} = {v!r}")
        else:
            print(f"    repr: {obj!r}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--place-real-order", action="store_true",
        help="post ONE 1-share resting BUY and cancel it (real write path)",
    )
    parser.add_argument(
        "--price", type=float, default=0.02,
        help="limit price for the resting order (default 0.02, deep/no-fill)",
    )
    args = parser.parse_args()

    config = Config.load()
    if not config.polymarket_private_key:
        print("FAIL: POLYMARKET_PRIVATE_KEY empty in .env")
        return 1

    client = PolymarketClient(config)
    await client.connect()
    if not client.is_authenticated:
        print("FAIL: CLOB auth failed")
        return 1
    print("OK: authenticated")

    # 1) Wallet balance — validates the get_balance params-object fix.
    bal = await client.get_balance()
    print(f"\nget_balance(): ${bal:.2f} USDC", "  <-- expect ~ your funded amount")
    if bal <= 0:
        print("  WARN: balance read 0 — live sizing would fall back to default!")

    # 2) Current market + token ids.
    disco = MarketDiscovery()
    market = await disco.find_active_market()
    if not market:
        print("\nWARN: no active 5m market right now — skipping orderbook/order")
        await disco.close()
        return 0
    print(f"\nactive market: {market.slug}")
    print(f"  YES(Up) token:  {market.yes_token_id}")
    print(f"  NO(Down) token: {market.no_token_id}")

    # 3) Orderbook shape — validates best_bid/best_ask against reality.
    book = client.get_orderbook(market.yes_token_id)
    dump("get_orderbook(YES) raw", book)
    print(f"\n  best_bid extractor: {best_bid(book)}")
    print(f"  best_ask extractor: {best_ask(book)}")

    if not args.place_real_order:
        print("\nREAD-ONLY run complete. Re-run with --place-real-order to "
              "exercise place/status/cancel.")
        await disco.close()
        return 0

    # 4) WRITE PATH — minimum-size resting BUY far below market, then cancel.
    # The market reports min_order_size = 5 shares; honour it or the post is
    # rejected (the very issue this smoke test surfaced for live sizing).
    size = 5.0
    price = round(max(0.01, min(args.price, 0.05)), 2)
    print(f"\n>>> placing REAL resting BUY: {size:.0f} shares @ ${price:.2f} "
          f"on {market.yes_token_id[:16]}... (max exposure ${price * size:.2f})")
    resp = await client.place_order(
        token_id=market.yes_token_id, side="BUY", price=price,
        size=size, order_type="GTC",
    )
    dump("place_order raw response", resp)
    order_id = (resp or {}).get("orderID", (resp or {}).get("id", "")) if isinstance(resp, dict) else ""
    if not order_id:
        print("\nFAIL/Note: no order id parsed from the response above — "
              "check the real field name and update place_order/_execute_gtc.")
        await disco.close()
        return 1
    print(f"\norder id: {order_id}")

    await asyncio.sleep(2.0)
    status = await client.get_order_status(order_id)
    dump("get_order_status raw response", status)
    print("\n  CHECK: does it expose 'status' (MATCHED/FILLED/LIVE/...) and "
          "'size_matched'/'price'? _execute_gtc keys on exactly those.")

    print(f"\n>>> cancelling {order_id[:16]}...")
    ok = await client.cancel_order(order_id)
    print(f"  cancel returned: {ok}")
    await asyncio.sleep(1.0)
    dump("get_order_status AFTER cancel", await client.get_order_status(order_id))

    await disco.close()
    print("\nPASS: write-path smoke test complete. Verify the shapes above "
          "match core/execution.py before the unattended probe.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
