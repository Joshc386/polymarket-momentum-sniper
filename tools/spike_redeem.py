"""Read-only SPIKE: scope the auto-redeem build (vault to-do 2026-06-15).

Answers the unknowns that decide whether redemption is a clean web3 tx or a
relayer reverse-engineering job — WITHOUT sending anything on-chain:

  1. proxy/signer setup: signature_type, funder (proxy) addr, signer EOA addr,
     and whether the funder is a smart contract (eth_getCode) — i.e. a proxy.
  2. gas: MATIC/POL balance of signer EOA and funder (does anyone hold gas?).
  3. neg-risk: are the current positions' tokens neg_risk (NegRiskAdapter
     redeem) or standard CTF (CTF.redeemPositions)?
  4. live test case: dump the Data API positions for the funder so we can see
     redeemable/resolved winnings sitting there right now, and the field shapes.

Never prints the private key. Throwaway scoping tool — safe to delete after the
auto-redeem design lands.

Run:  .venv\\Scripts\\python.exe -m tools.spike_redeem
"""

from __future__ import annotations

import json

import httpx
from eth_account import Account

from core.config import Config
from core.ctf_balances import DEFAULT_RPCS

DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def _rpc(http: httpx.Client, method: str, params: list):
    for url in DEFAULT_RPCS:
        try:
            r = http.post(url, json={"jsonrpc": "2.0", "method": method,
                                     "params": params, "id": 1}, timeout=8.0)
            body = r.json()
            if "result" in body:
                return body["result"]
        except Exception:
            continue
    return None


def main() -> None:
    cfg = Config.load()
    funder = (cfg.polymarket_funder_address or "").strip()
    signer = Account.from_key(cfg.polymarket_private_key).address if cfg.polymarket_private_key else "(no key)"

    print("=== 1. Proxy / signer setup ===")
    print(f"  signature_type : {cfg.polymarket_signature_type}")
    print(f"  funder (proxy) : {funder}")
    print(f"  signer EOA     : {signer}")

    with httpx.Client() as http:
        code = _rpc(http, "eth_getCode", [funder, "latest"]) if funder else None
        is_contract = bool(code and code not in ("0x", "0x0"))
        print(f"  funder is a contract (proxy)? {is_contract} "
              f"(code len={len(code) if code else 0})")

        print("\n=== 2. Gas (MATIC/POL) ===")
        for label, addr in (("signer EOA", signer), ("funder/proxy", funder)):
            if not addr or addr == "(no key)":
                continue
            bal = _rpc(http, "eth_getBalance", [addr, "latest"])
            matic = int(bal, 16) / 1e18 if bal else None
            print(f"  {label:13s}: {matic if matic is not None else 'RPC failed'} MATIC")

        print("\n=== 4. Data API positions (funder) ===")
        positions = []
        if funder:
            try:
                r = http.get(f"{DATA_API}/positions", params={"user": funder}, timeout=10.0)
                positions = r.json() if r.status_code == 200 else []
            except Exception as e:
                print(f"  positions query failed: {e}")
        print(f"  count: {len(positions)}")
        if positions:
            print("  field keys:", sorted(positions[0].keys()))
            for p in positions[:6]:
                tid = p.get("asset") or p.get("token_id") or p.get("tokenId")
                print("  -", json.dumps({
                    "token": str(tid)[:18],
                    "size": p.get("size"),
                    "redeemable": p.get("redeemable"),
                    "outcome": p.get("outcome"),
                    "title": str(p.get("title", ""))[:40],
                    "negRisk": p.get("negativeRisk", p.get("negRisk")),
                }))

        print("\n=== 3. neg-risk for held tokens ===")
        tokens = [str(p.get("asset") or p.get("token_id") or "") for p in positions[:5]]
        tokens = [t for t in tokens if t]
        if not tokens:
            print("  (no held tokens to check)")
        for t in tokens:
            try:
                r = http.get(f"{CLOB}/neg-risk", params={"token_id": t}, timeout=8.0)
                print(f"  {t[:18]}… -> {r.status_code} {r.text[:80]}")
            except Exception as e:
                print(f"  {t[:18]}… -> query failed: {e}")


if __name__ == "__main__":
    main()
