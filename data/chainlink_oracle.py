import asyncio
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# Chainlink BTC/USD Price Feed on Polygon
AGGREGATOR_ADDRESS = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
LATEST_ROUND_DATA_SELECTOR = "0xfeaf968c"
DECIMALS = 8

# Free Polygon RPCs (ordered by reliability)
DEFAULT_RPCS = [
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
]


@dataclass
class ChainlinkOracle:
    """Tracks Chainlink BTC/USD oracle price on Polygon.

    This is the resolution source for Polymarket 5-min BTC markets.
    """

    price: float = 0.0
    updated_at: float = 0.0       # Unix timestamp of last oracle update
    last_fetch_time: float = 0.0  # When we last polled
    window_open_price: float = 0.0  # Oracle price at start of current 5-min window
    _rpcs: list[str] = field(default_factory=lambda: list(DEFAULT_RPCS))
    _http_client: httpx.AsyncClient | None = None
    _running: bool = False

    async def start(self, poll_interval: float = 2.0):
        """Start polling the oracle. Runs until stop() is called."""
        self._http_client = httpx.AsyncClient(timeout=5.0)
        self._running = True
        while self._running:
            await self._fetch_price()
            await asyncio.sleep(poll_interval)

    async def stop(self):
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def fetch_once(self) -> float:
        """Single fetch — useful for grabbing window open price."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        await self._fetch_price()
        return self.price

    def set_window_open_price(self):
        """Snapshot current oracle price as the window opening price."""
        if self.price > 0:
            self.window_open_price = self.price
            logger.info(f"Window open oracle price set: ${self.price:,.2f}")

    async def _fetch_price(self):
        """Call latestRoundData() on the Chainlink aggregator."""
        for rpc_url in self._rpcs:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [
                        {
                            "to": AGGREGATOR_ADDRESS,
                            "data": LATEST_ROUND_DATA_SELECTOR,
                        },
                        "latest",
                    ],
                    "id": 1,
                }
                resp = await self._http_client.post(rpc_url, json=payload)
                data = resp.json()
                result = data.get("result", "")

                if not result or len(result) < 130:
                    continue

                hex_data = result[2:]
                answer = int(hex_data[64:128], 16)
                updated = int(hex_data[192:256], 16)

                self.price = answer / (10**DECIMALS)
                self.updated_at = float(updated)
                self.last_fetch_time = time.time()
                return

            except Exception as e:
                logger.debug(f"Chainlink fetch failed via {urlsplit(rpc_url).hostname or '<rpc>'}: {e}")
                continue

        logger.warning("All Chainlink RPC endpoints failed")
