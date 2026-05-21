"""Binance.US WebSocket feed — BTC/USD spot price.

Binance.com (international) does NOT offer BTC/USD spot — only BTC/USDT,
BTC/USDC, etc. To get a USD-native price from "Binance" requires the
separate Binance.US entity.

Caveat: Binance.US is licensed/operated in the US and may geo-restrict
non-US IPs. If you see persistent disconnects, this is likely the cause
- swap to a different feed (Coinbase + Kraken can stand alone, or use
Binance.com BTC/USDC for closer-to-USD pricing).

Volume disclaimer: Binance.US BTC/USD does ~$50-300M/day. Binance.com
BTC/USDT does $1B+/day. So this feed represents a smaller venue's view
of the price, but it is USD-native (no stablecoin basis distortion).
"""

import asyncio
import logging
import time

import websockets

from core import fast_json as json

logger = logging.getLogger(__name__)

# Binance.US uses the same WebSocket protocol as Binance.com, just
# a different domain and a separate symbol universe.
BINANCE_US_TRADE_STREAM = "wss://stream.binance.us:9443/ws/btcusd@trade"


class BinanceUSFeed:
    """Real-time BTC/USD trade-price stream from Binance.US.

    Minimal implementation — just exposes `price`, `received_at`, and
    `is_connected` for the price comparison monitor. Doesn't subscribe
    to klines/liquidations/etc. because those aren't needed for the
    monitor's purpose.
    """

    def __init__(self):
        self.price: float = 0.0
        self.price_time: float = 0.0      # Exchange-side timestamp (T field)
        self.received_at: float = 0.0     # Our local time when message arrived
        self._running: bool = False

    @property
    def is_connected(self) -> bool:
        ref = self.received_at or self.price_time
        return self.price > 0 and ref > 0 and (time.time() - ref) < 10

    async def start(self):
        """Connect and stream BTC/USD trade prices."""
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_US_TRADE_STREAM, ping_interval=20
                ) as ws:
                    logger.info("Binance.US BTC/USD trade stream connected")
                    backoff = 1
                    async for msg in ws:
                        if not self._running:
                            break
                        recv_at = time.time()
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        # Trade event schema: {"e":"trade","E":...,"s":"BTCUSD",
                        #   "t":..., "p":"...", "q":"...", "T":..., "m":bool}
                        if data.get("e") == "trade":
                            try:
                                self.price = float(data["p"])
                                self.price_time = data["T"] / 1000.0
                                self.received_at = recv_at
                            except (KeyError, ValueError, TypeError):
                                pass
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(
                    f"Binance.US disconnected: {e}. Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    f"Binance.US error: {e}. Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def stop(self):
        self._running = False
