"""Kraken WebSocket feed — BTC/USD ticker.

Kraken's WS v2 endpoint exposes a free public BTC/USD ticker channel.
Native USD pair (not USDT), so no stablecoin basis distortion.

Built to mirror the interface of `data.coinbase_feed.CoinbaseFeed` so
it can drop into the oracle aggregation without bespoke handling.
"""

import asyncio
import logging
import time

import websockets

from core import fast_json as json

logger = logging.getLogger(__name__)

KRAKEN_WS = "wss://ws.kraken.com/v2"


class KrakenFeed:
    """Real-time BTC/USD price from Kraken Exchange (WebSocket v2)."""

    def __init__(self):
        self.price: float = 0.0           # Last trade price
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.price_time: float = 0.0      # Exchange-side timestamp
        self.received_at: float = 0.0     # Our local time when message arrived
        self._running: bool = False
        # Track recent prices for direction (mirrors CoinbaseFeed shape)
        self._price_history: list[tuple[float, float]] = []
        self._max_history = 60

    @property
    def is_connected(self) -> bool:
        ref = self.received_at or self.price_time
        return self.price > 0 and ref > 0 and (time.time() - ref) < 10

    async def start(self):
        """Connect and stream BTC/USD ticker."""
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(KRAKEN_WS, ping_interval=20) as ws:
                    # Subscribe to BTC/USD ticker
                    sub_msg = json.dumps({
                        "method": "subscribe",
                        "params": {
                            "channel": "ticker",
                            "symbol": ["BTC/USD"],
                        },
                    })
                    await ws.send(sub_msg)
                    logger.info("Kraken BTC/USD ticker connected")
                    backoff = 1

                    async for msg in ws:
                        if not self._running:
                            break
                        recv_at = time.time()
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        self._handle_message(data, recv_at)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(
                    f"Kraken disconnected: {e}. Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Kraken error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _handle_message(self, data: dict, recv_at: float) -> None:
        """Handle one ticker update or snapshot message."""
        # Kraken v2 ticker format:
        # {"channel": "ticker", "type": "snapshot"|"update", "data": [{symbol, bid, ask, last, ...}]}
        if data.get("channel") != "ticker":
            return
        ticks = data.get("data", [])
        if not isinstance(ticks, list):
            return
        for tick in ticks:
            if tick.get("symbol") != "BTC/USD":
                continue
            last = tick.get("last")
            if last is not None:
                try:
                    self.price = float(last)
                    self.price_time = recv_at  # Kraken doesn't include trade ts in ticker
                    self.received_at = recv_at
                    self._price_history.append((recv_at, self.price))
                    if len(self._price_history) > self._max_history:
                        self._price_history = self._price_history[-self._max_history:]
                except (TypeError, ValueError):
                    pass
            bid = tick.get("bid")
            if bid is not None:
                try:
                    self.bid = float(bid)
                except (TypeError, ValueError):
                    pass
            ask = tick.get("ask")
            if ask is not None:
                try:
                    self.ask = float(ask)
                except (TypeError, ValueError):
                    pass

    async def stop(self):
        self._running = False
