"""Coinbase WebSocket feed — secondary BTC price for cross-exchange confirmation.

Subscribes to BTC-USD ticker on Coinbase Exchange WebSocket.
Used to confirm directional signals from Binance — if both exchanges
agree on direction, signal confidence is higher.
"""

import asyncio
import json
import logging
import time

import websockets

logger = logging.getLogger(__name__)

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"


class CoinbaseFeed:
    """Real-time BTC-USD price from Coinbase Exchange."""

    def __init__(self):
        self.price: float = 0.0
        self.price_time: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self._running: bool = False
        # Track recent prices for directional confirmation
        self._price_history: list[tuple[float, float]] = []  # (timestamp, price)
        self._max_history = 60  # Keep last 60 ticks

    @property
    def is_connected(self) -> bool:
        return self.price > 0 and (time.time() - self.price_time) < 10

    @property
    def price_direction(self) -> float:
        """Recent price direction as a signal in [-1, 1].

        Compares current price to price ~5 seconds ago.
        """
        if len(self._price_history) < 2:
            return 0.0
        now = time.time()
        # Find price from ~5 seconds ago
        ref_price = None
        for ts, px in reversed(self._price_history):
            if now - ts >= 4.0:
                ref_price = px
                break
        if ref_price is None or ref_price <= 0:
            return 0.0
        delta = (self.price - ref_price) / ref_price
        # Normalize: 0.05% move = full signal
        return max(-1.0, min(1.0, delta / 0.0005))

    async def start(self):
        """Connect and stream BTC-USD ticker."""
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(COINBASE_WS, ping_interval=20) as ws:
                    # Subscribe to BTC-USD ticker
                    subscribe_msg = json.dumps({
                        "type": "subscribe",
                        "product_ids": ["BTC-USD"],
                        "channels": ["ticker"],
                    })
                    await ws.send(subscribe_msg)
                    logger.info("Coinbase BTC-USD ticker connected")
                    backoff = 1

                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if data.get("type") == "ticker":
                            self._handle_ticker(data)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(f"Coinbase disconnected: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Coinbase error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _handle_ticker(self, data: dict):
        px = data.get("price")
        if px:
            self.price = float(px)
            self.price_time = time.time()
            self._price_history.append((self.price_time, self.price))
            if len(self._price_history) > self._max_history:
                self._price_history = self._price_history[-self._max_history:]
        bid = data.get("best_bid")
        ask = data.get("best_ask")
        if bid:
            self.bid = float(bid)
        if ask:
            self.ask = float(ask)

    async def stop(self):
        self._running = False
