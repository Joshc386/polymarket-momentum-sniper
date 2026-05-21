"""Bitstamp WebSocket feed — BTC/USD order book + trades.

Bitstamp's public WebSocket (`wss://ws.bitstamp.net`) is free, unauthenticated,
and not geo-restricted from UK/EU. Native USD pair (not USDT), so no
stablecoin basis distortion.

Subscribes to two channels:
- `order_book_btcusd` — top-of-book bids/asks on every change. Used to
  compute the canonical `self.price` as the bid/ask midpoint (most
  responsive — updates many times per second).
- `live_trades_btcusd` — completed trades. Used to populate
  `self.last_trade_price` for callers that want the trade-print value.

Mirrors the interface of `data.kraken_feed.KrakenFeed` so it can drop
into the price monitor (and later the oracle aggregation) without
bespoke handling.
"""

import asyncio
import logging
import time

import websockets

from core import fast_json as json

logger = logging.getLogger(__name__)

BITSTAMP_WS = "wss://ws.bitstamp.net"
ORDER_BOOK_CHANNEL = "order_book_btcusd"
TRADES_CHANNEL = "live_trades_btcusd"


class BitstampFeed:
    """Real-time BTC/USD price from Bitstamp (WebSocket public API).

    `self.price` is the bid/ask midpoint when both sides are available
    (most responsive), falling back to the last trade price if not.
    `self.last_trade_price` is exposed separately for callers that want
    the trade-print value specifically.
    """

    def __init__(self):
        self.price: float = 0.0           # Mid (preferred) or last trade
        self.last_trade_price: float = 0.0
        self.bid: float = 0.0
        self.ask: float = 0.0
        self.price_time: float = 0.0      # Exchange-side timestamp
        self.received_at: float = 0.0     # Our local time when message arrived
        self._running: bool = False
        # Track recent prices for direction (mirrors KrakenFeed shape)
        self._price_history: list[tuple[float, float]] = []
        self._max_history = 60

    @property
    def is_connected(self) -> bool:
        ref = self.received_at or self.price_time
        return self.price > 0 and ref > 0 and (time.time() - ref) < 10

    async def start(self):
        """Connect and stream BTC/USD order book + trades."""
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(BITSTAMP_WS, ping_interval=20) as ws:
                    # Subscribe to both channels on the same connection.
                    for channel in (ORDER_BOOK_CHANNEL, TRADES_CHANNEL):
                        sub_msg = json.dumps({
                            "event": "bts:subscribe",
                            "data": {"channel": channel},
                        })
                        await ws.send(sub_msg)
                    logger.info("Bitstamp BTC/USD order book + trades connected")
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
                    f"Bitstamp disconnected: {e}. Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Bitstamp error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _handle_message(self, data: dict, recv_at: float) -> None:
        """Handle one order-book or trade message."""
        event = data.get("event")
        channel = data.get("channel")
        # Ignore non-data events (subscription confirmations, heartbeats).
        if event != "data":
            return
        payload = data.get("data", {})
        if not isinstance(payload, dict):
            return

        if channel == ORDER_BOOK_CHANNEL:
            # Format: {"timestamp": "...", "microtimestamp": "...",
            #          "bids": [["price","amount"], ...],
            #          "asks": [["price","amount"], ...]}
            bids = payload.get("bids") or []
            asks = payload.get("asks") or []
            if bids:
                try:
                    self.bid = float(bids[0][0])
                except (TypeError, ValueError, IndexError):
                    pass
            if asks:
                try:
                    self.ask = float(asks[0][0])
                except (TypeError, ValueError, IndexError):
                    pass
            # Exchange-side timestamp (microseconds, string).
            mts = payload.get("microtimestamp")
            if mts:
                try:
                    self.price_time = int(mts) / 1_000_000.0
                except (TypeError, ValueError):
                    self.price_time = recv_at
            else:
                self.price_time = recv_at

        elif channel == TRADES_CHANNEL:
            # Format: {"price": float, "amount": float, "timestamp": "...",
            #          "microtimestamp": "...", "type": 0|1, "id": ...}
            px = payload.get("price")
            if px is not None:
                try:
                    self.last_trade_price = float(px)
                except (TypeError, ValueError):
                    pass
            mts = payload.get("microtimestamp")
            if mts:
                try:
                    self.price_time = int(mts) / 1_000_000.0
                except (TypeError, ValueError):
                    self.price_time = recv_at
        else:
            return

        # Canonical `price` = midpoint when both sides available, else last trade.
        if self.bid > 0 and self.ask > 0:
            self.price = (self.bid + self.ask) / 2.0
        elif self.last_trade_price > 0:
            self.price = self.last_trade_price

        if self.price > 0:
            self.received_at = recv_at
            self._price_history.append((recv_at, self.price))
            if len(self._price_history) > self._max_history:
                self._price_history = self._price_history[-self._max_history:]

    async def stop(self):
        self._running = False
