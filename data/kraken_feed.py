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
    """Real-time BTC/USD price from Kraken Exchange (WebSocket v2).

    Subscribes with `event_trigger: bbo` so updates fire on every change
    in the best bid/offer, not just on completed trades. Kraken BTC/USD
    trades infrequently relative to high-volume venues, so the default
    trade-only trigger produces slow (5-10s) updates that lag far behind
    Binance/Coinbase. The bbo trigger pushes updates many times per second.

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
                    # Subscribe to BTC/USD ticker with bbo event trigger.
                    # Default is "trades" which only updates on completed
                    # trades — BTC/USD trades infrequently on Kraken so
                    # this produces slow (5-10s) updates. bbo updates on
                    # every best-bid-offer change for sub-second freshness.
                    sub_msg = json.dumps({
                        "method": "subscribe",
                        "params": {
                            "channel": "ticker",
                            "symbol": ["BTC/USD"],
                            "event_trigger": "bbo",
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

        any_update = False
        for tick in ticks:
            if tick.get("symbol") != "BTC/USD":
                continue
            any_update = True

            last = tick.get("last")
            if last is not None:
                try:
                    self.last_trade_price = float(last)
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

        if not any_update:
            return

        # Canonical `price` = midpoint when both sides available, else last trade.
        # Midpoint is more responsive: updates every bbo change rather than
        # only on completed trades.
        if self.bid > 0 and self.ask > 0:
            self.price = (self.bid + self.ask) / 2.0
        elif self.last_trade_price > 0:
            self.price = self.last_trade_price

        if self.price > 0:
            self.price_time = recv_at  # Kraken ticker doesn't include trade ts
            self.received_at = recv_at
            self._price_history.append((recv_at, self.price))
            if len(self._price_history) > self._max_history:
                self._price_history = self._price_history[-self._max_history:]

    async def stop(self):
        self._running = False
