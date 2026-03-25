import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import websockets

logger = logging.getLogger(__name__)

TRADE_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@trade"
KLINE_STREAM = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


@dataclass
class BinanceFeed:
    """Real-time BTC/USDT price and 1-min candle data from Binance."""

    price: float = 0.0
    price_time: float = 0.0
    candles: deque = field(default_factory=lambda: deque(maxlen=30))
    current_candle: Candle | None = None
    _running: bool = False

    async def start(self):
        self._running = True
        await asyncio.gather(
            self._run_trade_stream(),
            self._run_kline_stream(),
        )

    async def stop(self):
        self._running = False

    async def _run_trade_stream(self):
        while self._running:
            try:
                async with websockets.connect(TRADE_STREAM, ping_interval=20) as ws:
                    logger.info("Binance trade stream connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        self.price = float(data["p"])
                        self.price_time = data["T"] / 1000.0
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(f"Binance trade stream disconnected: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Binance trade stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _run_kline_stream(self):
        while self._running:
            try:
                async with websockets.connect(KLINE_STREAM, ping_interval=20) as ws:
                    logger.info("Binance kline stream connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        k = data["k"]
                        candle = Candle(
                            open_time=k["t"],
                            open=float(k["o"]),
                            high=float(k["h"]),
                            low=float(k["l"]),
                            close=float(k["c"]),
                            volume=float(k["v"]),
                            is_closed=k["x"],
                        )
                        self.current_candle = candle
                        if candle.is_closed:
                            self.candles.append(candle)
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                if not self._running:
                    break
                logger.warning(f"Binance kline stream disconnected: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2)
            except Exception as e:
                if not self._running:
                    break
                logger.error(f"Binance kline stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
