"""Polymarket resolution oracle — exchange-aggregated live price + PolyBackTest window open.

Live BTC price: average of Binance and Coinbase WebSocket feeds, updated
every tick (sub-second). Chainlink Data Streams aggregates from multiple
exchanges including these two, so the average closely approximates the
actual resolution price (~$10 accuracy).

Window open price: fetched from PolyBackTest Pro API's btc_price_start
field, which captures the exact Chainlink Data Streams price Polymarket
uses for resolution. Set once per window via fetch_and_set_window_open().

Resolution: read from Polymarket's Gamma API (authoritative on-chain
result), not computed from prices.
"""

import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# ── PolyBackTest API (window open prices) ──
PBT_API_BASE = "https://api.polybacktest.com"

_API_KEY = os.environ.get("POLYBACKTEST_API_KEY", "")
if not _API_KEY:
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("POLYBACKTEST_API_KEY="):
                    _API_KEY = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break


@dataclass
class PolymarketOracle:
    """Exchange-aggregated oracle + PolyBackTest window open price.

    Live price (.price): Average of Binance and Coinbase prices, fed
    each tick via update_price(). When only one exchange is available,
    uses that single price. Approximates Chainlink Data Streams within
    ~$10.

    Window open (.window_open_price): Fetched from PolyBackTest Pro's
    btc_price_start field — the exact Chainlink Data Streams price
    Polymarket uses for resolution.

    Interface:
      .price, .window_open_price, .updated_at, .last_fetch_time
      .update_price(), .start(), .stop(), .fetch_once(),
      .set_window_open_price(), .fetch_and_set_window_open()
    """

    price: float = 0.0
    updated_at: float = 0.0
    last_fetch_time: float = 0.0
    window_open_price: float = 0.0
    window_open_source: str = ""  # "polybacktest" or "exchange_avg"
    _api_key: str = field(default_factory=lambda: _API_KEY)
    _http_client: httpx.AsyncClient | None = None
    _running: bool = False
    _coin: str = "btc"
    _bg_fetch_task: object = field(default=None, repr=False)

    def update_price(
        self,
        binance_price: float = 0.0,
        coinbase_price: float = 0.0,
    ) -> None:
        """Update the oracle price from exchange feeds.

        Computes a 50/50 average when both exchanges are available.
        Falls back to whichever single feed is connected.

        Args:
            binance_price: Current Binance BTC price (0 if unavailable).
            coinbase_price: Current Coinbase BTC price (0 if unavailable).
        """
        if binance_price > 0 and coinbase_price > 0:
            self.price = (binance_price + coinbase_price) / 2.0
        elif binance_price > 0:
            self.price = binance_price
        elif coinbase_price > 0:
            self.price = coinbase_price
        else:
            return  # No data — don't update

        self.updated_at = time.time()
        self.last_fetch_time = time.time()

    async def start(self, poll_interval: float = 1.0) -> None:
        """Keepalive coroutine for backwards compatibility.

        The oracle price is now fed each tick via update_price(),
        so this just keeps the task alive.
        """
        import asyncio
        self._running = True
        logger.info("Oracle started — price from Binance/Coinbase average")
        while self._running:
            await asyncio.sleep(60)

    async def stop(self) -> None:
        """Stop the oracle."""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def fetch_once(self) -> float:
        """Return current price. For backwards compatibility."""
        return self.price

    def set_window_open_price(self) -> None:
        """Snapshot current price as the window opening price.

        Called at window transitions by the main loop.
        """
        if self.price > 0:
            self.window_open_price = self.price
            logger.info(f"Window open price set: ${self.price:,.2f}")

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    def start_window_open_fetch(self, expected_slug: str = "") -> None:
        """Set exchange average immediately, then fetch exact price in background.

        Called at window transitions. Sets the exchange-average price as a
        temporary value so the bot can start immediately, then spawns a
        background task that polls PolyBackTest until it gets the exact
        Chainlink Data Streams price. The background task updates
        window_open_price in-place when it succeeds.

        Args:
            expected_slug: Slug of the current market window.
        """
        import asyncio

        # Immediate: set exchange average as temporary value
        if self.price > 0:
            self.window_open_price = self.price
            self.window_open_source = "exchange_avg"
            logger.info(
                f"Window open (temporary): ${self.price:,.2f} "
                f"(exchange avg, polling PolyBackTest...)"
            )

        # Cancel any prior background fetch
        if self._bg_fetch_task and not self._bg_fetch_task.done():
            self._bg_fetch_task.cancel()

        # Spawn background poll for exact price
        self._bg_fetch_task = asyncio.create_task(
            self._poll_polybacktest(expected_slug)
        )

    async def _poll_polybacktest(self, expected_slug: str) -> None:
        """Background task: poll PolyBackTest until exact price is available.

        Tries up to 10 times over ~60 seconds. Once the exact price is
        found, updates window_open_price in-place. If all attempts fail,
        the exchange-average remains as the fallback.

        Args:
            expected_slug: Slug of the current market window.
        """
        import asyncio

        # Delays between attempts: 0, 2, 3, 5, 5, 8, 8, 10, 10, 10 = ~61s total
        delays = [0, 2, 3, 5, 5, 8, 8, 10, 10, 10]
        for attempt, delay in enumerate(delays):
            if delay > 0:
                await asyncio.sleep(delay)

            try:
                price = await self._fetch_window_open_once(expected_slug)
                if price > 0:
                    self.window_open_price = price
                    self.window_open_source = "polybacktest"
                    logger.info(
                        f"Window open (exact): ${price:,.2f} "
                        f"from PolyBackTest ({expected_slug}) "
                        f"[attempt {attempt + 1}]"
                    )
                    return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"PolyBackTest poll attempt {attempt + 1} error: {e}")

        logger.warning(
            f"PolyBackTest: exact price unavailable after {len(delays)} attempts "
            f"for {expected_slug}, using exchange avg: "
            f"${self.window_open_price:,.2f}"
        )

    async def fetch_and_set_window_open(
        self, expected_slug: str = "",
    ) -> float:
        """Fetch the accurate window open price from PolyBackTest.

        DEPRECATED: Use start_window_open_fetch() instead for non-blocking
        behaviour. This method is kept for backwards compatibility but now
        just does a single attempt and returns.

        Args:
            expected_slug: Slug of the current market window.

        Returns:
            The window open price, or 0.0 if unavailable.
        """
        price = await self._fetch_window_open_once(expected_slug)
        if price > 0:
            return price
        return 0.0

    async def _fetch_window_open_once(self, expected_slug: str = "") -> float:
        """Single attempt to fetch window open price from PolyBackTest.

        Args:
            expected_slug: If set, only accept markets matching this slug.

        Returns:
            The window open price, or 0.0 if unavailable/wrong market.
        """
        if not self._api_key:
            return 0.0

        client = self._get_client()
        headers = {"X-API-Key": self._api_key}

        try:
            resp = await client.get(
                f"{PBT_API_BASE}/v2/markets",
                headers=headers,
                params={
                    "coin": self._coin,
                    "market_type": "5m",
                    "limit": 2,
                },
            )
            if resp.status_code != 200:
                logger.warning(
                    f"PolyBackTest /v2/markets returned {resp.status_code}"
                )
                return 0.0

            markets = resp.json().get("markets", [])
            if not markets:
                return 0.0

            # If we know the expected slug, find the matching market
            if expected_slug:
                for m in markets:
                    if m.get("slug") == expected_slug:
                        btc_start = m.get("btc_price_start")
                        if btc_start is not None:
                            self.window_open_price = float(btc_start)
                            logger.info(
                                f"Window open from PolyBackTest "
                                f"({expected_slug}): "
                                f"${self.window_open_price:,.2f}"
                            )
                            return self.window_open_price

                # Expected market not found — PolyBackTest hasn't indexed
                # this window yet. Return 0.0 so the retry logic can try
                # again, or the exchange-average fallback will handle it.
                slugs_found = [m.get("slug", "?") for m in markets]
                logger.info(
                    f"PolyBackTest: {expected_slug} not found "
                    f"(have: {slugs_found}), will retry"
                )
                return 0.0

            # No expected slug — take whatever the latest market gives us
            btc_start = markets[0].get("btc_price_start")
            if btc_start is not None:
                self.window_open_price = float(btc_start)
                logger.info(
                    f"Window open from PolyBackTest (latest): "
                    f"${self.window_open_price:,.2f}"
                )
                return self.window_open_price

            # No btc_price_start on latest market — return 0.0
            # so the exchange-average fallback handles it
            logger.info("PolyBackTest: latest market has no btc_price_start")

        except Exception as e:
            logger.warning(f"Failed to fetch window open from PolyBackTest: {e}")

        return 0.0
