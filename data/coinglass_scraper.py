"""CoinGlass liquidation heatmap scraper.

Built as a modular component with a clean interface so swapping the scraper
for an official API is a one-line change later.

Data contract (LiquidationData):
    - long_levels:  list of (price, estimated_volume_usd)  — long liquidation clusters
    - short_levels: list of (price, estimated_volume_usd)  — short liquidation clusters
    - timestamp: when the data was fetched

Fallback: If scraping fails, returns empty LiquidationData. Callers should
treat empty data as neutral (signal = 0). Bot continues on Layers 1+2 only.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# CoinGlass public liquidation endpoints (no auth required)
# Primary: liquidation heatmap aggregated data
COINGLASS_LIQUIDATION_URL = "https://fapi.coinglass.com/api/futures/liquidation/info"
COINGLASS_LIQMAP_URL = "https://fapi.coinglass.com/api/futures/liquidation/chart"
# Fallback: aggregated liq data
COINGLASS_LIQHEAT_URL = "https://fapi.coinglass.com/api/futures/liquidation/aggregated"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.coinglass.com/",
    "Origin": "https://www.coinglass.com",
}


@dataclass
class LiquidationLevel:
    """A liquidation cluster at a specific price level."""
    price: float
    volume_usd: float       # Estimated USD volume that would be liquidated
    side: str                # "long" or "short"


@dataclass
class LiquidationData:
    """Aggregated liquidation data for BTC."""
    long_levels: list[LiquidationLevel] = field(default_factory=list)
    short_levels: list[LiquidationLevel] = field(default_factory=list)
    timestamp: float = 0.0
    is_valid: bool = False

    @property
    def nearest_long_liq(self) -> LiquidationLevel | None:
        """Nearest long liquidation cluster (below current price)."""
        return self.long_levels[0] if self.long_levels else None

    @property
    def nearest_short_liq(self) -> LiquidationLevel | None:
        """Nearest short liquidation cluster (above current price)."""
        return self.short_levels[0] if self.short_levels else None

    @property
    def total_long_volume(self) -> float:
        return sum(l.volume_usd for l in self.long_levels)

    @property
    def total_short_volume(self) -> float:
        return sum(l.volume_usd for l in self.short_levels)


class CoinGlassScraper:
    """Scrapes BTC liquidation heatmap data from CoinGlass.

    Modular interface — swap this class for CoinGlassAPI later with
    identical LiquidationData output contract.

    Usage:
        scraper = CoinGlassScraper()
        await scraper.start(refresh_interval=60)  # background loop
        data = scraper.data  # always-current LiquidationData

    Or manual:
        data = await scraper.fetch_once(current_price=84000)
    """

    def __init__(self, refresh_interval: int = 60):
        self.refresh_interval = refresh_interval
        self.data = LiquidationData()
        self.last_fetch_time: float = 0.0
        self.consecutive_failures: int = 0
        self._running = False
        self._current_price: float = 0.0
        self._client = httpx.AsyncClient(timeout=15.0, headers=HEADERS)

    async def start(self, current_price_getter=None):
        """Start background refresh loop.

        Args:
            current_price_getter: callable returning current BTC price (float).
        """
        self._running = True
        self._price_getter = current_price_getter
        while self._running:
            try:
                price = self._price_getter() if self._price_getter else self._current_price
                if price > 0:
                    data = await self.fetch_once(price)
                    if data.is_valid:
                        self.consecutive_failures = 0
                    else:
                        self.consecutive_failures += 1
                else:
                    logger.debug("CoinGlass: no price available yet, skipping fetch")
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"CoinGlass fetch error ({self.consecutive_failures}): {e}")

            # Exponential backoff on failures, cap at 5 minutes
            wait = min(self.refresh_interval * (1.5 ** min(self.consecutive_failures, 5)), 300)
            await asyncio.sleep(wait)

    async def stop(self):
        self._running = False

    def set_current_price(self, price: float):
        """Update the current BTC price for distance calculations."""
        self._current_price = price

    async def fetch_once(self, current_price: float = 0.0) -> LiquidationData:
        """Fetch liquidation data once. Returns LiquidationData (possibly empty)."""
        if current_price > 0:
            self._current_price = current_price

        # Try multiple endpoints in order of preference
        data = await self._try_liquidation_chart()
        if not data.is_valid:
            data = await self._try_liquidation_info()
        if not data.is_valid:
            data = await self._try_liquidation_aggregated()

        if data.is_valid:
            data.timestamp = time.time()
            self.last_fetch_time = time.time()
            # Sort levels by proximity to current price
            if self._current_price > 0:
                data.long_levels.sort(key=lambda l: abs(l.price - self._current_price))
                data.short_levels.sort(key=lambda l: abs(l.price - self._current_price))
            self.data = data
            logger.debug(
                f"CoinGlass: {len(data.long_levels)} long levels, "
                f"{len(data.short_levels)} short levels"
            )
        else:
            logger.debug("CoinGlass: all endpoints failed, returning empty data")

        return data

    async def _try_liquidation_chart(self) -> LiquidationData:
        """Try the liquidation chart/heatmap endpoint."""
        try:
            resp = await self._client.get(
                COINGLASS_LIQMAP_URL,
                params={"symbol": "BTC", "timeType": 2},  # timeType 2 = 24h
            )
            if resp.status_code != 200:
                return LiquidationData()

            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                return LiquidationData()

            return self._parse_chart_data(body["data"])
        except Exception as e:
            logger.debug(f"CoinGlass chart endpoint failed: {e}")
            return LiquidationData()

    async def _try_liquidation_info(self) -> LiquidationData:
        """Try the liquidation info endpoint."""
        try:
            resp = await self._client.get(
                COINGLASS_LIQUIDATION_URL,
                params={"symbol": "BTC"},
            )
            if resp.status_code != 200:
                return LiquidationData()

            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                return LiquidationData()

            return self._parse_info_data(body["data"])
        except Exception as e:
            logger.debug(f"CoinGlass info endpoint failed: {e}")
            return LiquidationData()

    async def _try_liquidation_aggregated(self) -> LiquidationData:
        """Try the aggregated liquidation endpoint."""
        try:
            resp = await self._client.get(
                COINGLASS_LIQHEAT_URL,
                params={"symbol": "BTC", "timeType": 2},
            )
            if resp.status_code != 200:
                return LiquidationData()

            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                return LiquidationData()

            return self._parse_aggregated_data(body["data"])
        except Exception as e:
            logger.debug(f"CoinGlass aggregated endpoint failed: {e}")
            return LiquidationData()

    def _parse_chart_data(self, data) -> LiquidationData:
        """Parse liquidation chart/heatmap response into LiquidationData."""
        result = LiquidationData()
        try:
            # Chart data typically has price levels with long/short liq volumes
            # Format varies — try common structures
            if isinstance(data, dict):
                prices = data.get("priceArray", data.get("prices", []))
                longs = data.get("longLiqArray", data.get("longs", []))
                shorts = data.get("shortLiqArray", data.get("shorts", []))

                if prices and (longs or shorts):
                    for i, price in enumerate(prices):
                        p = float(price)
                        if i < len(longs) and float(longs[i]) > 0:
                            result.long_levels.append(LiquidationLevel(
                                price=p, volume_usd=float(longs[i]), side="long"
                            ))
                        if i < len(shorts) and float(shorts[i]) > 0:
                            result.short_levels.append(LiquidationLevel(
                                price=p, volume_usd=float(shorts[i]), side="short"
                            ))

            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    price = float(item.get("price", item.get("p", 0)))
                    long_vol = float(item.get("longLiqUsd", item.get("longVol", 0)))
                    short_vol = float(item.get("shortLiqUsd", item.get("shortVol", 0)))
                    if price > 0:
                        if long_vol > 0:
                            result.long_levels.append(LiquidationLevel(
                                price=price, volume_usd=long_vol, side="long"
                            ))
                        if short_vol > 0:
                            result.short_levels.append(LiquidationLevel(
                                price=price, volume_usd=short_vol, side="short"
                            ))

            if result.long_levels or result.short_levels:
                result.is_valid = True
                # Filter to significant levels only (> $1M)
                result.long_levels = [l for l in result.long_levels if l.volume_usd >= 1_000_000]
                result.short_levels = [l for l in result.short_levels if l.volume_usd >= 1_000_000]

        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to parse chart data: {e}")

        return result

    def _parse_info_data(self, data) -> LiquidationData:
        """Parse liquidation info response."""
        result = LiquidationData()
        try:
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    price = float(item.get("price", 0))
                    vol = float(item.get("volUsd", item.get("vol", 0)))
                    side = item.get("side", "").lower()
                    if price > 0 and vol >= 1_000_000:
                        level = LiquidationLevel(price=price, volume_usd=vol, side=side)
                        if side == "long":
                            result.long_levels.append(level)
                        elif side == "short":
                            result.short_levels.append(level)

            if result.long_levels or result.short_levels:
                result.is_valid = True

        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Failed to parse info data: {e}")

        return result

    def _parse_aggregated_data(self, data) -> LiquidationData:
        """Parse aggregated liquidation response."""
        # Same structure attempt as chart — CoinGlass APIs share patterns
        return self._parse_chart_data(data)

    async def close(self):
        await self._client.aclose()
