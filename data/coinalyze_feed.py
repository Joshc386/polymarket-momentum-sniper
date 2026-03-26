"""Coinalyze API feed — cross-exchange derivatives sentiment data.

Fetches aggregated open interest, long/short ratios, and funding rates
across major exchanges (Binance, Bybit, OKX, BitMEX, etc.) for BTC futures.

Data contract (CoinalyzeSnapshot):
    - open_interest_usd: Total OI across exchanges in USD
    - oi_change_pct: OI % change over lookback period
    - long_ratio: % of traders long (0-1)
    - short_ratio: % of traders short (0-1)
    - funding_rate: Current avg funding rate across exchanges
    - predicted_funding_rate: Next predicted funding rate
    - timestamp: When the snapshot was built

Rate limit: 40 calls/minute. We batch symbols to minimize calls.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.coinalyze.net/v1"

# BTC perpetual symbols across major exchanges
# Format: {PAIR}_{CONTRACT_TYPE}.{EXCHANGE_CODE}
# Exchange codes: A=Binance, 6=Bybit, K=OKX, 0=BitMEX, 4=Deribit
BTC_PERP_SYMBOLS = [
    "BTCUSDT_PERP.A",   # Binance USDT-M
    "BTCUSD_PERP.A",    # Binance COIN-M
    "BTCUSDC_PERP.A",   # Binance USDC-M
    "BTCUSDT.6",         # Bybit USDT
    "BTCUSD.6",          # Bybit Inverse
    "BTCUSDT_PERP.0",   # BitMEX USDT
    "BTCUSD_PERP.4",    # Deribit
]

# Symbols with long/short ratio data (verified)
BTC_LSR_SYMBOLS = [
    "BTCUSDT_PERP.A",   # Binance USDT-M
    "BTCUSD_PERP.A",    # Binance COIN-M
    "BTCUSDC_PERP.A",   # Binance USDC-M
    "BTCUSDT.6",         # Bybit
    "BTCUSD.6",          # Bybit Inverse
]


@dataclass
class CoinalyzeSnapshot:
    """Cross-exchange derivatives sentiment snapshot."""
    # Open interest
    open_interest_usd: float = 0.0
    oi_change_pct: float = 0.0           # % change from previous fetch
    oi_per_exchange: dict = field(default_factory=dict)  # {symbol: oi_usd}

    # Long/short ratio (aggregated across exchanges)
    long_ratio: float = 0.5              # 0-1, fraction of longs
    short_ratio: float = 0.5             # 0-1, fraction of shorts
    ls_ratio: float = 1.0                # long/short ratio (>1 = more longs)
    lsr_per_exchange: dict = field(default_factory=dict)  # {symbol: ratio}

    # Funding rates
    funding_rate: float = 0.0            # Current avg funding rate (%)
    predicted_funding_rate: float = 0.0  # Predicted next funding rate (%)
    funding_per_exchange: dict = field(default_factory=dict)  # {symbol: rate}

    timestamp: float = 0.0
    is_valid: bool = False

    @property
    def is_overleveraged_long(self) -> bool:
        """True if market is heavily long-biased (squeeze risk)."""
        return self.long_ratio > 0.60 and self.funding_rate > 0.01

    @property
    def is_overleveraged_short(self) -> bool:
        """True if market is heavily short-biased (squeeze risk)."""
        return self.short_ratio > 0.60 and self.funding_rate < -0.01

    @property
    def oi_expanding(self) -> bool:
        """True if OI is growing (new money entering)."""
        return self.oi_change_pct > 1.0

    @property
    def oi_contracting(self) -> bool:
        """True if OI is shrinking (positions closing)."""
        return self.oi_change_pct < -1.0


class CoinalyzeFeed:
    """Async Coinalyze API client for cross-exchange BTC derivatives data.

    Usage:
        feed = CoinalyzeFeed()
        await feed.start()  # background loop
        snap = feed.snapshot  # always-current CoinalyzeSnapshot

    Or manual:
        snap = await feed.fetch_once()
    """

    def __init__(self, refresh_interval: int = 60, api_key: str = ""):
        self.refresh_interval = refresh_interval
        self.api_key = api_key or os.getenv("COINALYZE_API_KEY", "")
        self.snapshot = CoinalyzeSnapshot()
        self._prev_oi: float = 0.0
        self._running = False
        self.consecutive_failures: int = 0
        self.last_fetch_time: float = 0.0
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers={"api_key": self.api_key},
        )

    async def start(self):
        """Start background refresh loop."""
        if not self.api_key:
            logger.warning("Coinalyze: No API key set — feed disabled")
            return

        self._running = True
        logger.info("Coinalyze feed starting...")

        while self._running:
            try:
                await self.fetch_once()
                if self.snapshot.is_valid:
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"Coinalyze fetch error ({self.consecutive_failures}): {e}")

            # Backoff on failures, cap at 5 minutes
            wait = min(
                self.refresh_interval * (1.5 ** min(self.consecutive_failures, 5)),
                300,
            )
            await asyncio.sleep(wait)

    async def stop(self):
        self._running = False

    async def close(self):
        self._running = False
        await self._client.aclose()

    async def fetch_once(self) -> CoinalyzeSnapshot:
        """Fetch all data points and build a snapshot."""
        if not self.api_key:
            return self.snapshot

        snap = CoinalyzeSnapshot()
        symbols_str = ",".join(BTC_PERP_SYMBOLS)
        lsr_symbols_str = ",".join(BTC_LSR_SYMBOLS)

        # Fetch OI, funding, and L/S ratio in parallel
        oi_task = self._fetch_open_interest(symbols_str)
        funding_task = self._fetch_funding_rate(symbols_str)
        pred_funding_task = self._fetch_predicted_funding(symbols_str)
        lsr_task = self._fetch_long_short_ratio(lsr_symbols_str)

        oi_data, funding_data, pred_funding_data, lsr_data = await asyncio.gather(
            oi_task, funding_task, pred_funding_task, lsr_task,
            return_exceptions=True,
        )

        # Process OI
        if isinstance(oi_data, list) and oi_data:
            total_oi = 0.0
            for item in oi_data:
                sym = item.get("symbol", "")
                val = float(item.get("value", 0))
                snap.oi_per_exchange[sym] = val
                total_oi += val
            snap.open_interest_usd = total_oi

            if self._prev_oi > 0:
                snap.oi_change_pct = ((total_oi - self._prev_oi) / self._prev_oi) * 100
            self._prev_oi = total_oi

        # Process funding rates
        if isinstance(funding_data, list) and funding_data:
            rates = []
            for item in funding_data:
                sym = item.get("symbol", "")
                val = float(item.get("value", 0))
                snap.funding_per_exchange[sym] = val
                rates.append(val)
            if rates:
                snap.funding_rate = sum(rates) / len(rates)

        # Process predicted funding
        if isinstance(pred_funding_data, list) and pred_funding_data:
            rates = [float(item.get("value", 0)) for item in pred_funding_data]
            if rates:
                snap.predicted_funding_rate = sum(rates) / len(rates)

        # Process long/short ratios
        if isinstance(lsr_data, list) and lsr_data:
            long_pcts = []
            short_pcts = []
            ratios = []
            for item in lsr_data:
                sym = item.get("symbol", "")
                # Current endpoint returns {symbol, value, update}
                # where value is the ratio
                ratio = float(item.get("value", 1.0))
                snap.lsr_per_exchange[sym] = ratio
                ratios.append(ratio)
                # Derive long/short % from ratio: ratio = longs/shorts
                # long% = ratio / (1 + ratio), short% = 1 / (1 + ratio)
                if ratio > 0:
                    long_pcts.append(ratio / (1 + ratio))
                    short_pcts.append(1.0 / (1 + ratio))

            if long_pcts:
                snap.long_ratio = sum(long_pcts) / len(long_pcts)
                snap.short_ratio = sum(short_pcts) / len(short_pcts)
                snap.ls_ratio = sum(ratios) / len(ratios)

        # Mark valid if we got at least OI or L/S data
        has_oi = snap.open_interest_usd > 0
        has_lsr = len(snap.lsr_per_exchange) > 0
        snap.is_valid = has_oi or has_lsr
        snap.timestamp = time.time()

        if snap.is_valid:
            self.snapshot = snap
            self.last_fetch_time = time.time()
            logger.debug(
                f"Coinalyze: OI=${snap.open_interest_usd:,.0f} "
                f"L/S={snap.ls_ratio:.2f} "
                f"FR={snap.funding_rate:.4f}%"
            )

        return snap

    async def _fetch_open_interest(self, symbols: str) -> list:
        """GET /open-interest — current OI per symbol."""
        try:
            resp = await self._client.get(
                f"{BASE_URL}/open-interest",
                params={"symbols": symbols, "convert_to_usd": "true"},
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"Coinalyze rate limited, retry after {retry_after}s")
                await asyncio.sleep(retry_after)
                return []
            if resp.status_code != 200:
                logger.debug(f"Coinalyze OI: HTTP {resp.status_code}")
                return []
            return resp.json()
        except Exception as e:
            logger.debug(f"Coinalyze OI fetch failed: {e}")
            return []

    async def _fetch_funding_rate(self, symbols: str) -> list:
        """GET /funding-rate — current funding rate per symbol."""
        try:
            resp = await self._client.get(
                f"{BASE_URL}/funding-rate",
                params={"symbols": symbols},
            )
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception as e:
            logger.debug(f"Coinalyze funding fetch failed: {e}")
            return []

    async def _fetch_predicted_funding(self, symbols: str) -> list:
        """GET /predicted-funding-rate — predicted next funding rate."""
        try:
            resp = await self._client.get(
                f"{BASE_URL}/predicted-funding-rate",
                params={"symbols": symbols},
            )
            if resp.status_code != 200:
                return []
            return resp.json()
        except Exception as e:
            logger.debug(f"Coinalyze predicted funding fetch failed: {e}")
            return []

    async def _fetch_long_short_ratio(self, symbols: str) -> list:
        """GET /long-short-ratio-history — most recent L/S ratio.

        Uses 5min interval with a short window to get the latest value.
        """
        try:
            now = int(time.time())
            resp = await self._client.get(
                f"{BASE_URL}/long-short-ratio-history",
                params={
                    "symbols": symbols,
                    "interval": "5min",
                    "from": str(now - 600),  # Last 10 minutes
                    "to": str(now),
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            # Extract most recent ratio for each symbol
            result = []
            for item in data:
                sym = item.get("symbol", "")
                history = item.get("history", [])
                if history:
                    latest = history[-1]  # Most recent
                    result.append({
                        "symbol": sym,
                        "value": latest.get("r", 1.0),
                        "long_pct": latest.get("l", 50.0),
                        "short_pct": latest.get("s", 50.0),
                    })
            return result
        except Exception as e:
            logger.debug(f"Coinalyze L/S ratio fetch failed: {e}")
            return []

    @property
    def status_str(self) -> str:
        """Short status string for dashboard display."""
        if not self.api_key:
            return "(no API key)"
        if not self.snapshot.is_valid:
            if self.consecutive_failures > 0:
                return f"(offline x{self.consecutive_failures})"
            return "(loading...)"
        age = int(time.time() - self.last_fetch_time)
        s = self.snapshot
        return (
            f"OI:${s.open_interest_usd/1e9:.1f}B "
            f"L/S:{s.ls_ratio:.2f} "
            f"FR:{s.funding_rate:+.4f}% "
            f"[{age}s]"
        )
