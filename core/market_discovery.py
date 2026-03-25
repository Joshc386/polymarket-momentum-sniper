import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class MarketState(Enum):
    UPCOMING = "upcoming"
    ACTIVE = "active"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    slug: str
    yes_token_id: str   # "Up" token
    no_token_id: str    # "Down" token
    start_time: float   # Unix timestamp
    end_time: float     # Unix timestamp
    state: MarketState
    outcomes: list[str]
    market_id: str = ""
    outcome_prices: list[str] | None = None

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def seconds_remaining(self) -> float:
        return self.time_remaining

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now <= self.end_time


class MarketDiscovery:
    """Discovers active 5-min BTC Up/Down markets on Polymarket.

    Markets follow the slug pattern: btc-updown-5m-{unix_timestamp}
    where the timestamp aligns to 5-minute boundaries.
    """

    def __init__(self):
        self.current_market: MarketInfo | None = None
        self.next_market: MarketInfo | None = None
        self._http_client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        await self._http_client.aclose()

    async def find_active_market(self) -> MarketInfo | None:
        """Find the currently active 5-min BTC Up/Down market.

        Uses the predictable slug pattern to minimise API calls.
        Only fetches 1 slug at a time — the current 5-min boundary.
        Falls back to adjacent windows only if the primary fails.
        """
        now = int(time.time())
        window_start = now - (now % 300)

        # Try current window first (1 API call)
        market = await self._fetch_by_slug(window_start)
        if market and market.is_active:
            self.current_market = market
            return market

        # Current window might have just expired — try next (1 more call)
        market = await self._fetch_by_slug(window_start + 300)
        if market and market.is_active:
            self.current_market = market
            return market

        # Fallback: slight misalignment (1 more call max)
        market = await self._fetch_by_slug(window_start - 300)
        if market and market.is_active:
            self.current_market = market
            return market

        logger.debug("No active 5-min BTC market found in nearby windows")
        return None

    async def _fetch_by_slug(self, timestamp: int) -> MarketInfo | None:
        """Fetch a specific 5-min market by its timestamp slug."""
        slug = f"btc-updown-5m-{timestamp}"
        try:
            resp = await self._http_client.get(
                f"{GAMMA_API_BASE}/events",
                params={"slug": slug},
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return None

            event = data[0] if isinstance(data, list) else data
            markets = event.get("markets", [])
            if not markets:
                return None

            m = markets[0]
            return self._parse_market(m, slug)

        except Exception as e:
            logger.debug(f"Failed to fetch market {slug}: {e}")
            return None

    def _parse_market(self, m: dict, event_slug: str = "") -> MarketInfo | None:
        """Parse a market from the Gamma API events response."""
        try:
            clob_ids = m.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                import json
                clob_ids = json.loads(clob_ids)

            if len(clob_ids) < 2:
                return None

            # Outcomes are ["Up", "Down"] — first token is Up (YES), second is Down (NO)
            yes_token = clob_ids[0]
            no_token = clob_ids[1]

            start_time = self._parse_time(m.get("startDate", ""))
            end_time = self._parse_time(m.get("endDate", ""))

            if not start_time or not end_time:
                return None

            now = time.time()
            if now < start_time:
                state = MarketState.UPCOMING
            elif now <= end_time:
                state = MarketState.ACTIVE
            else:
                state = MarketState.RESOLVED

            outcome_prices = m.get("outcomePrices", None)
            if isinstance(outcome_prices, str):
                import json
                outcome_prices = json.loads(outcome_prices)

            return MarketInfo(
                condition_id=m.get("conditionId", ""),
                question=m.get("question", ""),
                slug=event_slug or m.get("slug", ""),
                yes_token_id=yes_token,
                no_token_id=no_token,
                start_time=start_time,
                end_time=end_time,
                state=state,
                outcomes=m.get("outcomes", ["Up", "Down"]),
                market_id=m.get("id", ""),
                outcome_prices=outcome_prices,
            )
        except Exception as e:
            logger.debug(f"Failed to parse market: {e}")
            return None

    @staticmethod
    def _parse_time(time_str: str) -> float | None:
        if not time_str:
            return None
        try:
            if time_str.endswith("Z"):
                time_str = time_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(time_str)
            return dt.timestamp()
        except Exception:
            try:
                return float(time_str)
            except Exception:
                return None

    async def refresh(self) -> MarketInfo | None:
        """Re-discover the active market. Call periodically."""
        # If current market expired, promote next to current
        if self.current_market and self.current_market.time_remaining <= 0:
            if self.next_market and self.next_market.is_active:
                self.current_market = self.next_market
                self.next_market = None
                return self.current_market

        return await self.find_active_market()
