import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

CLOB_BOOK_URL = "https://clob.polymarket.com/book"


@dataclass
class OrderLevel:
    """A single price level in the orderbook."""
    price: float
    size: float


@dataclass
class OrderbookSummary:
    yes_best_bid: float
    yes_best_ask: float
    yes_midpoint: float
    no_best_bid: float
    no_best_ask: float
    no_midpoint: float
    yes_bid_depth: float  # Total size at top 5 bid levels
    yes_ask_depth: float  # Total size at top 5 ask levels
    spread: float         # Yes ask - yes bid
    last_trade_price_yes: float | None = None

    # Full multi-level depth data for L4 orderbook signal
    yes_bid_levels: list[OrderLevel] = field(default_factory=list)
    yes_ask_levels: list[OrderLevel] = field(default_factory=list)
    no_bid_levels: list[OrderLevel] = field(default_factory=list)
    no_ask_levels: list[OrderLevel] = field(default_factory=list)

    # Derived metrics for L4 signal
    yes_bid_total_size: float = 0.0   # Total size across ALL bid levels
    yes_ask_total_size: float = 0.0   # Total size across ALL ask levels
    no_bid_total_size: float = 0.0
    no_ask_total_size: float = 0.0
    yes_size_weighted_mid: float = 0.5  # Mid-price weighted by depth at each level
    num_yes_bid_levels: int = 0
    num_yes_ask_levels: int = 0

    @property
    def market_implied_prob_up(self) -> float:
        """Market-implied probability of UP from YES midpoint."""
        return self.yes_midpoint

    @property
    def bid_ask_imbalance(self) -> float:
        """Bid/ask volume imbalance for YES token.
        Positive = more bid volume (buyers), Negative = more ask volume (sellers).
        Range approximately [-1, 1].
        """
        total = self.yes_bid_total_size + self.yes_ask_total_size
        if total <= 0:
            return 0.0
        return (self.yes_bid_total_size - self.yes_ask_total_size) / total

    @property
    def depth_ratio(self) -> float:
        """Ratio of bid depth to ask depth. >1 = more bids, <1 = more asks."""
        if self.yes_ask_total_size <= 0:
            return 1.0
        return self.yes_bid_total_size / self.yes_ask_total_size


class OrderbookManager:
    """Fetches and parses Polymarket orderbook data via public REST API.

    Captures full multi-level depth for L4 orderbook signal analysis.
    """

    def __init__(self):
        self._http_client = httpx.Client(timeout=5.0)
        self.last_summary: OrderbookSummary | None = None
        # Track previous snapshots for order flow detection
        self._prev_yes_bid_total: float = 0.0
        self._prev_yes_ask_total: float = 0.0
        self._prev_no_bid_total: float = 0.0
        self._prev_no_ask_total: float = 0.0
        # Delta = change in depth between snapshots
        self.yes_bid_delta: float = 0.0
        self.yes_ask_delta: float = 0.0
        self.no_bid_delta: float = 0.0
        self.no_ask_delta: float = 0.0

    def close(self):
        self._http_client.close()

    def fetch(self, yes_token_id: str, no_token_id: str) -> OrderbookSummary | None:
        """Fetch orderbook for both YES and NO tokens, return summary."""
        yes_data = self._fetch_book(yes_token_id)
        no_data = self._fetch_book(no_token_id)

        if not yes_data and not no_data:
            logger.warning("Could not fetch any orderbook data")
            return None

        yes_result = self._parse_book_full(yes_data)
        no_result = self._parse_book_full(no_data)

        yes_best_bid = yes_result["best_bid"]
        yes_best_ask = yes_result["best_ask"]
        no_best_bid = no_result["best_bid"]
        no_best_ask = no_result["best_ask"]

        yes_mid = (yes_best_bid + yes_best_ask) / 2 if yes_best_bid and yes_best_ask else 0.5
        no_mid = (no_best_bid + no_best_ask) / 2 if no_best_bid and no_best_ask else 0.5
        spread = (yes_best_ask - yes_best_bid) if yes_best_ask and yes_best_bid else 0.0

        # Size-weighted mid: weight each price level by its size
        sw_mid = self._compute_size_weighted_mid(
            yes_result["bid_levels"], yes_result["ask_levels"], yes_mid
        )

        last_trade = None
        if yes_data:
            ltp = yes_data.get("last_trade_price")
            if ltp:
                last_trade = float(ltp)

        # Compute depth deltas (order flow)
        yes_bid_total = yes_result["total_bid_size"]
        yes_ask_total = yes_result["total_ask_size"]
        no_bid_total = no_result["total_bid_size"]
        no_ask_total = no_result["total_ask_size"]

        self.yes_bid_delta = yes_bid_total - self._prev_yes_bid_total
        self.yes_ask_delta = yes_ask_total - self._prev_yes_ask_total
        self.no_bid_delta = no_bid_total - self._prev_no_bid_total
        self.no_ask_delta = no_ask_total - self._prev_no_ask_total

        self._prev_yes_bid_total = yes_bid_total
        self._prev_yes_ask_total = yes_ask_total
        self._prev_no_bid_total = no_bid_total
        self._prev_no_ask_total = no_ask_total

        self.last_summary = OrderbookSummary(
            yes_best_bid=yes_best_bid or 0.0,
            yes_best_ask=yes_best_ask or 0.0,
            yes_midpoint=yes_mid,
            no_best_bid=no_best_bid or 0.0,
            no_best_ask=no_best_ask or 0.0,
            no_midpoint=no_mid,
            yes_bid_depth=yes_result["top5_bid_depth"],
            yes_ask_depth=yes_result["top5_ask_depth"],
            spread=spread,
            last_trade_price_yes=last_trade,
            # Full depth data
            yes_bid_levels=yes_result["bid_levels"],
            yes_ask_levels=yes_result["ask_levels"],
            no_bid_levels=no_result["bid_levels"],
            no_ask_levels=no_result["ask_levels"],
            yes_bid_total_size=yes_bid_total,
            yes_ask_total_size=yes_ask_total,
            no_bid_total_size=no_bid_total,
            no_ask_total_size=no_ask_total,
            yes_size_weighted_mid=sw_mid,
            num_yes_bid_levels=len(yes_result["bid_levels"]),
            num_yes_ask_levels=len(yes_result["ask_levels"]),
        )
        return self.last_summary

    def _fetch_book(self, token_id: str) -> dict | None:
        """Fetch raw orderbook from CLOB public API."""
        try:
            resp = self._http_client.get(CLOB_BOOK_URL, params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Failed to fetch book for token {token_id[:20]}: {e}")
            return None

    @staticmethod
    def _parse_book_full(ob: dict | None) -> dict:
        """Parse full orderbook into structured levels and summary stats."""
        result = {
            "best_bid": None,
            "best_ask": None,
            "bid_levels": [],
            "ask_levels": [],
            "top5_bid_depth": 0.0,
            "top5_ask_depth": 0.0,
            "total_bid_size": 0.0,
            "total_ask_size": 0.0,
        }

        if not ob:
            return result

        raw_bids = ob.get("bids", [])
        raw_asks = ob.get("asks", [])

        # Parse and sort bids (highest price first)
        if raw_bids:
            sorted_bids = sorted(
                raw_bids, key=lambda x: float(x.get("price", 0)), reverse=True
            )
            result["best_bid"] = float(sorted_bids[0]["price"])
            for b in sorted_bids:
                level = OrderLevel(
                    price=float(b.get("price", 0)),
                    size=float(b.get("size", 0)),
                )
                result["bid_levels"].append(level)
                result["total_bid_size"] += level.size
            result["top5_bid_depth"] = sum(
                l.size for l in result["bid_levels"][:5]
            )

        # Parse and sort asks (lowest price first)
        if raw_asks:
            sorted_asks = sorted(
                raw_asks, key=lambda x: float(x.get("price", 0))
            )
            result["best_ask"] = float(sorted_asks[0]["price"])
            for a in sorted_asks:
                level = OrderLevel(
                    price=float(a.get("price", 0)),
                    size=float(a.get("size", 0)),
                )
                result["ask_levels"].append(level)
                result["total_ask_size"] += level.size
            result["top5_ask_depth"] = sum(
                l.size for l in result["ask_levels"][:5]
            )

        return result

    @staticmethod
    def _compute_size_weighted_mid(
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        simple_mid: float,
    ) -> float:
        """Compute size-weighted midpoint using top levels.

        If bid side has 10x the volume of ask side, the 'true' fair value
        is closer to the ask price (buyers are aggressive).
        """
        if not bid_levels or not ask_levels:
            return simple_mid

        # Use top 3 levels on each side
        top_bids = bid_levels[:3]
        top_asks = ask_levels[:3]

        bid_vwap = sum(l.price * l.size for l in top_bids)
        bid_vol = sum(l.size for l in top_bids)
        ask_vwap = sum(l.price * l.size for l in top_asks)
        ask_vol = sum(l.size for l in top_asks)

        if bid_vol <= 0 or ask_vol <= 0:
            return simple_mid

        # Weight mid by relative volume
        # More bid volume → mid shifts toward ask (buyers pushing up)
        total_vol = bid_vol + ask_vol
        bid_weight = bid_vol / total_vol
        ask_weight = ask_vol / total_vol

        weighted_bid_price = bid_vwap / bid_vol
        weighted_ask_price = ask_vwap / ask_vol

        # Weighted mid: more weight to the side with less volume
        # (that side's price is more "informative" as it's the scarce side)
        return weighted_ask_price * bid_weight + weighted_bid_price * ask_weight

    @staticmethod
    def _parse_book(ob: dict | None) -> tuple[float | None, float | None, float, float]:
        """Legacy: Extract best bid, best ask, bid depth, ask depth."""
        if not ob:
            return None, None, 0.0, 0.0

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])

        best_bid = None
        bid_depth = 0.0
        if bids:
            sorted_bids = sorted(bids, key=lambda x: float(x.get("price", 0)), reverse=True)
            best_bid = float(sorted_bids[0]["price"])
            bid_depth = sum(float(b.get("size", 0)) for b in sorted_bids[:5])

        best_ask = None
        ask_depth = 0.0
        if asks:
            sorted_asks = sorted(asks, key=lambda x: float(x.get("price", 0)))
            best_ask = float(sorted_asks[0]["price"])
            ask_depth = sum(float(a.get("size", 0)) for a in sorted_asks[:5])

        return best_bid, best_ask, bid_depth, ask_depth
