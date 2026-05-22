"""3-exchange BTC/USD price aggregator (Coinbase + Kraken + Bitstamp).

Drop-in replacement for `binance.price` as the BTC reference price feeding
into L1 oracle_lag, L3 liquidation distance, risk stops, trade logging,
and the TUI display. All three constituent feeds are USD-native (no
stablecoin basis distortion).

Why aggregate: Binance.com offers only BTC/USDT (not BTC/USD), so its
price is ~$5-15 above the USD reference Polymarket settles on
(Chainlink BTC/USD). The price_compare monitor confirmed that a
USD-native mean tracks Chainlink within ~$5-15. Using it as L1's
exchange_price input cleans the chronic phantom-lag signal that was
contributing to NO-side miscalibration (see decisions_BTC J13).

Binance keeps streaming — it's still required for L2 momentum (OHLCV
candles), L7 taker_ratio (microstructure), and liquidations. This
class only replaces "the BTC reference price."

Aggregation rule: mean of healthy feeds. Health = price > 0 AND
received_at within `staleness_threshold` seconds. Fallback: 3 healthy
-> mean; 2 healthy -> mean; 1 healthy -> that one; 0 healthy ->
price=0.0 (existing "no price" sentinel — strategy already handles
this by skipping the tick).
"""

import time
from dataclasses import dataclass


# Match the existing is_connected staleness threshold across feed classes
DEFAULT_STALENESS_THRESHOLD = 10.0


@dataclass
class PriceAggregator:
    """Mean-of-healthy-feeds BTC reference price.

    Exposes `.price` and `.received_at` to match the shape of the
    individual feed classes — call sites don't need to special-case it.
    """

    coinbase: object       # CoinbaseFeed (has .price, .received_at)
    kraken: object         # KrakenFeed
    bitstamp: object       # BitstampFeed
    staleness_threshold: float = DEFAULT_STALENESS_THRESHOLD

    def _healthy(self) -> list[tuple[float, float]]:
        """Return [(price, received_at), ...] for feeds that are
        currently healthy (price > 0 AND not stale)."""
        now = time.time()
        out: list[tuple[float, float]] = []
        for feed in (self.coinbase, self.kraken, self.bitstamp):
            p = getattr(feed, "price", 0.0)
            r = getattr(feed, "received_at", 0.0)
            if p > 0 and r > 0 and (now - r) < self.staleness_threshold:
                out.append((p, r))
        return out

    @property
    def price(self) -> float:
        """Mean of healthy feeds, or 0.0 if none healthy."""
        h = self._healthy()
        if not h:
            return 0.0
        return sum(p for p, _ in h) / len(h)

    @property
    def received_at(self) -> float:
        """Most recent received_at among healthy feeds, or 0.0 if none."""
        h = self._healthy()
        if not h:
            return 0.0
        return max(r for _, r in h)

    @property
    def n_healthy_feeds(self) -> int:
        """Diagnostic — how many feeds contributed to the current price."""
        return len(self._healthy())

    @property
    def is_connected(self) -> bool:
        """Drop-in for is_connected on individual feeds — true when at
        least one feed is healthy."""
        return self.n_healthy_feeds > 0
