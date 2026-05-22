"""Tests for PriceAggregator — drop-in replacement for binance.price as the
BTC reference price, computed as the mean of healthy USD-native feeds
(Coinbase, Kraken, Bitstamp).

Health rule: a feed is healthy when feed.price > 0 AND
(now - feed.received_at) < staleness_threshold (default 10s, matches the
existing is_connected convention across feed classes).

Fallback rule: 3 healthy -> mean of 3; 2 healthy -> mean of 2;
1 healthy -> that one; 0 healthy -> price=0.0 (existing "no price"
sentinel that the strategy already handles by skipping the tick).
"""

import time

import pytest

from data.price_aggregator import PriceAggregator


class FakeFeed:
    """Mimics the shape of BitstampFeed / KrakenFeed / CoinbaseFeed
    (.price, .received_at) so the aggregator can be unit-tested without
    real websocket connections."""

    def __init__(self, price: float = 0.0, received_at: float = 0.0):
        self.price = price
        self.received_at = received_at


@pytest.fixture
def now() -> float:
    return time.time()


class TestThreeHealthy:
    def test_mean_of_three(self, now: float) -> None:
        cb = FakeFeed(price=80_000.0, received_at=now)
        kr = FakeFeed(price=80_010.0, received_at=now)
        bs = FakeFeed(price=80_020.0, received_at=now)
        agg = PriceAggregator(cb, kr, bs)
        assert agg.price == pytest.approx(80_010.0)
        assert agg.n_healthy_feeds == 3

    def test_received_at_is_freshest(self, now: float) -> None:
        cb = FakeFeed(price=80_000.0, received_at=now - 2.0)
        kr = FakeFeed(price=80_010.0, received_at=now - 1.0)
        bs = FakeFeed(price=80_020.0, received_at=now)  # freshest
        agg = PriceAggregator(cb, kr, bs)
        # received_at should reflect the most recent healthy update
        assert agg.received_at == pytest.approx(now)


class TestTwoHealthy:
    def test_one_stale_excluded(self, now: float) -> None:
        cb = FakeFeed(price=80_000.0, received_at=now)
        kr = FakeFeed(price=80_010.0, received_at=now)
        bs = FakeFeed(price=80_020.0, received_at=now - 60.0)  # stale
        agg = PriceAggregator(cb, kr, bs)
        assert agg.price == pytest.approx(80_005.0)
        assert agg.n_healthy_feeds == 2

    def test_one_zero_price_excluded(self, now: float) -> None:
        cb = FakeFeed(price=80_000.0, received_at=now)
        kr = FakeFeed(price=0.0, received_at=now)         # never connected
        bs = FakeFeed(price=80_020.0, received_at=now)
        agg = PriceAggregator(cb, kr, bs)
        assert agg.price == pytest.approx(80_010.0)
        assert agg.n_healthy_feeds == 2


class TestOneHealthy:
    def test_single_survivor(self, now: float) -> None:
        cb = FakeFeed(price=0.0, received_at=now)
        kr = FakeFeed(price=80_010.0, received_at=now)    # only survivor
        bs = FakeFeed(price=80_020.0, received_at=now - 60.0)
        agg = PriceAggregator(cb, kr, bs)
        assert agg.price == pytest.approx(80_010.0)
        assert agg.n_healthy_feeds == 1


class TestZeroHealthy:
    def test_all_unhealthy_returns_zero(self, now: float) -> None:
        cb = FakeFeed(price=0.0, received_at=0.0)
        kr = FakeFeed(price=80_010.0, received_at=now - 60.0)
        bs = FakeFeed(price=0.0, received_at=now)
        agg = PriceAggregator(cb, kr, bs)
        assert agg.price == 0.0
        assert agg.n_healthy_feeds == 0
        # received_at must be 0 when nothing is healthy (existing convention
        # for "no data" — strategy treats price<=0 as no-active-market)
        assert agg.received_at == 0.0


class TestStalenessThreshold:
    def test_custom_threshold(self, now: float) -> None:
        # Custom threshold of 2 seconds — a 3-second-old feed is now stale
        cb = FakeFeed(price=80_000.0, received_at=now - 3.0)
        kr = FakeFeed(price=80_010.0, received_at=now)
        bs = FakeFeed(price=80_020.0, received_at=now)
        agg = PriceAggregator(cb, kr, bs, staleness_threshold=2.0)
        assert agg.n_healthy_feeds == 2
        assert agg.price == pytest.approx(80_015.0)


class TestDropInCompatibility:
    """Aggregator must expose the same interface as BitstampFeed/etc so
    multi_runner and DataSnapshot don't have to special-case it."""

    def test_has_price_attribute(self, now: float) -> None:
        agg = PriceAggregator(FakeFeed(80_000.0, now),
                              FakeFeed(80_010.0, now),
                              FakeFeed(80_020.0, now))
        assert hasattr(agg, "price")
        assert isinstance(agg.price, float)

    def test_has_received_at_attribute(self, now: float) -> None:
        agg = PriceAggregator(FakeFeed(80_000.0, now),
                              FakeFeed(80_010.0, now),
                              FakeFeed(80_020.0, now))
        assert hasattr(agg, "received_at")
        assert isinstance(agg.received_at, float)
