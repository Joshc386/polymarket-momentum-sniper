"""Regression tests for the 2026-06-10/11 resolution-integrity incident (J28).

Three defects under test:
1. Window-open fallback was dead: oracle.price was never fed after the May-21
   refactor, so when PolyBackTest stopped serving btc_price_start the open
   stayed 0.0 (TUI blank, L1 neutralised).
2. detect_resolution polled the Gamma API exactly once, then fell back to a
   price comparison — which with a zero open degenerates to always-UP and
   fabricated resolutions (verified vs Binance: ~20-30% of Jun-10/11
   resolutions wrong).
3. The fallback had no input validation at all.
"""
import asyncio
from types import SimpleNamespace

from data.polymarket_oracle import PolymarketOracle
from multi_runner import detect_resolution


# --------------------------------------------------------------------------- #
# 1. window-open fallback
# --------------------------------------------------------------------------- #
def test_oracle_open_stays_zero_without_seeded_price():
    """Documents the bug trigger: with oracle.price unfed, the exchange-average
    fallback cannot fire and the open stays 0 when PolyBackTest is dead."""
    async def run():
        o = PolymarketOracle()
        o._api_key = ""  # PolyBackTest unavailable
        o.start_window_open_fetch(expected_slug="btc-updown-5m-1")
        return o.window_open_price
    assert asyncio.run(run()) == 0.0


def test_oracle_open_set_from_seeded_aggregator_price():
    """The fix: seed oracle.price from the 3-feed aggregator at window start
    -> the open is set immediately, PolyBackTest or not."""
    async def run():
        o = PolymarketOracle()
        o._api_key = ""
        o.update_price(binance_price=61234.5)  # runner seeds from aggregator
        o.start_window_open_fetch(expected_slug="btc-updown-5m-1")
        return o.window_open_price, o.window_open_source
    price, source = asyncio.run(run())
    assert price == 61234.5
    assert source == "exchange_avg"


# --------------------------------------------------------------------------- #
# 2/3. detect_resolution hardening
# --------------------------------------------------------------------------- #
class FakeDiscovery:
    """fetch_resolution returns scripted values in order, repeats the last."""
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    async def fetch_resolution(self, slug):
        self.calls += 1
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


def _oracle(open_price: float) -> PolymarketOracle:
    o = PolymarketOracle()
    o.window_open_price = open_price
    return o


def test_detect_resolution_retries_until_api_settles():
    """Polymarket settles a few seconds after window end — retry instead of
    falling back after a single miss."""
    disc = FakeDiscovery("UNKNOWN", "UNKNOWN", "UP")
    res = asyncio.run(detect_resolution(
        disc, "slug", _oracle(61000.0),
        SimpleNamespace(price=60000.0), retry_delays=(0, 0, 0)))
    assert res == "UP"
    assert disc.calls == 3


def test_detect_resolution_never_fabricates_on_zero_open():
    """THE always-UP regression: API unknown + open==0 must return UNKNOWN,
    not 'UP'."""
    disc = FakeDiscovery("UNKNOWN")
    res = asyncio.run(detect_resolution(
        disc, "slug", _oracle(0.0),
        SimpleNamespace(price=61000.0), retry_delays=(0,)))
    assert res == "UNKNOWN"


def test_detect_resolution_unknown_when_aggregator_dead():
    disc = FakeDiscovery("UNKNOWN")
    res = asyncio.run(detect_resolution(
        disc, "slug", _oracle(61000.0),
        SimpleNamespace(price=0.0), retry_delays=(0,)))
    assert res == "UNKNOWN"


def test_detect_resolution_fallback_with_valid_inputs():
    disc = FakeDiscovery("UNKNOWN")
    up = asyncio.run(detect_resolution(
        disc, "slug", _oracle(61000.0),
        SimpleNamespace(price=61500.0), retry_delays=(0,)))
    dn = asyncio.run(detect_resolution(
        FakeDiscovery("UNKNOWN"), "slug", _oracle(61000.0),
        SimpleNamespace(price=60500.0), retry_delays=(0,)))
    assert (up, dn) == ("UP", "DOWN")


def test_detect_resolution_api_first_no_fallback_needed():
    disc = FakeDiscovery("DOWN")
    res = asyncio.run(detect_resolution(
        disc, "slug", _oracle(0.0),
        SimpleNamespace(price=0.0), retry_delays=(0, 0)))
    assert res == "DOWN"
    assert disc.calls == 1
