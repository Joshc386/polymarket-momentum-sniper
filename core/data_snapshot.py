"""Immutable per-tick snapshot of all shared data feeds.

Built once per tick by the multi-bot runner and passed to every bot.
Prevents bots from mutating shared state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DataSnapshot:
    """Frozen snapshot of all market data at one point in time.

    Each bot receives the same snapshot per tick. Because it's frozen,
    no bot can accidentally corrupt state that another bot reads.
    """

    # ── Timestamp ──
    timestamp: float = 0.0

    # ── Binance spot ──
    binance_price: float = 0.0
    binance_price_time: float = 0.0      # Exchange-side timestamp (T field)
    binance_received_at: float = 0.0     # Local time when WS msg arrived
    binance_candles: tuple = ()          # tuple of Candle (immutable copy)
    binance_current_candle: Any = None   # Candle | None

    # ── Aggregated BTC reference price (Coinbase + Kraken + Bitstamp) ──
    # Mean of healthy USD-native feeds. Drop-in replacement for binance_price
    # at the "BTC reference price" call sites (L1 oracle_lag, L3 liquidation
    # distance, risk stops, trade logging, TUI). Binance keeps its slot
    # above because L2 momentum + L7 taker + liquidations still need it.
    aggregated_price: float = 0.0
    aggregated_received_at: float = 0.0
    aggregated_n_feeds: int = 0          # How many of the 3 USD-native feeds are healthy

    # ── Chainlink Oracle ──
    oracle_price: float = 0.0
    oracle_window_open_price: float = 0.0
    oracle_updated_at: float = 0.0

    # ── Coinbase ──
    coinbase_price: float = 0.0
    coinbase_price_time: float = 0.0     # Exchange-side timestamp
    coinbase_received_at: float = 0.0    # Local time when WS msg arrived
    coinbase_direction: float = 0.0
    coinbase_connected: bool = False

    # ── CoinGlass liquidation levels ──
    coinglass_data: Any = None           # LiquidationData

    # ── Coinalyze cross-exchange ──
    coinalyze_snapshot: Any = None       # CoinalyzeSnapshot

    # ── Binance live liquidations ──
    binance_liq_stats: Any = None        # LiquidationStats

    # ── Polymarket orderbook ──
    orderbook: Any = None                # OrderbookSummary | None
    ob_yes_bid_delta: float = 0.0
    ob_yes_ask_delta: float = 0.0
    ob_no_bid_delta: float = 0.0
    ob_no_ask_delta: float = 0.0

    # ── Polymarket CLOB trade flow (from WebSocket) ──
    clob_trade_flow: Any = None          # CLOBTradeFlow | None

    # ── Market ──
    market: Any = None                   # MarketInfo | None
    market_slug: str = ""
    seconds_remaining: float = 0.0

    # ── Health ──
    feeds_healthy: bool = True
    health_issues: tuple = ()


def build_snapshot(
    binance,
    oracle,
    coinbase,
    coinglass,
    coinalyze,
    binance_liqs,
    orderbook_mgr,
    market,
    health,
    clob_trade_flow=None,
    price_aggregator=None,
) -> DataSnapshot:
    """Build an immutable snapshot from live feed objects.

    Args:
        binance: BinanceFeed instance.
        oracle: ChainlinkOracle instance.
        coinbase: CoinbaseFeed instance.
        coinglass: CoinGlassScraper instance.
        coinalyze: CoinalyzeFeed instance.
        binance_liqs: BinanceLiquidationFeed instance.
        orderbook_mgr: OrderbookManager instance.
        market: MarketInfo or None.
        health: HealthMonitor instance.

    Returns:
        Frozen DataSnapshot with all current feed state.
    """
    feeds_healthy, health_issues = health.check_health()

    return DataSnapshot(
        timestamp=time.time(),
        # Binance
        binance_price=binance.price,
        binance_price_time=getattr(binance, "price_time", 0.0),
        binance_received_at=getattr(binance, "received_at", 0.0),
        binance_candles=tuple(binance.candles),
        binance_current_candle=binance.current_candle,
        # Aggregated BTC reference price (3-feed USD-native mean)
        aggregated_price=(price_aggregator.price if price_aggregator else 0.0),
        aggregated_received_at=(
            price_aggregator.received_at if price_aggregator else 0.0
        ),
        aggregated_n_feeds=(
            price_aggregator.n_healthy_feeds if price_aggregator else 0
        ),
        # Oracle
        oracle_price=oracle.price,
        oracle_window_open_price=oracle.window_open_price,
        oracle_updated_at=getattr(oracle, "updated_at", 0.0),
        # Coinbase
        coinbase_price=coinbase.price,
        coinbase_price_time=getattr(coinbase, "price_time", 0.0),
        coinbase_received_at=getattr(coinbase, "received_at", 0.0),
        coinbase_direction=coinbase.price_direction if coinbase.is_connected else 0.0,
        coinbase_connected=coinbase.is_connected,
        # CoinGlass
        coinglass_data=coinglass.data,
        # Coinalyze
        coinalyze_snapshot=coinalyze.snapshot,
        # Binance liquidations
        binance_liq_stats=binance_liqs.stats,
        # Orderbook
        orderbook=orderbook_mgr.last_summary,
        ob_yes_bid_delta=orderbook_mgr.yes_bid_delta,
        ob_yes_ask_delta=orderbook_mgr.yes_ask_delta,
        ob_no_bid_delta=orderbook_mgr.no_bid_delta,
        ob_no_ask_delta=orderbook_mgr.no_ask_delta,
        # CLOB trade flow
        clob_trade_flow=clob_trade_flow,
        # Market
        market=market,
        market_slug=market.slug if market else "",
        seconds_remaining=market.seconds_remaining if market else 0.0,
        # Health
        feeds_healthy=feeds_healthy,
        health_issues=tuple(health_issues),
    )
