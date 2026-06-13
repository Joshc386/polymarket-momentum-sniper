"""Shared feature-snapshot builder.

Single source of truth for the feature vector recorded both per trade
(the enriched `trades` row) and per tick (the `signal_diag` row). Keeping
one builder prevents the two stores from drifting apart — drift between
two copies of the snapshot logic is what caused the May-2026 silent
logging bug.

Design notes (see docs/adr/0001-full-feature-snapshot-on-trade-record.md):
- NULL-not-zero: a layer value is emitted only when the layer genuinely
  computed on that tick (object present AND its data guard satisfied);
  otherwise None. `0.0` means "computed, genuinely neutral", never "absent".
- EV uses the Polymarket-doc fee `0.07*p*(1-p)` (taker fee, charged at entry
  on every trade), NOT the 2% winner-fee the live `best_ev`/gate still uses.
  This is a logging value only; it does not feed any entry/sizing decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Polymarket crypto-market taker fee rate, charged at entry on every trade.
FEE_RATE = 0.07


def fee_per_share_v2(price: float, rate: float, exponent: float) -> float:
    """CLOB V2 taker fee per contract: ``rate * (p*(1-p))**exponent``.

    This is the exact platform-fee-per-share implied by py-clob-client-v2's
    ``fees.adjust_buy_amount_for_fees`` (platform_fee = (amount/price) *
    rate * (p*(1-p))**exponent, and amount/price == shares for a buy). The
    legacy ``fee_per_share`` is this with rate=0.07, exponent=1.

    Used by the LIVE PnL resolvers with the market's live rate/exponent
    (fetched per window). The entry EV gate and paper PnL keep the validated
    ``fee_per_share`` (0.07) — see CLOB V2 migration ADR / decisions_BTC.
    """
    return rate * (price * (1.0 - price)) ** exponent


def fee_per_share(price: float) -> float:
    """Polymarket crypto taker fee per contract: FEE_RATE * p * (1 - p).

    Charged once at entry, on every trade regardless of outcome. Peaks at
    p = 0.50 (FEE_RATE * 0.25). Multiply by the number of contracts (shares)
    for the total position fee. The validated cost basis for the entry EV
    gate, EV logging, and paper PnL. Equivalent to fee_per_share_v2(price,
    FEE_RATE, 1.0).
    """
    return fee_per_share_v2(price, FEE_RATE, 1.0)


@dataclass(frozen=True)
class SnapshotInputs:
    """Raw state the bot saw at one instant, before NULL-vs-zero is applied.

    All fields default so tests and callers set only what they need; real
    callers populate every field from the strategy at the decision point.
    """

    # ── Market / decision context ──
    side: str = "YES"
    entry_price: float = 0.5
    est_prob_up: float = 0.5
    market_prob_up: float = 0.5
    btc_price: float = 0.0
    oracle_price: float = 0.0
    oracle_open_price: float = 0.0
    secs_remaining: float = 0.0
    regime: str = ""
    schedule_override: str = ""
    required_edge: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)  # full L1–L12

    # ── Raw layer values (as the combiner saw them; 0.0-coerced) ──
    l1_oracle_lag: float = 0.0
    l1_lag_component: float = 0.0
    l1_open_component: float = 0.0
    l2_momentum: float = 0.0
    l3_liquidation: float = 0.0
    l4_orderbook: float = 0.0
    l4_imbalance: float = 0.0
    l4_flow: float = 0.0
    l4_mid_dev: float = 0.0
    l4_top_pressure: float = 0.0
    l4_thickness: float = 0.0
    l5_sentiment: float = 0.0
    l6_fade: float = 0.0
    l7_taker_ratio: float = 0.0
    l8_clob_flow: float = 0.0
    l9b_absorption: float = 0.0
    l10_exhaustion: float = 0.0
    l11_trade_size: float = 0.0
    l12_wallet_flow: float = 0.0
    combined_signal: float = 0.0
    coinbase_direction: float = 0.0

    # ── Presence flags driving builder-side active detection ──
    has_orderbook: bool = False
    yes_bid_depth: float = 0.0
    has_coinalyze: bool = False
    has_clob_flow: bool = False
    has_wallet_monitor: bool = False
    absorption_on: bool = False
    exhaustion_on: bool = False
    trade_size_on: bool = False
    wallet_flow_on: bool = False


def _net_ev_per_share(side: str, entry_price: float, est_prob_up: float) -> float:
    """Fee-correct EV per share: (q - p) - FEE_RATE*p*(1-p).

    q is the model's win probability for the chosen side; p is the entry price.
    """
    q = est_prob_up if side == "YES" else (1.0 - est_prob_up)
    p = entry_price
    return (q - p) - fee_per_share(p)


def _gated(value: float, active: bool) -> float | None:
    """Return the value only if the layer genuinely computed (active); else
    None. This is the NULL-not-zero rule: absence is None, never 0.0."""
    return value if active else None


def build_feature_snapshot(inp: SnapshotInputs) -> dict[str, float | str | None]:
    """Build the flat feature dict recorded for a trade / tick."""
    snap: dict[str, float | str | None] = {}
    snap["net_ev_per_share"] = _net_ev_per_share(
        inp.side, inp.entry_price, inp.est_prob_up
    )
    snap["prob_edge"] = abs(inp.est_prob_up - inp.market_prob_up)

    # ── Market / decision context ──
    snap["side"] = inp.side
    snap["entry_price"] = inp.entry_price
    snap["est_prob_up"] = inp.est_prob_up
    snap["market_implied_prob"] = inp.market_prob_up
    snap["btc_price"] = inp.btc_price
    snap["oracle_price"] = inp.oracle_price
    snap["oracle_open_price"] = inp.oracle_open_price
    snap["secs_remaining"] = inp.secs_remaining
    snap["secs_into_window"] = 300.0 - inp.secs_remaining
    snap["regime"] = inp.regime
    snap["schedule_override"] = inp.schedule_override
    snap["required_edge"] = inp.required_edge

    # ── Always-present layers: computed every tick, so 0.0 is a real value ──
    snap["l1_oracle_lag"] = inp.l1_oracle_lag
    snap["l1_lag_component"] = inp.l1_lag_component
    snap["l1_open_component"] = inp.l1_open_component
    snap["l2_momentum"] = inp.l2_momentum
    snap["l3_liquidation"] = inp.l3_liquidation
    snap["l7_taker_ratio"] = inp.l7_taker_ratio
    snap["l8_clob_flow"] = inp.l8_clob_flow
    snap["combined_signal"] = inp.combined_signal
    snap["coinbase_direction"] = inp.coinbase_direction

    # ── Orderbook-gated layers (L4 family): need a live orderbook ──
    snap["l4_orderbook"] = _gated(inp.l4_orderbook, inp.has_orderbook)
    snap["l4_imbalance"] = _gated(inp.l4_imbalance, inp.has_orderbook)
    snap["l4_flow"] = _gated(inp.l4_flow, inp.has_orderbook)
    snap["l4_mid_dev"] = _gated(inp.l4_mid_dev, inp.has_orderbook)
    snap["l4_top_pressure"] = _gated(inp.l4_top_pressure, inp.has_orderbook)
    snap["l4_thickness"] = _gated(inp.l4_thickness, inp.has_orderbook)

    # ── L5 sentiment: needs a valid Coinalyze snapshot ──
    snap["l5_sentiment"] = _gated(inp.l5_sentiment, inp.has_coinalyze)

    # ── L6 fade: live orderbook AND yes_bid_depth > 0 (its data guard) ──
    snap["l6_fade"] = _gated(
        inp.l6_fade, inp.has_orderbook and inp.yes_bid_depth > 0.0
    )

    # ── Weight-gated layers: object only exists when weight > 0, AND each
    #    needs its own data feed present that tick ──
    snap["l9b_absorption"] = _gated(
        inp.l9b_absorption, inp.absorption_on and inp.has_clob_flow
    )
    snap["l10_exhaustion"] = _gated(
        inp.l10_exhaustion, inp.exhaustion_on and inp.has_orderbook
    )
    snap["l11_trade_size"] = _gated(
        inp.l11_trade_size, inp.trade_size_on and inp.has_clob_flow
    )
    snap["l12_wallet_flow"] = _gated(
        inp.l12_wallet_flow, inp.wallet_flow_on and inp.has_wallet_monitor
    )

    return snap
