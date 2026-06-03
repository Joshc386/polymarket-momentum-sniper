"""Tests for the shared feature-snapshot builder (2026-06-01).

`build_feature_snapshot` is the single source of truth for the feature
vector recorded at trade entry (the `trades` row) and per tick (the
`signal_diag` row). It computes the fee-correct EV, records `prob_edge`
distinct from the legacy `edge` (=best_ev), and applies NULL-not-zero:
a layer value is recorded only when the layer genuinely computed this
tick, else None. See docs/adr/0001-full-feature-snapshot-on-trade-record.md.
"""

import pytest

from strategy.feature_snapshot import SnapshotInputs, build_feature_snapshot


class TestNetEvPerShare:
    """net_ev_per_share = (q - p) - 0.07*p*(1-p), Polymarket-doc fee."""

    def test_yes_trade_known_input(self) -> None:
        # YES: q = est_prob_up = 0.60, p = entry_price = 0.55
        # gross = 0.60 - 0.55 = 0.05
        # fee   = 0.07 * 0.55 * 0.45 = 0.017325
        # net   = 0.05 - 0.017325 = 0.032675
        snap = build_feature_snapshot(
            SnapshotInputs(side="YES", entry_price=0.55, est_prob_up=0.60)
        )
        assert snap["net_ev_per_share"] == pytest.approx(0.032675)

    def test_no_trade_uses_complement_win_prob(self) -> None:
        # NO: q = 1 - est_prob_up = 0.65, p = entry_price = 0.60
        # gross = 0.65 - 0.60 = 0.05
        # fee   = 0.07 * 0.60 * 0.40 = 0.016800
        # net   = 0.05 - 0.016800 = 0.033200
        snap = build_feature_snapshot(
            SnapshotInputs(side="NO", entry_price=0.60, est_prob_up=0.35)
        )
        assert snap["net_ev_per_share"] == pytest.approx(0.033200)


class TestProbEdge:
    """prob_edge = |est_prob_up - market_prob_up| — the gate metric, distinct
    from the legacy `edge` column (which stores best_ev) and from net EV."""

    def test_prob_edge_is_model_market_disagreement(self) -> None:
        snap = build_feature_snapshot(
            SnapshotInputs(est_prob_up=0.60, market_prob_up=0.52)
        )
        assert snap["prob_edge"] == pytest.approx(0.08)

    def test_prob_edge_distinct_from_net_ev(self) -> None:
        snap = build_feature_snapshot(
            SnapshotInputs(
                side="YES", entry_price=0.55, est_prob_up=0.60, market_prob_up=0.52
            )
        )
        assert snap["prob_edge"] != snap["net_ev_per_share"]


class TestNullNotZero:
    """A layer is recorded only when it genuinely computed this tick; else
    None. 0.0 means 'computed, neutral' and must never be confused with
    'absent'. Active detection lives in the builder, driven by presence flags.
    """

    def test_always_on_layer_records_genuine_zero(self) -> None:
        # L7 taker-ratio and L8 CLOB-flow compute every tick, unconditionally.
        # A 0.0 here is a real neutral reading, not absence.
        snap = build_feature_snapshot(
            SnapshotInputs(l7_taker_ratio=0.0, l8_clob_flow=0.0)
        )
        assert snap["l7_taker_ratio"] == 0.0
        assert snap["l7_taker_ratio"] is not None
        assert snap["l8_clob_flow"] == 0.0

    def test_orderbook_layers_null_when_no_orderbook(self) -> None:
        snap = build_feature_snapshot(
            SnapshotInputs(has_orderbook=False, l4_orderbook=0.0, l4_imbalance=0.0)
        )
        assert snap["l4_orderbook"] is None
        assert snap["l4_imbalance"] is None
        assert snap["l4_flow"] is None
        assert snap["l4_mid_dev"] is None
        assert snap["l4_top_pressure"] is None
        assert snap["l4_thickness"] is None

    def test_orderbook_layers_record_zero_when_present(self) -> None:
        # has_orderbook True + value 0.0 → recorded as 0.0, distinguishable
        # from the absent case above.
        snap = build_feature_snapshot(
            SnapshotInputs(has_orderbook=True, l4_orderbook=0.0, l4_imbalance=0.0)
        )
        assert snap["l4_orderbook"] == 0.0
        assert snap["l4_imbalance"] == 0.0

    def test_sentiment_gated_on_coinalyze(self) -> None:
        assert build_feature_snapshot(
            SnapshotInputs(has_coinalyze=False, l5_sentiment=0.0)
        )["l5_sentiment"] is None
        assert build_feature_snapshot(
            SnapshotInputs(has_coinalyze=True, l5_sentiment=0.0)
        )["l5_sentiment"] == 0.0

    def test_fade_needs_orderbook_and_bid_depth(self) -> None:
        # L6 fade: requires a live orderbook AND yes_bid_depth > 0 (its data guard)
        assert build_feature_snapshot(
            SnapshotInputs(has_orderbook=True, yes_bid_depth=0.0, l6_fade=0.0)
        )["l6_fade"] is None
        assert build_feature_snapshot(
            SnapshotInputs(has_orderbook=True, yes_bid_depth=5.0, l6_fade=0.0)
        )["l6_fade"] == 0.0

    def test_weight_gated_layers_off_are_null(self) -> None:
        # L9b/L10/L11/L12 only instantiate when their weight > 0.
        snap = build_feature_snapshot(SnapshotInputs())  # all *_on default False
        assert snap["l9b_absorption"] is None
        assert snap["l10_exhaustion"] is None
        assert snap["l11_trade_size"] is None
        assert snap["l12_wallet_flow"] is None

    def test_weight_gated_layers_need_their_data_guard(self) -> None:
        # On but data absent → still None (object present AND data guard).
        assert build_feature_snapshot(
            SnapshotInputs(trade_size_on=True, has_clob_flow=False)
        )["l11_trade_size"] is None
        assert build_feature_snapshot(
            SnapshotInputs(exhaustion_on=True, has_orderbook=False)
        )["l10_exhaustion"] is None
        assert build_feature_snapshot(
            SnapshotInputs(wallet_flow_on=True, has_wallet_monitor=False)
        )["l12_wallet_flow"] is None

    def test_weight_gated_layers_record_when_on_with_data(self) -> None:
        snap = build_feature_snapshot(
            SnapshotInputs(
                absorption_on=True, has_clob_flow=True, l9b_absorption=0.0,
                trade_size_on=True, l11_trade_size=0.0,
                exhaustion_on=True, has_orderbook=True, l10_exhaustion=0.0,
                wallet_flow_on=True, has_wallet_monitor=True, l12_wallet_flow=0.0,
            )
        )
        assert snap["l9b_absorption"] == 0.0
        assert snap["l10_exhaustion"] == 0.0
        assert snap["l11_trade_size"] == 0.0
        assert snap["l12_wallet_flow"] == 0.0


class TestContextFields:
    """Market/decision context recorded alongside the layers."""

    def test_secs_into_window_is_derived(self) -> None:
        # 5-minute window: secs_into_window = 300 - secs_remaining
        snap = build_feature_snapshot(SnapshotInputs(secs_remaining=188.0))
        assert snap["secs_into_window"] == pytest.approx(112.0)

    def test_btc_price_passes_through(self) -> None:
        # Regression for the May-2026 bug: the builder sources BTC price from
        # its input, never an out-of-scope variable.
        snap = build_feature_snapshot(SnapshotInputs(btc_price=72554.70))
        assert snap["btc_price"] == 72554.70

    def test_context_passthrough(self) -> None:
        snap = build_feature_snapshot(
            SnapshotInputs(
                regime="ranging", schedule_override="ranging_sched",
                required_edge=0.04, side="NO", entry_price=0.61,
                est_prob_up=0.42, market_prob_up=0.47,
                oracle_price=72500.0, oracle_open_price=72480.0,
            )
        )
        assert snap["regime"] == "ranging"
        assert snap["schedule_override"] == "ranging_sched"
        assert snap["required_edge"] == 0.04
        assert snap["side"] == "NO"
        assert snap["entry_price"] == 0.61
        assert snap["est_prob_up"] == 0.42
        assert snap["market_implied_prob"] == 0.47
        assert snap["oracle_price"] == 72500.0
        assert snap["oracle_open_price"] == 72480.0
