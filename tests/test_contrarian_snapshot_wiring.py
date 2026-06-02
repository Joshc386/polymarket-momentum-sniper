"""Wiring tests: the strategy feeds the shared feature-snapshot builder.

These pin (a) the May-2026 bug fix — BTC price reaches the snapshot via an
explicit argument, never an out-of-scope variable — and (b) that NULL-not-zero
holds end-to-end when real feeds are absent. See ADR-0001.
"""

from types import SimpleNamespace

from strategies.contrarian_ev import ContrarianEvStrategy
from strategy.feature_snapshot import build_feature_snapshot


def _strategy(tmp_path) -> ContrarianEvStrategy:
    return ContrarianEvStrategy("t", {"db_path": str(tmp_path / "t.db")})


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        coinalyze_snapshot=None,          # → L5 inactive
        clob_trade_flow=None,             # → L8/L11 data absent
        oracle_price=72500.0,
        oracle_window_open_price=72480.0,
        coinbase_price=72510.0,
        binance_candles=[],
        aggregated_price=72554.70,
        binance_price=72554.70,
    )


def _ob() -> SimpleNamespace:
    return SimpleNamespace(
        yes_bid_depth=5.0,
        yes_best_bid=0.50,
        yes_best_ask=0.52,
        market_implied_prob_up=0.51,
    )


class TestSnapshotWiring:
    def test_btc_price_flows_from_argument(self, tmp_path) -> None:
        s = _strategy(tmp_path)
        try:
            inp = s._collect_snapshot_inputs(_snapshot(), _ob(), 200.0, "", 72554.70)
            snap = build_feature_snapshot(inp)
            assert snap["btc_price"] == 72554.70
            assert snap["secs_into_window"] == 100.0
        finally:
            s._db.close()

    def test_inactive_layers_null_end_to_end(self, tmp_path) -> None:
        s = _strategy(tmp_path)
        try:
            inp = s._collect_snapshot_inputs(_snapshot(), _ob(), 200.0, "", 72554.70)
            snap = build_feature_snapshot(inp)
            assert snap["l5_sentiment"] is None      # no coinalyze feed
            assert snap["l4_imbalance"] is not None   # orderbook present
        finally:
            s._db.close()

    def test_diag_logging_exception_never_propagates(self, tmp_path) -> None:
        # Money-safety contract: best-effort logging must never crash the
        # trading loop, even if the diag store raises.
        s = _strategy(tmp_path)

        class _BoomLogger:
            def log(self, _tick) -> None:
                raise RuntimeError("disk full")

        s._signal_diag = _BoomLogger()
        try:
            # Must not raise despite the logger blowing up.
            s._log_signal_diagnostic(_snapshot(), _ob(), 200.0, "", 72554.70)
        finally:
            s._db.close()
