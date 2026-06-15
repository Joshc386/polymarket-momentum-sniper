"""Regression: the entry path must forward the market's token IDs to the executor.

The live engine's ``execute_trade`` needs ``yes_token_id``/``no_token_id`` to
know which CTF token to buy; without them it logs "No token ID for side ..."
and places no order. Paper tolerated the omission (it simulates fills and never
touches a token id), so the gap only surfaced when Bot K2 went live
(2026-06-15): every live entry failed before placement. This pins the wiring so
``_execute_trade`` always hands the chosen market's token ids to the executor.
"""

from types import SimpleNamespace

from strategies.contrarian_ev import ContrarianEvStrategy, EntryDecision


def _snapshot(market) -> SimpleNamespace:
    # Mirrors tests/test_contrarian_snapshot_wiring.py (a shape the real
    # _collect_snapshot_inputs / build_feature_snapshot accept), plus the market.
    return SimpleNamespace(
        coinalyze_snapshot=None,
        clob_trade_flow=None,
        oracle_price=72500.0,
        oracle_window_open_price=72480.0,
        coinbase_price=72510.0,
        binance_candles=[],
        aggregated_price=72554.70,
        binance_price=72554.70,
        coinbase_direction=0.0,
        market=market,
    )


def _ob() -> SimpleNamespace:
    return SimpleNamespace(
        yes_bid_depth=5.0,
        yes_ask_depth=5.0,
        yes_best_bid=0.50,
        yes_best_ask=0.52,
        no_best_bid=0.48,
        no_best_ask=0.50,
        market_implied_prob_up=0.51,
        spread=0.02,
    )


class _CapturingExecutor:
    """Paper-style executor double that records the execute_trade kwargs."""

    is_paper = True
    bankroll = 100.0
    total_trades = 0

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def execute_trade(self, **kwargs):
        self.kwargs = kwargs
        return None  # falsy -> _execute_trade skips the paper book-keeping block


def _drive_entry(tmp_path, side: str) -> dict:
    """Run _execute_trade for a decided entry; return the executor's kwargs."""
    s = ContrarianEvStrategy("t", {"db_path": str(tmp_path / "t.db")})
    try:
        ex = _CapturingExecutor()
        s._executor = ex
        # Isolate the sizer (its own tests cover sizing) — force a positive size.
        s._sizer = SimpleNamespace(
            decide=lambda **kw: SimpleNamespace(size=2.0, bumped=False, raw=2.0)
        )
        s._risk_state = SimpleNamespace(size_multiplier=1.0)
        s._est_prob_up = 0.55
        s._entry_decision = EntryDecision(
            should_enter=True, side=side, price=0.50,
            best_ev=0.02, prob_edge=0.03, required_edge=0.01, reason="",
        )
        market = SimpleNamespace(
            condition_id="cond1", slug="btc-updown",
            yes_token_id="YES_TOK_123", no_token_id="NO_TOK_456",
        )
        s._execute_trade(_snapshot(market), _ob(), 200.0)
        return ex.kwargs
    finally:
        s._db.close()


def test_entry_forwards_both_token_ids_to_executor(tmp_path) -> None:
    kwargs = _drive_entry(tmp_path, side="YES")
    assert kwargs.get("yes_token_id") == "YES_TOK_123"
    assert kwargs.get("no_token_id") == "NO_TOK_456"


def test_chosen_side_has_a_nonempty_token_id(tmp_path) -> None:
    # The live engine selects token_id = yes_token_id if side=="YES" else no.
    # Both sides must arrive non-empty so neither hits the "No token ID" guard.
    yes_kwargs = _drive_entry(tmp_path, side="YES")
    assert yes_kwargs.get("yes_token_id")  # truthy for a YES entry
    no_kwargs = _drive_entry(tmp_path, side="NO")
    assert no_kwargs.get("no_token_id")    # truthy for a NO entry
