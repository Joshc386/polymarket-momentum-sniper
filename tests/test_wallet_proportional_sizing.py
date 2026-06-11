"""Wallet-proportional position sizing (vault to-do 1b, agreed spec Jun 11).

Band: floor = max($1 exchange minimum, 1% of sizing wallet),
ceiling = 5% of sizing wallet, sizing wallet = min(bankroll, $200 cap).
Above a $200 wallet the band freezes at $2-$10. Default OFF — legacy
absolute $1/$5 clamps are untouched unless enabled in config.
"""

from core.execution import PaperExecutionEngine
from logging_db.database import Database
from strategy.sizing import PositionSizer

# A strong edge that pushes raw Kelly well above any ceiling we test:
# p=0.9 at price 0.5 -> kelly = 0.8, quarter-kelly = 0.2 -> 20% of bankroll.
STRONG = dict(est_prob=0.9, share_price=0.5)


def make_sizer(**overrides) -> PositionSizer:
    params = dict(
        kelly_multiplier=0.25,
        min_bet_usdc=1.0,
        max_bet_usdc=5.0,
        wallet_proportional=True,
    )
    params.update(overrides)
    return PositionSizer(**params)


def test_ceiling_scales_with_wallet() -> None:
    # 5% of a $150 wallet -> $7.50 cap (was $5 flat).
    assert make_sizer().compute(bankroll=150.0, **STRONG) == 7.5


def test_band_freezes_above_wallet_cap() -> None:
    # Above $200 the sizing wallet stays $200 -> ceiling stays $10.
    assert make_sizer().compute(bankroll=1000.0, **STRONG) == 10.0


# A real but tiny edge: p=0.51 at price 0.5 -> quarter-kelly = 0.5% of
# bankroll, always below the 1% floor -> bet clamps UP to the floor.
WEAK = dict(est_prob=0.51, share_price=0.5)


def test_floor_is_one_pct_of_wallet() -> None:
    assert make_sizer().compute(bankroll=150.0, **WEAK) == 1.5


def test_floor_freezes_at_two_dollars_above_cap() -> None:
    assert make_sizer().compute(bankroll=1000.0, **WEAK) == 2.0


def test_floor_never_below_exchange_minimum() -> None:
    # 1% of an $80 wallet is $0.80 -- below Polymarket's $1 minimum order.
    assert make_sizer().compute(bankroll=80.0, **WEAK) == 1.0


_TRADE = dict(
    side="YES", price=0.50, size_usdc=1.0, market_id="m", market_slug="s",
    oracle_lag_signal=0.0, momentum_signal=0.0, liquidation_signal=0.0,
    combined_signal=0.0, estimated_prob_up=0.60, market_implied_prob=0.50,
    edge=0.01, time_remaining_secs=188.0, btc_price=72000.0,
    oracle_price=0.0, oracle_open_price=0.0,
)


def _record_resolved_trade(db: Database, timestamp: str, pnl: float) -> None:
    """Insert a resolved paper trade at a controlled timestamp."""
    eng = PaperExecutionEngine(db, initial_bankroll=100.0)
    eng.execute_paper_trade(**_TRADE)
    db.conn.execute(
        "UPDATE trades SET timestamp = ?, pnl = ? "
        "WHERE id = (SELECT MAX(id) FROM trades)",
        (timestamp, pnl),
    )
    db.conn.commit()


def test_bankroll_epoch_seeds_from_post_epoch_pnl(tmp_path) -> None:
    # Wallet survives restarts: bankroll = initial + PnL since the epoch.
    # Pre-epoch history is ignored (fresh $100 epoch).
    db = Database(str(tmp_path / "t.db"))
    db.connect()
    _record_resolved_trade(db, "2026-06-10T09:00:00+00:00", pnl=50.0)   # pre-epoch
    _record_resolved_trade(db, "2026-06-11T13:00:00+00:00", pnl=8.25)   # post-epoch
    _record_resolved_trade(db, "2026-06-12T01:00:00+00:00", pnl=-3.00)  # post-epoch

    eng = PaperExecutionEngine(
        db, initial_bankroll=100.0,
        bankroll_epoch="2026-06-11T12:00:00+00:00",
    )
    eng.restore_from_db()
    assert eng.bankroll == 105.25


def test_no_epoch_keeps_legacy_fresh_start(tmp_path) -> None:
    db = Database(str(tmp_path / "t.db"))
    db.connect()
    _record_resolved_trade(db, "2026-06-10T09:00:00+00:00", pnl=50.0)

    eng = PaperExecutionEngine(db, initial_bankroll=100.0)
    eng.restore_from_db()
    assert eng.bankroll == 100.0


def test_legacy_mode_is_default_and_unchanged() -> None:
    # Other strategies construct PositionSizer without the new kwargs;
    # absolute $1/$5 clamps must behave exactly as before.
    legacy = PositionSizer(kelly_multiplier=0.25, min_bet_usdc=1.0, max_bet_usdc=5.0)
    assert legacy.compute(bankroll=150.0, **STRONG) == 5.0   # absolute cap
    assert legacy.compute(bankroll=1000.0, **WEAK) == 5.0    # kelly on full bankroll, capped
    assert legacy.compute(bankroll=150.0, **WEAK) == 1.0     # clamps up to $1 min


def test_strategy_wires_proportional_sizing_from_config(tmp_path) -> None:
    from strategies.contrarian_ev import ContrarianEvStrategy

    cfg = {
        "db_path": str(tmp_path / "t.db"),
        "sizing": {
            "wallet_proportional": True,
            "bankroll_epoch": "2026-06-11T12:00:00+00:00",
        },
    }
    s = ContrarianEvStrategy("t", cfg)
    assert s._sizer.wallet_proportional is True
    assert s._executor.bankroll_epoch == "2026-06-11T12:00:00+00:00"


def test_strategy_defaults_to_legacy_sizing(tmp_path) -> None:
    from strategies.contrarian_ev import ContrarianEvStrategy

    s = ContrarianEvStrategy("t", {"db_path": str(tmp_path / "t.db")})
    assert s._sizer.wallet_proportional is False
    assert s._executor.bankroll_epoch is None
