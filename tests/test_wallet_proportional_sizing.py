"""Wallet-proportional position sizing (revised spec Jun 12).

- max risk/trade (ceiling) = 4% of sizing wallet = min(bankroll, $200);
  $4 @ $100, $8 @ $200, flat $8 above.
- min risk/trade (floor):
    * wallet < $200: the dynamic exchange minimum = max(5 x price, $1).
      Kelly results below it are BUMPED up to it and the trade is tagged.
    * wallet >= $200: flat $5.
- skip (size 0) when the exchange floor cannot fit under the ceiling
  (high price on a small wallet) — never breach the max-risk cap.
Default OFF — legacy absolute $1/$5 clamps untouched unless enabled.
"""

from core.execution import PaperExecutionEngine
from logging_db.database import Database
from strategy.sizing import PositionSizer

# A strong edge that pushes raw Kelly well above any ceiling we test:
# p=0.9 at price 0.5 -> kelly = 0.8, quarter-kelly = 0.2 -> 20% of bankroll.
STRONG = dict(est_prob=0.9, share_price=0.5)
# A real but tiny edge: p=0.51 at price 0.5 -> quarter-kelly = 0.5% of
# wallet, always below the floor -> bet bumps UP to the floor.
WEAK = dict(est_prob=0.51, share_price=0.5)


def make_sizer(**overrides) -> PositionSizer:
    params = dict(
        kelly_multiplier=0.25,
        min_bet_usdc=1.0,
        max_bet_usdc=5.0,
        wallet_proportional=True,
    )
    params.update(overrides)
    return PositionSizer(**params)


# ── Ceiling: 4% of the capped sizing wallet ──────────────────────────

def test_ceiling_is_four_pct_at_hundred() -> None:
    assert make_sizer().compute(bankroll=100.0, **STRONG) == 4.0


def test_ceiling_scales_to_eight_at_two_hundred() -> None:
    assert make_sizer().compute(bankroll=150.0, **STRONG) == 6.0   # 4% of 150
    assert make_sizer().compute(bankroll=200.0, **STRONG) == 8.0   # 4% of 200


def test_ceiling_freezes_at_eight_above_cap() -> None:
    assert make_sizer().compute(bankroll=1000.0, **STRONG) == 8.0


# ── Floor below the cap: the dynamic exchange minimum (5 x price) ─────

def test_floor_below_cap_is_five_shares_times_price() -> None:
    # Tiny edge (est_prob just over the price) so Kelly sizes below the floor
    # and the bet bumps up to the dynamic exchange minimum.
    s = make_sizer()
    assert s.compute(bankroll=100.0, est_prob=0.31, share_price=0.30) == 1.5  # 5*0.30
    assert s.compute(bankroll=100.0, est_prob=0.51, share_price=0.50) == 2.5  # 5*0.50


def test_floor_below_cap_does_not_scale_with_wallet() -> None:
    # Price-driven, not wallet-driven, anywhere below $200.
    s = make_sizer()
    assert s.compute(bankroll=120.0, est_prob=0.51, share_price=0.50) == 2.5
    assert s.compute(bankroll=190.0, est_prob=0.51, share_price=0.50) == 2.5


def test_floor_backstopped_at_one_dollar_for_cheap_shares() -> None:
    # 5 * 0.10 = $0.50 -> $1 exchange-notional backstop (tiny edge -> bump).
    assert make_sizer().compute(bankroll=100.0, est_prob=0.11, share_price=0.10) == 1.0


# ── Floor at/above the cap: flat $5 ──────────────────────────────────

def test_floor_is_flat_five_at_and_above_cap() -> None:
    s = make_sizer()
    assert s.compute(bankroll=200.0, **WEAK) == 5.0
    assert s.compute(bankroll=1000.0, **WEAK) == 5.0


# ── Skip when the exchange floor exceeds the ceiling ─────────────────

def test_skip_when_five_shares_exceeds_the_cap() -> None:
    # $100 wallet, price 0.85: floor = 5*0.85 = $4.25 > $4 ceiling -> skip
    # (independent of edge — the band can't fit a compliant order).
    assert make_sizer().compute(bankroll=100.0, est_prob=0.90, share_price=0.85) == 0.0
    # price 0.80 fits exactly (5 shares = $4.00 = ceiling).
    assert make_sizer().compute(bankroll=100.0, est_prob=0.81, share_price=0.80) == 4.0


# ── decide(): bump / skip flags for trade tagging ────────────────────

def test_decide_flags_bump_cap_and_skip() -> None:
    s = make_sizer()
    bumped = s.decide(bankroll=100.0, est_prob=0.51, share_price=0.50)
    assert bumped.bumped is True and bumped.skipped is False and bumped.size == 2.5
    capped = s.decide(bankroll=100.0, **STRONG)
    assert capped.bumped is False and capped.size == 4.0   # clamped down, not bumped
    skipped = s.decide(bankroll=100.0, est_prob=0.51, share_price=0.85)
    assert skipped.skipped is True and skipped.size == 0.0


def test_compute_matches_decide_size() -> None:
    s = make_sizer()
    assert s.compute(bankroll=100.0, **WEAK) == s.decide(bankroll=100.0, **WEAK).size


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
    assert legacy.wallet_proportional is False


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
