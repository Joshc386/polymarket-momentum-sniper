"""Flat min-size mode (ADR-0005 / decisions_BTC J38).

The K2 live-probe sizing mode: every entry sizes to the 5-share exchange
minimum (max(min_order_shares x price, $1)), BYPASSING the 4% ceiling and the
J37 floor>ceiling skip. It affects SIZING ONLY — every entry/EV gate, the
bankroll guard, and streak HALTS are preserved. Default OFF.
"""

from strategy.sizing import PositionSizer

# Strong edge: quarter-Kelly well above any floor/ceiling we test.
STRONG = dict(est_prob=0.9, share_price=0.5)


def make_sizer(**overrides) -> PositionSizer:
    params = dict(kelly_multiplier=0.25, flat_min_size=True)
    params.update(overrides)
    return PositionSizer(**params)


def test_min_size_trades_where_wallet_proportional_would_skip() -> None:
    # price 0.85, $100 wallet: wallet-proportional SKIPS (floor 5*0.85=$4.25 >
    # $4 ceiling) — the J37 starvation path. Min-size TRADES at the floor.
    assert make_sizer().compute(bankroll=100.0, est_prob=0.90, share_price=0.85) == 4.25


def test_min_size_ignores_ceiling_even_on_a_strong_edge() -> None:
    # A STRONG edge would size to 20% of bankroll (legacy) or the 4% ceiling
    # ($4) under wallet-proportional. Min-size is always the floor: 5*0.50=$2.50.
    assert make_sizer().compute(bankroll=100.0, **STRONG) == 2.5


def test_min_size_still_skips_a_negative_ev_trade() -> None:
    # The -EV gate (kelly_fraction <= 0) lives above the min-size branch.
    # est_prob below the price -> no edge -> skip, never floored to min.
    d = make_sizer().decide(bankroll=100.0, est_prob=0.40, share_price=0.50)
    assert d.skipped is True and d.size == 0.0


def test_min_size_skips_invalid_price_or_prob() -> None:
    s = make_sizer()
    assert s.decide(bankroll=100.0, est_prob=0.9, share_price=1.0).skipped is True
    assert s.decide(bankroll=100.0, est_prob=0.9, share_price=0.0).skipped is True
    assert s.decide(bankroll=100.0, est_prob=0.0, share_price=0.5).skipped is True


def test_min_size_one_dollar_backstop_for_cheap_shares() -> None:
    # 5 * 0.10 = $0.50 -> $1 exchange-notional backstop.
    assert make_sizer().compute(bankroll=100.0, est_prob=0.20, share_price=0.10) == 1.0


def test_min_size_skips_when_floor_exceeds_bankroll() -> None:
    # Can't place an unaffordable order: floor 5*0.50=$2.50 > $2 wallet -> skip.
    d = make_sizer().decide(bankroll=2.0, est_prob=0.90, share_price=0.50)
    assert d.skipped is True and d.size == 0.0


def test_min_size_honours_streak_halt() -> None:
    # A streak HALT (size_multiplier -> 0) must SKIP, not floor-to-min. Streak
    # size-reduction is a no-op at min size, but a halt still stops trading.
    d = make_sizer().decide(bankroll=100.0, est_prob=0.90, share_price=0.50,
                            size_multiplier=0.0)
    assert d.skipped is True and d.size == 0.0


def test_min_size_unaffected_by_streak_size_reduction() -> None:
    # A partial reduction (0 < mult < 1) cannot push below the exchange min, so
    # the size is still the floor — reduction is a no-op, NOT a skip.
    assert make_sizer().compute(bankroll=100.0, est_prob=0.90, share_price=0.50,
                                size_multiplier=0.5) == 2.5


def test_strategy_wires_flat_min_size_from_config(tmp_path) -> None:
    from strategies.contrarian_ev import ContrarianEvStrategy
    s = ContrarianEvStrategy("t", {"db_path": str(tmp_path / "t.db"),
                                   "sizing": {"flat_min_size": True}})
    assert s._sizer.flat_min_size is True


def test_strategy_defaults_flat_min_size_off(tmp_path) -> None:
    from strategies.contrarian_ev import ContrarianEvStrategy
    s = ContrarianEvStrategy("t", {"db_path": str(tmp_path / "t.db")})
    assert s._sizer.flat_min_size is False


def test_flat_min_size_defaults_off_and_other_modes_untouched() -> None:
    # Default OFF — every other bot is unaffected.
    assert PositionSizer().flat_min_size is False
    # Legacy absolute clamp: STRONG edge -> $5 max, unchanged.
    legacy = PositionSizer(kelly_multiplier=0.25, min_bet_usdc=1.0, max_bet_usdc=5.0)
    assert legacy.compute(bankroll=150.0, **STRONG) == 5.0
    # Wallet-proportional still SKIPS the J37 case when min-size is off.
    wp = PositionSizer(kelly_multiplier=0.25, wallet_proportional=True)
    assert wp.compute(bankroll=100.0, est_prob=0.90, share_price=0.85) == 0.0
