"""Known-input regression tests for the regime monitor's pure aggregation logic.

These lock the L1 bucketing and the daily/PnL aggregation maths so refactors
can't silently change what the drift monitor reports.
"""
import numpy as np
import pandas as pd
import pytest

from agents.regime_monitor import (
    daily_profile,
    drift_signals,
    expected_edge,
    l1_bucket,
    psi,
    regime_pnl,
    zone_fractions,
    zone_mean_pnl,
)


# --- l1_bucket boundaries -------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (-1.0, "strong_bear"),
    (-0.4, "strong_bear"),     # boundary: -0.4 falls into strong_bear
    (-0.39, "mild_bear"),
    (-0.1, "mild_bear"),       # boundary: -0.1 falls into mild_bear
    (-0.09, "neutral"),
    (0.0, "neutral"),
    (0.09, "neutral"),
    (0.1, "mild_bull"),        # boundary: 0.1 falls into mild_bull
    (0.39, "mild_bull"),
    (0.4, "strong_bull"),      # boundary: 0.4 falls into strong_bull
    (1.0, "strong_bull"),
])
def test_l1_bucket_boundaries(value, expected):
    assert l1_bucket(value) == expected


def test_l1_bucket_nan_is_unknown():
    assert l1_bucket(float("nan")) == "unknown"
    assert l1_bucket(None) == "unknown"


# --- fixtures -------------------------------------------------------------- #
def _frame() -> pd.DataFrame:
    """Two days, hand-computable aggregates."""
    rows = [
        # day 1: L1 = +0.5, +0.5  -> both strong_bull, both bullish
        ("2026-06-01T00:00:00+00:00", 0.5, "ranging", 100.0, "UP", 1.0),
        ("2026-06-01T00:01:00+00:00", 0.5, "ranging", 102.0, "DOWN", -1.0),
        # day 2: L1 = -0.5, 0.0   -> one strong_bear, one neutral
        ("2026-06-02T00:00:00+00:00", -0.5, "high_vol", 200.0, "UP", 2.0),
        ("2026-06-02T00:01:00+00:00", 0.0, "high_vol", 210.0, "DOWN", -3.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "oracle_lag_signal", "regime", "btc_price_at_entry",
        "resolution", "pnl"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    df["l1"] = df["oracle_lag_signal"]
    df["l1_zone"] = df["l1"].map(l1_bucket)
    return df


def test_daily_profile_known_values():
    prof = daily_profile(_frame()).set_index("day")
    d1 = prof.loc[pd.Timestamp("2026-06-01").date()]
    assert d1["n"] == 2
    assert d1["l1_mean"] == 0.5
    assert d1["pct_bull"] == 1.0
    assert d1["pct_bear"] == 0.0
    # btc range% = (102-100)/101 * 100
    assert d1["btc_range_pct"] == pytest.approx((2 / 101) * 100, abs=0.01)
    assert d1["dom_regime"] == "ranging"

    d2 = prof.loc[pd.Timestamp("2026-06-02").date()]
    assert d2["l1_mean"] == -0.25
    assert d2["pct_bull"] == 0.0   # 0.0 is not > 0
    assert d2["pct_bear"] == 0.5   # one of two is < 0


def test_regime_pnl_by_l1zone_known_values():
    out = regime_pnl(_frame(), "l1_zone").set_index("l1_zone")
    # strong_bull: two trades, pnl +1 and -1 -> mean 0, total 0, win 0.5
    assert out.loc["strong_bull", "n"] == 2
    assert out.loc["strong_bull", "win_pct"] == 0.5
    assert out.loc["strong_bull", "mean_pnl"] == 0.0
    assert out.loc["strong_bull", "total_pnl"] == 0.0
    # strong_bear: one trade pnl +2
    assert out.loc["strong_bear", "total_pnl"] == 2.0
    assert out.loc["strong_bear", "win_pct"] == 1.0
    # neutral: one trade pnl -3
    assert out.loc["neutral", "total_pnl"] == -3.0


def test_regime_pnl_excludes_unresolved():
    df = _frame()
    extra = df.iloc[[0]].copy()
    extra["resolution"] = "PENDING"
    extra["pnl"] = np.nan
    df2 = pd.concat([df, extra], ignore_index=True)
    out = regime_pnl(df2, "l1_zone")
    # total resolved count unchanged at 4
    assert out["n"].sum() == 4


# --- Phase 3: drift detection ---------------------------------------------- #
def test_zone_fractions_sum_to_one_and_canonical():
    df = _frame()  # zones: strong_bull, strong_bull, strong_bear, neutral
    f = zone_fractions(df)
    assert set(f) == {"strong_bear", "mild_bear", "neutral", "mild_bull", "strong_bull"}
    assert f["strong_bull"] == 0.5
    assert f["strong_bear"] == 0.25
    assert f["neutral"] == 0.25
    assert pytest.approx(sum(f.values())) == 1.0


def test_psi_zero_for_identical_distributions():
    f = {"strong_bear": 0.2, "mild_bear": 0.2, "neutral": 0.2,
         "mild_bull": 0.2, "strong_bull": 0.2}
    assert psi(f, f) == pytest.approx(0.0, abs=1e-9)


def test_psi_known_two_bucket_shift():
    # Only neutral & strong_bull populated; flips 0.8/0.2 -> 0.2/0.8.
    exp = {"neutral": 0.8, "strong_bull": 0.2}
    act = {"neutral": 0.2, "strong_bull": 0.8}
    # PSI = (0.2-0.8)ln(0.2/0.8) + (0.8-0.2)ln(0.8/0.2) = 2*0.6*ln4
    assert psi(exp, act) == pytest.approx(2 * 0.6 * np.log(4), abs=1e-6)


def test_expected_edge_is_weighted_dot_product():
    fracs = {"strong_bear": 0.0, "mild_bear": 0.0, "neutral": 0.5,
             "mild_bull": 0.0, "strong_bull": 0.5}
    pnl = {"strong_bear": 0.0, "mild_bear": 0.0, "neutral": 0.2,
           "mild_bull": 0.0, "strong_bull": 1.0}
    assert expected_edge(fracs, pnl) == pytest.approx(0.5 * 0.2 + 0.5 * 1.0)


def test_zone_mean_pnl_known():
    zp = zone_mean_pnl(_frame())
    assert zp["strong_bull"] == 0.0   # +1 and -1
    assert zp["strong_bear"] == 2.0
    assert zp["neutral"] == -3.0


def _two_period_frame() -> pd.DataFrame:
    """Baseline = strong-bull regime; recent = collapsed to neutral.
    Engineered so the edge-at-risk and PSI signals both fire."""
    rows = []
    # baseline days (strong_bull, profitable): 6 trades over 3 days
    for d in range(1, 4):
        for i, pnl in enumerate([1.0, 1.0]):
            rows.append((f"2026-06-0{d}T0{i}:00:00+00:00", 0.6, "ranging",
                         100.0 + i, "UP", pnl))
    # recent days (neutral, flat): 4 trades over 2 days
    for d in range(5, 7):
        for i, pnl in enumerate([0.0, 0.0]):
            rows.append((f"2026-06-0{d}T0{i}:00:00+00:00", 0.0, "ranging",
                         100.0 + i, "UP", pnl))
    df = pd.DataFrame(rows, columns=[
        "timestamp", "oracle_lag_signal", "regime", "btc_price_at_entry",
        "resolution", "pnl"])
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["day"] = df["ts"].dt.date
    df["l1"] = df["oracle_lag_signal"]
    df["l1_zone"] = df["l1"].map(l1_bucket)
    return df


def test_drift_signals_flags_regime_collapse():
    df = _two_period_frame()
    baseline = df[df["day"] < pd.Timestamp("2026-06-05").date()]
    recent = df[df["day"] >= pd.Timestamp("2026-06-05").date()]
    d = drift_signals(baseline, recent)
    assert d["ok"] is False  # something must fire
    by_name = {s["name"]: s for s in d["signals"]}
    # strong_bull (baseline payoff +1.0) -> neutral (payoff 0): edge collapses
    assert by_name["expected-edge-at-risk"]["alert"] is True
    # distribution moved entirely strong_bull -> neutral
    assert by_name["L1 zone-mix (PSI)"]["alert"] is True


def test_drift_signals_insufficient_data():
    df = _two_period_frame()
    d = drift_signals(df.iloc[:0], df)
    assert d["ok"] is None
    assert "insufficient" in d["reason"]
