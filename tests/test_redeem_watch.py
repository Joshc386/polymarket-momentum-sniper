"""Unit tests for the redeem_watch report aggregation (pure, wallet-free)."""

from __future__ import annotations

from tools.redeem_watch import summarize


def _row(ts, usdc, n_red, halt=0, val=0.0):
    return {
        "ts_utc": ts, "usdc": str(usdc), "n_positions": "1",
        "n_redeemable": str(n_red), "redeemable_value": str(val),
        "halt": str(halt), "redeemable_titles": "",
    }


def test_empty():
    assert summarize([]) == {"rows": 0}


def test_no_lingering_is_clean():
    # Redeemable blips to 1 for a single reading, then clears — the good case.
    rows = [
        _row("2026-06-15T10:00:00+00:00", 100.0, 0),
        _row("2026-06-15T10:01:00+00:00", 98.0, 1, val=2.0),
        _row("2026-06-15T10:02:00+00:00", 102.0, 0),
    ]
    s = summarize(rows)
    assert s["longest_redeemable_streak_rows"] == 1
    assert s["danger_rows"] == 0
    assert s["min_usdc"] == 98.0
    assert s["peak_n_redeemable"] == 1


def test_lingering_streak_spans_minutes():
    # Redeemable stays >0 across three consecutive readings → 2-minute streak.
    rows = [
        _row("2026-06-15T10:00:00+00:00", 90.0, 1, val=2.0),
        _row("2026-06-15T10:01:00+00:00", 90.0, 2, val=4.0),
        _row("2026-06-15T10:02:00+00:00", 90.0, 1, val=2.0),
        _row("2026-06-15T10:03:00+00:00", 95.0, 0),
    ]
    s = summarize(rows)
    assert s["longest_redeemable_streak_rows"] == 3
    assert s["longest_redeemable_streak_min"] == 2.0
    assert s["peak_n_redeemable"] == 2
    assert s["peak_redeemable_value"] == 4.0


def test_danger_combo_flagged():
    # Redeemable winnings present WHILE the bot is halted — the failure mode.
    rows = [
        _row("2026-06-15T10:00:00+00:00", 5.0, 2, halt=1, val=8.0),
        _row("2026-06-15T10:01:00+00:00", 5.0, 2, halt=1, val=8.0),
    ]
    s = summarize(rows)
    assert s["danger_rows"] == 2


def test_handles_blank_usdc():
    # RPC failure leaves usdc blank — must not crash, just skip it.
    rows = [
        _row("2026-06-15T10:00:00+00:00", "", 0),
        _row("2026-06-15T10:01:00+00:00", 50.0, 0),
    ]
    s = summarize(rows)
    assert s["min_usdc"] == 50.0
