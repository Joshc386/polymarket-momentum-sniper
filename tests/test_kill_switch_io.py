"""Tests for core.kill_switch_io — the kill switch's file signalling channel.

This module is the single source of truth for the heartbeat.json / HALT file
formats (ADR-0002, KILL_SWITCH_PLAN.md §2). It is written by the trading
process and read by the standalone watchdog/kill-switch processes, so the
contract (atomic writes, fail-safe reads, sticky HALT) is load-bearing.
"""

import json

from core import kill_switch_io as ksio


# --- Heartbeat ------------------------------------------------------------

def test_heartbeat_round_trip(tmp_path):
    p = tmp_path / "heartbeat.json"
    written = ksio.write_heartbeat(
        window_end_ts=1780432200.0,
        token_ids=["0xYES", "0xNO"],
        ts=1780431948.84,
        path=p,
    )
    assert written == {
        "ts": 1780431948.84,
        "window_end_ts": 1780432200.0,
        "token_ids": ["0xYES", "0xNO"],
    }
    assert ksio.read_heartbeat(path=p) == written


def test_heartbeat_write_is_atomic_no_temp_left(tmp_path):
    p = tmp_path / "heartbeat.json"
    ksio.write_heartbeat(window_end_ts=1.0, token_ids=["a"], ts=2.0, path=p)
    # Only the final file exists — no leftover temp artifact.
    names = sorted(f.name for f in tmp_path.iterdir())
    assert names == ["heartbeat.json"]


def test_heartbeat_overwrites_previous(tmp_path):
    p = tmp_path / "heartbeat.json"
    ksio.write_heartbeat(window_end_ts=1.0, token_ids=["a"], ts=1.0, path=p)
    ksio.write_heartbeat(window_end_ts=2.0, token_ids=["b"], ts=2.0, path=p)
    hb = ksio.read_heartbeat(path=p)
    assert hb["window_end_ts"] == 2.0 and hb["token_ids"] == ["b"]


def test_heartbeat_filters_empty_token_ids(tmp_path):
    p = tmp_path / "heartbeat.json"
    written = ksio.write_heartbeat(
        window_end_ts=1.0, token_ids=["0xYES", "", None, "0xNO"], ts=1.0, path=p
    )
    assert written["token_ids"] == ["0xYES", "0xNO"]


def test_heartbeat_allows_none_window_end(tmp_path):
    # No active market → window_end_ts None, empty tokens. Must still write.
    p = tmp_path / "heartbeat.json"
    written = ksio.write_heartbeat(window_end_ts=None, token_ids=[], ts=5.0, path=p)
    assert written["window_end_ts"] is None and written["token_ids"] == []
    assert ksio.read_heartbeat(path=p) == written


def test_read_missing_heartbeat_returns_none(tmp_path):
    assert ksio.read_heartbeat(path=tmp_path / "nope.json") is None


def test_read_corrupt_heartbeat_returns_none(tmp_path):
    p = tmp_path / "heartbeat.json"
    p.write_text("{not valid json")
    assert ksio.read_heartbeat(path=p) is None


# --- HALT flag ------------------------------------------------------------

def test_halt_inactive_when_absent(tmp_path):
    assert ksio.halt_active(path=tmp_path / "HALT") is False


def test_write_halt_makes_it_active_and_sticky(tmp_path):
    p = tmp_path / "HALT"
    rec = ksio.write_halt(source="watchdog", reason="heartbeat stale 27s", ts=99.0, path=p)
    assert rec == {"ts": 99.0, "source": "watchdog", "reason": "heartbeat stale 27s"}
    assert ksio.halt_active(path=p) is True
    # Sticky: reading/checking it does not remove it.
    assert ksio.halt_active(path=p) is True
    on_disk = json.loads(p.read_text())
    assert on_disk["source"] == "watchdog" and on_disk["reason"] == "heartbeat stale 27s"


def test_write_halt_is_atomic_no_temp_left(tmp_path):
    p = tmp_path / "HALT"
    ksio.write_halt(source="manual", reason="x", ts=1.0, path=p)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["HALT"]


def test_clear_halt_removes_flag(tmp_path):
    p = tmp_path / "HALT"
    ksio.write_halt(source="manual", reason="x", ts=1.0, path=p)
    assert ksio.halt_active(path=p) is True
    ksio.clear_halt(path=p)
    assert ksio.halt_active(path=p) is False
    # Idempotent — clearing an absent flag is not an error.
    ksio.clear_halt(path=p)
