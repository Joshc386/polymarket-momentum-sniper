"""Tests for tools.watchdog — the heartbeat monitor that auto-fires the kill.

The watchdog (ADR-0002 §4) polls heartbeat.json. It ARMS only after seeing one
valid heartbeat (so it never fires on a just-booting bot); once armed it fires
after N consecutive stale polls; a missing/unreadable heartbeat counts as stale
(fail-safe: pause beats running blind). On fire it calls the kill action once.

evaluate(now) is the pure state machine (fake clock + fake heartbeat); run() is
exercised with a controllable clock + a mocked kill for the "fake hung bot"
integration.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from core import kill_switch_io as ksio
from tools.watchdog import Watchdog


def _wd(tmp_path, **kw):
    return Watchdog(
        on_fire=kw.pop("on_fire", AsyncMock()),
        heartbeat_path=tmp_path / "heartbeat.json",
        poll_secs=kw.pop("poll_secs", 2.0),
        staleness_secs=kw.pop("staleness_secs", 20.0),
        stale_checks_to_fire=kw.pop("stale_checks_to_fire", 3),
        **kw,
    )


def _beat(tmp_path, ts):
    ksio.write_heartbeat(window_end_ts=ts + 300, token_ids=["0xA"], ts=ts,
                         path=tmp_path / "heartbeat.json")


# --- arming ---------------------------------------------------------------

def test_does_not_arm_or_fire_without_a_heartbeat(tmp_path):
    wd = _wd(tmp_path)
    # No heartbeat file at all — must stay disarmed (booting bot), never fire.
    for t in range(0, 100, 2):
        assert wd.evaluate(float(t)) == "waiting"
    assert wd.armed is False


def test_arms_after_first_valid_heartbeat(tmp_path):
    wd = _wd(tmp_path)
    _beat(tmp_path, ts=100.0)
    assert wd.evaluate(100.0) == "armed"
    assert wd.armed is True


# --- staleness / firing ---------------------------------------------------

def test_fresh_heartbeat_does_not_fire(tmp_path):
    wd = _wd(tmp_path)
    _beat(tmp_path, ts=100.0)
    wd.evaluate(100.0)  # arm
    # Heartbeat keeps updating; age stays small.
    for t in (101.0, 103.0, 105.0):
        _beat(tmp_path, ts=t)
        assert wd.evaluate(t) == "fresh"
    assert wd.stale_count == 0


def test_fires_after_three_consecutive_stale(tmp_path):
    wd = _wd(tmp_path, staleness_secs=20.0, stale_checks_to_fire=3)
    _beat(tmp_path, ts=100.0)
    assert wd.evaluate(100.0) == "armed"
    # Heartbeat frozen at ts=100; clock advances past staleness.
    assert wd.evaluate(125.0) == "stale"   # age 25 > 20 (1)
    assert wd.evaluate(127.0) == "stale"   # (2)
    assert wd.evaluate(129.0) == "fire"    # (3) -> fire


def test_stale_counter_resets_on_fresh(tmp_path):
    wd = _wd(tmp_path, staleness_secs=20.0, stale_checks_to_fire=3)
    _beat(tmp_path, ts=100.0)
    wd.evaluate(100.0)
    assert wd.evaluate(125.0) == "stale"   # (1)
    assert wd.evaluate(127.0) == "stale"   # (2)
    _beat(tmp_path, ts=128.0)              # bot recovers
    assert wd.evaluate(128.0) == "fresh"
    assert wd.stale_count == 0
    # Needs three fresh stales again, not one.
    assert wd.evaluate(150.0) == "stale"   # (1)


def test_missing_heartbeat_when_armed_counts_as_stale(tmp_path):
    wd = _wd(tmp_path, stale_checks_to_fire=2)
    _beat(tmp_path, ts=100.0)
    wd.evaluate(100.0)  # arm
    (tmp_path / "heartbeat.json").unlink()  # file vanishes
    assert wd.evaluate(101.0) == "stale"    # fail-safe (1)
    assert wd.evaluate(102.0) == "fire"     # (2) -> fire


def test_corrupt_heartbeat_when_armed_counts_as_stale(tmp_path):
    wd = _wd(tmp_path, stale_checks_to_fire=1)
    _beat(tmp_path, ts=100.0)
    wd.evaluate(100.0)
    (tmp_path / "heartbeat.json").write_text("{garbage")
    assert wd.evaluate(101.0) == "fire"


# --- integration: fake hung bot fires the kill exactly once ---------------

def test_run_fires_kill_once_on_hung_bot(tmp_path):
    fire = AsyncMock()
    wd = _wd(tmp_path, on_fire=fire, poll_secs=0.0,
             staleness_secs=20.0, stale_checks_to_fire=3)
    _beat(tmp_path, ts=100.0)  # one heartbeat, then the "bot" hangs

    # Deterministic clock: 100 (arm), then advances 5s each poll.
    times = iter([100.0, 125.0, 127.0, 129.0, 131.0, 133.0])
    wd.clock = lambda: next(times)

    asyncio.run(wd.run(max_polls=6))
    fire.assert_awaited_once()
    assert wd.fired is True
