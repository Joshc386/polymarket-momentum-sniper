"""Tests for per-bot execution_mode wiring + live-fleet guards (ADR-0003 p4).

config_multi.yaml gains a per-bot ``execution_mode: live | paper`` key
(default paper). Guards (grilled 2026-06-12): the runner refuses to start
when a live bot lacks CLOB auth (no silent paper fallback) and when more
than one bot declares live (v1). The live engine gets the shared
authenticated client, the bot's name as bot_id for J27 telemetry, and a
bankroll sync at startup + each window end.
"""

import asyncio
from unittest.mock import MagicMock

from core.execution import LiveExecutionEngine, PaperExecutionEngine
from core.live_mode import validate_live_fleet
from strategies.contrarian_ev import ContrarianEvStrategy


class TestValidateLiveFleet:
    def test_all_paper_is_fine(self):
        cfg = {"k": {"enabled": True}, "k2": {"enabled": True}}
        assert validate_live_fleet(cfg, authenticated=False) is None

    def test_one_live_authenticated_is_fine(self):
        cfg = {
            "k": {"enabled": True, "execution_mode": "live"},
            "k2": {"enabled": True},
        }
        assert validate_live_fleet(cfg, authenticated=True) is None

    def test_live_without_auth_refused(self):
        cfg = {"k": {"enabled": True, "execution_mode": "live"}}
        err = validate_live_fleet(cfg, authenticated=False)
        assert err is not None and "auth" in err.lower()

    def test_two_live_bots_refused(self):
        cfg = {
            "k": {"enabled": True, "execution_mode": "live"},
            "k2": {"enabled": True, "execution_mode": "live"},
        }
        err = validate_live_fleet(cfg, authenticated=True)
        assert err is not None and "one live bot" in err.lower()

    def test_disabled_live_bot_does_not_count(self):
        cfg = {
            "k": {"enabled": True, "execution_mode": "live"},
            "old": {"enabled": False, "execution_mode": "live"},
        }
        assert validate_live_fleet(cfg, authenticated=True) is None


class TestExecutorSelection:
    def test_default_is_paper(self):
        s = ContrarianEvStrategy("bot_t", {"db_path": ":memory:"})
        assert isinstance(s._executor, PaperExecutionEngine)

    def test_live_mode_builds_live_engine_with_shared_client(self):
        poly = MagicMock()
        s = ContrarianEvStrategy("bot_t_live", {
            "db_path": ":memory:",
            "execution_mode": "live",
            "_poly_client": poly,
        })
        assert isinstance(s._executor, LiveExecutionEngine)
        assert s._executor.poly is poly
        assert s._executor.bot_id == "bot_t_live"   # J27 telemetry attribution

    def test_live_mode_without_client_refuses_loudly(self):
        try:
            ContrarianEvStrategy("bot_t", {
                "db_path": ":memory:", "execution_mode": "live",
            })
            raised = False
        except ValueError:
            raised = True
        assert raised, "live without an injected client must not construct"


class _FakeLiveExecutor:
    is_paper = False
    pending_trade = None
    bankroll = 50.0

    def __init__(self):
        self.sync_calls = 0

    def resolve_pending_trade(self, resolution):
        return None

    async def sync_bankroll(self):
        self.sync_calls += 1


def test_window_end_syncs_live_bankroll():
    """Sizing reads the wallet at entries in fresh windows -> sync after
    each window-end resolution (grilled decision: startup + window end)."""
    s = ContrarianEvStrategy("bot_t", {"db_path": ":memory:"})
    fake = _FakeLiveExecutor()
    s._executor = fake

    async def scenario():
        s.on_window_end("UP")
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert fake.sync_calls == 1
