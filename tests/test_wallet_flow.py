"""Tests for Layer 12: On-chain Wallet Flow.

Tests cover:
1. WalletFlowMonitor — log parsing, trade classification, accumulation,
   reset, window transitions, RPC failure handling.
2. WalletFlowSignal — concentration, repeat traders, wallet asymmetry,
   min trades, decay, reset, late-window suppression.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

from data.wallet_flow_monitor import (
    WalletFlowMonitor,
    WalletFlowState,
    WalletTrade,
)
from data.sm_trade_monitor import _parse_log
from signals.wallet_flow import WalletFlowSignal


# ── Helpers ──────────────────────────────────────────────────────


def _make_log(
    from_addr: str,
    to_addr: str,
    token_id: int,
    value: int,
) -> dict:
    """Build a raw TransferSingle log dict matching eth_getLogs format."""
    from data.sm_trade_monitor import TRANSFER_SINGLE_TOPIC

    # Operator topic (topics[1]) — not used in our parsing, just pad
    operator = "0x" + "0" * 64

    # Pad addresses to 32 bytes (64 hex chars)
    from_topic = "0x" + from_addr.lower().replace("0x", "").zfill(64)
    to_topic = "0x" + to_addr.lower().replace("0x", "").zfill(64)

    # Data: ABI-encoded (uint256 id, uint256 value)
    data = "0x" + hex(token_id)[2:].zfill(64) + hex(value)[2:].zfill(64)

    return {
        "topics": [TRANSFER_SINGLE_TOPIC, operator, from_topic, to_topic],
        "data": data,
    }


WALLET_A = "0x" + "a" * 40
WALLET_B = "0x" + "b" * 40
WALLET_C = "0x" + "c" * 40
WALLET_D = "0x" + "d" * 40
YES_TOKEN = 12345
NO_TOKEN = 67890


# ── WalletFlowMonitor tests ─────────────────────────────────────


class TestLogParsing:
    """Verify _parse_log extracts correct fields."""

    def test_valid_log(self) -> None:
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)
        result = _parse_log(log)
        assert result is not None
        from_addr, to_addr, token_id, value = result
        assert from_addr == WALLET_A
        assert to_addr == WALLET_B
        assert token_id == YES_TOKEN
        assert value == 1_000_000

    def test_wrong_topic_skipped(self) -> None:
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)
        log["topics"][0] = "0x" + "f" * 64
        assert _parse_log(log) is None

    def test_short_data_skipped(self) -> None:
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 100)
        log["data"] = "0x1234"
        assert _parse_log(log) is None


class TestTradeClassification:
    """Verify bull/bear classification from token + direction."""

    def _monitor(self) -> WalletFlowMonitor:
        m = WalletFlowMonitor()
        m._yes_token_id = YES_TOKEN
        m._no_token_id = NO_TOKEN
        return m

    def test_buying_yes_is_bullish(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)
        m._process_logs([log])
        state = m.get_flow_state()
        assert state.total_bull_volume > 0
        assert WALLET_B in state.bull_wallets

    def test_selling_yes_is_bearish(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)
        m._process_logs([log])
        state = m.get_flow_state()
        # Sender (WALLET_A) is selling YES = bearish
        assert WALLET_A in state.bear_wallets

    def test_buying_no_is_bearish(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, NO_TOKEN, 1_000_000)
        m._process_logs([log])
        state = m.get_flow_state()
        assert WALLET_B in state.bear_wallets

    def test_selling_no_is_bullish(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, NO_TOKEN, 1_000_000)
        m._process_logs([log])
        state = m.get_flow_state()
        assert WALLET_A in state.bull_wallets

    def test_unknown_token_ignored(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, 99999, 1_000_000)
        m._process_logs([log])
        state = m.get_flow_state()
        assert state.num_wallets == 0

    def test_zero_value_ignored(self) -> None:
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 0)
        m._process_logs([log])
        assert m.get_flow_state().num_wallets == 0


class TestFlowAccumulation:
    """Verify per-wallet volume tracking."""

    def _monitor(self) -> WalletFlowMonitor:
        m = WalletFlowMonitor()
        m._yes_token_id = YES_TOKEN
        m._no_token_id = NO_TOKEN
        return m

    def test_multiple_trades_accumulate(self) -> None:
        m = self._monitor()
        logs = [
            _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000),
            _make_log(WALLET_C, WALLET_B, YES_TOKEN, 2_000_000),
        ]
        m._process_logs(logs)
        state = m.get_flow_state()
        # WALLET_B received YES twice = 3.0 shares bullish
        assert state.wallet_volumes[WALLET_B]["bull"] == pytest.approx(3.0)

    def test_distinct_wallet_count(self) -> None:
        m = self._monitor()
        logs = [
            _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000),
            _make_log(WALLET_A, WALLET_C, YES_TOKEN, 1_000_000),
        ]
        m._process_logs(logs)
        state = m.get_flow_state()
        # WALLET_A (seller, bear) + WALLET_B (buyer, bull) + WALLET_C (buyer, bull)
        assert state.num_wallets == 3

    def test_same_wallet_both_sides(self) -> None:
        """Wallet can have both bull and bear volume."""
        m = self._monitor()
        logs = [
            _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000),  # B buys YES (bull)
            _make_log(WALLET_B, WALLET_C, YES_TOKEN, 500_000),    # B sells YES (bear)
        ]
        m._process_logs(logs)
        state = m.get_flow_state()
        assert state.wallet_volumes[WALLET_B]["bull"] == pytest.approx(1.0)
        assert state.wallet_volumes[WALLET_B]["bear"] == pytest.approx(0.5)

    def test_value_scaled_correctly(self) -> None:
        """Raw value divided by 1e6 (TOKEN_DECIMALS)."""
        m = self._monitor()
        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 5_500_000)
        m._process_logs([log])
        state = m.get_flow_state()
        assert state.wallet_volumes[WALLET_B]["bull"] == pytest.approx(5.5)


class TestMonitorReset:
    """Reset and window transition."""

    def test_reset_clears_all(self) -> None:
        m = WalletFlowMonitor()
        m._yes_token_id = YES_TOKEN
        m._no_token_id = NO_TOKEN
        m._process_logs([_make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)])
        assert m.get_flow_state().num_wallets > 0

        m.reset()
        state = m.get_flow_state()
        assert state.num_wallets == 0
        assert state.total_bull_volume == 0.0
        assert state.total_bear_volume == 0.0
        assert len(state.trades) == 0

    def test_new_window_triggers_reset(self) -> None:
        m = WalletFlowMonitor()
        m.set_market(str(YES_TOKEN), str(NO_TOKEN), "window-1")
        m._process_logs([_make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)])
        assert m.get_flow_state().num_wallets > 0

        m.set_market(str(YES_TOKEN), str(NO_TOKEN), "window-2")
        assert m.get_flow_state().num_wallets == 0

    def test_same_window_no_reset(self) -> None:
        m = WalletFlowMonitor()
        m.set_market(str(YES_TOKEN), str(NO_TOKEN), "window-1")
        m._process_logs([_make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)])
        count = m.get_flow_state().num_wallets

        m.set_market(str(YES_TOKEN), str(NO_TOKEN), "window-1")
        assert m.get_flow_state().num_wallets == count


class _MockHttpClient:
    """Minimal async httpx client mock for RPC tests."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._call_idx = 0

    async def post(self, url: str, json: dict | None = None) -> "_MockResp":
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return _MockResp(resp)
        raise Exception("No more mock responses")


class _MockRespFail:
    """Mock that always raises on post."""

    async def post(self, url: str, json: dict | None = None):
        raise Exception("Connection refused")


class _MockResp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def json(self) -> dict:
        return self._data


class TestMonitorRPC:
    """RPC integration with mocked responses."""

    def test_fetch_once_processes_logs(self) -> None:
        import asyncio

        m = WalletFlowMonitor(_rpcs=["http://fake-rpc"])
        m._yes_token_id = YES_TOKEN
        m._no_token_id = NO_TOKEN
        m._last_block = 100

        log = _make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)
        m._http_client = _MockHttpClient([
            {"result": hex(105)},        # eth_blockNumber
            {"result": [log]},           # eth_getLogs
        ])

        state = asyncio.run(m.fetch_once())
        assert state.num_wallets >= 1
        assert m._last_block == 105

    def test_all_rpcs_fail_returns_partial(self) -> None:
        import asyncio

        m = WalletFlowMonitor(_rpcs=["http://bad1", "http://bad2"])
        m._yes_token_id = YES_TOKEN
        m._no_token_id = NO_TOKEN

        # Pre-accumulate some data
        m._process_logs([_make_log(WALLET_A, WALLET_B, YES_TOKEN, 1_000_000)])
        existing = m.get_flow_state()

        m._http_client = _MockRespFail()

        state = asyncio.run(m.fetch_once())
        # Should return existing accumulated state
        assert state.num_wallets == existing.num_wallets

    def test_first_fetch_sets_block(self) -> None:
        import asyncio

        m = WalletFlowMonitor(_rpcs=["http://fake-rpc"])
        m._yes_token_id = YES_TOKEN
        m._last_block = 0  # First call

        m._http_client = _MockHttpClient([
            {"result": hex(200)},  # eth_blockNumber
        ])

        state = asyncio.run(m.fetch_once())
        assert m._last_block == 200
        assert state.num_wallets == 0  # No logs fetched on first call


# ── WalletFlowSignal tests ──────────────────────────────────────


def _make_trades(
    wallet: str,
    direction: str,
    count: int,
    value: float = 100.0,
) -> list[WalletTrade]:
    """Create multiple WalletTrade records for one wallet."""
    return [
        WalletTrade(wallet=wallet, direction=direction, token="YES", value=value)
        for _ in range(count)
    ]


class TestConcentration:
    """Sub-signal 1: top wallet share per side."""

    def test_one_dominant_bull_is_bullish(self) -> None:
        """One wallet with 80% of bull volume -> positive concentration."""
        sig = WalletFlowSignal(min_trades=1)
        # WALLET_A has 800, WALLET_B has 200 on bull side
        wallet_volumes = {
            WALLET_A: {"bull": 800.0, "bear": 0.0},
            WALLET_B: {"bull": 200.0, "bear": 0.0},
            WALLET_C: {"bear": 500.0, "bull": 0.0},
            WALLET_D: {"bear": 500.0, "bull": 0.0},
        }
        trades = (
            _make_trades(WALLET_A, "BULL", 3)
            + _make_trades(WALLET_B, "BULL", 1)
            + _make_trades(WALLET_C, "BEAR", 2)
            + _make_trades(WALLET_D, "BEAR", 2)
        )
        val = sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=1000.0,
            total_bear_volume=1000.0,
            bull_wallets={WALLET_A, WALLET_B},
            bear_wallets={WALLET_C, WALLET_D},
            trades=trades,
        )
        assert val > 0  # Bull concentration > bear concentration

    def test_one_dominant_bear_is_bearish(self) -> None:
        sig = WalletFlowSignal(min_trades=1)
        wallet_volumes = {
            WALLET_A: {"bull": 500.0, "bear": 0.0},
            WALLET_B: {"bull": 500.0, "bear": 0.0},
            WALLET_C: {"bear": 900.0, "bull": 0.0},
            WALLET_D: {"bear": 100.0, "bull": 0.0},
        }
        trades = (
            _make_trades(WALLET_A, "BULL", 2)
            + _make_trades(WALLET_B, "BULL", 2)
            + _make_trades(WALLET_C, "BEAR", 3)
            + _make_trades(WALLET_D, "BEAR", 1)
        )
        val = sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=1000.0,
            total_bear_volume=1000.0,
            bull_wallets={WALLET_A, WALLET_B},
            bear_wallets={WALLET_C, WALLET_D},
            trades=trades,
        )
        assert val < 0


class TestRepeatTraders:
    """Sub-signal 2: wallets trading same direction multiple times."""

    def test_repeat_bull_is_bullish(self) -> None:
        """Wallet with 3+ bull trades -> positive repeat signal."""
        sig = WalletFlowSignal(min_trades=1, repeat_trade_min=2)
        wallet_volumes = {
            WALLET_A: {"bull": 300.0, "bear": 0.0},
            WALLET_B: {"bear": 100.0, "bull": 0.0},
        }
        # WALLET_A trades 3 times bullish (repeat), WALLET_B once bearish
        trades = (
            _make_trades(WALLET_A, "BULL", 3)
            + _make_trades(WALLET_B, "BEAR", 1)
        )
        val = sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=300.0,
            total_bear_volume=100.0,
            bull_wallets={WALLET_A},
            bear_wallets={WALLET_B},
            trades=trades,
        )
        assert val > 0

    def test_repeat_bear_is_bearish(self) -> None:
        sig = WalletFlowSignal(min_trades=1, repeat_trade_min=2)
        wallet_volumes = {
            WALLET_A: {"bull": 100.0, "bear": 0.0},
            WALLET_B: {"bear": 300.0, "bull": 0.0},
        }
        trades = (
            _make_trades(WALLET_A, "BULL", 1)
            + _make_trades(WALLET_B, "BEAR", 3)
        )
        val = sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=100.0,
            total_bear_volume=300.0,
            bull_wallets={WALLET_A},
            bear_wallets={WALLET_B},
            trades=trades,
        )
        assert val < 0


class TestWalletAsymmetry:
    """Sub-signal 3: unique wallet count per side."""

    def test_more_bull_wallets_is_bullish(self) -> None:
        """Isolate wallet_asym by making concentration and repeat neutral."""
        sig = WalletFlowSignal(
            min_trades=1,
            # Zero out other sub-signals to isolate wallet asymmetry
            concentration_weight=0.0,
            repeat_weight=0.0,
            wallet_asym_weight=1.0,
        )
        wallet_volumes = {
            WALLET_A: {"bull": 100.0, "bear": 0.0},
            WALLET_B: {"bull": 100.0, "bear": 0.0},
            WALLET_C: {"bull": 100.0, "bear": 0.0},
            WALLET_D: {"bear": 100.0, "bull": 0.0},
        }
        trades = (
            _make_trades(WALLET_A, "BULL", 1)
            + _make_trades(WALLET_B, "BULL", 1)
            + _make_trades(WALLET_C, "BULL", 1)
            + _make_trades(WALLET_D, "BEAR", 1)
        )
        val = sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=300.0,
            total_bear_volume=100.0,
            bull_wallets={WALLET_A, WALLET_B, WALLET_C},
            bear_wallets={WALLET_D},
            trades=trades,
        )
        assert val > 0  # 3 bull wallets vs 1 bear wallet


class TestSignalEdgeCases:
    """Min trades, decay, reset, late window."""

    def test_below_min_trades_decays(self) -> None:
        sig = WalletFlowSignal(min_trades=10)
        val = sig.compute(
            wallet_volumes={},
            total_bull_volume=0.0,
            total_bear_volume=0.0,
            bull_wallets=set(),
            bear_wallets=set(),
            trades=[],
        )
        assert val == 0.0

    def test_late_window_suppressed(self) -> None:
        """Signal decays when seconds_remaining < min."""
        sig = WalletFlowSignal(
            min_trades=1, min_seconds_remaining=30.0,
        )
        trades = _make_trades(WALLET_A, "BULL", 5)
        wallet_volumes = {WALLET_A: {"bull": 500.0, "bear": 0.0}}

        # First: build signal with enough time
        sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=500.0,
            total_bear_volume=0.0,
            bull_wallets={WALLET_A},
            bear_wallets=set(),
            trades=trades,
            seconds_remaining=200.0,
        )
        peak = abs(sig.last_signal)
        assert peak > 0

        # Now late in window
        sig.compute(
            wallet_volumes=wallet_volumes,
            total_bull_volume=500.0,
            total_bear_volume=0.0,
            bull_wallets={WALLET_A},
            bear_wallets=set(),
            trades=trades,
            seconds_remaining=10.0,
        )
        assert abs(sig.last_signal) < peak

    def test_reset_clears_all(self) -> None:
        sig = WalletFlowSignal(min_trades=1)
        trades = _make_trades(WALLET_A, "BULL", 5)
        sig.compute(
            wallet_volumes={WALLET_A: {"bull": 500.0, "bear": 0.0}},
            total_bull_volume=500.0,
            total_bear_volume=0.0,
            bull_wallets={WALLET_A},
            bear_wallets=set(),
            trades=trades,
        )
        assert sig.last_signal != 0.0

        sig.reset()
        assert sig.last_signal == 0.0

    def test_empty_state_zero(self) -> None:
        sig = WalletFlowSignal(min_trades=0)
        val = sig.compute(
            wallet_volumes={},
            total_bull_volume=0.0,
            total_bear_volume=0.0,
            bull_wallets=set(),
            bear_wallets=set(),
            trades=[],
        )
        assert val == 0.0

    def test_signal_clamped(self) -> None:
        """Signal stays within [-1, 1] with extreme weights."""
        sig = WalletFlowSignal(
            min_trades=1,
            concentration_weight=10.0,
            repeat_weight=10.0,
            wallet_asym_weight=10.0,
        )
        trades = _make_trades(WALLET_A, "BULL", 10)
        val = sig.compute(
            wallet_volumes={WALLET_A: {"bull": 1000.0, "bear": 0.0}},
            total_bull_volume=1000.0,
            total_bear_volume=0.0,
            bull_wallets={WALLET_A},
            bear_wallets=set(),
            trades=trades,
        )
        assert -1.0 <= val <= 1.0
