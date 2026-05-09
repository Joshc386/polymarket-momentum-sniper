"""Tests for SM Trade Monitor (Phase 2 of L9 signal layer)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.sm_trade_monitor import (
    CTF_CONTRACT,
    TOKEN_DECIMALS,
    TRANSFER_SINGLE_TOPIC,
    SMTradeMonitor,
    _addr_from_topic,
    _parse_log,
)
from data.sm_wallets import SMWalletRegistry
from strategy.sm_confirmation import SMFlowState


# --- Test fixtures ---

SM_WALLET_A = "0xaaaa1111111111111111111111111111aaaaaaaa"
SM_WALLET_B = "0xbbbb2222222222222222222222222222bbbbbbbb"
NON_SM_WALLET = "0xcccc3333333333333333333333333333cccccccc"

YES_TOKEN_ID = 12345
NO_TOKEN_ID = 67890


def _pad_address(addr: str) -> str:
    """Pad an address to 32-byte hex topic."""
    raw = addr.lower().replace("0x", "")
    return "0x" + raw.zfill(64)


def _encode_data(token_id: int, value: int) -> str:
    """ABI-encode (uint256 id, uint256 value) as hex data field."""
    return "0x" + hex(token_id)[2:].zfill(64) + hex(value)[2:].zfill(64)


def _make_log(
    from_addr: str,
    to_addr: str,
    token_id: int,
    value: int,
    operator: str = "0x0000000000000000000000000000000000000001",
    block_number: int = 100,
) -> dict:
    """Build a realistic TransferSingle log entry."""
    return {
        "address": CTF_CONTRACT,
        "topics": [
            TRANSFER_SINGLE_TOPIC,
            _pad_address(operator),
            _pad_address(from_addr),
            _pad_address(to_addr),
        ],
        "data": _encode_data(token_id, value),
        "blockNumber": hex(block_number),
        "transactionHash": "0x" + "ab" * 32,
        "logIndex": "0x0",
    }


@pytest.fixture
def sm_registry() -> SMWalletRegistry:
    """Registry with two known SM wallets."""
    reg = SMWalletRegistry(csv_path=Path("/nonexistent"))
    reg.load_from_list([SM_WALLET_A, SM_WALLET_B])
    return reg


@pytest.fixture
def monitor(sm_registry: SMWalletRegistry) -> SMTradeMonitor:
    """Monitor with test RPCs and known market tokens."""
    mon = SMTradeMonitor(
        _sm_registry=sm_registry,
        _rpcs=["http://test-rpc:8545"],
    )
    mon.set_market(str(YES_TOKEN_ID), str(NO_TOKEN_ID), "test-window-1")
    mon._last_block = 99  # simulate having fetched an initial block
    return mon


# --- Test classes ---


class TestAddrFromTopic:
    def test_extracts_address(self) -> None:
        topic = _pad_address("0xAABBccDD11223344556677889900aabbccddeeff")
        assert _addr_from_topic(topic) == "0xaabbccdd11223344556677889900aabbccddeeff"

    def test_lowercase(self) -> None:
        topic = _pad_address("0xAABBCCDD11223344556677889900AABBCCDDEEFF")
        assert _addr_from_topic(topic) == "0xaabbccdd11223344556677889900aabbccddeeff"


class TestParseLog:
    def test_valid_log(self) -> None:
        log = _make_log(SM_WALLET_A, SM_WALLET_B, YES_TOKEN_ID, 1_000_000)
        result = _parse_log(log)
        assert result is not None
        from_addr, to_addr, token_id, value = result
        assert from_addr == SM_WALLET_A
        assert to_addr == SM_WALLET_B
        assert token_id == YES_TOKEN_ID
        assert value == 1_000_000

    def test_wrong_topic_returns_none(self) -> None:
        log = _make_log(SM_WALLET_A, SM_WALLET_B, YES_TOKEN_ID, 1_000_000)
        log["topics"][0] = "0x" + "00" * 32
        assert _parse_log(log) is None

    def test_missing_topics_returns_none(self) -> None:
        log = {"topics": [TRANSFER_SINGLE_TOPIC], "data": "0x" + "00" * 64}
        assert _parse_log(log) is None

    def test_short_data_returns_none(self) -> None:
        log = _make_log(SM_WALLET_A, SM_WALLET_B, YES_TOKEN_ID, 100)
        log["data"] = "0x00"
        assert _parse_log(log) is None


class TestTokenClassification:
    def test_yes_token(self, monitor: SMTradeMonitor) -> None:
        assert monitor._classify_token(YES_TOKEN_ID) == "YES"

    def test_no_token(self, monitor: SMTradeMonitor) -> None:
        assert monitor._classify_token(NO_TOKEN_ID) == "NO"

    def test_unknown_token(self, monitor: SMTradeMonitor) -> None:
        assert monitor._classify_token(99999) is None


class TestFlowAccumulation:
    def test_sm_buys_yes(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 5_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == pytest.approx(5.0)
        assert flow.no_volume == 0.0
        assert flow.num_wallets == 1

    def test_sm_buys_no(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, NO_TOKEN_ID, 3_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0
        assert flow.no_volume == pytest.approx(3.0)

    def test_sm_sells_yes_is_bearish(self, monitor: SMTradeMonitor) -> None:
        """Selling YES = bearish → counts as NO volume."""
        logs = [_make_log(SM_WALLET_A, NON_SM_WALLET, YES_TOKEN_ID, 2_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0
        assert flow.no_volume == pytest.approx(2.0)

    def test_sm_sells_no_is_bullish(self, monitor: SMTradeMonitor) -> None:
        """Selling NO = bullish → counts as YES volume."""
        logs = [_make_log(SM_WALLET_A, NON_SM_WALLET, NO_TOKEN_ID, 4_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == pytest.approx(4.0)
        assert flow.no_volume == 0.0

    def test_distinct_wallet_count(self, monitor: SMTradeMonitor) -> None:
        logs = [
            _make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 1_000_000),
            _make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 2_000_000),
            _make_log(NON_SM_WALLET, SM_WALLET_B, NO_TOKEN_ID, 1_000_000),
        ]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.num_wallets == 2

    def test_non_sm_wallet_ignored(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, NON_SM_WALLET, YES_TOKEN_ID, 10_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0
        assert flow.no_volume == 0.0
        assert flow.num_wallets == 0

    def test_zero_value_ignored(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 0)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0

    def test_unknown_token_ignored(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, 99999, 5_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0
        assert flow.no_volume == 0.0

    def test_sm_to_sm_transfer(self, monitor: SMTradeMonitor) -> None:
        """Both from and to are SM — both get counted."""
        logs = [_make_log(SM_WALLET_A, SM_WALLET_B, YES_TOKEN_ID, 1_000_000)]
        monitor._process_logs(logs)
        flow = monitor.get_flow_state()
        # A sells YES (bearish → NO) + B buys YES (bullish → YES)
        assert flow.yes_volume == pytest.approx(1.0)
        assert flow.no_volume == pytest.approx(1.0)
        assert flow.num_wallets == 2

    def test_reset_clears_all(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 5_000_000)]
        monitor._process_logs(logs)
        assert monitor.get_flow_state().yes_volume > 0

        monitor.reset()
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0
        assert flow.no_volume == 0.0
        assert flow.num_wallets == 0

    def test_set_market_resets_on_new_window(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 5_000_000)]
        monitor._process_logs(logs)
        assert monitor.get_flow_state().yes_volume > 0

        monitor.set_market(str(YES_TOKEN_ID), str(NO_TOKEN_ID), "new-window-2")
        flow = monitor.get_flow_state()
        assert flow.yes_volume == 0.0

    def test_set_market_no_reset_same_window(self, monitor: SMTradeMonitor) -> None:
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 5_000_000)]
        monitor._process_logs(logs)
        vol_before = monitor.get_flow_state().yes_volume

        monitor.set_market(str(YES_TOKEN_ID), str(NO_TOKEN_ID), "test-window-1")
        assert monitor.get_flow_state().yes_volume == vol_before


class TestFlowStateType:
    def test_returns_sm_flow_state(self, monitor: SMTradeMonitor) -> None:
        flow = monitor.get_flow_state()
        assert isinstance(flow, SMFlowState)

    def test_flow_state_is_frozen(self, monitor: SMTradeMonitor) -> None:
        flow = monitor.get_flow_state()
        with pytest.raises(AttributeError):
            flow.yes_volume = 999.0  # type: ignore[misc]


# --- Async RPC integration tests ---

def _mock_rpc_response(method: str, result: object) -> dict:
    """Build a JSON-RPC response."""
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def _setup_mock_client(responses: list[dict]) -> AsyncMock:
    """Create a mock httpx client that returns sequential responses."""
    mock_resp = MagicMock()
    mock_resp.json = MagicMock(side_effect=responses)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


class TestRPCIntegration:
    def test_fetch_once_processes_logs(self, monitor: SMTradeMonitor) -> None:
        log = _make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 2_000_000, block_number=101)
        monitor._http_client = _setup_mock_client([
            _mock_rpc_response("eth_blockNumber", hex(105)),
            _mock_rpc_response("eth_getLogs", [log]),
        ])

        flow = asyncio.run(monitor.fetch_once())
        assert flow.yes_volume == pytest.approx(2.0)
        assert flow.num_wallets == 1

    def test_fetch_once_empty_logs(self, monitor: SMTradeMonitor) -> None:
        monitor._http_client = _setup_mock_client([
            _mock_rpc_response("eth_blockNumber", hex(105)),
            _mock_rpc_response("eth_getLogs", []),
        ])

        flow = asyncio.run(monitor.fetch_once())
        assert flow.yes_volume == 0.0
        assert flow.no_volume == 0.0
        assert flow.num_wallets == 0

    def test_fetch_once_advances_block(self, monitor: SMTradeMonitor) -> None:
        monitor._http_client = _setup_mock_client([
            _mock_rpc_response("eth_blockNumber", hex(110)),
            _mock_rpc_response("eth_getLogs", []),
        ])

        asyncio.run(monitor.fetch_once())
        assert monitor._last_block == 110

    def test_all_rpcs_fail_returns_partial(self, monitor: SMTradeMonitor) -> None:
        # Accumulate some data first
        logs = [_make_log(NON_SM_WALLET, SM_WALLET_A, YES_TOKEN_ID, 3_000_000)]
        monitor._process_logs(logs)

        # Now all RPCs fail
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        monitor._http_client = mock_client

        flow = asyncio.run(monitor.fetch_once())
        assert flow.yes_volume == pytest.approx(3.0)

    def test_first_fetch_sets_last_block(self, sm_registry: SMWalletRegistry) -> None:
        """First fetch for a new window just records the block number."""
        mon = SMTradeMonitor(
            _sm_registry=sm_registry,
            _rpcs=["http://test:8545"],
        )
        mon.set_market(str(YES_TOKEN_ID), str(NO_TOKEN_ID), "window-1")
        assert mon._last_block == 0

        mon._http_client = _setup_mock_client([
            _mock_rpc_response("eth_blockNumber", hex(200)),
        ])

        flow = asyncio.run(mon.fetch_once())
        assert mon._last_block == 200
        assert flow.yes_volume == 0.0

    def test_no_fetch_when_block_unchanged(self, monitor: SMTradeMonitor) -> None:
        monitor._last_block = 105
        mock_client = _setup_mock_client([
            _mock_rpc_response("eth_blockNumber", hex(105)),
        ])
        monitor._http_client = mock_client

        flow = asyncio.run(monitor.fetch_once())
        assert mock_client.post.call_count == 1
        assert flow.yes_volume == 0.0
