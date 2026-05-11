"""Layer 12: On-chain Wallet Flow Signal — wallet-level conviction.

Analyses per-wallet trading patterns from Polygon blockchain data
to detect conviction asymmetry that anonymous WebSocket data cannot
reveal.

Three sub-signals:
1. Wallet concentration: when a small number of wallets dominate
   volume on one side, that's informed money — not diffuse retail.
   Measures top-1 wallet share of each side's volume.
2. Repeat trade detection: wallets placing 2+ trades in the same
   direction within the window. These are "building a position in
   pieces" — the SM pattern observed from Dune analysis.
3. Unique wallet asymmetry: count of distinct wallets on buy vs
   sell side. More unique buyers than sellers = broader conviction.

Data source: WalletFlowState from data.wallet_flow_monitor, which
polls Polygon eth_getLogs for TransferSingle events on the CTF contract.

Output is [-1.0, 1.0]:
  Positive = bullish wallet conviction
  Negative = bearish wallet conviction
  Zero = balanced or insufficient data
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WalletFlowSignal:
    """Detects wallet-level conviction from on-chain trade data.

    Args:
        concentration_weight: Weight of wallet concentration sub-signal.
        repeat_weight: Weight of repeat trader sub-signal.
        wallet_asym_weight: Weight of unique wallet asymmetry sub-signal.
        min_trades: Minimum number of on-chain trades before signal
            fires. Below this there isn't enough data.
        repeat_trade_min: A wallet needs this many trades in the same
            direction to count as a "repeat" trader.
        min_seconds_remaining: Don't fire late in the window — not
            enough time for the conviction to play out.
        decay_rate: Signal decay when conditions stop being met.
    """

    concentration_weight: float = 0.40
    repeat_weight: float = 0.35
    wallet_asym_weight: float = 0.25
    min_trades: int = 5
    repeat_trade_min: int = 2
    min_seconds_remaining: float = 30.0
    decay_rate: float = 0.3

    # Internal state
    _current_signal: float = 0.0

    def reset(self) -> None:
        """Clear all state for a new window."""
        self._current_signal = 0.0

    def compute(
        self,
        wallet_volumes: dict[str, dict[str, float]],
        total_bull_volume: float,
        total_bear_volume: float,
        bull_wallets: set[str],
        bear_wallets: set[str],
        trades: list,
        seconds_remaining: float = 300.0,
    ) -> float:
        """Compute wallet flow conviction signal.

        Args:
            wallet_volumes: {wallet: {"bull": float, "bear": float}}.
            total_bull_volume: Sum of all bullish volume.
            total_bear_volume: Sum of all bearish volume.
            bull_wallets: Set of wallets with bullish trades.
            bear_wallets: Set of wallets with bearish trades.
            trades: List of WalletTrade records.
            seconds_remaining: Seconds left in the 5-min window.

        Returns:
            Signal in [-1.0, 1.0].
            Positive = bullish wallet conviction.
            Negative = bearish wallet conviction.
        """
        # Late in window — decay, don't build
        if seconds_remaining < self.min_seconds_remaining:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # Need minimum trades
        if len(trades) < self.min_trades:
            self._current_signal *= (1.0 - self.decay_rate)
            return self._current_signal

        # ── Sub-signal 1: Wallet concentration ──
        concentration = self._compute_concentration(
            wallet_volumes, total_bull_volume, total_bear_volume,
        )

        # ── Sub-signal 2: Repeat trade detection ──
        repeat = self._compute_repeat_traders(trades)

        # ── Sub-signal 3: Unique wallet asymmetry ──
        wallet_asym = self._compute_wallet_asymmetry(
            bull_wallets, bear_wallets,
        )

        # Blend sub-signals
        raw = (
            self.concentration_weight * concentration
            + self.repeat_weight * repeat
            + self.wallet_asym_weight * wallet_asym
        )

        self._current_signal = _clamp(raw)
        return self._current_signal

    @property
    def last_signal(self) -> float:
        """Most recently computed signal value."""
        return self._current_signal

    def _compute_concentration(
        self,
        wallet_volumes: dict[str, dict[str, float]],
        total_bull: float,
        total_bear: float,
    ) -> float:
        """Directional signal from top-wallet concentration.

        If the top wallet on the bull side holds a larger share than
        the top wallet on the bear side, that's a bullish concentration
        signal (one dominant informed buyer vs diffuse sellers).

        Returns [-1, 1].
        """
        if not wallet_volumes:
            return 0.0

        # Top wallet share on bull side
        bull_share = 0.0
        if total_bull > 0:
            top_bull = max(
                (v.get("bull", 0.0) for v in wallet_volumes.values()),
                default=0.0,
            )
            bull_share = top_bull / total_bull

        # Top wallet share on bear side
        bear_share = 0.0
        if total_bear > 0:
            top_bear = max(
                (v.get("bear", 0.0) for v in wallet_volumes.values()),
                default=0.0,
            )
            bear_share = top_bear / total_bear

        # Higher concentration on bull side → positive
        # Higher concentration on bear side → negative
        # Both range [0, 1], so difference ranges [-1, 1]
        return bull_share - bear_share

    def _compute_repeat_traders(self, trades: list) -> float:
        """Directional signal from wallets trading multiple times.

        Counts wallets with repeat_trade_min+ trades in the same
        direction. These are the "building a position in pieces"
        wallets — the key SM behaviour pattern.

        Returns [-1, 1].
        """
        if not trades:
            return 0.0

        # Count trades per wallet per direction
        wallet_dir_counts: dict[str, dict[str, int]] = {}
        for trade in trades:
            wallet = trade.wallet
            direction = trade.direction
            if wallet not in wallet_dir_counts:
                wallet_dir_counts[wallet] = {"BULL": 0, "BEAR": 0}
            wallet_dir_counts[wallet][direction] += 1

        # Count wallets meeting the repeat threshold per direction
        repeat_bull = 0
        repeat_bear = 0
        for counts in wallet_dir_counts.values():
            if counts.get("BULL", 0) >= self.repeat_trade_min:
                repeat_bull += 1
            if counts.get("BEAR", 0) >= self.repeat_trade_min:
                repeat_bear += 1

        total_repeat = repeat_bull + repeat_bear
        if total_repeat == 0:
            return 0.0

        return (repeat_bull - repeat_bear) / total_repeat

    @staticmethod
    def _compute_wallet_asymmetry(
        bull_wallets: set[str],
        bear_wallets: set[str],
    ) -> float:
        """Ratio of unique bull wallets to bear wallets.

        More distinct wallets on one side = broader conviction.

        Returns [-1, 1].
        """
        n_bull = len(bull_wallets)
        n_bear = len(bear_wallets)
        total = n_bull + n_bear

        if total == 0:
            return 0.0

        return (n_bull - n_bear) / total


def _clamp(x: float) -> float:
    """Clamp to [-1.0, 1.0]."""
    return max(-1.0, min(1.0, x))
