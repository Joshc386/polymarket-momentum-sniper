"""BotStrategy protocol — the interface every bot must implement.

The multi-bot runner calls these methods. Each bot owns its entire
signal-to-trade pipeline internally.
"""

from __future__ import annotations

from typing import Protocol, Any

from core.data_snapshot import DataSnapshot


class BotStrategy(Protocol):
    """Interface for a trading bot strategy.

    The runner handles shared concerns (data feeds, window detection,
    resolution). Each bot handles independent concerns (signals, entry,
    sizing, risk, execution, P&L tracking).
    """

    name: str

    def on_new_window(self, snapshot: DataSnapshot) -> None:
        """Called when a new 5-min window starts.

        Reset per-window state (position flag, signal history, etc.).
        """
        ...

    def on_tick(self, snapshot: DataSnapshot) -> None:
        """Called every tick (~1s). Compute signals, evaluate entry, execute.

        The snapshot contains all current market data. The bot should:
        1. Compute its signal layers from snapshot data
        2. Combine signals into a probability estimate
        3. Evaluate entry conditions
        4. Execute a trade if conditions are met
        5. Update internal state for dashboard display
        """
        ...

    def on_window_end(self, resolution: str) -> None:
        """Called when the window resolves ('UP' or 'DOWN').

        Resolve any pending trade and record the outcome.
        """
        ...

    def get_dashboard_state(self) -> dict[str, Any]:
        """Return current state for the multi-bot dashboard.

        Expected keys:
            name: str — bot display name
            signals: dict — {layer_name: float} signal values
            combined_signal: float — raw combined signal
            prob_up: float — estimated P(UP)
            entry_decision: str — current entry status/reason
            side: str — 'YES'/'NO'/'' if pending
            bankroll: float
            session_pnl: float
            session_wins: int
            session_losses: int
            win_rate: float
            total_trades: int
            equity_curve: list[float] — bankroll history
            pending_trade: dict | None — active trade info
            recent_trades: list[dict] — last N resolved trades
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources (close DB connections, etc.)."""
        ...
