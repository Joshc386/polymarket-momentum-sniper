"""Per-tick signal diagnostic logger.

Captures every signal layer value and the bot's entry decision every
tick (not just on actual trades) for offline analysis. Intended to
diagnose situations like the 100% YES bias seen in May 11+ Bot G data,
where the snapshot audit shows balanced signals across all windows but
the bot only enters on one side.

Schema captures:
- All L1-L12 signal values
- L4 sub-components (imbalance, flow, mid_dev, top_pressure, thickness)
- L1 sub-components (lag_component, open_component)
- Combined signal + est_prob_up + market_implied_prob
- Entry decision + filter outcome
- Market context (btc, oracle, regime, secs_remaining)

Storage is per-bot SQLite at data_runtime/<bot>_signal_diag.db.
Disabled by default; enable via config `signal_diagnostic: enabled: true`.
"""

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    unix_time REAL NOT NULL,
    bot_name TEXT,
    market_id TEXT,
    market_slug TEXT,
    secs_remaining REAL,
    secs_into_window REAL,

    btc_price REAL,
    oracle_price REAL,
    oracle_open_price REAL,
    coinbase_price REAL,

    regime TEXT,
    schedule_override TEXT,

    -- Core L1-L5 signal values
    l1_oracle_lag REAL,
    l2_momentum REAL,
    l3_liquidation REAL,
    l4_orderbook REAL,
    l5_sentiment REAL,

    -- L1 sub-components (the suspected bug source)
    l1_lag_component REAL,   -- (binance - oracle_now) component, 60 percent of L1
    l1_open_component REAL,  -- (binance - oracle_open) component, 40 percent of L1

    -- L4 sub-components (the other suspected bug source)
    l4_imbalance REAL,
    l4_flow REAL,
    l4_mid_dev REAL,
    l4_top_pressure REAL,
    l4_thickness REAL,

    -- Additive modifier signals
    l6_fade REAL,
    l7_taker_ratio REAL,
    l8_clob_flow REAL,
    l9b_absorption REAL,
    l10_exhaustion REAL,
    l11_trade_size REAL,
    l12_wallet_flow REAL,

    -- Cross-exchange and combination
    coinbase_direction REAL,
    combined_signal REAL,
    est_prob_up REAL,
    market_implied_prob REAL,
    prob_edge REAL,
    required_edge REAL,

    -- Weights used
    w_oracle REAL, w_momentum REAL, w_liquidation REAL,
    w_orderbook REAL, w_sentiment REAL,

    -- What the bot would pick (entry side)
    would_pick_side TEXT,
    would_enter INTEGER,
    entry_reason TEXT,

    -- Filter / risk gating
    filter_blocked TEXT,
    risk_can_trade INTEGER,
    risk_reason TEXT,

    -- Actual outcome (1 if bot placed a trade this tick)
    trade_placed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_signal_ticks_unix ON signal_ticks(unix_time);
CREATE INDEX IF NOT EXISTS idx_signal_ticks_market ON signal_ticks(market_id);
"""


@dataclass
class SignalTick:
    """One tick of signal state. All fields optional for flexible logging."""
    timestamp: str = ""
    unix_time: float = 0.0
    bot_name: str = ""
    market_id: str = ""
    market_slug: str = ""
    secs_remaining: float = 0.0
    secs_into_window: float = 0.0

    btc_price: float = 0.0
    oracle_price: float = 0.0
    oracle_open_price: float = 0.0
    coinbase_price: float = 0.0

    regime: str = ""
    schedule_override: str = ""

    l1_oracle_lag: float = 0.0
    l2_momentum: float = 0.0
    l3_liquidation: float = 0.0
    l4_orderbook: float = 0.0
    l5_sentiment: float = 0.0

    l1_lag_component: float = 0.0
    l1_open_component: float = 0.0

    l4_imbalance: float = 0.0
    l4_flow: float = 0.0
    l4_mid_dev: float = 0.0
    l4_top_pressure: float = 0.0
    l4_thickness: float = 0.0

    l6_fade: float = 0.0
    l7_taker_ratio: float = 0.0
    l8_clob_flow: float = 0.0
    l9b_absorption: float = 0.0
    l10_exhaustion: float = 0.0
    l11_trade_size: float = 0.0
    l12_wallet_flow: float = 0.0

    coinbase_direction: float = 0.0
    combined_signal: float = 0.0
    est_prob_up: float = 0.5
    market_implied_prob: float = 0.5
    prob_edge: float = 0.0
    required_edge: float = 0.0

    w_oracle: float = 0.0
    w_momentum: float = 0.0
    w_liquidation: float = 0.0
    w_orderbook: float = 0.0
    w_sentiment: float = 0.0

    would_pick_side: str = ""
    would_enter: int = 0
    entry_reason: str = ""

    filter_blocked: str = ""
    risk_can_trade: int = 1
    risk_reason: str = ""

    trade_placed: int = 0


class SignalDiagnosticLogger:
    """SQLite per-tick signal logger.

    Use:
        logger = SignalDiagnosticLogger("data_runtime/bot_g_signal_diag.db",
                                        bot_name="bot_g_signal_aligned",
                                        enabled=True,
                                        sample_every_n=1)
        logger.log(SignalTick(...))
        ...
        logger.close()

    Set `sample_every_n=5` to log only every 5th tick (cheaper if needed).
    """

    def __init__(
        self,
        db_path: str,
        bot_name: str = "",
        enabled: bool = True,
        sample_every_n: int = 1,
    ):
        self.enabled = enabled
        self.bot_name = bot_name
        self.sample_every_n = max(1, int(sample_every_n))
        self._tick_counter = 0
        self._conn: Optional[sqlite3.Connection] = None

        if not self.enabled:
            return

        # Ensure parent dir exists
        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            logger.info(
                "Signal diagnostic logger enabled for %s -> %s "
                "(sample_every_n=%d)",
                bot_name, db_path, self.sample_every_n,
            )
        except Exception as e:
            logger.error("Failed to init signal diagnostic DB at %s: %s",
                         db_path, e)
            self.enabled = False
            self._conn = None

    def log(self, tick: SignalTick) -> None:
        """Log a tick. No-op if disabled. Best-effort: swallows exceptions."""
        if not self.enabled or self._conn is None:
            return

        self._tick_counter += 1
        if self._tick_counter % self.sample_every_n != 0:
            return

        # Default bot_name from constructor if not provided
        if not tick.bot_name:
            tick.bot_name = self.bot_name

        # Auto-fill timestamp if missing
        if not tick.unix_time:
            tick.unix_time = time.time()
        if not tick.timestamp:
            from datetime import datetime, timezone
            tick.timestamp = datetime.fromtimestamp(
                tick.unix_time, tz=timezone.utc
            ).isoformat()

        try:
            self._conn.execute(
                """
                INSERT INTO signal_ticks (
                    timestamp, unix_time, bot_name, market_id, market_slug,
                    secs_remaining, secs_into_window,
                    btc_price, oracle_price, oracle_open_price, coinbase_price,
                    regime, schedule_override,
                    l1_oracle_lag, l2_momentum, l3_liquidation, l4_orderbook, l5_sentiment,
                    l1_lag_component, l1_open_component,
                    l4_imbalance, l4_flow, l4_mid_dev, l4_top_pressure, l4_thickness,
                    l6_fade, l7_taker_ratio, l8_clob_flow,
                    l9b_absorption, l10_exhaustion, l11_trade_size, l12_wallet_flow,
                    coinbase_direction, combined_signal,
                    est_prob_up, market_implied_prob, prob_edge, required_edge,
                    w_oracle, w_momentum, w_liquidation, w_orderbook, w_sentiment,
                    would_pick_side, would_enter, entry_reason,
                    filter_blocked, risk_can_trade, risk_reason, trade_placed
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    tick.timestamp, tick.unix_time, tick.bot_name,
                    tick.market_id, tick.market_slug,
                    tick.secs_remaining, tick.secs_into_window,
                    tick.btc_price, tick.oracle_price,
                    tick.oracle_open_price, tick.coinbase_price,
                    tick.regime, tick.schedule_override,
                    tick.l1_oracle_lag, tick.l2_momentum, tick.l3_liquidation,
                    tick.l4_orderbook, tick.l5_sentiment,
                    tick.l1_lag_component, tick.l1_open_component,
                    tick.l4_imbalance, tick.l4_flow, tick.l4_mid_dev,
                    tick.l4_top_pressure, tick.l4_thickness,
                    tick.l6_fade, tick.l7_taker_ratio, tick.l8_clob_flow,
                    tick.l9b_absorption, tick.l10_exhaustion,
                    tick.l11_trade_size, tick.l12_wallet_flow,
                    tick.coinbase_direction, tick.combined_signal,
                    tick.est_prob_up, tick.market_implied_prob,
                    tick.prob_edge, tick.required_edge,
                    tick.w_oracle, tick.w_momentum, tick.w_liquidation,
                    tick.w_orderbook, tick.w_sentiment,
                    tick.would_pick_side, tick.would_enter, tick.entry_reason,
                    tick.filter_blocked, tick.risk_can_trade,
                    tick.risk_reason, tick.trade_placed,
                ),
            )
            # Commit every 50 rows for crash safety without too much I/O
            if self._tick_counter % 50 == 0:
                self._conn.commit()
        except Exception as e:
            logger.debug("Signal diag log write failed: %s", e)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
            self._conn = None
