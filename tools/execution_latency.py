"""Execution latency tracker — measures end-to-end time for trade actions.

Captures the full chain from "data arrived" to "order acknowledged":

  T0  exchange_msg_time   — exchange-side WS timestamp (when they sent it)
  T1  msg_received_time   — when we received the WS message on our machine
  T2  decision_time       — when our strategy decided to enter
  T3  order_submitted     — when we sent the order to Polymarket
  T4  order_acknowledged  — when Polymarket confirmed the order

Latencies derived:
  L1  network_in   = T1 - T0   (exchange → us, network RTT one-way)
  L2  processing   = T2 - T1   (Python work between WS msg and decision)
  L3  signing      = T3 - T2   (order construction + signing time)
  L4  submission   = T4 - T3   (us → Polymarket → ack, network RTT)
  Lt  total        = T4 - T0   (full reaction time from exchange to fill)

Usage from a strategy:
    tracker = ExecutionLatencyTracker(output_dir="data_runtime")
    # When data arrives:
    tracker.start_trade(trade_id, exchange_msg_time, msg_received_time)
    # When decision made:
    tracker.mark_decision(trade_id)
    # Just before order submit:
    tracker.mark_submitted(trade_id)
    # When acknowledged:
    tracker.mark_acknowledged(trade_id)
    # Or on timeout/cancel:
    tracker.cancel(trade_id, reason="loss_cut")
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeLatencyRecord:
    trade_id: str
    primary_exchange: str            # "coinbase" or "binance"
    exchange_msg_time: float         # T0 — exchange-side
    msg_received_time: float         # T1 — our machine
    decision_time: Optional[float] = None      # T2
    order_submitted: Optional[float] = None    # T3
    order_acknowledged: Optional[float] = None # T4
    cancelled: bool = False
    cancel_reason: str = ""
    # Optional context
    side: str = ""
    move_bps: float = 0.0
    clob_gap_bps: float = 0.0


class ExecutionLatencyTracker:
    """Per-trade latency timeline tracking. CSV output for analysis."""

    def __init__(self, output_dir: str = "data_runtime") -> None:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(output_dir, f"execution_latency_{ts}.csv")
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "trade_id", "primary_exchange",
            "exchange_msg_time", "msg_received_time",
            "decision_time", "order_submitted", "order_acknowledged",
            "L1_network_in_ms", "L2_processing_ms",
            "L3_signing_ms", "L4_submission_ms", "Lt_total_ms",
            "side", "move_bps", "clob_gap_bps",
            "cancelled", "cancel_reason",
        ])
        self._active: dict[str, TradeLatencyRecord] = {}
        self._completed_count = 0
        logger.info(f"ExecutionLatencyTracker → {self._csv_path}")

    def start_trade(
        self,
        trade_id: str,
        exchange_msg_time: float,
        msg_received_time: float,
        primary_exchange: str = "coinbase",
        side: str = "",
        move_bps: float = 0.0,
        clob_gap_bps: float = 0.0,
    ) -> None:
        """Begin tracking a trade. Called when the triggering data arrived."""
        self._active[trade_id] = TradeLatencyRecord(
            trade_id=trade_id,
            primary_exchange=primary_exchange,
            exchange_msg_time=exchange_msg_time,
            msg_received_time=msg_received_time,
            side=side,
            move_bps=move_bps,
            clob_gap_bps=clob_gap_bps,
        )

    def mark_decision(self, trade_id: str) -> None:
        """Mark when the strategy decided to enter."""
        if trade_id in self._active:
            self._active[trade_id].decision_time = time.time()

    def mark_submitted(self, trade_id: str) -> None:
        """Mark when the order was sent to Polymarket."""
        if trade_id in self._active:
            self._active[trade_id].order_submitted = time.time()

    def mark_acknowledged(self, trade_id: str) -> None:
        """Mark when Polymarket acknowledged the order. Writes CSV row."""
        if trade_id not in self._active:
            return
        rec = self._active.pop(trade_id)
        rec.order_acknowledged = time.time()
        self._write_row(rec)

    def cancel(self, trade_id: str, reason: str = "") -> None:
        """Trade was cancelled (e.g., risk gate, paper-trade direct fill)."""
        if trade_id not in self._active:
            return
        rec = self._active.pop(trade_id)
        rec.cancelled = True
        rec.cancel_reason = reason
        self._write_row(rec)

    def _write_row(self, rec: TradeLatencyRecord) -> None:
        def ms(a: Optional[float], b: Optional[float]) -> str:
            if a is None or b is None or a <= 0 or b <= 0:
                return ""
            return f"{(b - a) * 1000:.1f}"

        self._writer.writerow([
            rec.trade_id, rec.primary_exchange,
            f"{rec.exchange_msg_time:.3f}",
            f"{rec.msg_received_time:.3f}",
            f"{rec.decision_time:.3f}" if rec.decision_time else "",
            f"{rec.order_submitted:.3f}" if rec.order_submitted else "",
            f"{rec.order_acknowledged:.3f}" if rec.order_acknowledged else "",
            ms(rec.exchange_msg_time, rec.msg_received_time),  # L1
            ms(rec.msg_received_time, rec.decision_time),      # L2
            ms(rec.decision_time, rec.order_submitted),        # L3
            ms(rec.order_submitted, rec.order_acknowledged),   # L4
            ms(rec.exchange_msg_time, rec.order_acknowledged), # Lt
            rec.side, f"{rec.move_bps:.2f}", f"{rec.clob_gap_bps:.2f}",
            "Y" if rec.cancelled else "N", rec.cancel_reason,
        ])
        self._file.flush()
        self._completed_count += 1

    def close(self) -> None:
        try:
            # Flush any incomplete records as cancelled
            for trade_id in list(self._active.keys()):
                self.cancel(trade_id, "shutdown")
            self._file.close()
        except Exception:
            pass
        logger.info(
            f"ExecutionLatencyTracker closed — "
            f"{self._completed_count} records written"
        )
