"""Multi-bot side-by-side dashboard renderer.

Displays shared market data at the top and per-bot panels below,
using ANSI escape codes for in-place terminal rendering.
"""

import re
import sys
import time as _time
from datetime import datetime, timezone
from typing import Any

# ANSI codes
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
RESET = "\033[0m"
CLEAR_LINE = "\033[K"
CLEAR_SCREEN = "\033[2J\033[H"

# Regex to strip ANSI escape sequences for visible-width calculation
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def render_multi_dashboard(
    bot_states: list[dict[str, Any]],
    binance_price: float,
    oracle_price: float,
    coinbase_price: float,
    market_question: str,
    seconds_remaining: float,
    orderbook: Any,
    regime: str,
    regime_confidence: float,
    window_open: float = 0.0,
    window_high: float = 0.0,
    window_low: float = 0.0,
    ws_connected: bool = False,
    ws_updates: int = 0,
    ws_last_update: float = 0.0,
    window_open_source: str = "",
) -> None:
    """Render the multi-bot dashboard to the terminal.

    Args:
        bot_states: List of dicts from each bot's get_dashboard_state().
        binance_price: Current Binance BTC price.
        oracle_price: Current Chainlink oracle price.
        coinbase_price: Current Coinbase price.
        market_question: Active market question text.
        seconds_remaining: Seconds left in current window.
        orderbook: OrderbookSummary or None.
        regime: Current regime string.
        regime_confidence: Regime confidence 0-1.
        window_open: Oracle price at window open.
        window_high: Highest oracle price during this window.
        window_low: Lowest oracle price during this window.
    """
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    lines: list[str] = []

    # ── Header ──
    lines.append(
        f"{BOLD}{CYAN}  POLYMARKET MULTI-BOT RUNNER -- Paper Trading{RESET}"
    )

    # ── Prices ──
    oracle_lag_pct = 0.0
    if oracle_price > 0 and binance_price > 0:
        oracle_lag_pct = ((binance_price - oracle_price) / oracle_price) * 100

    lag_color = GREEN if oracle_lag_pct > 0 else RED if oracle_lag_pct < 0 else DIM
    price_line = (
        f"  {now}  "
        f"Oracle: ${oracle_price:,.2f}  "
        f"Binance: ${binance_price:,.2f} "
        f"({lag_color}{oracle_lag_pct:+.3f}%{RESET})"
    )
    if coinbase_price > 0:
        price_line += f"  CB: ${coinbase_price:,.2f}"

    # WS feed status
    if ws_connected:
        ws_age = _time.time() - ws_last_update if ws_last_update > 0 else 999
        ws_age_str = f"{ws_age:.1f}s ago" if ws_age < 60 else "stale"
        price_line += f"  {GREEN}WS:{RESET}{ws_updates} ({ws_age_str})"
    else:
        price_line += f"  {RED}WS:OFF{RESET}"
    lines.append(price_line)

    # ── Window OHLC ──
    if window_open > 0:
        change = oracle_price - window_open
        change_pct = (change / window_open) * 100 if window_open > 0 else 0.0
        cc = GREEN if change >= 0 else RED
        src_tag = ""
        if window_open_source == "polybacktest":
            src_tag = f"  {GREEN}[PBT]{RESET}"
        elif window_open_source == "exchange_avg":
            src_tag = f"  {YELLOW}[~avg]{RESET}"
        lines.append(
            f"  Window: O:${window_open:,.0f}  "
            f"H:{GREEN}${window_high:,.0f}{RESET}  "
            f"L:{RED}${window_low:,.0f}{RESET}  "
            f"{cc}{change:+,.0f} ({change_pct:+.3f}%){RESET}"
            f"{src_tag}"
        )

    # ── Market ──
    if market_question:
        mins = int(seconds_remaining) // 60
        secs = int(seconds_remaining) % 60
        ob_str = ""
        if orderbook:
            ob_str = (
                f"  YES {orderbook.yes_best_bid:.2f}/{orderbook.yes_best_ask:.2f}  "
                f"NO {orderbook.no_best_bid:.2f}/{orderbook.no_best_ask:.2f}  "
                f"Spr: {orderbook.spread:.3f}"
            )
        lines.append(
            f"  {MAGENTA}Market:{RESET} {market_question}  "
            f"{BOLD}{mins}:{secs:02d}{RESET} remaining{ob_str}"
        )
        if regime:
            rc = (
                GREEN if "trending" in regime
                else YELLOW if regime == "ranging"
                else RED if "vol" in regime
                else DIM
            )
            lines.append(
                f"  Regime: {rc}{regime}{RESET} "
                f"(conf={regime_confidence:.2f})"
            )
    else:
        lines.append(f"  {DIM}Waiting for market...{RESET}")

    lines.append("")

    # ── Bot panels ──
    if not bot_states:
        lines.append(f"  {DIM}No bots running{RESET}")
    else:
        # Build each bot's panel lines
        panels = [_build_bot_panel(state) for state in bot_states]
        max_panel_width = 42
        num_bots = len(panels)

        # Arrange in columns (3 per row for 3+ bots, 2 for 2)
        cols_per_row = 3 if num_bots >= 3 else 2
        for row_start in range(0, num_bots, cols_per_row):
            row_panels = panels[row_start:row_start + cols_per_row]
            max_lines = max(len(p) for p in row_panels)

            # Pad shorter panels
            for p in row_panels:
                while len(p) < max_lines:
                    p.append(" " * max_panel_width)

            # Merge columns (use visible-width padding, not ljust)
            for i in range(max_lines):
                parts = []
                for p in row_panels:
                    cell = p[i]
                    visible_len = len(_ANSI_RE.sub("", cell))
                    pad = max(0, max_panel_width - visible_len)
                    parts.append(cell + " " * pad)
                lines.append("  " + " ".join(parts))
            lines.append("")

    lines.append(f"  {DIM}Ctrl+C to stop{RESET}")

    # ── Render in-place ──
    buf = []
    for row_idx, line in enumerate(lines):
        buf.append(f"\033[{row_idx + 1};1H{line}{CLEAR_LINE}")
    # Clear leftover rows
    for row_idx in range(len(lines), 70):
        buf.append(f"\033[{row_idx + 1};1H{CLEAR_LINE}")
    sys.stdout.write("".join(buf))
    sys.stdout.flush()


def _build_bot_panel(state: dict) -> list[str]:
    """Build display lines for a single bot panel."""
    lines = []
    name = state.get("name", "???")
    BOX_W = 40

    # Header
    header = f" {name} "
    pad = BOX_W - 4 - len(header)
    lines.append(f"{CYAN}+--{header}{'-' * max(pad, 0)}+{RESET}")

    # Signals
    signals = state.get("signals", {})
    for layer_name, val in signals.items():
        bar = _signal_bar(val)
        c = GREEN if val > 0 else RED if val < 0 else DIM
        lines.append(f"{DIM}|{RESET} {layer_name:>9s}: {c}{val:+.3f}{RESET} {bar} {DIM}|{RESET}")

    # Combined
    combined = state.get("combined_signal", 0.0)
    prob_up = state.get("prob_up", 0.5)
    c = GREEN if combined > 0 else RED if combined < 0 else DIM
    lines.append(
        f"{DIM}|{RESET} Combined: {c}{combined:+.3f}{RESET}  "
        f"P(Up): {prob_up:.1%}   {DIM}|{RESET}"
    )

    # Kalman extras (if present)
    kf = state.get("kalman_extras")
    if kf:
        vel = kf.get("velocity", 0.0)
        brk = "BRK" if kf.get("breakout", False) else "   "
        vc = GREEN if vel > 0 else RED if vel < 0 else DIM
        lines.append(
            f"{DIM}|{RESET} KF vel: {vc}{vel:+.2f}{RESET}$/s  "
            f"{YELLOW}{brk}{RESET}            {DIM}|{RESET}"
        )

    # Entry decision
    entry = state.get("entry_decision", "")
    side = state.get("side", "")
    if side:
        lines.append(
            f"{DIM}|{RESET} {GREEN}SIGNAL: {side}{RESET} "
            f"{DIM}{entry[:22]}{RESET}  {DIM}|{RESET}"
        )
    else:
        entry_short = entry[:32] if entry else "Waiting"
        lines.append(f"{DIM}|{RESET} {DIM}{entry_short}{RESET}  {DIM}|{RESET}")

    # Stats
    bankroll = state.get("bankroll", 0.0)
    pnl = state.get("session_pnl", 0.0)
    wins = state.get("session_wins", 0)
    losses = state.get("session_losses", 0)
    wr = state.get("win_rate", 0.0)
    total = state.get("total_trades", 0)
    pnl_c = GREEN if pnl >= 0 else RED
    lines.append(
        f"{DIM}|{RESET} ${bankroll:.2f} "
        f"{pnl_c}{pnl:+.2f}{RESET} "
        f"{total}T {wins}W/{losses}L "
        f"{wr:.0%}  {DIM}|{RESET}"
    )

    # Equity curve (mini sparkline)
    curve = state.get("equity_curve", [])
    if len(curve) >= 2:
        sparkline = _mini_sparkline(curve[-20:], width=24)
        lines.append(f"{DIM}|{RESET} {sparkline}  {DIM}|{RESET}")

    # Pending trade
    pending = state.get("pending_trade")
    if pending:
        lines.append(
            f"{DIM}|{RESET} {YELLOW}PEND:{RESET} {pending['side']} "
            f"@ ${pending['entry_price']:.3f} "
            f"${pending['size_usdc']:.2f}  {DIM}|{RESET}"
        )

    # Recent trades (last 3)
    recent = state.get("recent_trades", [])[-3:]
    for t in recent:
        pnl_val = t.get("pnl", 0)
        won = pnl_val is not None and pnl_val > 0
        wc = GREEN if won else RED
        marker = "W" if won else "L"
        lines.append(
            f"{DIM}|{RESET} {wc}{marker}{RESET} "
            f"{t['side']:>3s} @ ${t['entry_price']:.3f} "
            f"{wc}${pnl_val:+.2f}{RESET}  {DIM}|{RESET}"
        )

    # Footer
    lines.append(f"{CYAN}+{'-' * (BOX_W - 2)}+{RESET}")

    return lines


def _signal_bar(val: float, width: int = 10) -> str:
    """ASCII bar chart for signal value [-1, 1]."""
    half = width // 2
    pos = int(abs(val) * half)
    pos = min(pos, half)

    if val >= 0:
        left = "." * half
        right = "+" * pos + "." * (half - pos)
    else:
        filled = "." * (half - pos) + "-" * pos
        left = filled
        right = "." * half

    return f"[{left}|{right}]"


def _mini_sparkline(data: list[float], width: int = 24) -> str:
    """Tiny ASCII sparkline for equity curve."""
    if not data:
        return " " * width

    chars = " _.,:-=!#@"
    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx > mn else 1.0

    result = []
    # Resample if needed
    step = max(1, len(data) // width)
    samples = data[::step][:width]

    for v in samples:
        idx = int((v - mn) / rng * (len(chars) - 1))
        idx = max(0, min(idx, len(chars) - 1))
        result.append(chars[idx])

    line = "".join(result)

    # Color based on trend
    c = GREEN if data[-1] >= data[0] else RED
    return f"{c}{line}{RESET}"
