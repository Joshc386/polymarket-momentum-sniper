"""File-based signalling channel between the trading process and the kill switch.

The kill switch runs as a SEPARATE OS process (ADR-0002), so the bot and the
watchdog/kill-switch communicate through plain files in ``data_runtime/`` — the
most robust primitive when one side may be wedged (no socket to refuse, no lock
to deadlock on, atomic via ``os.replace``, human-inspectable mid-incident).

This module is the single source of truth for those two file formats:

- ``heartbeat.json`` — ``{ts, window_end_ts, token_ids[]}``, rewritten atomically
  by the bot every loop iteration. The watchdog reads ``ts`` to detect staleness
  and ``window_end_ts``/``token_ids`` to drive the flatten guard.
- ``HALT`` — a sticky flag. Its presence means trading is disabled. Written by
  the kill switch; the bot checks it before every order and exits gracefully.
  Removed only by a human.

All writes are atomic (temp file + ``os.replace``). All reads fail safe.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_RUNTIME = Path("data_runtime")
HEARTBEAT_PATH = DATA_RUNTIME / "heartbeat.json"
HALT_PATH = DATA_RUNTIME / "HALT"


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic on the same filesystem


def write_heartbeat(
    window_end_ts: float | None,
    token_ids: list[str],
    *,
    ts: float | None = None,
    path: Path = HEARTBEAT_PATH,
) -> dict:
    """Atomically write the bot's heartbeat. Returns the payload written.

    ``ts`` defaults to the current wall-clock; pass it explicitly in tests.
    Empty/falsy token ids are dropped so the watchdog never matches on ``""``.
    """
    payload = {
        "ts": time.time() if ts is None else ts,
        "window_end_ts": window_end_ts,
        "token_ids": [t for t in token_ids if t],
    }
    _atomic_write(Path(path), json.dumps(payload))
    return payload


def read_heartbeat(path: Path = HEARTBEAT_PATH) -> dict | None:
    """Read the heartbeat, or ``None`` if missing/unreadable/unparseable.

    Fail-safe by design: the watchdog treats ``None`` as stale (pause beats
    running blind), so a corrupt or absent file must never raise.
    """
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return None


def halt_active(path: Path = HALT_PATH) -> bool:
    """True if the sticky HALT flag is present (trading disabled)."""
    return Path(path).exists()


def write_halt(
    source: str,
    reason: str,
    *,
    ts: float | None = None,
    path: Path = HALT_PATH,
) -> dict:
    """Atomically write the sticky HALT flag. Returns the record written.

    ``source`` is ``"manual"`` or ``"watchdog"``; ``reason`` is a short
    human-readable explanation. The flag is removed only by a human
    (:func:`clear_halt` exists for tests / explicit re-enable tooling).
    """
    record = {
        "ts": time.time() if ts is None else ts,
        "source": source,
        "reason": reason,
    }
    _atomic_write(Path(path), json.dumps(record))
    return record


def clear_halt(path: Path = HALT_PATH) -> None:
    """Remove the HALT flag. Idempotent (no error if already absent)."""
    Path(path).unlink(missing_ok=True)
