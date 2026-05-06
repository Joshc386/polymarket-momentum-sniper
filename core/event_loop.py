"""Event loop bootstrap — selects the fastest available asyncio loop.

Priority:
1. uvloop (Linux/macOS) — 2-4x faster than default asyncio
2. winloop (Windows) — Windows port of uvloop with similar perf gains
3. asyncio default — fallback if neither is available

Call setup_event_loop() at the very start of any async entry point
(before asyncio.run / asyncio.get_event_loop).
"""

import logging
import sys

logger = logging.getLogger(__name__)

_installed_loop: str = ""


def setup_event_loop() -> str:
    """Install the fastest available asyncio loop. Idempotent.

    Returns:
        Name of the installed loop ("uvloop", "winloop", or "asyncio").
    """
    global _installed_loop
    if _installed_loop:
        return _installed_loop

    # Try uvloop first (Linux/macOS — used in production VPS)
    try:
        import uvloop
        uvloop.install()
        _installed_loop = "uvloop"
        logger.info("Event loop: uvloop installed")
        return _installed_loop
    except ImportError:
        pass

    # Try winloop (Windows — used in dev)
    try:
        import winloop
        winloop.install()
        _installed_loop = "winloop"
        logger.info("Event loop: winloop installed")
        return _installed_loop
    except ImportError:
        pass

    _installed_loop = "asyncio"
    logger.warning(
        "Event loop: using stdlib asyncio (slower). "
        "Install uvloop (Linux/macOS) or winloop (Windows) for ~2-4x perf."
    )
    return _installed_loop


def get_installed_loop() -> str:
    """Return the name of the currently-installed loop."""
    return _installed_loop or "asyncio"
