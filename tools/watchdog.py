"""Heartbeat watchdog — auto-fires the kill action when the bot goes dark.

Runs as a SEPARATE OS-supervised process (Windows Scheduled Task, step 6),
independent of the trading process so it can act precisely when the bot is the
thing that is broken (ADR-0002 §4).

Behaviour:
- Poll ``heartbeat.json`` every ``poll_secs``.
- **Arm only after seeing one valid heartbeat** — never fire on a just-booting
  bot that hasn't written a heartbeat yet.
- Once armed, a heartbeat older than ``staleness_secs`` increments a stale
  counter; a fresh one resets it. Fire after ``stale_checks_to_fire`` consecutive
  stale polls (~``staleness`` + ``checks``×``poll`` worst-case detection).
- **Fail-safe:** once armed, a missing / unreadable / unparseable heartbeat
  counts as stale (pausing trading beats running blind).
- On fire, call the kill action once (same code path as a manual kill).

``evaluate(now)`` is the pure state machine (unit-tested with a fake clock and
fake heartbeat contents); ``run()`` is the polling loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from core.kill_switch_io import HEARTBEAT_PATH, read_heartbeat

logger = logging.getLogger(__name__)


class Watchdog:
    """Monitors the bot heartbeat and fires ``on_fire`` when it goes stale."""

    def __init__(
        self,
        *,
        on_fire: Callable[[], Awaitable[None]],
        heartbeat_path: Path = HEARTBEAT_PATH,
        poll_secs: float = 2.0,
        staleness_secs: float = 20.0,
        stale_checks_to_fire: int = 3,
        clock: Callable[[], float] = time.time,
    ):
        self.on_fire = on_fire
        self.heartbeat_path = Path(heartbeat_path)
        self.poll_secs = poll_secs
        self.staleness_secs = staleness_secs
        self.stale_checks_to_fire = stale_checks_to_fire
        self.clock = clock

        self.armed = False
        self.stale_count = 0
        self.fired = False

    def evaluate(self, now: float) -> str:
        """One poll's decision. Returns 'waiting'|'armed'|'fresh'|'stale'|'fire'.

        Updates ``armed``/``stale_count`` as a side effect. Does NOT fire — the
        caller fires when this returns 'fire'.
        """
        hb = read_heartbeat(path=self.heartbeat_path)
        valid = isinstance(hb, dict) and isinstance(hb.get("ts"), (int, float))

        if not self.armed:
            if valid:
                self.armed = True
                logger.info("Watchdog armed (first heartbeat seen).")
                return "armed"
            return "waiting"

        # Armed.
        if not valid:
            self.stale_count += 1  # fail-safe: missing/corrupt == stale
            logger.warning(
                "Watchdog: heartbeat missing/unreadable (stale %d/%d).",
                self.stale_count, self.stale_checks_to_fire,
            )
        else:
            age = now - hb["ts"]
            if age > self.staleness_secs:
                self.stale_count += 1
                logger.warning(
                    "Watchdog: heartbeat stale %.1fs > %.1fs (%d/%d).",
                    age, self.staleness_secs, self.stale_count, self.stale_checks_to_fire,
                )
            else:
                self.stale_count = 0
                return "fresh"

        if self.stale_count >= self.stale_checks_to_fire:
            return "fire"
        return "stale"

    async def run(self, max_polls: int | None = None) -> None:
        """Poll until the kill fires (or ``max_polls`` reached, for tests)."""
        logger.info(
            "Watchdog starting: poll=%.1fs staleness=%.1fs fire-after=%d "
            "heartbeat=%s",
            self.poll_secs, self.staleness_secs, self.stale_checks_to_fire,
            self.heartbeat_path,
        )
        polls = 0
        while not self.fired:
            status = self.evaluate(self.clock())
            if status == "fire":
                self.fired = True
                logger.critical(
                    "Watchdog FIRING kill action: %d consecutive stale checks.",
                    self.stale_count,
                )
                await self.on_fire()
                break
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            await asyncio.sleep(self.poll_secs)


async def _main() -> None:
    """Build a real client and run the watchdog (OS-supervised invocation)."""
    from core.config import Config
    from core.polymarket_client import PolymarketClient
    from tools.kill_switch import run_kill

    config = Config.load()
    poly = PolymarketClient(config)
    await poly.connect()

    async def on_fire() -> None:
        await run_kill(
            trigger="watchdog",
            reason="heartbeat stale -- bot presumed hung/dead",
            poly=poly,
            guard_secs=config.ks_flatten_guard_secs,
            max_retries=config.ks_flatten_max_retries,
        )

    wd = Watchdog(
        on_fire=on_fire,
        poll_secs=config.ks_watchdog_poll_secs,
        staleness_secs=config.ks_staleness_secs,
        stale_checks_to_fire=config.ks_stale_checks_to_fire,
    )
    await wd.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(_main())
