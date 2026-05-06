"""Agent Scheduler — runs agents on schedules and file-change triggers.

Scheduled (daily):
    - data_pipeline status + validate
    - regime_researcher analyse
    - reconciliation compare

Triggered (file change):
    - strategy_prototyper compare  — when new .py in strategies/
    - signal_lab scan              — when new .py in signals/hypotheses/
    - data_pipeline unify          — when data files in backtest/data/ are modified

Usage:
    python -m agents.scheduler                    # Run in foreground
    python -m agents.scheduler --run-now          # Run all daily tasks immediately, then schedule
    python -m agents.scheduler --daily-hour 6     # Set daily run time (UTC, default: 6)
    python -m agents.scheduler --poll-interval 60 # File poll interval in seconds (default: 60)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging to both console and file
LOG_DIR = PROJECT_ROOT / "data_runtime"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_PATH), encoding="utf-8"),
    ],
)
logger = logging.getLogger("agents.scheduler")

# Suppress noisy loggers
logging.getLogger("apscheduler").setLevel(logging.WARNING)

# ANSI
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


# ─── Task Definitions ──────────────────────────────────────────────

def run_agent_command(module: str, command: str, description: str) -> bool:
    """Run an agent CLI command as a subprocess.

    Args:
        module: Python module path (e.g. 'agents.data_pipeline').
        command: Subcommand (e.g. 'status').
        description: Human-readable description for logging.

    Returns:
        True if the command succeeded, False otherwise.
    """
    cmd = [sys.executable, "-m", module, command]
    logger.info(f"Running: {description} ({' '.join(cmd)})")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per task
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode == 0:
            logger.info(f"Completed: {description}")
            # Log abbreviated output
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[:20]:
                logger.debug(f"  {line}")
            if len(output_lines) > 20:
                logger.debug(f"  ... ({len(output_lines) - 20} more lines)")
            return True
        else:
            logger.error(f"Failed: {description} (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:10]:
                    logger.error(f"  {line}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {description} (exceeded 300s)")
        return False
    except Exception as e:
        logger.error(f"Error running {description}: {e}")
        return False


def daily_tasks() -> None:
    """Run all daily scheduled tasks."""
    logger.info(f"{BOLD}{CYAN}Starting daily agent tasks{RESET}")
    start = time.time()

    tasks = [
        ("agents.data_pipeline", "status", "Data Pipeline: status check"),
        ("agents.data_pipeline", "validate", "Data Pipeline: validation"),
        ("agents.regime_researcher", "analyse", "Regime Researcher: analyse"),
        ("agents.reconciliation", "compare", "Reconciliation: compare"),
    ]

    results = []
    for module, command, description in tasks:
        success = run_agent_command(module, command, description)
        results.append((description, success))

    elapsed = time.time() - start
    passed = sum(1 for _, s in results if s)
    failed = len(results) - passed

    logger.info(
        f"{BOLD}Daily tasks complete:{RESET} "
        f"{GREEN}{passed} passed{RESET}, "
        f"{RED}{failed} failed{RESET} "
        f"({elapsed:.1f}s)"
    )

    for desc, success in results:
        status = f"{GREEN}OK{RESET}" if success else f"{RED}FAIL{RESET}"
        logger.info(f"  [{status}] {desc}")


# ─── File Watcher ───────────────────────────────────────────────────

class FileWatcher:
    """Polls directories for new or modified .py files and triggers agent runs.

    Avoids the watchdog dependency by using simple mtime polling.
    """

    def __init__(self) -> None:
        self._watched: dict[str, dict[str, float]] = {}
        # track: dir_path -> {filename: mtime}

        self._triggers: list[dict] = [
            {
                "directory": str(PROJECT_ROOT / "strategies"),
                "pattern": "*.py",
                "ignore": ["__init__.py"],
                "module": "agents.strategy_prototyper",
                "command": "compare",
                "args": "--strategies",
                "description": "Strategy Prototyper: compare (new strategy detected)",
                "needs_dynamic_args": True,
            },
            {
                "directory": str(PROJECT_ROOT / "signals" / "hypotheses"),
                "pattern": "*.py",
                "ignore": ["__init__.py"],
                "module": "agents.signal_lab",
                "command": "scan",
                "description": "Signal Lab: scan (new hypothesis detected)",
                "needs_dynamic_args": False,
            },
            {
                "directory": str(PROJECT_ROOT / "backtest" / "data"),
                "pattern": "*.csv",
                "ignore": ["regime_timeline.csv", "prototype_*.csv"],
                "module": "agents.data_pipeline",
                "command": "unify",
                "description": "Data Pipeline: unify (data files changed)",
                "needs_dynamic_args": False,
                "cooldown_secs": 300,  # Don't re-trigger within 5 min
            },
        ]

        self._last_triggered: dict[str, float] = {}

        # Take initial snapshot of all watched directories
        for trigger in self._triggers:
            self._snapshot_dir(trigger["directory"], trigger["pattern"], trigger["ignore"])

    def _snapshot_dir(self, directory: str, pattern: str, ignore: list[str]) -> dict[str, float]:
        """Take a snapshot of file mtimes in a directory."""
        snapshot: dict[str, float] = {}
        dir_path = Path(directory)
        if not dir_path.exists():
            return snapshot

        for f in dir_path.glob(pattern):
            if f.name in ignore:
                continue
            # Check ignore patterns with wildcards
            skip = False
            for ign in ignore:
                if "*" in ign and f.match(ign):
                    skip = True
                    break
            if skip:
                continue
            snapshot[f.name] = f.stat().st_mtime

        self._watched[directory] = snapshot
        return snapshot

    def check(self) -> list[dict]:
        """Check all watched directories for changes.

        Returns:
            List of triggers that should fire.
        """
        fired: list[dict] = []
        now = time.time()

        for trigger in self._triggers:
            directory = trigger["directory"]
            pattern = trigger["pattern"]
            ignore = trigger["ignore"]
            cooldown = trigger.get("cooldown_secs", 60)

            # Check cooldown
            last = self._last_triggered.get(directory, 0)
            if now - last < cooldown:
                continue

            old_snapshot = self._watched.get(directory, {})
            new_snapshot = self._snapshot_dir(directory, pattern, ignore)

            # Detect new or modified files
            changes: list[str] = []
            for filename, mtime in new_snapshot.items():
                if filename not in old_snapshot:
                    changes.append(f"new: {filename}")
                elif mtime > old_snapshot[filename]:
                    changes.append(f"modified: {filename}")

            if changes:
                logger.info(
                    f"{YELLOW}File changes detected in {directory}:{RESET} "
                    f"{', '.join(changes)}"
                )
                fired.append(trigger)
                self._last_triggered[directory] = now

        return fired

    def handle_triggers(self, triggers: list[dict]) -> None:
        """Execute the agent commands for fired triggers."""
        for trigger in triggers:
            module = trigger["module"]
            command = trigger["command"]
            description = trigger["description"]

            if trigger.get("needs_dynamic_args"):
                # For strategy compare, build the strategy list dynamically
                strat_dir = Path(trigger["directory"])
                strat_files = [
                    f.stem for f in strat_dir.glob("*.py")
                    if f.name != "__init__.py"
                ]
                if strat_files:
                    # Also include built-in strategies
                    all_strats = list(set(
                        ["contrarian_ev", "signal_follow", "orderbook_fade"] + strat_files
                    ))
                    strat_list = ",".join(all_strats)
                    cmd = [
                        sys.executable, "-m", module, command,
                        trigger["args"], strat_list,
                    ]
                    logger.info(f"Running: {description} ({' '.join(cmd)})")
                    try:
                        subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            capture_output=True,
                            text=True,
                            timeout=600,
                            encoding="utf-8",
                            errors="replace",
                        )
                        logger.info(f"Completed: {description}")
                    except Exception as e:
                        logger.error(f"Error: {description}: {e}")
                else:
                    logger.info(f"Skipped: {description} (no user strategies found)")
            else:
                run_agent_command(module, command, description)


# ─── Main Loop ──────────────────────────────────────────────────────

def run_scheduler(daily_hour: int = 6, poll_interval: int = 60, run_now: bool = False) -> None:
    """Main scheduler loop.

    Args:
        daily_hour: UTC hour to run daily tasks (0-23).
        poll_interval: Seconds between file change polls.
        run_now: If True, run daily tasks immediately on startup.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    logger.info(f"{BOLD}{CYAN}Agent Scheduler starting{RESET}")
    logger.info(f"  Daily tasks at: {daily_hour:02d}:00 UTC")
    logger.info(f"  File poll interval: {poll_interval}s")
    logger.info(f"  Log file: {LOG_PATH}")
    logger.info(f"  Watched directories:")
    logger.info(f"    - strategies/ (triggers: strategy_prototyper compare)")
    logger.info(f"    - signals/hypotheses/ (triggers: signal_lab scan)")
    logger.info(f"    - backtest/data/*.csv (triggers: data_pipeline unify)")

    # Init file watcher
    watcher = FileWatcher()

    # Run daily tasks immediately if requested
    if run_now:
        logger.info(f"{YELLOW}--run-now: executing daily tasks immediately{RESET}")
        daily_tasks()

    # Set up APScheduler
    scheduler = BlockingScheduler()

    # Daily tasks at the configured hour
    scheduler.add_job(
        daily_tasks,
        CronTrigger(hour=daily_hour, minute=0, timezone="UTC"),
        id="daily_tasks",
        name="Daily agent tasks",
    )

    # File watcher poll
    def poll_files() -> None:
        triggers = watcher.check()
        if triggers:
            watcher.handle_triggers(triggers)

    scheduler.add_job(
        poll_files,
        "interval",
        seconds=poll_interval,
        id="file_watcher",
        name="File change watcher",
    )

    logger.info(f"{GREEN}Scheduler running. Ctrl+C to stop.{RESET}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m agents.scheduler",
        description="Agent Scheduler — runs agents on schedules and file-change triggers",
    )
    parser.add_argument(
        "--daily-hour",
        type=int,
        default=6,
        help="UTC hour for daily tasks (0-23, default: 6)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="File poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run all daily tasks immediately on startup, then continue scheduling",
    )

    args = parser.parse_args()

    if args.daily_hour < 0 or args.daily_hour > 23:
        logger.error("--daily-hour must be 0-23")
        sys.exit(1)

    run_scheduler(
        daily_hour=args.daily_hour,
        poll_interval=args.poll_interval,
        run_now=args.run_now,
    )


if __name__ == "__main__":
    main()
