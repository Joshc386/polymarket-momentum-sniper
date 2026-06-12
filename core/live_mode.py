"""Live-fleet startup guards (ADR-0003).

Pure validation so the rules are regression-testable without importing
the runner: a live intent that silently ran paper would corrupt the very
record the T1 probe exists to create, and one yaml typo must not double
live exposure.
"""


def live_bot_names(bots_cfg: dict) -> list[str]:
    """Names of enabled bots declaring ``execution_mode: live``."""
    return [
        name for name, cfg in bots_cfg.items()
        if cfg.get("enabled", True) and cfg.get("execution_mode", "paper") == "live"
    ]


def validate_live_fleet(bots_cfg: dict, authenticated: bool) -> str | None:
    """Refuse-to-start errors for the live fleet; None when fine.

    - More than one live bot (v1 limit — lifted deliberately, not by typo).
    - A live bot without CLOB auth (no silent paper fallback).
    """
    live = live_bot_names(bots_cfg)
    if len(live) > 1:
        return (
            f"at most ONE live bot in v1, got {len(live)}: {live} — "
            "flip the extras back to execution_mode: paper"
        )
    if live and not authenticated:
        return (
            f"bot '{live[0]}' declares execution_mode: live but the CLOB "
            "client is not authenticated — set POLYMARKET_PRIVATE_KEY/"
            "POLYMARKET_FUNDER_ADDRESS in .env (refusing to fall back to "
            "paper silently)"
        )
    return None
