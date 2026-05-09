import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml
from dotenv import load_dotenv


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _deep_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


@dataclass
class Config:
    mode: str = "paper"

    # Market
    asset: str = "btc"
    period: str = "5m"

    # Signal weights — oracle lag
    oracle_max_expected_lag: float = 0.001
    oracle_weight_early: float = 0.35
    oracle_weight_late: float = 0.25

    # Signal weights — momentum
    momentum_roc_weight: float = 0.30
    momentum_direction_weight: float = 0.25
    momentum_volume_weight: float = 0.25
    momentum_body_ratio_weight: float = 0.10
    momentum_rsi_weight: float = 0.10
    momentum_rsi_period: int = 5
    momentum_lookback_candles: int = 10
    momentum_weight_early: float = 0.40
    momentum_weight_late: float = 0.65

    # Signal weights — liquidation
    liquidation_weight_early: float = 0.25
    liquidation_weight_late: float = 0.10
    liquidation_refresh_interval_sec: int = 60

    # Signal combination
    max_adjustment: float = 0.15

    # Entry
    min_edge: float = 0.015
    max_edge: float = 0.08
    fee_adjustment: float = 0.02
    min_confidence: float = 0.02       # Minimum signal confidence to trade
    preferred_entry_secs: int = 180    # Target entry at 3min remaining
    latest_entry_secs: int = 60        # Don't enter after 1min remaining
    earliest_entry_secs: int = 270     # Start scanning at 4:30 remaining
    gtc_timeout_sec: int = 10
    fok_slippage: float = 0.005

    # Sizing
    kelly_multiplier: float = 0.25
    min_bet_usdc: float = 1.0
    max_bet_usdc: float = 5.0
    initial_bankroll: float = 100.0

    # Risk
    daily_loss_cap_pct: float = 0.20
    daily_loss_warn_pct: float = 0.15
    min_bankroll: float = 10.0
    streak_reduce_at: int = 3
    streak_reduce_factor: float = 0.75
    streak_heavy_reduce_at: int = 5
    streak_heavy_reduce_factor: float = 0.5
    streak_pause_at: int = 7
    streak_pause_minutes: int = 30
    streak_reset_wins: int = 3
    low_volatility_threshold: float = 0.0001
    high_volatility_threshold: float = 0.02
    high_volatility_size_factor: float = 0.5

    # Telegram
    telegram_enabled: bool = True
    telegram_notify_every_trade: bool = True
    telegram_notify_daily_summary: bool = True
    telegram_paper_prefix: str = "[PAPER] "

    # SM Confirmation (L9)
    sm_confirmation_enabled: bool = False
    sm_check_minutes: list[int] = field(default_factory=lambda: [3, 4])
    sm_agreement_threshold: float = 0.60
    sm_min_volume: float = 100.0
    sm_min_wallets: int = 2
    sm_price_floor: float = 0.65
    sm_price_ceiling: float = 0.80
    sm_poll_interval: float = 3.0

    # Logging
    db_path: str = "./data_runtime/trades.db"
    signal_log_path: str = "./data_runtime/signals.csv"
    log_level: str = "INFO"

    # Secrets (from .env)
    polymarket_private_key: str = ""
    polymarket_funder_address: str = ""
    polymarket_signature_type: int = 1
    polygon_rpc_url: str = "https://polygon-rpc.com"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    coinalyze_api_key: str = ""

    @classmethod
    def load(cls, config_path: str = "config.yaml", env_path: str = ".env") -> "Config":
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / env_path)
        raw = _load_yaml(project_root / config_path)

        c = cls()
        c.mode = raw.get("mode", c.mode)

        # Market
        c.asset = _deep_get(raw, "market", "asset", default=c.asset)
        c.period = _deep_get(raw, "market", "period", default=c.period)

        # Signals — oracle lag
        c.oracle_max_expected_lag = _deep_get(raw, "signals", "oracle_lag", "max_expected_lag", default=c.oracle_max_expected_lag)
        c.oracle_weight_early = _deep_get(raw, "signals", "oracle_lag", "weight_early", default=c.oracle_weight_early)
        c.oracle_weight_late = _deep_get(raw, "signals", "oracle_lag", "weight_late", default=c.oracle_weight_late)

        # Signals — momentum
        s_m = _deep_get(raw, "signals", "momentum", default={})
        if isinstance(s_m, dict):
            c.momentum_roc_weight = s_m.get("roc_weight", c.momentum_roc_weight)
            c.momentum_direction_weight = s_m.get("direction_weight", c.momentum_direction_weight)
            c.momentum_volume_weight = s_m.get("volume_weight", c.momentum_volume_weight)
            c.momentum_body_ratio_weight = s_m.get("body_ratio_weight", c.momentum_body_ratio_weight)
            c.momentum_rsi_weight = s_m.get("rsi_weight", c.momentum_rsi_weight)
            c.momentum_rsi_period = s_m.get("rsi_period", c.momentum_rsi_period)
            c.momentum_lookback_candles = s_m.get("lookback_candles", c.momentum_lookback_candles)
            c.momentum_weight_early = s_m.get("weight_early", c.momentum_weight_early)
            c.momentum_weight_late = s_m.get("weight_late", c.momentum_weight_late)

        # Signals — liquidation
        c.liquidation_weight_early = _deep_get(raw, "signals", "liquidation", "weight_early", default=c.liquidation_weight_early)
        c.liquidation_weight_late = _deep_get(raw, "signals", "liquidation", "weight_late", default=c.liquidation_weight_late)
        c.liquidation_refresh_interval_sec = _deep_get(raw, "signals", "liquidation", "refresh_interval_sec", default=c.liquidation_refresh_interval_sec)

        c.max_adjustment = _deep_get(raw, "signals", "max_adjustment", default=c.max_adjustment)

        # Entry
        e = raw.get("entry", {})
        if isinstance(e, dict):
            c.min_edge = e.get("min_edge", c.min_edge)
            c.max_edge = e.get("max_edge", c.max_edge)
            c.fee_adjustment = e.get("fee_adjustment", c.fee_adjustment)
            c.min_confidence = e.get("min_confidence", c.min_confidence)
            c.preferred_entry_secs = e.get("preferred_entry_secs", c.preferred_entry_secs)
            c.latest_entry_secs = e.get("latest_entry_secs", c.latest_entry_secs)
            c.earliest_entry_secs = e.get("earliest_entry_secs", c.earliest_entry_secs)
            c.gtc_timeout_sec = e.get("gtc_timeout_sec", c.gtc_timeout_sec)
            c.fok_slippage = e.get("fok_slippage", c.fok_slippage)

        # Sizing
        sz = raw.get("sizing", {})
        if isinstance(sz, dict):
            c.kelly_multiplier = sz.get("kelly_multiplier", c.kelly_multiplier)
            c.min_bet_usdc = sz.get("min_bet_usdc", c.min_bet_usdc)
            c.max_bet_usdc = sz.get("max_bet_usdc", c.max_bet_usdc)
            c.initial_bankroll = sz.get("initial_bankroll", c.initial_bankroll)

        # Risk
        r = raw.get("risk", {})
        if isinstance(r, dict):
            c.daily_loss_cap_pct = r.get("daily_loss_cap_pct", c.daily_loss_cap_pct)
            c.daily_loss_warn_pct = r.get("daily_loss_warn_pct", c.daily_loss_warn_pct)
            c.min_bankroll = r.get("min_bankroll", c.min_bankroll)
            c.streak_reduce_at = r.get("streak_reduce_at", c.streak_reduce_at)
            c.streak_reduce_factor = r.get("streak_reduce_factor", c.streak_reduce_factor)
            c.streak_heavy_reduce_at = r.get("streak_heavy_reduce_at", c.streak_heavy_reduce_at)
            c.streak_heavy_reduce_factor = r.get("streak_heavy_reduce_factor", c.streak_heavy_reduce_factor)
            c.streak_pause_at = r.get("streak_pause_at", c.streak_pause_at)
            c.streak_pause_minutes = r.get("streak_pause_minutes", c.streak_pause_minutes)
            c.streak_reset_wins = r.get("streak_reset_wins", c.streak_reset_wins)
            c.low_volatility_threshold = r.get("low_volatility_threshold", c.low_volatility_threshold)
            c.high_volatility_threshold = r.get("high_volatility_threshold", c.high_volatility_threshold)
            c.high_volatility_size_factor = r.get("high_volatility_size_factor", c.high_volatility_size_factor)

        # SM Confirmation (L9)
        sm = raw.get("sm_confirmation", {})
        if isinstance(sm, dict):
            c.sm_confirmation_enabled = sm.get("enabled", c.sm_confirmation_enabled)
            c.sm_check_minutes = sm.get("check_minutes", c.sm_check_minutes)
            c.sm_agreement_threshold = sm.get("agreement_threshold", c.sm_agreement_threshold)
            c.sm_min_volume = sm.get("min_volume", c.sm_min_volume)
            c.sm_min_wallets = sm.get("min_wallets", c.sm_min_wallets)
            c.sm_price_floor = sm.get("price_floor", c.sm_price_floor)
            c.sm_price_ceiling = sm.get("price_ceiling", c.sm_price_ceiling)
            c.sm_poll_interval = sm.get("poll_interval", c.sm_poll_interval)

        # Telegram
        t = raw.get("telegram", {})
        if isinstance(t, dict):
            c.telegram_enabled = t.get("enabled", c.telegram_enabled)
            c.telegram_notify_every_trade = t.get("notify_every_trade", c.telegram_notify_every_trade)
            c.telegram_notify_daily_summary = t.get("notify_daily_summary", c.telegram_notify_daily_summary)
            c.telegram_paper_prefix = t.get("paper_prefix", c.telegram_paper_prefix)

        # Logging
        lg = raw.get("logging", {})
        if isinstance(lg, dict):
            c.db_path = lg.get("db_path", c.db_path)
            c.signal_log_path = lg.get("signal_log_path", c.signal_log_path)
            c.log_level = lg.get("log_level", c.log_level)

        # Secrets from env
        c.polymarket_private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        c.polymarket_funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
        c.polymarket_signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))
        c.polygon_rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        c.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        c.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        c.coinalyze_api_key = os.getenv("COINALYZE_API_KEY", "")

        return c
