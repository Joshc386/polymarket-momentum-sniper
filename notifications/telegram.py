import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    """Sends notifications to a Telegram chat via Bot API."""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True, paper_prefix: str = "[PAPER] "):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self.paper_prefix = paper_prefix
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        await self._client.aclose()

    async def send(self, message: str):
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            return
        try:
            url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"
            await self._client.post(url, json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            })
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")

    async def notify_trade(
        self,
        is_paper: bool,
        side: str,
        price: float,
        size_usdc: float,
        edge: float,
        est_prob: float,
        market_prob: float,
        time_remaining: float,
        oracle_lag: float,
        momentum: float,
        liquidation: float,
        bankroll: float,
        session_pnl: float,
    ):
        prefix = self.paper_prefix if is_paper else ""
        mins = int(time_remaining // 60)
        secs = int(time_remaining % 60)
        msg = (
            f"{'📝' if is_paper else '🟢'} <b>{prefix}TRADE PLACED</b>\n"
            f"Side: <b>{side}</b> ({'Up' if side == 'YES' else 'Down'})\n"
            f"Price: ${price:.4f} | Size: ${size_usdc:.2f} USDC\n"
            f"Edge: {edge:+.1%} | Model: {est_prob:.1%} | Market: {market_prob:.1%}\n"
            f"Time left: {mins}m {secs:02d}s\n"
            f"Signals: Oracle {oracle_lag:+.2f} | Mom {momentum:+.2f} | Liq {liquidation:+.2f}\n"
            f"---\n"
            f"Bankroll: ${bankroll:.2f} | Session P&L: ${session_pnl:+.2f}"
        )
        await self.send(msg)

    async def notify_resolution(
        self,
        is_paper: bool,
        side: str,
        resolution: str,
        won: bool,
        pnl: float,
        bankroll: float,
        session_pnl: float,
        win_rate: float,
        total_trades: int,
    ):
        prefix = self.paper_prefix if is_paper else ""
        emoji = "✅" if won else "❌"
        msg = (
            f"{emoji} <b>{prefix}RESOLVED: {resolution}</b>\n"
            f"Side: {side} | {'WIN' if won else 'LOSS'}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"---\n"
            f"Bankroll: ${bankroll:.2f} | Session: ${session_pnl:+.2f}\n"
            f"Record: {total_trades} trades | {win_rate:.0%} win rate"
        )
        await self.send(msg)

    async def notify_streak_alert(self, consecutive_losses: int, action: str):
        msg = (
            f"⚠️ <b>STREAK ALERT</b>\n"
            f"{consecutive_losses} consecutive losses\n"
            f"Action: {action}"
        )
        await self.send(msg)

    async def notify_daily_summary(
        self,
        is_paper: bool,
        total_trades: int,
        wins: int,
        losses: int,
        pnl: float,
        bankroll: float,
    ):
        prefix = self.paper_prefix if is_paper else ""
        win_rate = wins / total_trades if total_trades > 0 else 0
        msg = (
            f"📊 <b>{prefix}DAILY SUMMARY</b>\n"
            f"Trades: {total_trades} | Wins: {wins} | Losses: {losses}\n"
            f"Win Rate: {win_rate:.0%}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Bankroll: ${bankroll:.2f}"
        )
        await self.send(msg)

    async def notify_bot_event(self, event: str, details: str = ""):
        msg = f"🤖 <b>{event}</b>"
        if details:
            msg += f"\n{details}"
        await self.send(msg)
