"""
J13 over-performance investigation — Tests 1 and 2.

Test 1: Contamination check.
  Re-run the weekday NO audit excluding the 2026-05-08 to 2026-05-15
  bug window (NO-side AttributeError silently rejected NO trades). If
  pre-J13 mild-L1 NO WR jumps to 40%+ once bug-week is dropped, then
  most of the "J13 improvement" is just measurement artifact.

Test 2: Bot K vs Bot G post-J13 comparison.
  Bot G has J13 disabled (A/B preservation). For the same calendar
  window (Bot K's pre vs post J13 dates), did Bot G also improve?
  If yes → improvement is regime + infrastructure. If only Bot K
  improved → J13 is genuinely driving it.

Read-only.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_BOT_K = REPO / "data_runtime" / "bot_k_sm_confirmation.db"
DB_BOT_G = REPO / "data_runtime" / "bot_g_signal_aligned.db"

J13_CUTOFF_UTC = pd.Timestamp("2026-05-22 00:00:00", tz="UTC")
BUG_WINDOW_START = pd.Timestamp("2026-05-08 00:00:00", tz="UTC")
BUG_WINDOW_END = pd.Timestamp("2026-05-15 23:59:59", tz="UTC")


def load(db_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT id, timestamp, side, pnl, oracle_lag_signal
        FROM trades
        WHERE pnl IS NOT NULL AND oracle_lag_signal IS NOT NULL
    """, con)
    con.close()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["weekend"] = df["ts"].dt.dayofweek >= 5
    df["mild_l1"] = df["oracle_lag_signal"].abs() <= 0.4
    df["post_j13"] = df["ts"] >= J13_CUTOFF_UTC
    df["bug_window"] = (df["ts"] >= BUG_WINDOW_START) & (df["ts"] <= BUG_WINDOW_END)
    df["win"] = (df["pnl"] > 0).astype(int)
    return df


def stat(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "WR_%": None, "pnl": 0.0, "pnl_per_day": 0.0}
    days = max(1, (sub["ts"].max() - sub["ts"].min()).days + 1)
    return {
        "n": n,
        "WR_%": round(100 * sub["win"].mean(), 1),
        "pnl": round(sub["pnl"].sum(), 2),
        "pnl_per_day": round(sub["pnl"].sum() / days, 2),
        "days": days,
    }


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------- TEST 1 ----------

def test1_contamination(bot_k: pd.DataFrame) -> None:
    section("TEST 1 -- Contamination check (pre-J13 with/without bug window)")
    print(f"Bug window dropped: {BUG_WINDOW_START.date()} to {BUG_WINDOW_END.date()}")
    print()

    wk_no = bot_k[(bot_k["side"] == "NO") & (~bot_k["weekend"])]

    print("Weekday NO -- pre-J13 ONLY, with and without the bug window:")
    rows = []
    for mild in [True, False]:
        label_l1 = "mild (|L1|<=0.4)" if mild else "strong (|L1|>0.4)"
        full = wk_no[(~wk_no["post_j13"]) & (wk_no["mild_l1"] == mild)]
        clean = wk_no[(~wk_no["post_j13"]) & (wk_no["mild_l1"] == mild)
                      & (~wk_no["bug_window"])]
        rows.append({
            "L1": label_l1,
            "sample": "pre-J13 full",
            **stat(full),
        })
        rows.append({
            "L1": label_l1,
            "sample": "pre-J13 clean (bug-week dropped)",
            **stat(clean),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("Side-by-side: full pre-J13 vs clean pre-J13 vs post-J13 by L1 mag:")
    rows = []
    for mild in [True, False]:
        label_l1 = "mild (|L1|<=0.4)" if mild else "strong (|L1|>0.4)"
        for sample_name, mask in [
            ("pre-J13 full", ~wk_no["post_j13"]),
            ("pre-J13 clean", (~wk_no["post_j13"]) & (~wk_no["bug_window"])),
            ("post-J13", wk_no["post_j13"]),
        ]:
            sub = wk_no[mask & (wk_no["mild_l1"] == mild)]
            rows.append({"L1": label_l1, "sample": sample_name, **stat(sub)})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("Interpretation guide:")
    print("  If pre-J13 mild-L1 WR jumps from ~37% to ~45%+ when bug-week is")
    print("  dropped, the 'improvement' to 48% post-J13 is mostly artifact.")
    print("  If pre-J13 mild-L1 WR is roughly unchanged, contamination is not")
    print("  the explanation -- need to check regime / J13 indirect effects.")
    print()


# ---------- TEST 2 ----------

def test2_bot_g_comparison(bot_k: pd.DataFrame, bot_g: pd.DataFrame) -> None:
    section("TEST 2 -- Bot K vs Bot G (J13 enabled vs disabled)")

    # Restrict Bot G to the same calendar window as Bot K
    k_start = bot_k["ts"].min()
    k_end = bot_k["ts"].max()
    bot_g_window = bot_g[(bot_g["ts"] >= k_start) & (bot_g["ts"] <= k_end)].copy()

    print(f"Bot K window         : {k_start} to {k_end}")
    print(f"Bot G total trades   : {len(bot_g)} (April 14 onwards)")
    print(f"Bot G in K's window  : {len(bot_g_window)}")
    print()
    print("Bot G has directional_signal_gating: FALSE (A/B preservation).")
    print("If Bot G also improved post-J13, the improvement is regime.")
    print("If only Bot K improved, J13 is doing it.")
    print()

    print("Weekday NO performance by era, both bots:")
    rows = []
    for bot_name, df in [("Bot K (J13 ON)", bot_k), ("Bot G (J13 OFF)", bot_g_window)]:
        wk_no = df[(df["side"] == "NO") & (~df["weekend"])]
        for era_name, mask in [("pre-J13", ~wk_no["post_j13"]),
                                ("post-J13", wk_no["post_j13"])]:
            rows.append({"bot": bot_name, "era": era_name, **stat(wk_no[mask])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("Weekday NO by L1 magnitude × era × bot:")
    rows = []
    for bot_name, df in [("Bot K (J13 ON)", bot_k), ("Bot G (J13 OFF)", bot_g_window)]:
        wk_no = df[(df["side"] == "NO") & (~df["weekend"])]
        for mild in [True, False]:
            mild_label = "mild" if mild else "strong"
            for era_name, mask in [("pre-J13", ~wk_no["post_j13"]),
                                    ("post-J13", wk_no["post_j13"])]:
                sub = wk_no[mask & (wk_no["mild_l1"] == mild)]
                rows.append({
                    "bot": bot_name, "L1": mild_label, "era": era_name,
                    **stat(sub),
                })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("Whole-bot post-J13 vs pre-J13 (both sides, all conditions):")
    rows = []
    for bot_name, df in [("Bot K (J13 ON)", bot_k), ("Bot G (J13 OFF)", bot_g_window)]:
        for era_name, mask in [("pre-J13", ~df["post_j13"]),
                                ("post-J13", df["post_j13"])]:
            rows.append({"bot": bot_name, "era": era_name, **stat(df[mask])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("Interpretation guide:")
    print("  Bot G's improvement (or lack thereof) post-J13 isolates the regime")
    print("  component. Bot K improvement minus Bot G improvement = J13 effect.")
    print()


def main() -> int:
    print()
    bot_k = load(DB_BOT_K)
    bot_g = load(DB_BOT_G)
    print(f"Bot K loaded: {len(bot_k)} trades")
    print(f"Bot G loaded: {len(bot_g)} trades")

    test1_contamination(bot_k)
    test2_bot_g_comparison(bot_k, bot_g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
