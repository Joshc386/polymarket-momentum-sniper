"""Phase B: the execution engine persists the full feature snapshot.

A trade row must carry the feature vector at entry as flat columns, with
NULL preserved for inactive layers (distinct from a genuine 0.0).
See docs/adr/0001-full-feature-snapshot-on-trade-record.md.
"""

from core.execution import PaperExecutionEngine
from logging_db.database import Database

_BASE = dict(
    side="YES", price=0.50, size_usdc=1.0, market_id="m", market_slug="s",
    oracle_lag_signal=0.0, momentum_signal=0.0, liquidation_signal=0.0,
    combined_signal=0.0, estimated_prob_up=0.60, market_implied_prob=0.50,
    edge=0.01, time_remaining_secs=188.0, btc_price=72000.0,
    oracle_price=0.0, oracle_open_price=0.0,
)


def test_feature_snapshot_columns_persist(tmp_path) -> None:
    db = Database(str(tmp_path / "t.db"))
    db.connect()
    eng = PaperExecutionEngine(db, initial_bankroll=100.0)
    snap = {
        "net_ev_per_share": 0.032,
        "prob_edge": 0.08,
        "l4_imbalance": None,       # inactive layer → NULL
        "l6_fade": 0.0,             # genuinely-computed neutral → 0.0
        "secs_into_window": 112.0,
        "schedule_override": "ranging",
    }
    eng.execute_paper_trade(feature_snapshot=snap, **_BASE)

    row = db.conn.execute(
        "SELECT net_ev_per_share, prob_edge, l4_imbalance, l6_fade, "
        "secs_into_window, schedule_override FROM trades"
    ).fetchone()
    assert row["net_ev_per_share"] == 0.032
    assert row["prob_edge"] == 0.08
    assert row["l4_imbalance"] is None      # NULL, not 0.0
    assert row["l6_fade"] == 0.0            # distinguishable from absent
    assert row["secs_into_window"] == 112.0
    assert row["schedule_override"] == "ranging"
    db.close()


def test_no_feature_snapshot_is_harmless(tmp_path) -> None:
    # Backward-compat: omitting the snapshot still records a trade (new
    # columns simply stay NULL).
    db = Database(str(tmp_path / "t.db"))
    db.connect()
    eng = PaperExecutionEngine(db, initial_bankroll=100.0)
    eng.execute_paper_trade(**_BASE)
    row = db.conn.execute(
        "SELECT side, net_ev_per_share FROM trades"
    ).fetchone()
    assert row["side"] == "YES"
    assert row["net_ev_per_share"] is None
    db.close()
