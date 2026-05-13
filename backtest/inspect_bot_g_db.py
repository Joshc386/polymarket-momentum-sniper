"""Quick DB inspection for Bot G trade schema."""
import sqlite3

conn = sqlite3.connect("data_runtime/bot_g_signal_aligned.db")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

for t in tables:
    cur.execute(f"PRAGMA table_info({t})")
    cols = cur.fetchall()
    print(f"\nTable: {t}")
    for c in cols:
        print(f"  {c[1]:30s} {c[2]}")
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    print(f"  rows: {cur.fetchone()[0]}")

# Sample a few completed trades
print("\n--- Sample completed trades ---")
cur.execute("""
    SELECT * FROM trades
    WHERE pnl IS NOT NULL
    LIMIT 3
""")
cols = [d[0] for d in cur.description]
for row in cur.fetchall():
    print()
    for c, v in zip(cols, row):
        print(f"  {c}: {v}")

conn.close()
