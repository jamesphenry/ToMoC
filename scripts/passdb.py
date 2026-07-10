#!/usr/bin/env python3
"""passdb — store every training/eval pass's metrics + meta.

One SQLite file at benchmarks/passes.db. Three tables:
  passes      run header (what we trained)
  pass_metrics per-eval score (the balance numbers we tune)
  pass_meta   free-form key/value (git_commit, data_hash, notes...)

KISS: stdlib sqlite3 only. Helpers return row ids so callers chain
new_pass() -> id, then log_metric(id, ...) / log_meta(id, ...).

Usage:
    from passdb import PassDB
    db = PassDB()                      # opens/creates benchmarks/passes.db
    pid = db.new_pass(base_model="smollm:135m", lora_r=8, ...)
    db.log_metric(pid, "over_call_rate", 0.12)
    db.log_meta(pid, "git_commit", "d0dd03f")
    db.close()
"""
import os
import sqlite3
from datetime import datetime, timezone


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "passes.db")


class PassDB:
    def __init__(self, path: str = DB_PATH):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS passes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                base_model  TEXT,
                lora_r      INTEGER,
                lora_alpha  INTEGER,
                epochs      REAL,
                lr          REAL,
                num_cards   INTEGER,
                a_ratio     REAL,
                loss_final  REAL,
                loss_train  REAL,
                status      TEXT DEFAULT 'done'
            );
            CREATE TABLE IF NOT EXISTS pass_metrics (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                pass_id   INTEGER NOT NULL REFERENCES passes(id),
                metric    TEXT NOT NULL,
                value     REAL,
                detail    TEXT
            );
            CREATE TABLE IF NOT EXISTS pass_meta (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                pass_id   INTEGER NOT NULL REFERENCES passes(id),
                key       TEXT NOT NULL,
                value     TEXT
            );
            """
        )
        self.conn.commit()

    def new_pass(self, **fields) -> int:
        """Insert a passes row. Unknown keys are ignored (schema-safe)."""
        cols = {
            "base_model", "lora_r", "lora_alpha", "epochs", "lr",
            "num_cards", "a_ratio", "loss_final", "loss_train", "status",
        }
        f = {k: fields[k] for k in cols if k in fields}
        f["created_at"] = datetime.now(timezone.utc).isoformat()
        keys = ", ".join(f.keys())
        ph = ", ".join("?" * len(f))
        cur = self.conn.execute(
            f"INSERT INTO passes ({keys}) VALUES ({ph})", tuple(f.values())
        )
        self.conn.commit()
        return cur.lastrowid

    def log_metric(self, pass_id: int, metric: str, value, detail: str = None):
        self.conn.execute(
            "INSERT INTO pass_metrics (pass_id, metric, value, detail) VALUES (?,?,?,?)",
            (pass_id, metric, value, detail),
        )
        self.conn.commit()

    def log_meta(self, pass_id: int, key: str, value: str):
        self.conn.execute(
            "INSERT INTO pass_meta (pass_id, key, value) VALUES (?,?,?)",
            (pass_id, key, str(value)),
        )
        self.conn.commit()

    def summarize(self, pass_id: int = None):
        """Print a compact report. If pass_id None, show latest pass only."""
        if pass_id is None:
            row = self.conn.execute(
                "SELECT id FROM passes ORDER BY id DESC LIMIT 1"
            ).fetchone()
            pass_id = row["id"] if row else None
        if pass_id is None:
            print("(no passes yet)")
            return
        p = self.conn.execute(
            "SELECT * FROM passes WHERE id=?", (pass_id,)
        ).fetchone()
        print(f"=== pass {pass_id} ({p['created_at']}) ===")
        for k in ("base_model", "lora_r", "lora_alpha", "epochs", "lr",
                  "num_cards", "a_ratio", "loss_final", "status"):
            if p[k] is not None:
                print(f"  {k}: {p[k]}")
        print("  -- metrics --")
        for m in self.conn.execute(
            "SELECT metric, value, detail FROM pass_metrics WHERE pass_id=?",
            (pass_id,),
        ):
            d = f"  ({m['detail']})" if m["detail"] else ""
            print(f"    {m['metric']}: {m['value']}{d}")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # smoke test: exercise the API without polluting real data
    import tempfile
    import os as _os
    tmp = tempfile.mktemp(suffix=".db")
    db = PassDB(tmp)
    pid = db.new_pass(base_model="smollm:135m", lora_r=8, num_cards=60, a_ratio=0.5)
    db.log_metric(pid, "over_call_rate", 0.04)
    db.log_metric(pid, "call_rate_when_should", 0.92)
    db.log_meta(pid, "git_commit", "smoke")
    db.summarize(pid)
    db.close()
    _os.remove(tmp)
    print("(smoke test passed, temp db removed)")
