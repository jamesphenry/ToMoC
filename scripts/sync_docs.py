#!/usr/bin/env python3
"""sync_docs.py — regenerate derivable doc fragments from benchmarks/passes.db.

DERIVABLE (written here, idempotently):
  - README.md  cost banner        (pass count / cost / GPU-hrs)
  - AGENTS.md  "Cost tracking live" line
  - runs.md    "Totals" block      (count / cost / GPU-hrs / avg)
  - runs.md    NEW per-pass rows   (any id > max already in the table)

NOT touched (curated / hand-written by the assistant):
  - runs.md narrative tables (call-rate arc, resolver end-to-end)
  - wiki/JOURNAL.md               (the assistant changelog)
  - all prose / commentary

The per-pass table is APPENDED to, never regenerated, so the hand-curated
early rows (pass 1-35, with "-> vN" labels and BUG footnotes) are preserved.
New rows get base_model + status-derived type; JOURNAL carries the narrative.

Idempotent: a file is only rewritten when its content actually changes.
Usage:
  python scripts/sync_docs.py [path-to-passes.db]
The optional db path is for testing only; default = benchmarks/passes.db.
"""
import os
import re
import sqlite3
import sys
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(REPO, "benchmarks", "passes.db")

TYPE_MAP = {
    "trained": "train",
    "resolver-eval": "resolver-eval",
    "orchestrate-eval": "orchestrate-eval",
    "benchmark-ref": "benchmark-ref",
}


def load(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, created_at, base_model, num_cards, loss_final, walltime_s, "
        "gpu_mem_used_mb, cost_usd, status FROM passes ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def totals(rows):
    n = len(rows)
    cost = sum((r["cost_usd"] or 0) for r in rows)
    wt = sum((r["walltime_s"] or 0) for r in rows)
    return n, cost, wt / 3600.0


def fmt_time(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except Exception:
        return "??:??"


def table_row(r):
    rid = r["id"]
    t = fmt_time(r["created_at"])
    typ = TYPE_MAP.get(r["status"], r["status"] or "eval")
    model = r["base_model"] or "—"
    cards = r["num_cards"] if r["num_cards"] is not None else "—"
    loss = f"{r['loss_final']:.4f}" if r["loss_final"] is not None else "—"
    wall = f"{r['walltime_s']:.1f}" if r["walltime_s"] is not None else "—"
    gpu = r["gpu_mem_used_mb"] if r["gpu_mem_used_mb"] is not None else "—"
    cost = f"{r['cost_usd']:.5f}" if r["cost_usd"] is not None else "—"
    return f"| {rid} | {t} | {typ} | {model} | {cards} | {loss} | {wall} | {gpu} | {cost} |"


def write_if_changed(path, new_text):
    old = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = f.read()
    if old == new_text:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return True


def sync_readme(text, n, cost, gpu_h):
    lines = text.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if "Sovereign compute cost so far" in line:
            nl = f'> ## 🔌 Sovereign compute cost so far: **${cost:.4f}**'
            if lines[i] != nl:
                lines[i] = nl
                changed = True
            if i + 1 < len(lines):
                nl2 = (f'> {n} training/eval passes · {gpu_h:.2f} GPU-hrs · '
                       f'14¢/kWh · ~90W over server idle')
                if lines[i + 1] != nl2:
                    lines[i + 1] = nl2
                    changed = True
            break
    return "\n".join(lines), changed


def sync_agents(text, n, cost):
    lines = text.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if "Cost tracking live" in line:
            nl = (f'- **Cost tracking live**: total **${cost:.4f}** '
                  f'across {n} passes (README banner).')
            if lines[i] != nl:
                lines[i] = nl
                changed = True
            break
    return "\n".join(lines), changed


def sync_runs_totals(text, n, cost, gpu_h):
    lines = text.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if line.startswith("## Totals (all"):
            nh = f"## Totals (all {n} passes)"
            if lines[i] != nh:
                lines[i] = nh
                changed = True
            for j in range(i + 1, min(i + 8, len(lines))):
                if lines[j].startswith("| total cost |"):
                    nl = f"| total cost | **${cost:.4f}** |"
                    if lines[j] != nl:
                        lines[j] = nl
                        changed = True
                elif lines[j].startswith("| total GPU time |"):
                    nl = f"| total GPU time | {gpu_h:.2f} h |"
                    if lines[j] != nl:
                        lines[j] = nl
                        changed = True
                elif lines[j].startswith("| avg cost / pass |"):
                    nl = f"| avg cost / pass | ${cost/n:.5f} |"
                    if lines[j] != nl:
                        lines[j] = nl
                        changed = True
            break
    return "\n".join(lines), changed


def sync_runs_table(text, rows):
    existing = [int(x) for x in re.findall(r'^\|\s*(\d+)\s*\|', text, re.M)]
    max_id = max(existing) if existing else 0
    new_rows = [table_row(r) for r in rows if r["id"] > max_id]
    if not new_rows:
        return text, False, 0
    lines = text.split("\n")
    # find the last existing table-row line to append after it
    last_idx = None
    for i, line in enumerate(lines):
        if re.match(rf'\|\s*{max_id}\s*\|', line):
            last_idx = i
    if last_idx is None:
        # no rows yet: insert right after the header row
        for i, line in enumerate(lines):
            if line.startswith("| pass |"):
                last_idx = i
                break
    if last_idx is None:
        last_idx = len(lines) - 1
    for off, nr in enumerate(new_rows):
        lines.insert(last_idx + 1 + off, nr)
    return "\n".join(lines), True, len(new_rows)


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    if not os.path.exists(db_path):
        print(f"[sync_docs] no DB at {db_path}; skipping")
        return 0
    # SAFETY: if a custom db path is given (testing), write outputs to /tmp
    # sandbox copies instead of the real repo files, so a test never mutates
    # the repo. Default db path = live mode (writes the real repo).
    sandbox = (db_path != DEFAULT_DB)
    rows = load(db_path)
    n, cost, gpu_h = totals(rows)
    print(f"[sync_docs] DB: {n} passes, ${cost:.4f}, {gpu_h:.2f} GPU-h"
          + ("  [SANDBOX — writing to /tmp]" if sandbox else ""))
    if sandbox:
        import shutil
        for fn in ("README.md", "AGENTS.md", "runs.md"):
            src = os.path.join(REPO, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join("/tmp", "sync_" + fn))

    def repo_path(fn):
        return os.path.join("/tmp", "sync_" + fn) if sandbox else os.path.join(REPO, fn)

    changed_any = False

    p = repo_path("README.md")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            t = f.read()
        new_t, ch = sync_readme(t, n, cost, gpu_h)
        if ch:
            write_if_changed(p, new_t)
            print("  README.md: updated")
            changed_any = True
        else:
            print("  README.md: ok")

    p = repo_path("AGENTS.md")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            t = f.read()
        new_t, ch = sync_agents(t, n, cost)
        if ch:
            write_if_changed(p, new_t)
            print("  AGENTS.md: updated")
            changed_any = True
        else:
            print("  AGENTS.md: ok")

    p = repo_path("runs.md")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            t = f.read()
        # diagnose
        existing = [int(x) for x in re.findall(r'^\|\s*(\d+)\s*\|', t, re.M)]
        max_id = max(existing) if existing else 0
        print(f"  [diag] runs.md parsed max per-pass id = {max_id} "
              f"(rows > max will be appended)")
        new_t, ch1 = sync_runs_totals(t, n, cost, gpu_h)
        new_t, ch2, k = sync_runs_table(new_t, rows)
        if ch1 or ch2:
            write_if_changed(p, new_t)
            print(f"  runs.md: updated (totals={ch1}, +{k} new per-pass rows)")
            changed_any = True
        else:
            print("  runs.md: ok")

    if sandbox:
        print("[sync_docs] SANDBOX complete — inspect /tmp/sync_*.md; "
              "repo files untouched")
    else:
        print("[sync_docs] done" + ("" if changed_any else " (no changes)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
