#!/usr/bin/env python3
"""probe_three — dated, git-stored PROOF of a model's behavior.

We have metrics (eval_resolver logs, router precision...) but no *evidence* of
what the model actually SAYS. This script runs:
  (1) 3 FIXED prompts (lookup / run_code / KB-miss) through the real ToMoC loop,
  (2) optionally N random gsm8k rows as an AUDIT, split into PASS / FAIL examples,
and writes a dated markdown to `probe_logs/` (tracked, not gitignored) with the
VERBATIM output of every turn. Commit the file so each run leaves a permanent,
inspectable trace of what the model got right AND wrong.

The 3 fixed prompts live in `data/probe/three_prompts.jsonl` (stable across runs)
so we compare adapters apples-to-apples over time.

Usage:
  source .venv/bin/activate
  python -u scripts/probe_three.py --model adapters/v11
  python -u scripts/probe_three.py --model adapters/v11 --audit 8   # + pass/fail examples
  python -u scripts/probe_three.py --model adapters/v10 --audit 8   # for comparison
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from orchestrate import Engine, KB, run_question

PROBE_FILE = os.path.join(ROOT, "data", "probe", "three_prompts.jsonl")
GSM8K = os.path.expanduser("~/llm_eval/datasets/gsm8k_test.jsonl")
OUT_DIR = os.path.join(ROOT, "probe_logs")


def load_probes():
    with open(PROBE_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def render_rec(p, r):
    """Verbatim markdown for one question/result pair (reused everywhere)."""
    out = [
        f"**prompt:** {p.get('q', r.get('q', ''))}",
        "",
        "**turn 1 (model):**",
        "",
        "```",
        r.get("turn1") or "(no output)",
        "```",
        "",
    ]
    if r.get("called"):
        res = r.get("resolved") or {}
        result_str = (res.get("answer") if res.get("verdict") == "hit"
                      else "No answer found in the knowledge base.")
        out += [
            f"**tool call:** `{r.get('tool')}`  query=`{r.get('query')}`  "
            f"verdict=`{res.get('verdict')}`",
            "",
            "**tool result fed back:**",
            "",
            "```",
            str(result_str),
            "```",
            "",
        ]
    out += [
        "**turn 2 (final answer):**",
        "",
        "```",
        r.get("final_answer") or "(empty)",
        "```",
        "",
        f"**gold:** {r.get('gold')}",
        f"**final_correct:** {r.get('final_correct')}  "
        f"**canonical_correct:** {r.get('canonical_correct')}",
        "",
    ]
    return out


def render_md(model, probes, recs, audit_pass, audit_fail):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    lines = [
        f"# probe_three — {now}",
        "",
        f"model: `{model}`",
        f"generated: {now}",
        "",
        "Three fixed prompts through the real ToMoC loop, PLUS a random gsm8k "
        "audit split into PASS/FAIL examples. Verbatim output — no "
        "summarization. Commit this file as proof of behavior for this model.",
        "",
        "---",
        "",
    ]
    lines.append("## Fixed probes")
    lines.append("")
    for p, r in zip(probes, recs):
        lines += [
            f"### {p.get('label', '?')}  _({p.get('kind', '?')})_",
            "",
        ]
        lines += render_rec(p, r)
        lines.append("---")
        lines.append("")

    if audit_pass or audit_fail:
        npass, nfail = len(audit_pass), len(audit_fail)
        lines += [
            f"## Audit — {npass + nfail} random gsm8k rows "
            f"({npass} pass / {nfail} fail)",
            "",
        ]
        if audit_pass:
            lines.append(f"### PASS examples ({npass})")
            lines.append("")
            for p, r in audit_pass:
                lines += [f"**Q:** {p['q']}", ""] + render_rec(p, r)
                lines.append("---")
                lines.append("")
        if audit_fail:
            lines.append(f"### FAIL examples ({nfail})")
            lines.append("")
            for p, r in audit_fail:
                lines += [f"**Q:** {p['q']}", ""] + render_rec(p, r)
                lines.append("---")
                lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", default=PROBE_FILE)
    ap.add_argument("--audit", type=int, default=0,
                    help="run N random gsm8k_test rows, split into PASS/FAIL "
                         "examples (default 0 = fixed probes only)")
    ap.add_argument("--audit-seed", type=int, default=0,
                    help="deterministic seed for the audit shuffle")
    args = ap.parse_args()

    probes = load_probes()
    kb = KB.get()
    engine = Engine(args.model, max_new_tokens=160)

    recs = []
    for p in probes:
        rec = run_question(engine, kb, p["q"], p.get("gold", ""), verbose=True)
        recs.append(rec)

    audit_pass, audit_fail = [], []
    if args.audit > 0:
        with open(GSM8K) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        rng = random.Random(args.audit_seed)
        rng.shuffle(rows)
        taken = rows[:args.audit]
        for row in taken:
            p = {"q": row["prompt"], "gold": row.get("expected", "")}
            r = run_question(engine, kb, p["q"], p["gold"], verbose=True)
            (audit_pass if r.get("final_correct") else audit_fail).append((p, r))

    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    slug = os.path.basename(os.path.normpath(args.model)).replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"probe_{now}_{slug}.md")
    with open(out_path, "w") as f:
        f.write(render_md(model=args.model, probes=probes, recs=recs,
                          audit_pass=audit_pass, audit_fail=audit_fail))
    print(f"\nwrote proof log -> {out_path}")
    print(f"  fixed probes: {len(recs)}  audit: +{len(audit_pass)} pass / "
          f"{len(audit_fail)} fail")
    print("git add it to leave a permanent trace of this model's behavior.")


if __name__ == "__main__":
    main()
