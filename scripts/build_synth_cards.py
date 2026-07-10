#!/usr/bin/env python3
"""build_synth_cards.py — synthesize a bigger, balanced tool-habit dataset.

Mines on-disk sources into Type A (lookup) / Type B (answer) flashcards in the
SAME {type,src,q,a} shape as data/raw/flashcards.jsonl.

Type A (call the tool) — drawn from 135m's FAILURES:
  - gsm8k_train.jsonl  : thousands of math Qs; 135m scores ~2% -> ALL lookups
  - eval.db runitems   : knowledge_qa & reasoning_logic items where correct=0

Type B (answer, no tool) — drawn from 135m's WINS:
  - eval.db runitems   : coding_func, brainteasers, summarization, mmlu,
                         knowledge_qa, reasoning_logic items where correct=1

We bias A-HEAVY on purpose: our measured failure is UNDER-call (call_rate 0.000),
not over-call. The 50/50 in flashcards_spec.md prevents over-call; since we don't
over-call, we weight toward A to install the habit. Knobs below.

Query is a VERBATIM copy of the question (KISS — model only decides
call-vs-answer and copies text; it never composes a search query).

Usage:
  python scripts/build_synth_cards.py --out data/raw/flashcards2.jsonl
"""
import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LLM_EVAL = os.path.expanduser("~/llm_eval")
GSM_TRAIN = os.path.join(LLM_EVAL, "datasets", "gsm8k_train.jsonl")
EVAL_DB = os.path.join(LLM_EVAL, "eval.db")

# max characters we keep in a lookup query so the full `query="..."` (with the
# closing quote) fits inside the eval's max_new_tokens budget. ~40 tokens of headroom
# for "TOOL lookup query=\"" + closing quote at 64 new tokens (BUG-008).
MAX_Q = 180


def load_gsm(n):
    out = []
    with open(GSM_TRAIN) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            r = json.loads(line)
            q = r.get("prompt") or r.get("question") or ""
            q = q.strip()
            if not q:
                continue
            out.append(mk_A(q))
    return out


def mk_A(q, src="gsm8k_train"):
    """Build a Type A (lookup) card, truncating the query so the full
    `query="..."` (with closing quote) fits the eval token budget (BUG-008)."""
    if len(q) > MAX_Q:
        # leave room for the ellipsis, cut at a word boundary
        q = q[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
    return {"type": "A", "src": src,
            "q": q, "a": f'TOOL lookup query="{q}"'}


def load_from_eval():
    c = sqlite3.connect(EVAL_DB)
    c.row_factory = sqlite3.Row
    # pick the most recent completed run (status='done')
    run = c.execute(
        "SELECT id FROM evalruns WHERE status='done' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    rid = run["id"] if run else 1

    a, b = [], []
    for it in c.execute(
        "SELECT task_id, question, correct FROM runitems WHERE run_id=?", (rid,)
    ):
        q = (it["question"] or "").strip()
        if not q:
            continue
        if it["correct"] == 0 and it["task_id"] in ("knowledge_qa", "reasoning_logic"):
            a.append(mk_A(q, src=f"eval.{it['task_id']}"))
        elif it["correct"] == 1 and it["task_id"] in (
            "coding_func", "brainteasers", "summarization",
            "knowledge_qa", "reasoning_logic",
            "dataset_mmlu_abstract_algebra_test",
        ):
            b.append({"type": "B", "src": f"eval.{it['task_id']}",
                      "q": q, "a": "<answer>"})
    c.close()
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "data", "raw", "flashcards2.jsonl"))
    ap.add_argument("--gsm", type=int, default=500,
                    help="how many gsm8k_train math cards (Type A)")
    ap.add_argument("--b-cap", type=int, default=300,
                    help="cap on Type B cards (answer)")
    args = ap.parse_args()

    a_gsm = load_gsm(args.gsm)
    a_eval, b_eval = load_from_eval()

    A = a_gsm + a_eval
    B = b_eval[:args.b_cap]

    cards = A + B
    with open(args.out, "w") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")

    from collections import Counter
    print(f"wrote {len(cards)} cards -> {args.out}")
    print(f"  Type A (lookup): {len(A)}  [gsm8k={len(a_gsm)} eval-wrong={len(a_eval)}]")
    print(f"  Type B (answer): {len(B)}  [eval-right capped at {args.b_cap}]")
    print("  A:B ratio =", round(len(A) / max(1, len(B)), 2))


if __name__ == "__main__":
    main()
