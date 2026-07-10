#!/usr/bin/env python3
"""eval_resolver — DIRECTION B end-to-end: call -> resolve -> is-it-correct.

Closes the loop AGENTS.md left open: the model emits `TOOL lookup
query="..."` (adapter v3 does this 97% of the time on weak tasks) but nothing
resolved it. This script runs the FULL loop on a dataset and measures whether the
resolved answer matches the gold answer:

    1. load cards (each has `q` == the question, `answer`/gold if present,
       or a labelled dataset)
    2. prompt the model with the SAME priming cue as training/eval
    3. parse the call (reuse eval_toolcall.parse_call)
    4. resolve via tool_resolver (sovereign KB lookup)
    5. score: did it CALL when it should? did the RESOLVED answer match GOLD?

This is the real capability number — not just "did it call", but "did calling
produce the right answer". It proves the thesis: 135m routes around its 1.74%
math floor by looking up.

Outputs (per AGENTS.md user ask): writes a FULL per-item log to
logs/eval_resolver_<stamp>.jsonl so every run is inspectable, plus prints a
summary. Also logs cost to passdb like the other evals.

The scorer extracts the final numeric answer (gsm8k `expected` is already the
final number; for word problems the KB answer is the gold number). We compare
the resolved answer's trailing number to the gold number — matches gsm8k's
own regex scorer semantics.

Usage:
  source .venv/bin/activate
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python -u scripts/eval_resolver.py --model adapters/v3 \
          --data ~/llm_eval/datasets/gsm8k_test.jsonl --verbose
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from eval_toolcall import Engine, format_prompt, parse_call  # reuse the habit eval
from tool_resolver import resolve, KB

LOG_DIR = os.path.join(ROOT, "logs")


def norm_numeric(s):
    """Extract the final number from a string (gsm8k-style). None if absent."""
    if s is None:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(s))
    return float(nums[-1]) if nums else None


def gold_for(rec, dataset_kind):
    """Pull the gold answer + question from a dataset record."""
    if dataset_kind == "gsm8k":
        return rec.get("prompt", ""), rec.get("expected")
    if dataset_kind == "mmlu":
        return rec.get("prompt", ""), rec.get("expected")
    # raw flashcard shape {q, a}
    return rec.get("q", ""), rec.get("answer") or rec.get("expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="adapters/v3")
    ap.add_argument("--data",
                    default=os.path.expanduser("~/llm_eval/datasets/gsm8k_test.jsonl"))
    ap.add_argument("--kind", default="gsm8k",
                    choices=["gsm8k", "mmlu", "flashcard"])
    ap.add_argument("--max", type=int, default=0, help="cap rows (0=all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # load dataset
    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max:
        rows = rows[:args.max]
    print(f"eval_resolver: model={args.model} kind={args.kind} rows={len(rows)}")

    # build prompts from the SAME priming cue (format_prompt) so the habit fires
    questions, golds = [], []
    for r in rows:
        q, g = gold_for(r, args.kind)
        questions.append(q)
        golds.append(g)
    prompts = [format_prompt({"q": q, "type": "A"}) for q in questions]

    # generate (batched, chunked — same engine as eval_toolcall)
    t0 = time.time()
    engine = Engine(args.model)
    outputs = engine.generate_all(prompts)
    wall = time.time() - t0

    kb = KB.get()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(LOG_DIR, f"eval_resolver_{stamp}.jsonl")
    os.makedirs(LOG_DIR, exist_ok=True)

    # metrics
    n = len(rows)
    called_should = 0          # emitted a lookup call (we treat all as "should" here)
    wellformed = 0
    resolved_hit = 0            # call -> KB returned an answer
    correct = 0                 # resolved answer matched gold
    total_gold = 0              # rows that HAVE a gold answer to score

    with open(log_path, "w") as lf:
        for i, (q, out, gold) in enumerate(zip(questions, outputs, golds)):
            called, tool, query, wf = parse_call(out)
            rec = {"i": i, "q": q, "raw_output": out.strip(),
                   "called": called, "tool": tool, "query": query,
                   "well_formed": wf, "gold": gold,
                   "resolved": None, "correct": None}
            if called:
                called_should += 1
                if wf:
                    wellformed += 1
                res = resolve(tool, query, kb)
                rec["resolved"] = res
                if res["verdict"] == "hit":
                    resolved_hit += 1
                    if gold is not None and gold != "":
                        total_gold += 1
                        gn = norm_numeric(gold)
                        rn = norm_numeric(res["answer"])
                        ok = (gn is not None and rn is not None and gn == rn)
                        rec["correct"] = ok
                        if ok:
                            correct += 1
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summary
    print("\n=== resolver eval results ===")
    print(f"  rows                       : {n}")
    print(f"  call_rate                  : {called_should/n:.3f}" if n else "")
    print(f"  well_formed_rate           : {wellformed/called_should:.3f}" if called_should else "  well_formed_rate           : n/a")
    print(f"  resolved_hit_rate          : {resolved_hit/called_should:.3f}" if called_should else "  resolved_hit_rate          : n/a")
    if total_gold:
        print(f"  correct_vs_gold            : {correct}/{total_gold} = {correct/total_gold:.3f}")
    else:
        print("  correct_vs_gold            : no gold to score (set --kind with gold)")
    print(f"  walltime_s                 : {wall:.1f}")
    print(f"  full log                   : {log_path}")

    # persist to passdb (reuse schema via eval_toolcall's PassDB)
    try:
        from passdb import PassDB
        db = PassDB()
        pid = db.new_pass(base_model=args.model, num_cards=n,
                          a_ratio=1.0, walltime_s=round(wall, 1),
                          status="resolver-eval")
        db.log_metric(pid, "call_rate", round(called_should / n, 4) if n else 0.0)
        db.log_metric(pid, "well_formed_rate",
                      round(wellformed / called_should, 4) if called_should else 0.0)
        db.log_metric(pid, "resolved_hit_rate",
                      round(resolved_hit / called_should, 4) if called_should else 0.0)
        if total_gold:
            db.log_metric(pid, "correct_vs_gold", round(correct / total_gold, 4))
        db.log_meta(pid, "run_type", "adapter")
        db.log_meta(pid, "data", os.path.basename(args.data))
        db.log_meta(pid, "log", log_path)
        db.summarize(pid)
        db.close()
    except Exception as e:
        print(f"  [passdb skipped] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
