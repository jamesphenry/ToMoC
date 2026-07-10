#!/usr/bin/env python3
"""eval_gsm8k_hf — batched gsm8k_test eval on an HF model (base OR adapter).

Reuses eval_toolcall.Engine.generate_all so we never generate one row at a
time (that's why the Ollama run crawls — per-call generate pegs a CPU core).
Chunks the 1319-row test into GPU-sized batches; full pass runs in minutes.

Scorer mirrors llm_eval's `regex` scorer: pull the last number from the
response, compare to `expected`. Reports accuracy + walltime.

Usage:
  python scripts/eval_gsm8k_hf.py
  python scripts/eval_gsm8k_hf.py --model adapters/v2 --batch 24
  python scripts/eval_gsm8k_hf.py --data ~/llm_eval/datasets/gsm8k_test.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_toolcall import Engine, DEFAULT_BASE

DEFAULT_DATA = os.path.expanduser("~/llm_eval/datasets/gsm8k_test.jsonl")

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def build_prompt(eng, row):
    """Apply the instruct chat template so the HF base sees the same shape
    Ollama would send via its API."""
    msgs = [{"role": "user", "content": row["prompt"]}]
    return eng.tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def extract_pred(text):
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def score_row(row, out):
    pred = extract_pred(out)
    if pred is None:
        return False
    exp = str(row["expected"]).replace(",", "").strip()
    try:
        return abs(float(pred) - float(exp)) < 1e-6
    except ValueError:
        return pred == exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE,
                    help="HF base dir or LoRA adapter dir")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="eval only first N rows (smoke test)")
    ap.add_argument("--out", default="",
                    help="optional path to write a JSON summary")
    args = ap.parse_args()

    with open(args.data) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"eval_gsm8k_hf: model={args.model} rows={len(rows)} batch={args.batch}")

    eng = Engine(args.model, max_new_tokens=args.max_new, max_len=320)
    dev = next(eng.mdl.parameters()).device
    print(f"  model on {dev}")

    correct = 0
    total = 0
    t0 = time.time()
    # full per-item log for every run (user ask: log full output going forward)
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(ROOT, "logs", f"eval_gsm8k_hf_{stamp}.jsonl")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_f = open(log_path, "w")
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        prompts = [build_prompt(eng, r) for r in chunk]
        outs = eng.generate_all(prompts)
        for r, o in zip(chunk, outs):
            total += 1
            ok = score_row(r, o)
            correct += int(ok)
            log_f.write(json.dumps({
                "i": total - 1, "prompt": r.get("prompt"),
                "expected": r.get("expected"), "raw_output": o.strip(),
                "correct": ok,
            }, ensure_ascii=False) + "\n")
        acc = correct / total if total else 0
        print(f"  {total}/{len(rows)}  acc={acc:.3f}  "
              f"{time.time() - t0:.1f}s", flush=True)
    log_f.close()

    wall = time.time() - t0
    # capture GPU mem if on cuda (mirrors other eval scripts)
    gpu_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass

    acc = correct / total if total else 0
    print(f"\n=== RESULT: acc={acc:.4f} ({correct}/{total}) "
          f"wall={wall:.1f}s ===")
    print(f"  full log: {log_path}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "rows": total,
                       "correct": correct, "acc": acc,
                       "wall_s": round(wall, 1)}, f, indent=2)
        print(f"  wrote {args.out}")

    # persist to passdb so EVERY GPU run tracks cost (sovereignty metric)
    try:
        from passdb import PassDB
        db = PassDB()
        pid = db.new_pass(base_model=args.model, num_cards=total,
                          a_ratio=1.0, walltime_s=round(wall, 1),
                          gpu_mem_used_mb=gpu_mb, status="gsm8k-bench")
        db.log_metric(pid, "acc", round(acc, 4))
        db.log_meta(pid, "run_type", "adapter" if os.path.isdir(args.model) else "base")
        db.log_meta(pid, "data", os.path.basename(args.data))
        db.log_meta(pid, "log", log_path)
        db.summarize(pid)
        db.cost_report()
        db.close()
    except Exception as e:
        print(f"  [passdb skipped] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
