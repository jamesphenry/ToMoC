#!/usr/bin/env python3
"""Capability audit across the ~/llm_eval/datasets capability suite.

Runs the REAL ToMoC loop (scripts/orchestrate.run_question) on each item of
each dataset, so the lookup -> vault -> web fallback and the wiki_write gate
all apply exactly as in production. Then scores per the dataset's declared
scorer:

  contains  : expected substring present in output (case-insensitive, weak)
  regex     : last number in output vs expected (exact)
  llm_judge : a JUDGE model grades {task, thinking, output} via the dataset's
              judge_prompt rubric. Judge backend is pluggable:

                --judge local      -> LOCAL smollm-1.7b (SOVEREIGN, default).
                                      Weak directional signal.
                --judge ollama[:m] -> a LOCAL ollama model (default m=
                                      qwen2.5:1.5b), queried over the local
                                      ollama HTTP API. Still fully local
                                      inference (the model is pulled once, then
                                      runs offline) -- this is the setup the
                                      ollama-benchmark projects use.

Outputs a per-item JSONL log under logs/audit_<model>_<ts>.jsonl and prints a
summary table. With >1 --model it enters COMPARE mode and also writes
benchmarks/adapter_comparison.md (version-vs-version table).

Usage:
  python -u scripts/audit_capabilities.py --model adapters/v17
  python -u scripts/audit_capabilities.py --model adapters/v17 \
        --judge ollama:qwen2.5:1.5b
  python -u scripts/audit_capabilities.py --model adapters/v16 adapters/v17 \
        --judge ollama
  python -u scripts/audit_capabilities.py --model adapters/v17 \
        --data-dir ~/llm_eval/datasets --only math_gsm summarization
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from orchestrate import run_question, Engine  # reuse the production loop

# datasets live in ~/llm_eval/datasets by default
DEFAULT_DATA = os.path.join(os.path.expanduser("~"), "llm_eval", "datasets")
DEFAULT_JUDGE = os.path.join(ROOT, "models", "smollm-1.7b-instruct")
KNOWN = ["brainteasers", "hallucinations", "reasoning_logic", "coding_func",
         "knowledge_qa", "summarization", "math_gsm"]
BENCH_DIR = os.path.join(ROOT, "benchmarks")


def dataset_path(name, data_dir):
    """Resolve a dataset .json, tolerating singular filenames
    (hallucination.json vs hallucinations)."""
    p = os.path.join(data_dir, name + ".json")
    if os.path.exists(p):
        return p
    sing = name.rstrip("s")
    p2 = os.path.join(data_dir, sing + ".json")
    return p2 if os.path.exists(p2) else p


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------
def score_contains(output, expected):
    if expected is None:
        return None
    return expected.strip().lower() in (output or "").strip().lower()


def score_regex(output, expected):
    """Robust final-number extraction (matches project norm_numeric).

    The dataset ships a malformed `answer_pattern` (`(?:$)?(-?\\d+...)` where
    `$` is an end anchor that breaks re.search). We ignore it and just take
    the LAST number in the output -- that's the model's final answer, which is
    what `expected` is. Exact numeric compare.
    """
    if expected is None:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", output or "")
    got = nums[-1] if nums else None
    try:
        return float(got) == float(expected)
    except (ValueError, TypeError):
        return got == expected


def parse_verdict(out):
    """Return (verdict, reason) from a judge's raw text. Reason keeps case."""
    raw = out or ""
    up = raw.upper()
    if "VERDICT: YES" in up:
        # locate in the ORIGINAL-case text so reason isn't mangled
        i = raw.upper().index("VERDICT: YES") + len("VERDICT: YES")
        return ("YES", raw[i:].strip()[:200])
    if "VERDICT: NO" in up:
        i = raw.upper().index("VERDICT: NO") + len("VERDICT: NO")
        return ("NO", raw[i:].strip()[:200])
    return ("UNK", raw.strip()[:200])


# --------------------------------------------------------------------------
# judge backends (pluggable)
# --------------------------------------------------------------------------
class LocalJudge:
    """Wraps a local HF Engine as the grader."""
    label = "local:smollm-1.7b"

    def __init__(self, judge_model):
        self.engine = Engine(judge_model, max_new_tokens=120, max_len=1024)

    def grade(self, prompt):
        (out,) = self.engine.generate_all([prompt], chunk=1)
        return out or ""


class OllamaJudge:
    """Queries a local ollama model over its HTTP API (no pip dep)."""
    def __init__(self, model="qwen2.5:1.5b"):
        self.model = model
        self.label = f"ollama:{model}"
        self.url = "http://localhost:11434/api/generate"

    def grade(self, prompt):
        # temperature:0 -> greedy/deterministic judging (reproducible scores).
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r).get("response", "")
        except Exception as e:
            return f"ERR: ollama call failed: {e}"


def build_prompt(rubric, task, thinking, output):
    if rubric:
        try:
            return rubric.format(task=task, thinking=thinking, output=output)
        except Exception:
            pass
    return ("You are a strict grader. Given the task, the model's reasoning, "
            "and its answer, decide if the answer is correct/acceptable.\n\n"
            f"TASK:\n{task}\n\nMODEL REASONING:\n{thinking}\n\n"
            f"MODEL ANSWER:\n{output}\n\n"
            "Reply with exactly one line:\nVERDICT: YES\nor\nVERDICT: NO\n"
            "Then optionally one short reason.")


def judge_one(judge, task, thinking, output, rubric):
    """Returns (verdict, reason). judge is a LocalJudge/OllamaJudge."""
    prompt = build_prompt(rubric, task, thinking, output)
    out = judge.grade(prompt)
    if out.startswith("ERR:"):
        return ("ERR", out)
    return parse_verdict(out)


# --------------------------------------------------------------------------
# audit one model across all sets
# --------------------------------------------------------------------------
# -------------------------------------------------------------------------
# audit one model across all sets  (BATCHED two-pass loop, BUG-005/007)
# -------------------------------------------------------------------------
def run_audit(engine, judge, sets, args, model_label):
    from orchestrate import (format_prompt, parse_call, resolve,
                             build_turn2_prompt, CHUNK)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(ROOT, "logs",
                            f"audit_{model_label}_{ts}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    summary = []
    print(f"\n==== Capability audit: {model_label} (judge={getattr(judge,'label','none')}) ====\n",
          flush=True)
    for name in sets:
        path = dataset_path(name, args.data_dir)
        if not os.path.exists(path):
            print(f"[warn] dataset not found: {path} (skipped)", flush=True)
            continue
        d = json.load(open(path))
        items = d.get("items", [])
        ds_scorer = d.get("scorer")
        rubric = d.get("judge_prompt")
        if args.max:
            items = items[:args.max]
        print(f"--- {name} (dataset scorer={ds_scorer}, n={len(items)}) ---",
              flush=True)

        # --- extract q / gold / scorer per item ---
        qs, golds, scorers = [], [], []
        for it in items:
            q = it.get("prompt") or it.get("question")
            gold = it.get("expected") or it.get("answer")
            scorer = it.get("scorer") or ds_scorer
            qs.append(q)
            golds.append(gold)
            scorers.append(scorer)

        # --- PASS 1: all turn-1 (emit TOOL call or direct answer), batched ---
        t1_prompts = [format_prompt({"q": q}) for q in qs]
        t1_outs = engine.generate_all(t1_prompts, chunk=CHUNK)

        # --- resolve each call (cheap, CPU) ---
        results, t2_prompts = [], []
        for q, out1 in zip(qs, t1_outs):
            out1 = out1.strip()
            did_call, tool, query, wf = parse_call(out1)
            if did_call:
                res = resolve(tool, query, None)
                results.append(res)
                result_str = (res.get("answer") if res.get("verdict") == "hit"
                              else "No answer found in the knowledge base.")
                t2_prompts.append(
                    build_turn2_prompt(format_prompt({"q": q}), out1, result_str))
            else:
                results.append(None)
                t2_prompts.append(None)

        # --- PASS 2: all turn-2 (feed result back -> final answer), batched ---
        t2_outs = [None] * len(items)
        batch_t2 = [(i, p) for i, p in enumerate(t2_prompts) if p is not None]
        if batch_t2:
            prompts = [p for _, p in batch_t2]
            gen = engine.generate_all(prompts, chunk=CHUNK)
            for (i, _), o in zip(batch_t2, gen):
                t2_outs[i] = o.strip()

        # --- score + log each row (logic identical to run_question) ---
        correct, rows = 0, []
        for i, (q, gold, sc) in enumerate(zip(qs, golds, scorers)):
            out1 = t1_outs[i].strip()
            did_call, tool, query, wf = parse_call(out1)
            res = results[i]
            rec = {"called": did_call, "tool": tool, "query": query,
                   "turn1": out1, "resolved": res}
            if not did_call:
                rec["final_answer"] = out1
            else:
                if res and res.get("verdict") == "proposed_write":
                    rec["final_answer"] = (
                        f"[PROPOSED wiki write] key={res.get('matched')!r} "
                        f"body={res.get('answer')!r} (needs human approval)")
                else:
                    rec["final_answer"] = t2_outs[i]

            ok = None
            if sc == "contains":
                ok = score_contains(rec["final_answer"], gold)
            elif sc == "regex":
                ok = score_regex(rec["final_answer"], gold)
            elif sc == "llm_judge":
                if judge is None:
                    ok = None
                else:
                    verd, reason = judge_one(
                        judge, q, rec["turn1"] or "",
                        rec["final_answer"] or "", rubric)
                    ok = (verd == "YES")
                    rec["judge_verdict"] = verd
                    rec["judge_reason"] = reason
            if ok:
                correct += 1
            row = {"set": name, "scorer": sc, "idx": i,
                   "prompt": q, "gold": gold, "correct": ok,
                   "called": rec["called"], "tool": rec["tool"],
                   "verdict": (rec.get("resolved") or {}).get("verdict"),
                   "final_answer": rec["final_answer"]}
            rows.append(row)
            mark = "OK " if ok else ("NO " if ok is False else "?? ")
            print(f"  {mark}[{i+1}/{len(items)}] ({sc}) {q[:54]!r} "
                  f"-> {str(rec['final_answer'])[:46]!r}", flush=True)

        with open(log_path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        pct = correct / len(items) if items else 0.0
        summary.append((name, ds_scorer, len(items), correct, pct))
        print(f"  => {name}: {correct}/{len(items)} = {pct:.2f} "
              f"(dataset scorer={ds_scorer})\n", flush=True)
    return summary, log_path


def render_one(summary):
    lines = []
    lines.append(f"{'set':16} {'scorer':10} {'n':>3} {'ok':>3} {'pct':>6}")
    for name, sc, n, ok, pct in summary:
        lines.append(f"{name:16} {sc:10} {n:>3} {ok:>3} {pct:>6.2f}")
    overall = sum(s[3] for s in summary)
    total = sum(s[2] for s in summary)
    lines.append(f"{'OVERALL':16} {'':10} {total:>3} {overall:>3} "
                 f"{overall/total if total else 0:>6.2f}")
    return "\n".join(lines)


def write_comparison(results, judge_label):
    """results: list of (model_label, summary). Writes markdown table."""
    os.makedirs(BENCH_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sets_order = [s[0] for s in results[0][1]]
    head = ("# Adapter Capability Comparison\n\n"
            f"Generated {ts} by `scripts/audit_capabilities.py` "
            f"(judge backend: **{judge_label}**).\n\n"
            "> **Reading this:** `contains`/`regex` columns are deterministic "
            "and trustworthy. `llm_judge` columns are graded by the judge "
            "backend above -- if it is the local 1.7b, treat those numbers as "
            "**directional only** (a weak grader). If it is `ollama:qwen2.5`, "
            "they are a much stronger signal but still automated.\n\n"
            "Cells show `correct/total` and pct. The right-most column is the "
            "macro-average across all 7 sets.\n\n")
    # per-set columns
    cols = "".join(f"| {s:18} " for s in sets_order) + "| OVERALL |\n"
    sep = "".join("|" + "-" * 20 for s in sets_order) + "|" + "-" * 10 + "|\n"
    rows = head + "| adapter " + cols + "|" + "-" * 9 + sep
    for label, summary in results:
        by_set = {s[0]: s for s in summary}
        cells = ""
        for s in sets_order:
            if s in by_set:
                _, _, n, ok, pct = by_set[s]
                cells += f"| {ok}/{n} ({pct:.0%}) "
            else:
                cells += "|  -   "
        # overall
        overall = sum(x[3] for x in summary)
        total = sum(x[2] for x in summary)
        opct = overall / total if total else 0
        rows += f"| {label:7} {cells}| {overall}/{total} ({opct:.0%}) |\n"
    rows += ("\n## Notes\n"
             "- Deterministic scorers: `knowledge_qa`, `reasoning_logic` "
             "(`contains`); `math_gsm` (`regex`).\n"
             "- Judge scorers: `brainteasers`, `coding_func`, `summarization`, "
             "`hallucination` traps (`llm_judge`).\n"
             "- `hallucination` mixes 10 closed-fact `contains` items + 10 "
             "trap `llm_judge` items (truthful-decline graded by the judge).\n"
             "- Per-item logs: `logs/audit_<adapter>_*.jsonl`.\n")
    out = os.path.join(BENCH_DIR, "adapter_comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(rows)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=[os.path.join(ROOT, "adapters", "v17")],
                    help="one or more adapter dirs (multiple => compare mode)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: " + " ".join(KNOWN))
    ap.add_argument("--max", type=int, default=0, help="cap items per set")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip llm_judge sets (no 2nd model load)")
    ap.add_argument("--judge", default="local",
                    help="local (default 1.7b) or ollama[:model]")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE,
                    help="HF path for --judge local")
    args = ap.parse_args()

    sets = args.only or [s for s in KNOWN if s != "hallucinations"]
    # warn about missing
    for s in sets:
        p = dataset_path(s, args.data_dir)
        if not os.path.exists(p):
            print(f"[warn] dataset not found: {p} (skipped)")

    # judge backend
    judge = None
    judge_label = "none"
    if not args.no_judge:
        if args.judge == "local" or args.judge.startswith("local"):
            try:
                judge = LocalJudge(args.judge_model)
                judge_label = judge.label
            except Exception as e:
                print(f"[judge] local FAILED ({e}); llm_judge -> UNK")
        elif args.judge.startswith("ollama"):
            model = args.judge.split(":", 1)[1] if ":" in args.judge else "qwen2.5:1.5b"
            judge = OllamaJudge(model)
            judge_label = judge.label
        else:
            print(f"[judge] unknown --judge {args.judge!r}; llm_judge -> UNK")
        if judge:
            print(f"[judge] backend = {judge_label}")

    models = args.model
    results = []
    try:
        for m in models:
            engine = Engine(m, max_new_tokens=160, max_len=512)
            summary, log = run_audit(engine, judge, sets, args,
                                     os.path.basename(m.rstrip("/")))
            results.append((os.path.basename(m.rstrip("/")), summary))
            print(f"\nper-item log: {log}\n")
            del engine
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    finally:
        if judge is not None and hasattr(judge, "engine"):
            del judge.engine

    if len(results) == 1:
        print("==== SUMMARY ====")
        print(render_one(results[0][1]))
    else:
        out = write_comparison(results, judge_label)
        print("==== COMPARISON ====")
        for label, summary in results:
            print(f"\n## {label}\n" + render_one(summary))
        print(f"\ncomparison written: {out}")


if __name__ == "__main__":
    main()
