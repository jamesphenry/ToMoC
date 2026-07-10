#!/usr/bin/env python3
"""eval_toolcall — custom console eval for the tool-call habit.

Separate from llm_eval (no UI, verbose console). Measures the CALL DECISION,
not raw accuracy:
  - call_rate_when_should : of Type A cards, how many emitted a TOOL call
  - over_call_rate        : of Type B cards, how many wrongly emitted a TOOL call
  - correct_tool_rate     : of calls emitted, how many named `lookup` (only tool)
  - well_formed_rate      : of calls emitted, how many matched the mini-format
Also logs GPU/walltime via passdb when a model is loaded on the P4.

Runs on the BASE model (baseline) or an adapter (after training). The base
model has no tool habit, so baseline should show ~0 call_rate / ~0 over_call
(a known floor to beat).

Usage:
  python scripts/eval_toolcall.py --model smollm:135m --data data/raw/flashcards.jsonl --verbose
  python scripts/eval_toolcall.py --base-path adapters/my-lora --verbose
"""
import argparse
import json
import os
import re
import time

from passdb import PassDB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CALL_RE = re.compile(r'TOOL\s+lookup\s+query="(.+?)"\s*"$', re.DOTALL)
# looser first-pass detector: did it emit anything resembling a TOOL line?
TOOL_HINT_RE = re.compile(r'TOOL\s+(\w+)', re.IGNORECASE)


def load_cards(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def generate(model, prompt, max_new_tokens=64):
    """Generate one completion. Uses transformers if a local path is given,
    else falls back to Ollama's /api/generate for a named model."""
    if os.path.isdir(model) or model.endswith(".gguf"):
        return _generate_transformers(model, prompt, max_new_tokens)
    return _generate_ollama(model, prompt, max_new_tokens)


def _generate_ollama(model, prompt, max_new_tokens):
    import urllib.request
    import json as _json
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_new_tokens, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return _json.loads(r.read().decode())["response"]


def _generate_transformers(model_path, prompt, max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                               device_map="auto")
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens,
                           temperature=0.0, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def format_prompt(card):
    """Build the prompt we show the model for this card."""
    return f"Question: {card['q']}\nAnswer or call a tool:\n"


def parse_call(text):
    """Return (called: bool, tool: str|None, query: str|None, well_formed: bool)."""
    m = CALL_RE.search(text)
    if m:
        return True, "lookup", m.group(1), True
    hint = TOOL_HINT_RE.search(text)
    if hint:
        # emitted a TOOL line but not well-formed
        return True, hint.group(1).lower(), None, False
    return False, None, None, False


def evaluate(model, cards, verbose=False):
    a_cards = [c for c in cards if c["type"] == "A"]
    b_cards = [c for c in cards if c["type"] == "B"]
    stats = {"A_total": len(a_cards), "B_total": len(b_cards),
             "A_called": 0, "A_wellformed": 0, "A_correct_tool": 0,
             "B_called": 0, "B_wellformed": 0}

    def run(card, is_A):
        out = generate(model, format_prompt(card))
        called, tool, query, wf = parse_call(out)
        if is_A:
            stats["A_called"] += int(called)
            stats["A_wellformed"] += int(called and wf)
            stats["A_correct_tool"] += int(called and tool == "lookup")
        else:
            stats["B_called"] += int(called)
            stats["B_wellformed"] += int(called and wf)
        if verbose:
            tag = "A" if is_A else "B"
            print(f"[{tag}] called={called} tool={tool} wf={wf}")
            print(f"    Q: {card['q'][:70]}")
            print(f"    -> {out.strip()[:80]!r}\n")

    for c in a_cards:
        run(c, True)
    for c in b_cards:
        run(c, False)

    metrics = {}
    if stats["A_total"]:
        metrics["call_rate_when_should"] = stats["A_called"] / stats["A_total"]
        metrics["well_formed_rate"] = stats["A_wellformed"] / stats["A_total"]
        metrics["correct_tool_rate"] = (stats["A_correct_tool"] / stats["A_called"]
                                        if stats["A_called"] else 0.0)
    if stats["B_total"]:
        metrics["over_call_rate"] = stats["B_called"] / stats["B_total"]
    return metrics, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm:135m", help="Ollama model name or local path")
    ap.add_argument("--data", default=os.path.join(ROOT, "data/raw/flashcards.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cards = load_cards(args.data)
    print(f"eval_toolcall: model={args.model} cards={len(cards)}")
    t0 = time.time()
    metrics, stats = evaluate(args.model, cards, verbose=args.verbose)
    wall = time.time() - t0

    print("\n=== results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    print(f"  walltime_s: {wall:.1f}")

    # persist to passdb
    db = PassDB()
    pid = db.new_pass(base_model=args.model, num_cards=len(cards),
                      a_ratio=stats["A_total"] / len(cards),
                      walltime_s=round(wall, 1),
                      status="eval-only")
    for k, v in metrics.items():
        db.log_metric(pid, k, round(v, 4))
    db.log_meta(pid, "run_type", "baseline" if "smollm" in args.model and "/" not in args.model else "adapter")
    db.log_meta(pid, "data", os.path.basename(args.data))
    db.summarize(pid)
    db.close()


if __name__ == "__main__":
    main()
