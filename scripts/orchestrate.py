#!/usr/bin/env python3
"""orchestrate — Phase 6: the CLOSING half of the ToMoC loop.

Phase 5 proved the model LEARNS the tool-call habit: on gsm8k it emits
`TOOL lookup query="..."` (call_rate ~0.96-0.99) and `TOOL run_code code="..."`
for arithmetic, and the sovereign resolver computes the right answer. But
eval_resolver STOPPED at "did the resolved answer match gold?" — it extracted
the answer from the tool result itself. The model never SAW the result, and
never emitted a final answer of its own.

This script closes that loop (the missing half of ToMoC):
    q -> turn1 (model emits a TOOL call) -> resolve -> turn2 (feed the
    tool result back) -> model emits a FINAL ANSWER.

Two modes:
  --data <jsonl> --kind flashcard|gsm8k   batch: score final-answer accuracy
  --ask "some question"                    single live demo (prints each turn)

The 360m adapter (v6) is the default — it's the locked-in base (best
speed/accuracy sweet spot: 96.7% run_code, 99.2% lookup, ~18 min train).

The canonical answer is taken from the MODEL's turn-2 output (that's the point
of Phase 6 — it emits the answer). We ALSO report `canonical_correct`, which
accepts the resolver's own answer as the source of truth when the model's
turn-2 output isn't a clean match — so we can see how much the loop depends on
the model vs the resolver.

Usage:
  source .venv/bin/activate
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # live single question:
  python -u scripts/orchestrate.py --model adapters/v6 --ask "48 - 5 + 20"
  # batch score on the hard arithmetic set:
  python -u scripts/orchestrate.py --model adapters/v6 \
          --data data/raw/flashcards2.jsonl --kind flashcard
  # batch score on gsm8k (lookup loop):
  python -u scripts/orchestrate.py --model adapters/v6 \
          --data ~/llm_eval/datasets/gsm8k_test.jsonl --kind gsm8k
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

from eval_toolcall import Engine, format_prompt, parse_call
from tool_resolver import resolve, KB

DEFAULT_MODEL = os.path.join(ROOT, "adapters", "v8")
MAX_NEW = 160         # turn-1 / turn-2 continuation budget; 160 (was 64) so a
                       # show-your-work prefix (Type-F) can precede the TOOL call
                       # without being truncated mid-code (BUG-008 class).
CHUNK = 16            # batched forward chunk (BUG-007)


def norm_numeric(s):
    """Extract the final number from a string. None if absent."""
    if s is None:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(s))
    return float(nums[-1]) if nums else None


def gold_for(rec, kind):
    if kind == "gsm8k":
        return rec.get("prompt", ""), rec.get("expected")
    if kind == "mmlu":
        return rec.get("prompt", ""), rec.get("expected")
    return rec.get("q", ""), rec.get("answer") or rec.get("expected")


def build_turn2_prompt(turn1_prompt, call_text, result_str):
    """Append the model's call + the resolved result + a final-answer cue."""
    return (f"{turn1_prompt}{call_text.strip()}\n"
            f"Tool result: {result_str}\nFinal answer:")


def run_question(engine, kb, q, gold, verbose=False):
    """Run the full ToMoC loop for one question. Returns a result dict."""
    t1 = format_prompt({"q": q})
    (out1,) = engine.generate_all([t1], chunk=1)
    out1 = out1.strip()
    called, tool, query, wf = parse_call(out1)

    rec = {"q": q, "gold": gold, "called": called, "tool": tool,
           "query": query, "turn1": out1, "resolved": None,
           "final_answer": None, "final_correct": None,
           "canonical_answer": None, "canonical_correct": None,
           "from_tool": called}

    if not called:
        # No tool call -> the model answered directly (Type B card).
        rec["final_answer"] = out1
        rec["canonical_answer"] = out1
    else:
        res = resolve(tool, query, kb)
        rec["resolved"] = res
        if res.get("verdict") == "proposed_write":
            # Phase 7 #1: the model PROPOSED a wiki edit. Sovereign gate — we
            # never auto-commit. In interactive mode the user approves; in batch
            # mode we just record the proposal (no store mutation). The loop ends
            # here (no turn-2 generation needed for a write proposal).
            rec["final_answer"] = (f"[PROPOSED wiki write] key={res.get('matched')!r} "
                                   f"body={res.get('answer')!r} "
                                   f"(needs human approval)")
            rec["canonical_answer"] = rec["final_answer"]
            rec["from_tool"] = True
        elif res.get("verdict") == "hit":
            result_str = res.get("answer")
            # turn 2: feed the result back
            t2 = build_turn2_prompt(t1, out1, result_str)
            (out2,) = engine.generate_all([t2], chunk=1)
            rec["final_answer"] = out2.strip()
            rec["canonical_answer"] = result_str   # resolver's own answer
        else:
            # Honest miss: don't feed the model the literal word "miss" (it will
            # guess). A real orchestrator would surface "no answer found" and let
            # the model either give up or reason from its own weights.
            result_str = "No answer found in the knowledge base."
            # turn 2: feed the result back
            t2 = build_turn2_prompt(t1, out1, result_str)
            (out2,) = engine.generate_all([t2], chunk=1)
            rec["final_answer"] = out2.strip()
            rec["canonical_answer"] = result_str   # resolver's own answer

    # score
    if gold is not None and gold != "":
        gn = norm_numeric(gold)
        if rec["final_answer"] is not None:
            fn = norm_numeric(rec["final_answer"])
            rec["final_correct"] = (gn is not None and fn is not None and gn == fn)
        if rec["canonical_answer"] is not None:
            cn = norm_numeric(rec["canonical_answer"])
            rec["canonical_correct"] = (gn is not None and cn is not None and gn == cn)

    if verbose:
        print(f"\nQ: {q}")
        if called:
            print(f"  turn1: {out1}")
            print(f"  tool={tool} query={query!r} verdict={rec['resolved'].get('verdict')}")
            print(f"  result fed back: {rec['canonical_answer']}")
            print(f"  turn2 final: {rec['final_answer']}")
        else:
            print(f"  turn1 (direct answer): {out1}")
        print(f"  gold={gold}  final_correct={rec['final_correct']}  "
              f"canonical_correct={rec['canonical_correct']}")
    return rec


def chat_display(rec):
    """Clean, conversational render of one ToMoC turn for the playground.

    Surfaces the model's *reasoning* (its tool-call choice) and the tool's
    result as a thinking step, then the final answer as a reply. Only echoes
    data the model/tool actually produced — no fabricated prose.
    """
    print("  reasoning:")
    if rec["called"]:
        print(f"    • model called: {rec['turn1']}")
        res = rec.get("resolved")
        if res:
            if res.get("verdict") == "hit":
                print(f"    • tool returned: {res.get('answer')}")
            else:
                print("    • tool returned: (no answer found in the knowledge base)")
    else:
        print("    • model answered directly (no tool call)")
    print(f"\n  answer: {rec['final_answer']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data",
                    default=os.path.expanduser("~/llm_eval/datasets/gsm8k_test.jsonl"))
    ap.add_argument("--kind", default="gsm8k",
                    choices=["gsm8k", "mmlu", "flashcard"])
    ap.add_argument("--ask", default=None, help="single live question")
    ap.add_argument("--chat", action="store_true",
                    help="interactive REPL: type a question, watch the ToMoC loop")
    ap.add_argument("--max", type=int, default=0, help="cap rows (0=all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    kb = KB.get()
    engine = Engine(args.model, max_new_tokens=MAX_NEW)

    if args.chat:
        try:
            import readline  # arrow keys + history (no effect on plain pipes)
        except Exception:
            pass
        print(f"smol ToMoC playground — model={args.model}")
        print("type a question, 'quit'/'exit' to leave. The model will call a "
              "tool, get the result, then answer.")
        print("  /export [path]  save this conversation as markdown")
        print("  /mark <n> <seen|fixed>  mark a turn's review status\n")
        transcript = []  # each: dict(q, call, verdict, result, final)

        def render_md(path, turns, model):
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            out = [f"# smol ToMoC conversation", "",
                   f"- model: `{model}`", f"- exported: {ts}",
                   f"- turns: {len(turns)}", ""]
            for i, t in enumerate(turns, 1):
                out.append(f"## Turn {i}  _[status: {t.get('status','new')}]_")
                out.append("")
                out.append(f"**You:** {t['q']}")
                out.append("")
                out.append(f"**Model (turn-1 call):** `{t['call']}`")
                if t.get("verdict") is not None:
                    out.append(f"**Tool:** {t['verdict']}"
                               + (f" → `{t['result']}`" if t.get("result") is not None else ""))
                out.append(f"**Final answer:** {t['final']}")
                out.append("")
            out.append("---")
            out.append("_Exported by orchestrate.py --chat. Edit the `[status: ...]` "
                       "tags to seen/fixed and share back for review._")
            with open(path, "w") as f:
                f.write("\n".join(out) + "\n")

        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not q:
                continue
            low = q.lower()
            # exit: bare word or slash form (don't ctrl-c)
            if low in ("quit", "exit", "q", "/quit", "/exit", "/bye"):
                print("bye")
                break
            # slash commands
            if q.startswith("/export"):
                parts = q.split()
                path = parts[1] if len(parts) > 1 else \
                    os.path.join(ROOT, "logs", "chat_export.md")
                render_md(path, transcript, args.model)
                print(f"  exported {len(transcript)} turns -> {path}")
                continue
            if q.startswith("/mark"):
                parts = q.split()
                if len(parts) >= 3 and parts[1].isdigit():
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(transcript):
                        transcript[idx]["status"] = parts[2]
                        print(f"  turn {parts[1]} -> {parts[2]}")
                    else:
                        print("  no such turn")
                else:
                    print("  usage: /mark <n> <seen|fixed>")
                continue
            rec = run_question(engine, kb, q, None, verbose=False)
            chat_display(rec)
            transcript.append({
                "q": q,
                "call": rec["turn1"],
                "verdict": (rec["resolved"].get("verdict")
                            if rec.get("resolved") else None),
                "result": (rec["resolved"].get("answer")
                           if rec.get("resolved") and
                           rec["resolved"].get("verdict") == "hit" else None),
                "final": rec["final_answer"],
                "status": "new",
            })
        # auto-save on exit (non-destructive: only writes if we had turns)
        if transcript:
            auto = os.path.join(ROOT, "logs", "chat_last.md")
            render_md(auto, transcript, args.model)
            print(f"  auto-saved last session -> {auto}")
        return

    if args.ask:
        run_question(engine, kb, args.ask, None, verbose=True)
        return

    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max:
        rows = rows[:args.max]
    print(f"orchestrate: model={args.model} kind={args.kind} rows={len(rows)}")

    questions, golds = [], []
    for r in rows:
        q, g = gold_for(r, args.kind)
        questions.append(q)
        golds.append(g)

    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(ROOT, "logs", f"orchestrate_{stamp}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    n = len(rows)
    called = 0
    resolved_hit = 0
    final_correct = 0
    canonical_correct = 0
    total_gold = 0

    # Batched two-pass loop (BUG-005/007: batch generates, chunk to bound VRAM).
    # Pass 1: every row's turn-1 (emit TOOL call or direct answer).
    t1_prompts = [format_prompt({"q": q}) for q in questions]
    t1_outs = engine.generate_all(t1_prompts, chunk=CHUNK)

    # Resolve each turn-1 call (cheap, CPU).
    t2_prompts = []
    results = []
    for q, out1 in zip(questions, t1_outs):
        out1 = out1.strip()
        did_call, tool, query, wf = parse_call(out1)
        if did_call:
            res = resolve(tool, query, kb)
            results.append(res)
            result_str = (res.get("answer") if res.get("verdict") == "hit"
                          else "No answer found in the knowledge base.")
            t2_prompts.append(build_turn2_prompt(
                format_prompt({"q": q}), out1, result_str))
        else:
            results.append(None)
            t2_prompts.append(None)  # no second turn needed

    # Pass 2: every row's turn-2 (feed result back -> final answer). For rows
    # that didn't call a tool, reuse turn-1 output as the final answer.
    t2_outs = [None] * n
    idx = 0
    batch_t2 = [(i, p) for i, p in enumerate(t2_prompts) if p is not None]
    if batch_t2:
        prompts = [p for _, p in batch_t2]
        gen = engine.generate_all(prompts, chunk=CHUNK)
        for (i, _), o in zip(batch_t2, gen):
            t2_outs[i] = o.strip()

    with open(log_path, "w") as lf:
        for i, (q, gold) in enumerate(zip(questions, golds)):
            out1 = t1_outs[i].strip()
            did_call, tool, query, wf = parse_call(out1)
            res = results[i]
            rec = {"q": q, "gold": gold, "called": did_call, "tool": tool,
                   "query": query, "turn1": out1, "resolved": res,
                   "final_answer": None, "final_correct": None,
                   "canonical_answer": None, "canonical_correct": None,
                   "from_tool": did_call}
            if did_call:
                called += 1
                if res and res.get("verdict") == "hit":
                    resolved_hit += 1
                rec["final_answer"] = t2_outs[i]
                rec["canonical_answer"] = (res.get("answer") if res and
                                            res.get("verdict") == "hit" else None)
            else:
                rec["final_answer"] = out1
                rec["canonical_answer"] = out1
            # score
            if gold is not None and gold != "":
                gn = norm_numeric(gold)
                if rec["final_answer"] is not None:
                    fn = norm_numeric(rec["final_answer"])
                    rec["final_correct"] = (gn is not None and fn is not None
                                            and gn == fn)
                if rec["canonical_answer"] is not None:
                    cn = norm_numeric(rec["canonical_answer"])
                    rec["canonical_correct"] = (gn is not None and cn is not None
                                                and gn == cn)
            if rec["final_correct"] is not None:
                total_gold += 1
                if rec["final_correct"]:
                    final_correct += 1
            if rec["canonical_correct"] is not None and rec["canonical_correct"]:
                canonical_correct += 1
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    wall = time.time() - t0

    # persist to passdb (guardrail: every GPU run logs cost)
    try:
        from passdb import PassDB
        db = PassDB()
        pid = db.new_pass(base_model=args.model, num_cards=n,
                          a_ratio=1.0, walltime_s=round(wall, 1),
                          status="orchestrate-eval")
        if n:
            db.log_metric(pid, "call_rate", round(called / n, 4))
        if called:
            db.log_metric(pid, "resolved_hit_rate", round(resolved_hit / called, 4))
        if total_gold:
            db.log_metric(pid, "final_answer_correct",
                          round(final_correct / total_gold, 4))
            db.log_metric(pid, "canonical_correct",
                          round(canonical_correct / total_gold, 4))
        db.log_meta(pid, "run_type", "orchestrate")
        db.log_meta(pid, "kind", args.kind)
        print(f"  logged pass id={pid} cost=${db.compute_cost(round(wall,1)):.4f}")
    except Exception as e:
        print(f"  [warn] passdb log failed: {e}")
    print(f"  rows                  : {n}")
    print(f"  call_rate             : {called/n:.3f}" if n else "")
    print(f"  resolved_hit_rate     : {resolved_hit/called:.3f}" if called else "  resolved_hit_rate     : n/a")
    if total_gold:
        print(f"  final_answer_correct  : {final_correct}/{total_gold} = {final_correct/total_gold:.3f}")
        print(f"  canonical_correct     : {canonical_correct}/{total_gold} = {canonical_correct/total_gold:.3f}")
    print(f"  walltime_s            : {wall:.1f}")
    print(f"  full log              : {log_path}")


if __name__ == "__main__":
    main()
