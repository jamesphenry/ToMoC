#!/usr/bin/env python3
"""build_synth_cards.py — synthesize a bigger, balanced tool-habit dataset.

Mines on-disk sources into tool-habit flashcards in the {type,src,q,a} shape
consumed by train_adapter.py / eval_toolcall.py.

Card types (all share the SAME priming cue at train+eval time):
  Type A (lookup)  — drawn from 135m's FAILURES: emit `TOOL lookup query="..."`
  Type B (answer)  — drawn from 135m's WINS: emit a direct answer (no tool)
  Type C (run_code)— Phase 5: emit `TOOL run_code code="<expr>"` to COMPUTE
                     arithmetic instead of guessing or fetching.

We bias A-HEAVY on purpose: our measured failure is UNDER-call (call_rate 0.000),
not over-call. The 50/50 in the old spec prevented over-call; since we don't
over-call, we weight toward A to install the habit. Type C is added on top as a
DISJOINT skill (compute), sourced from sovereign synthetic arithmetic so it
never contradicts a Type A lookup card (same question, two targets = no learn).

Query is a VERBATIM copy of the question (KISS — model only decides
call-vs-answer and copies text; it never composes a search query).

Usage:
  python scripts/build_synth_cards.py --out data/raw/flashcards2.jsonl
"""
import argparse
import json
import os
import random
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


# --------------------------------------------------------------------------
# Type A (lookup) cards
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Type C (run_code) cards — sovereign synthetic arithmetic (no external deps)
# --------------------------------------------------------------------------
def mk_C(q, code, answer, src="synth.arith"):
    """Build a Type C (run_code) card. `code` is a SAFE arithmetic expression
    the resolver will execute; `answer` is its (precomputed) numeric result
    for training/eval scoring."""
    return {"type": "C", "src": src,
            "q": q, "a": f'TOOL run_code code="{code}"',
            "answer": str(answer), "code": code}


def gen_arith(seed=0, n=150):
    """Generate `n` simple arithmetic word problems with a known expression.

    Sovereign + deterministic: plain language templates + random integers.
    Expressions are always leaf-SAFE (no __import__/open/defs) so the sandbox
    accepts them. Returns list of (question, code, answer).
    """
    rng = random.Random(seed)
    out = []
    templates = [
        ("A baker made {a} loaves of bread and then baked {b} more. "
         "How many loaves does the baker have in total?",
         lambda a, b: (f"{a} + {b}", a + b)),
        ("There are {a} apples in a basket. If {b} apples are eaten, "
         "how many apples remain?",
         lambda a, b: (f"{a} - {b}", a - b)),
        ("A box holds {a} chocolates. If there are {b} such boxes, "
         "how many chocolates are there altogether?",
         lambda a, b: (f"{a} * {b}", a * b)),
        ("{a} candies are shared equally among {b} children. "
         "How many candies does each child get?",
         lambda a, b: (f"{a} / {b}", a / b)),
        ("A train travels {a} kilometers on Monday and {b} kilometers on "
         "Tuesday. What is the total distance travelled?",
         lambda a, b: (f"{a} + {b}", a + b)),
        ("A shirt costs ${a} and a pair of socks costs ${b}. "
         "What is the total cost of both items?",
         lambda a, b: (f"{a} + {b}", a + b)),
        ("Tom has {a} marbles. He buys {b} more marbles and then loses {c}. "
         "How many marbles does Tom have now?",
         lambda a, b, c: (f"{a} + {b} - {c}", a + b - c)),
        ("A classroom has {a} rows of desks with {b} desks in each row. "
         "How many desks are there in total?",
         lambda a, b: (f"{a} * {b}", a * b)),
        ("A library had {a} books. It received a donation of {b} books and "
         "then lent out {c} books. How many books remain?",
         lambda a, b, c: (f"{a} + {b} - {c}", a + b - c)),
        ("A recipe needs {a} grams of flour per batch. For {b} batches, "
         "how many grams of flour are needed?",
         lambda a, b: (f"{a} * {b}", a * b)),
    ]
    i = 0
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        tmpl, fn = templates[i % len(templates)]
        i += 1
        nargs = fn.__code__.co_argcount
        if nargs == 2:
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            if "/ {b}" in tmpl.replace(" ", ""):
                b = rng.randint(2, 12)  # keep division tidy
            if b == 0:
                b = rng.randint(2, 99)
            c = 0
            code, ans = fn(a, b)
        else:  # 3-arg templates
            a = rng.randint(2, 50)
            b = rng.randint(2, 50)
            c = rng.randint(1, a + b)
            code, ans = fn(a, b, c)
        # sanity: expression must evaluate to the claimed answer
        try:
            if eval(code) != ans:
                continue
        except Exception:
            continue
        # only keep integer/float-clean answers for scoring stability
        if isinstance(ans, float) and not ans.is_integer():
            continue
        ans = int(ans) if isinstance(ans, float) else ans
        code = code if isinstance(ans, int) else f"int({code})"
        q = tmpl.format(a=a, b=b, c=c)
        out.append((q, code, ans))
    return out


def load_run_code(n, seed=0):
    cards = []
    for q, code, ans in gen_arith(seed=seed, n=n):
        cards.append(mk_C(q, code, ans))
    return cards


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "data", "raw", "flashcards2.jsonl"))
    ap.add_argument("--gsm", type=int, default=500,
                    help="how many gsm8k_train math cards (Type A)")
    ap.add_argument("--b-cap", type=int, default=300,
                    help="cap on Type B cards (answer)")
    ap.add_argument("--c", type=int, default=150,
                    help="how many synthetic run_code (Type C) cards to add")
    ap.add_argument("--c-seed", type=int, default=0,
                    help="deterministic seed for the synthetic arithmetic")
    args = ap.parse_args()

    a_gsm = load_gsm(args.gsm)
    a_eval, b_eval = load_from_eval()
    c_cards = load_run_code(args.c, seed=args.c_seed)

    A = a_gsm + a_eval
    B = b_eval[:args.b_cap]
    C = c_cards

    cards = A + B + C
    with open(args.out, "w") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")

    from collections import Counter
    print(f"wrote {len(cards)} cards -> {args.out}")
    print(f"  Type A (lookup): {len(A)}  [gsm8k={len(a_gsm)} eval-wrong={len(a_eval)}]")
    print(f"  Type B (answer): {len(B)}  [eval-right capped at {args.b_cap}]")
    print(f"  Type C (run_code): {len(C)}  [synth.arith seed={args.c_seed}]")
    print("  A:B:C ratio =",
          round(len(A) / max(1, len(B)), 2), ":",
          round(len(B) / max(1, len(B)), 2), ":",
          round(len(C) / max(1, len(B)), 2))


if __name__ == "__main__":
    main()
