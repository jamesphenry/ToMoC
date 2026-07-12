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
        "SELECT task_id, question, expected, response, correct "
        "FROM runitems WHERE run_id=?", (rid,)
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
            # BUG-010 fix: use the real gold answer (expected), NOT the
            # literal "<answer>" placeholder. The placeholder taught the model
            # to emit the string "<answer>" verbatim on out-of-knowledge Qs.
            gold = (it["expected"] or "").strip() or (it["response"] or "").strip()
            if not gold:
                continue
            b.append({"type": "B", "src": f"eval.{it['task_id']}",
                      "q": q, "a": gold})
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


# Clean, UNAMBIGUOUS verb banks. Each template maps to ONE operator; joiners
# are chosen so + and * do NOT share phrasing (the v5 miss was "and" appearing
# in both add and mul templates -> model couldn't tell sum from product).
# Round-robin sampling => EVEN operator distribution (no sub/mixed skew, which
# pushed v5 toward subtract-by-default and dropped coverage 94.7% -> 87.6%).
ADD_TEMPLATES = [
    ("{a} children were at the park and {b} more arrived. How many are there now?",
     lambda a, b: (f"{a} + {b}", a + b)),
    ("A train travels {a} km on Monday and {b} km on Tuesday. Total distance?",
     lambda a, b: (f"{a} + {b}", a + b)),
    ("A shirt costs ${a} and socks cost ${b}. What is the total cost?",
     lambda a, b: (f"{a} + {b}", a + b)),
    ("There are {a} apples in one basket and {b} in another. How many in all?",
     lambda a, b: (f"{a} + {b}", a + b)),
    ("Sam had {a} cards and his friend gave him {b} more. How many does he have?",
     lambda a, b: (f"{a} + {b}", a + b)),
    ("A library had {a} books and received a donation of {b} books. Total books?",
     lambda a, b: (f"{a} + {b}", a + b)),
]
SUB_TEMPLATES = [
    ("There are {a} apples in a basket. If {b} apples are eaten, how many remain?",
     lambda a, b: (f"{a} - {b}", a - b)),
    ("A jar has {a} marbles. {b} marbles roll away. How many are left?",
     lambda a, b: (f"{a} - {b}", a - b)),
    ("{a} cookies were on a plate. {b} were eaten. How many cookies remain?",
     lambda a, b: (f"{a} - {b}", a - b)),
    ("Tom had ${a}. He spent ${b}. How much money is left?",
     lambda a, b: (f"{a} - {b}", a - b)),
    ("A store had {a} shirts. They sold {b} shirts. How many shirts remain?",
     lambda a, b: (f"{a} - {b}", a - b)),
    ("{a} birds were in a tree. {b} flew away. How many birds are still there?",
     lambda a, b: (f"{a} - {b}", a - b)),
]
MUL_TEMPLATES = [
    ("A recipe needs {a} g of flour per batch. For {b} batches, how many g?",
     lambda a, b: (f"{a} * {b}", a * b)),
    ("A classroom has {a} rows of desks with {b} desks in each row. Total desks?",
     lambda a, b: (f"{a} * {b}", a * b)),
    ("There are {a} packs with {b} candies in each pack. How many candies?",
     lambda a, b: (f"{a} * {b}", a * b)),
    ("A ticket costs ${a}. {b} tickets cost how much in total?",
     lambda a, b: (f"{a} * {b}", a * b)),
    ("Each bag has {a} oranges and there are {b} bags. How many oranges total?",
     lambda a, b: (f"{a} * {b}", a * b)),
    ("A box holds {a} chocolates and there are {b} such boxes. How many total?",
     lambda a, b: (f"{a} * {b}", a * b)),
]
DIV_TEMPLATES = [
    ("{a} candies are shared equally among {b} children. How many per child?",
     lambda a, b: (f"{a} / {b}", a / b)),
    ("{a} meters of ribbon are cut into {b} equal pieces. Length of each?",
     lambda a, b: (f"{a} / {b}", a / b)),
    ("{a} apples are split between {b} baskets evenly. How many per basket?",
     lambda a, b: (f"{a} / {b}", a / b)),
    ("{a} students form {b} equal teams. How many students per team?",
     lambda a, b: (f"{a} / {b}", a / b)),
]
MIXED_TEMPLATES = [
    ("A library had {a} books. It received {b} more and then lent out {c}. "
     "How many books remain?",
     lambda a, b, c: (f"{a} + {b} - {c}", a + b - c)),
    ("Tom has {a} marbles. He buys {b} more and then loses {c}. How many now?",
     lambda a, b, c: (f"{a} + {b} - {c}", a + b - c)),
    ("A shop had {a} hats. It sold {b} hats and then made {c} new hats. "
     "How many hats are there now?",
     lambda a, b, c: (f"{a} - {b} + {c}", a - b + c)),
    ("There were {a} people. {b} left and then {c} arrived. How many now?",
     lambda a, b, c: (f"{a} - {b} + {c}", a - b + c)),
    ("A farmer had {a} chickens. He bought {b} more and sold {c}. How many left?",
     lambda a, b, c: (f"{a} + {b} - {c}", a + b - c)),
    ("Sara earned ${a}. She spent ${b} and then earned ${c} more. "
     "How much money does she have?",
     lambda a, b, c: (f"{a} - {b} + {c}", a - b + c)),
]
# One flat round-robin pool => even operator distribution by construction.
_ALL_TEMPLATES = (ADD_TEMPLATES + SUB_TEMPLATES + MUL_TEMPLATES +
                 DIV_TEMPLATES + MIXED_TEMPLATES)


def gen_arith(seed=0, n=300):
    """Generate `n` synthetic arithmetic word problems (clean, balanced).

    Sovereign + deterministic. Round-robin over a flat pool of unambiguous
    templates so each operator gets ~equal exposure (no distribution skew).
    Expressions stay leaf-SAFE (no imports/open/defs) so the sandbox accepts
    them. Returns list of (question, code, answer).
    """
    rng = random.Random(seed)
    out = []
    attempts = 0
    i = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        tmpl, fn = _ALL_TEMPLATES[i % len(_ALL_TEMPLATES)]
        i += 1
        nargs = fn.__code__.co_argcount
        is_div = ("equally" in tmpl or "cut into" in tmpl
                  or "split between" in tmpl or "equal teams" in tmpl)
        if nargs == 2:
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            if is_div:
                # keep division tidy + integer-valued (a = b * k)
                b = rng.randint(2, 12)
                a = b * rng.randint(2, 12)
            c = 0
            code, ans = fn(a, b)
        else:  # 3-arg mixed templates
            a = rng.randint(2, 60)
            b = rng.randint(1, 40)
            c = rng.randint(1, 40)
            code, ans = fn(a, b, c)
        # sanity: expression must evaluate to the claimed answer
        try:
            if eval(code) != ans:
                continue
        except Exception:
            continue
        # only keep integer-clean answers for scoring stability
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
# Type D (two-turn) cards — Phase 6b: teach the CLOSING half of the loop.
# The model already learns turn-1 (emit a TOOL call). Phase 6 measured that at
# turn-2 — given the tool result fed back — v6 emits an EMPTY answer ~44% of the
# time (it was never trained on the `Tool result: X\nFinal answer:` format).
# A Type-D card supervises exactly that continuation: its `prompt_full` is the
# BYTE-IDENTICAL two-turn context orchestrate.build_turn2_prompt() produces at
# inference, and its target `a` is just the final answer. Loss lands only on the
# answer (the prompt — incl. the injected "Tool result:" — is masked), so the
# model learns to ECHO the result, NOT to hallucinate its own tool output.
# --------------------------------------------------------------------------
# Priming cue — MUST stay byte-identical to train_adapter.format/eval/orchestrate.
_CUE = ("If you are not certain of the answer, call the lookup tool "
        "instead of guessing.\n")


def _two_turn_prompt(q, call, result):
    """Reconstruct orchestrate.build_turn2_prompt() output byte-for-byte:
      <cue>Question: {q}\nAnswer or call a tool:\n<call>\nTool result: <r>\nFinal answer:
    """
    t1 = _CUE + f"Question: {q}\nAnswer or call a tool:\n"
    return f"{t1}{call.strip()}\nTool result: {result}\nFinal answer:"


def mk_D(q, call, result, answer, src="synth.twoturn"):
    """Build a Type D (two-turn) card. Target is the bare final answer so loss
    focuses on the turn-2 continuation (echo the tool result)."""
    return {"type": "D", "src": src, "q": q,
            "prompt_full": _two_turn_prompt(q, call, result),
            "a": str(answer), "answer": str(answer)}


def load_two_turn(n_runcode, n_lookup, seed=0):
    """Type-D cards for BOTH tools (the empty-turn-2 gap hit run_code AND lookup):
      - run_code: reuse the clean synth arithmetic; call=run_code, result=answer.
      - lookup:   gsm8k_train questions; call=lookup, result=gold expected number.
    Same-question overlap with Type C/A is intentional and NON-contradictory —
    the D prompt carries the tool-result context, so the target differs by design
    (reinforces the full trajectory rather than competing with turn-1).
    """
    cards = []
    # run_code two-turn (offset seed so we don't just clone the C set verbatim)
    for q, code, ans in gen_arith(seed=seed + 1000, n=n_runcode):
        cards.append(mk_D(q, f'TOOL run_code code="{code}"', ans, ans,
                          src="synth.twoturn.runcode"))
    # lookup two-turn from gsm8k_train (result = gold answer -> teach echo)
    added = 0
    with open(GSM_TRAIN) as f:
        for line in f:
            if added >= n_lookup:
                break
            r = json.loads(line)
            q = (r.get("prompt") or r.get("question") or "").strip()
            expected = r.get("expected")
            if not q or expected in (None, ""):
                continue
            # match mk_A truncation so the emitted query is consistent
            qq = q
            if len(qq) > MAX_Q:
                qq = qq[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
            ans = str(expected).strip()
            cards.append(mk_D(qq, f'TOOL lookup query="{qq}"', ans, ans,
                              src="synth.twoturn.lookup"))
            added += 1
    return cards


# -------------------------------------------------------------------------
# Type E (two-turn MISS) cards — Phase 7 lean fix: graceful KB-miss recovery.
# On a resolver miss, run_question feeds back the literal MISS_RESULT string
# (scripts/orchestrate.py run_question, miss branch). v8 was trained to ECHO
# the tool result, so facing that non-numeric string it either echoes the text
# or falls back to weak from-weights math and invents a number (the "wrong
# operation" symptom James saw). A Type-E card supervises the MISS-branch
# two-turn continuation with the SAME byte-identical mechanism as Type-D:
# prompt_full mirrors orchestrate.build_turn2_prompt(t1, call, MISS_RESULT);
# target = MISS_RESULT (echo honestly, do NOT guess). Covers BOTH miss flavors:
# lookup-miss (question not in KB) and run_code-miss (sandbox rejected code).
# --------------------------------------------------------------------------
# MUST stay byte-identical to orchestrate.run_question's miss result_str.
MISS_RESULT = "No answer found in the knowledge base."


def _miss_two_turn_prompt(q, call):
    """Mirror orchestrate.build_turn2_prompt on the MISS branch (result=MISS_RESULT)."""
    t1 = _CUE + f"Question: {q}\nAnswer or call a tool:\n"
    return f"{t1}{call.strip()}\nTool result: {MISS_RESULT}\nFinal answer:"


def mk_E(q, call, src="synth.miss"):
    """Build a Type E (miss-branch two-turn) card. Target is the honest miss
    string, so loss on turn-2 teaches the model to echo it instead of guessing."""
    return {"type": "E", "src": src, "q": q,
            "prompt_full": _miss_two_turn_prompt(q, call),
            "a": MISS_RESULT, "answer": MISS_RESULT}


def load_miss_two_turn(n_lookup, n_runcode, seed=0):
    cards = []
    # lookup misses: questions the resolver can't answer -> echo miss string
    added = 0
    with open(GSM_TRAIN) as f:
        for line in f:
            if added >= n_lookup:
                break
            r = json.loads(line)
            q = (r.get("prompt") or r.get("question") or "").strip()
            if not q:
                continue
            qq = q
            if len(qq) > MAX_Q:
                qq = qq[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
            cards.append(mk_E(qq, f'TOOL lookup query="{qq}"',
                              src="synth.miss.lookup"))
            added += 1
    # run_code misses: code the sandbox rejects -> echo miss string
    for q, code, ans in gen_arith(seed=seed + 2000, n=n_runcode):
        cards.append(mk_E(q, f'TOOL run_code code="{code}"',
                          src="synth.miss.runcode"))
    return cards


# -------------------------------------------------------------------------
# Type F (show-your-work) cards — human-authored grade-1-3 word problems.
# The user hand-annotates each with WORK (reasoning) + CODE (pure-arithmetic
# expr) + A (gold numeric answer). The model is trained to EMIT the work, then
# call run_code:  target = "<work> TOOL run_code code=\"<code>\""
# This teaches operation disambiguation (the "more/fewer/left/altogether" ->
# + - * / mapping) instead of guessing. Only NUMERIC rows live in the seed
# file; symbolic rows (comparison / geometry / clock / algebra / yes-no) were
# split out to f_cards_symbolic.txt (different schema) and are ignored here.
# Non-arithmetic CODE (N/A / eval-fail) is skipped defensively.
# -------------------------------------------------------------------------
F_SEED = os.path.join(ROOT, "data", "raw", "f_cards_seed.txt")
# cap the reasoning prefix so the full target fits the 256-token max-len
MAX_WORK = 140
# Rephrase F word-problems into direct "Compute this: <code>" prompts so the
# reasoning+run_code habit is triggered by computation requests, not word
# problems (which must route to lookup on gsm8k). See load_f_cards below.
REPHRASE_F_TO_COMPUTE = True


def load_f_cards(n, seed=0):
    """Parse f_cards_seed.txt into Type-F cards.

    Keeps only blocks whose CODE eval()s to a plain number (the file is
    already numeric-only after the split, but we defend against N/A / bad
    code so a stray row can't train a wrong run_code call). Returns up to `n`.
    """
    text = open(F_SEED).read()
    blocks, cur = [], {}
    def flush(c):
        if c.get("Q") is not None:
            blocks.append(c)
    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush(cur); cur = {}; continue
        if s.startswith("#"):
            continue
        if s.startswith("Q:"):      cur["Q"] = s[2:].strip()
        elif s.startswith("WORK:"): cur["WORK"] = s[5:].strip()
        elif s.startswith("CODE:"): cur["CODE"] = s[5:].strip()
        elif s.startswith("A:"):    cur["A"] = s[2:].strip()
    flush(cur)

    rng = random.Random(seed)
    rng.shuffle(blocks)

    cards = []
    for b in blocks:
        code = b["CODE"].strip()
        if code == "" or code.upper() == "N/A":
            continue
        try:
            val = eval(code, {"__builtins__": {}}, {})
        except Exception:
            continue
        if not isinstance(val, (int, float)):
            continue
        work = b["WORK"].strip()
        if len(work) > MAX_WORK:
            work = work[:MAX_WORK - 1].rsplit(" ", 1)[0] + "…"
        q = b["Q"].strip()
        # REPHRASE_F_TO_COMPUTE (default True): the seed problems are WORD
        # problems, but gsm8k is ALSO word problems — training "word problem ->
        # run_code" directly fights "word problem -> lookup" and poisoned v10/v11
        # routing on gsm8k (run_code fired on KB-answerable questions, ~2% correct).
        # Rephrase the F question into a direct-COMPUTE prompt so the model learns
        # "reasoning + run_code" is triggered by computation requests, NOT word
        # problems. Word problems keep routing to lookup (accurate on gsm8k);
        # the reasoning trail is preserved for the playground's arithmetic input.
        if REPHRASE_F_TO_COMPUTE:
            q = f"Compute this: {code}"
        cards.append(mk_F(q, work, code, b["A"].strip()))
        if len(cards) >= n:
            break
    return cards


def mk_F(q, work, code, answer, src="f_cards_seed"):
    """Type-F card: model narrates the work, then calls run_code."""
    return {"type": "F", "src": src, "q": q, "answer": answer,
            "a": f'{work} TOOL run_code code="{code}"'}


# -------------------------------------------------------------------------
# Type G (wiki) cards — Phase 7 #2: teach the model to ROUTE curated,
# general-knowledge questions to the disk-backed wiki (TOOL wiki) instead of
# the frozen gsm8k KB or run_code. Two card flavors (same as A/D split):
#   - single-turn (route):  q = wiki key -> a = `TOOL wiki query="<key>"`
#   - two-turn (echo):      q -> TOOL wiki query="key" ; result = body ;
#                           target = body  (closes the loop, mirrors Type-D)
# The wiki store (data/wiki/wiki.jsonl) is the single source of truth for the
# targets, so a human-edited wiki entry automatically updates training targets.
# -------------------------------------------------------------------------
WIKI_PATH = os.path.join(ROOT, "data", "wiki", "wiki.jsonl")


def load_wiki_cards(n_single, n_twoturn, seed=0):
    """Draw up to `n_single` routing cards and `n_twoturn` echo cards from the
    wiki store. Returns the combined list (each carries q/a per type)."""
    if not os.path.exists(WIKI_PATH):
        return []
    entries = []
    with open(WIKI_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    if not entries:
        return []

    rng = random.Random(seed)
    rng.shuffle(entries)

    cards = []
    # single-turn routing cards (the core new skill: q -> TOOL wiki)
    for e in entries:
        if len(cards) >= n_single:
            break
        key = e["key"].strip()
        body = e["body"].strip()
        if not key or not body:
            continue
        q = key
        if len(q) > MAX_Q:
            q = q[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
        cards.append({"type": "G", "src": "wiki.route",
                      "q": q, "a": f'TOOL wiki query="{q}"',
                      "answer": body})
    # two-turn echo cards (close the loop like Type-D)
    for e in entries:
        if len([c for c in cards if c["src"] == "wiki.echo"]) >= n_twoturn:
            break
        key = e["key"].strip()
        body = e["body"].strip()
        if not key or not body:
            continue
        q = key
        if len(q) > MAX_Q:
            q = q[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
        call = f'TOOL wiki query="{q}"'
        cards.append({"type": "G", "src": "wiki.echo",
                      "q": q,
                      "prompt_full": _two_turn_prompt(q, call, body),
                      "a": body, "answer": body})
    return cards


def load_write_cards(n, seed=0):
    """Phase 7 #1 (FLAG-TO-DATASET): teach the model to EMIT `TOOL wiki_write`
    when it has a verified fact to save.

    Single-turn cards only: the target is the write call itself (the human
    approval gate lives in tool_resolver.resolve, not in training). For each
    wiki entry we build a 'save this fact' instruction -> the write call.
    Returns up to `n` Type-H cards.
    """
    if not os.path.exists(WIKI_PATH) or n <= 0:
        return []
    entries = []
    with open(WIKI_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    if not entries:
        return []
    rng = random.Random(seed)
    rng.shuffle(entries)

    templates = [
        'Save this fact: {key} is {body}',
        'Remember that {key} is {body}.',
        'Store the following: {key} — {body}',
        'Add to your knowledge base: {key} = {body}',
    ]
    cards = []
    for e in entries:
        if len(cards) >= n:
            break
        key = e["key"].strip()
        body = e["body"].strip()
        if not key or not body:
            continue
        q = rng.choice(templates).format(key=key, body=body)
        # truncate to keep within MAX_Q so the model learns to close quotes
        if len(q) > MAX_Q:
            q = q[:MAX_Q - 1].rsplit(" ", 1)[0] + "…"
        call = f'TOOL wiki_write key="{key}" body="{body}"'
        cards.append({"type": "H", "src": "wiki.write",
                      "q": q, "a": call, "answer": body})
    return cards


# -------------------------------------------------------------------------
# assemble
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "data", "raw", "flashcards2.jsonl"))
    ap.add_argument("--gsm", type=int, default=500,
                    help="how many gsm8k_train math cards (Type A)")
    ap.add_argument("--b-cap", type=int, default=300,
                    help="cap on Type B cards (answer)")
    ap.add_argument("--c", type=int, default=300,
                    help="how many synthetic run_code (Type C) cards to add")
    ap.add_argument("--c-seed", type=int, default=0,
                    help="deterministic seed for the synthetic arithmetic")
    ap.add_argument("--d", type=int, default=0,
                    help="how many Type-D two-turn (final-answer) cards to add "
                         "[Phase 6b: closes the empty-turn-2 gap]")
    ap.add_argument("--d-runcode", type=int, default=150,
                    help="of --d, how many are run_code two-turn (rest: lookup)")
    ap.add_argument("--d-seed", type=int, default=0,
                    help="deterministic seed for the synthetic two-turn arithmetic")
    ap.add_argument("--e", type=int, default=0,
                    help="how many Type-E miss-branch two-turn cards to add "
                         "[Phase 7 lean fix: graceful KB-miss recovery]")
    ap.add_argument("--e-runcode", type=int, default=150,
                    help="of --e, how many are run_code-miss (rest: lookup-miss)")
    ap.add_argument("--e-seed", type=int, default=0,
                    help="deterministic seed for the synthetic run_code-miss cards")
    ap.add_argument("--f", type=int, default=0,
                    help="how many Type-F (show-your-work) cards to draw from "
                         "data/raw/f_cards_seed.txt [grade-1-3 word problems]")
    ap.add_argument("--f-seed", type=int, default=0,
                    help="deterministic seed for shuffling the Type-F seed file")
    ap.add_argument("--g", type=int, default=0,
                    help="how many Type-G (wiki routing) cards to draw from "
                         "data/wiki/wiki.jsonl [Phase 7 #2: route to TOOL wiki]")
    ap.add_argument("--g-twoturn", type=int, default=0,
                    help="of --g, how many are two-turn echo cards "
                         "(rest: single-turn routing)")
    ap.add_argument("--g-seed", type=int, default=0,
                    help="deterministic seed for shuffling the wiki store")
    ap.add_argument("--h", type=int, default=0,
                    help="how many Type-H (wiki WRITE) cards to draw from "
                         "data/wiki/wiki.jsonl [Phase 7 #1: emit TOOL wiki_write]")
    ap.add_argument("--h-seed", type=int, default=0,
                    help="deterministic seed for the Type-H write cards")
    args = ap.parse_args()

    a_gsm = load_gsm(args.gsm)
    a_eval, b_eval = load_from_eval()
    c_cards = load_run_code(args.c, seed=args.c_seed)
    d_cards = load_two_turn(args.d_runcode, max(0, args.d - args.d_runcode),
                            seed=args.d_seed) if args.d else []
    e_cards = load_miss_two_turn(max(0, args.e - args.e_runcode),
                                 args.e_runcode, seed=args.e_seed) if args.e else []
    f_cards = load_f_cards(args.f, seed=args.f_seed) if args.f else []
    g_cards = (load_wiki_cards(max(0, args.g - args.g_twoturn),
                               args.g_twoturn, seed=args.g_seed)
               if args.g else [])
    h_cards = load_write_cards(args.h, seed=args.h_seed) if args.h else []

    A = a_gsm + a_eval
    B = b_eval[:args.b_cap]
    C = c_cards
    D = d_cards
    E = e_cards
    F = f_cards
    G = g_cards
    H = h_cards

    cards = A + B + C + D + E + F + G + H
    with open(args.out, "w") as f:
        for c in cards:
            f.write(json.dumps(c) + "\n")

    from collections import Counter
    print(f"wrote {len(cards)} cards -> {args.out}")
    print(f"  Type A (lookup): {len(A)}  [gsm8k={len(a_gsm)} eval-wrong={len(a_eval)}]")
    print(f"  Type B (answer): {len(B)}  [eval-right capped at {args.b_cap}]")
    print(f"  Type C (run_code): {len(C)}  [synth.arith seed={args.c_seed}]")
    print(f"  Type D (two-turn): {len(D)}  [run_code="
          f"{sum(1 for x in D if 'runcode' in x.get('src',''))} "
          f"lookup={sum(1 for x in D if 'lookup' in x.get('src',''))} "
          f"seed={args.d_seed}]")
    print(f"  Type E (miss-branch): {len(E)}  [run_code="
          f"{sum(1 for x in E if 'runcode' in x.get('src',''))} "
          f"lookup={sum(1 for x in E if 'lookup' in x.get('src',''))} "
          f"seed={args.e_seed}]")
    print(f"  Type F (show-work): {len(F)}  [f_cards_seed seed={args.f_seed}]")
    print(f"  Type G (wiki routing): {len(G)}  [route="
          f"{sum(1 for x in G if 'route' in x.get('src',''))} "
          f"echo={sum(1 for x in G if 'echo' in x.get('src',''))} "
          f"seed={args.g_seed}]")
    print(f"  Type H (wiki WRITE): {len(H)}  [seed={args.h_seed}]")
    print("  A:B:C ratio =",
          round(len(A) / max(1, len(B)), 2), ":",
          round(len(B) / max(1, len(B)), 2), ":",
          round(len(C) / max(1, len(B)), 2))


if __name__ == "__main__":
    main()
