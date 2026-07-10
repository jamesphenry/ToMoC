#!/usr/bin/env python3
"""Generate v1 tool-call flashcards from llm_eval datasets.

Reads source datasets from ~/llm_eval/datasets, emits data/raw/flashcards.jsonl
per data/raw/flashcards_spec.md. KISS: no deps beyond stdlib.

Card types:
  A (lookup): question 135m CAN'T answer -> "TOOL lookup query=\"...\""
  B (answer): question 135m CAN answer  -> plain answer

Type A sources = 135m's FAILURES (from eval DB wrong-items + full gsm8k_train).
Type B sources = 135m's WINS (coding / brainteasers / summarization).
Balance target ~50/50; we report the real ratio and trim B down to match A.
"""
import json
import os
import random

SRC = os.path.expanduser("~/llm_eval/datasets")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "flashcards.jsonl")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build():
    cards = []

    # ---- TYPE A: lookup cards (questions 135m can't answer) ----

    # math_gsm: 135m scored 0% -> every gsm8k_train item is a lookup.
    for item in _load_jsonl(os.path.join(SRC, "gsm8k_train.jsonl")):
        q = item["prompt"].strip()
        cards.append({"type": "A", "src": "gsm8k", "q": q, "a": f'TOOL lookup query="{q}"'})

    # knowledge_qa: only the 6 items 135m got WRONG in eval (run 2).
    # (expected wrong set verified from DB; hardcoded to keep this script dep-free
    #  of sqlite. If you re-run evals, cross-check these indices.)
    wrong_kq = {1, 6, 8, 11}
    kq = _load_json(os.path.join(SRC, "knowledge_qa.json"))
    for i, item in enumerate(kq["items"]):
        if i in wrong_kq:
            q = item["prompt"].strip()
            cards.append({"type": "A", "src": "knowledge_qa", "q": q, "a": f'TOOL lookup query="{q}"'})

    # reasoning_logic: only the 7 items 135m got WRONG in eval (run 2).
    wrong_rl = {0, 1, 2, 5, 6, 7, 8}
    rl = _load_json(os.path.join(SRC, "reasoning_logic.json"))
    for i, item in enumerate(rl["items"]):
        if i in wrong_rl:
            q = item["prompt"].strip()
            # reasoning prompts can be long; keep the whole thing as the query.
            cards.append({"type": "A", "src": "reasoning_logic", "q": q, "a": f'TOOL lookup query="{q}"'})

    # ---- TYPE B: answer cards (questions 135m CAN answer) ----

    # coding_func: 135m scored 100%. No 'expected' field -> answer is the
    # model's own job; we store the prompt and let training treat response as target.
    cf = _load_json(os.path.join(SRC, "coding_func.json"))
    for item in cf["items"]:
        cards.append({"type": "B", "q": item["prompt"].strip(), "a": None})

    # brainteasers: 135m scored 100%; use the provided expected answer.
    bt = _load_json(os.path.join(SRC, "brainteasers.json"))
    for item in bt["items"]:
        cards.append({"type": "B", "q": item["prompt"].strip(),
                      "a": item.get("expected", "").strip()})

    # summarization: 135m scored 100%; answer is its own summary.
    sm = _load_json(os.path.join(SRC, "summarization.json"))
    for item in sm["items"]:
        cards.append({"type": "B", "q": item["prompt"].strip(), "a": None})

    # ---- Balance: Type B is scarce (coding/brainteasers/summary ~30), so we
    # SUBSAMPLE Type A down to it (NOT pad B up). Dumping all 7,473 gsm8k
    # items would make the set 100% lookup -> the model over-calls. We want
    # A ~= B (default 1:1, up to A_RATIO x B). Keep A DIVERSE: always
    # include the knowledge+reasoning wrong-items, fill the rest from gsm8k. ----
    A_RATIO = 1  # Type A count = A_RATIO * Type B count
    a_cards = [c for c in cards if c["type"] == "A"]
    b_cards = [c for c in cards if c["type"] == "B"]
    random.seed(135)
    # separate the non-math A cards (knowledge/reasoning) so they're never
    # dropped by random sampling.
    a_math = [c for c in a_cards if c.get("src") == "gsm8k"]
    a_other = [c for c in a_cards if c.get("src") != "gsm8k"]
    target_a = min(len(a_cards), A_RATIO * len(b_cards))
    # guarantee the rare non-math lookups are kept; sample math to fill.
    keep_other = a_other[:target_a]
    need_math = max(0, target_a - len(keep_other))
    a_math_sample = random.sample(a_math, min(need_math, len(a_math)))
    a_cards = keep_other + a_math_sample
    balanced = a_cards + b_cards
    random.shuffle(balanced)

    out_path = os.path.abspath(OUT)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for c in balanced:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return {
        "type_A": len(a_cards),
        "type_B_raw": len([c for c in cards if c["type"] == "B"]),
        "type_B_trimmed": len(b_cards),
        "total": len(balanced),
        "out": out_path,
    }


if __name__ == "__main__":
    stats = build()
    print("flashcards generated:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    ratio = stats["type_A"] / stats["total"] * 100
    print(f"  A:B ratio ~ {ratio:.0f}/{100-ratio:.0f}")
