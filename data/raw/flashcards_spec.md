# Flashcard spec — v1 LoRA (smolLM:135m, `lookup` only)

> Goal: teach 135m ONE habit — when it can't answer from memory, write a
> tool-call; when it can, just answer. The hard part is the BALANCE
> (don't over-call, don't under-call). This file defines the card shape
> and the mix. No code yet.

## The two card types

### TYPE A — "call the tool" (lookup)
Used for questions 135m CAN'T reliably answer from memory.
The model should emit the call and STOP (no answer from it).

    Q: <question>
    A: TOOL lookup query="<what to look up>"

Example (math — 135m scores 0% here, so always a lookup):
    Q: Natalia sold clips to 48 friends in April and half as many in May. How many altogether?
    A: TOOL lookup query="Natalia sold clips to 48 friends in April and half as many in May. How many altogether?"

Example (knowledge — 135m scores 60% here, so we pick the 40% it gets WRONG):
    Q: What year did the Ming dynasty begin?
    A: TOOL lookup query="What year did the Ming dynasty begin?"

### TYPE B — "answer normally" (no tool)
Used for questions 135m CAN answer (its strong tasks). Teaches it NOT
to call a tool for everything.

    Q: <question>
    A: <answer>

Example (coding — 135m scores 100%):
    Q: Write a Python function that returns the sum of a list.
    A: def sum_list(xs): return sum(xs)

Example (brainteasers — 135m scores 100%):
    Q: I have two coins that total 30 cents and one is not a nickel. What are they?
    A: A quarter and a nickel (one is not a nickel, the other is).

Example (summarization — 135m scores 100%):
    Q: Summarize: <text>
    A: <short summary>

## THE MIX — this is the balance knob
If Type A >> Type B, the model over-calls. If B >> A, it under-calls
and just guesses. Target a BALANCED set, roughly:

    Type A (lookup):  50%
    Type B (answer) :  50%

But Type B must span the model's STRONG tasks (coding, brainteasers,
summarization, mmlu-algebra) so it learns those are "answer" territory.
Type A must span its WEAK tasks (math_gsm, knowledge_qa misses,
reasoning_logic misses) so it learns those are "lookup" territory.

Within Type A, weight toward the tasks it fails hardest:
    math_gsm         : heavy   (0% -> almost always a lookup card)
    knowledge_qa     : medium (use only the items it got WRONG in eval)
    reasoning_logic   : medium (use only the items it got WRONG in eval)

## Source data (already on disk)
    llm_eval/datasets/gsm8k_train.jsonl   (7,473 math problems; prompt+answer)
        -> Type A math cards (lookup the known answer)
    llm_eval/datasets/knowledge_qa.json    (facts; 135m got 40% wrong)
        -> Type A cards from the WRONG ones only
    llm_eval/datasets/reasoning_logic.json (135m got 70% wrong)
        -> Type A cards from the WRONG ones only
    llm_eval/datasets/coding_func.json      (135m 100%)
        -> Type B cards
    llm_eval/datasets/brainteasers.json   (135m 100%)
        -> Type B cards
    llm_eval/datasets/summarization.json  (135m 100%)
        -> Type B cards

## Why this reaches the balance
- Type A draws from 135m's FAILURES -> it learns "when I'd be wrong, call."
- Type B draws from 135m's WINS -> it learns "when I'm right, just answer."
- The 50/50 split prevents the dominant behavior from being over-call.
- The query field is a verbatim (or lightly cleaned) copy of the question,
  so the model never has to COMPOSE a search query — only decide call-vs-answer
  and copy the text. That's the smallest possible learning burden for 135m.

## Open knobs (tune after first train)
- Exact A:B ratio (start 50/50, adjust from eval).
- Whether query should be cleaned/shortened vs verbatim.
- How many reasoning/knowledge WRONG-items to include (start: all of them).
- Negative examples: maybe a few Type B cards that LOOK like lookup tasks
  but are actually answerable (hard negatives) — optional v1.1.

## Out of scope (see future.md)
JSON format, grammar-constrained output, run_code tool, SearXNG, LLM-wiki.
This spec is mini-format + single `lookup` tool + balanced call/answer habit.
