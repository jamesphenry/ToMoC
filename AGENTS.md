# AGENTS.md — smol-lab checkpoint

> Read this first if you are a new/continued coding session. It is a
> resume-point, not a spec. The living journal is `wiki/JOURNAL.md`; bugs in
> `wiki/BUGS.md`; future ideas in `future.md`; **per-run cost breakdown in
> `runs.md`**.

## What this project is
Teach a *tiny* LLM (smollm-135m-instruct, fp16) to **look things up via a tool
call instead of guessing**. The model has NO native tool support (confirmed vs
Ollama) — so we teach the habit with a LoRA adapter that emits a mini call
script: `TOOL lookup query="<question>"`. Thesis: *functions ARE its knowledge*;
reasoning > raw smarts; sovereignty (homelab-only, no external APIs).

## Current state (as of 2026-07-11, after Phase 6 loop — 360m/v6 production base)
- **Best adapters (same 2-tool format, 1127-card clean-balanced set):**
  - `adapters/v7/` — LoRA on **smolLM-1.7B** (hidden 2048, 24 layers).
    **Best compute**: run_code **100% (300/300)** on the hard 300-card set, 0
    misses (incl. division + 2-step). Lookup gsm8k **99.0% (1253/1266)**,
    call_rate 0.964. Slow on the P4 (~59 min train / ~49 min eval) but fits
    (4236 MB of 7680). **Default best if you can spare the wall-clock.**
  - `adapters/v6/` — LoRA on **smolLM-360m-instruct** (hidden 960, 32 layers).
    The **speed/accuracy sweet spot**: run_code **96.7% (289/299)**, lookup
    **99.2% (1280/1290)**, trains in 18 min / evals in ~23 min, 1590 MB.
  - `adapters/v5b/` — LoRA on **smolLM-135m**. run_code **89.0% (266/299)**,
    lookup 98.5%. Use only if you MUST stay at 135m (fastest iteration).
  - `adapters/v4/` — 135m, 2 tools; on the hard 300-card set run_code only
    71.1% (never trained on division). Lookup 98.4%.
  - (v5, skewed sub/mixed dist, 87.6% — superseded by v5b. Don't use.)
  - Dataset: 1127 cards (527 lookup / 300 answer / 300 run_code).
- **`base` model scores math gsm8k_test = 1.74% (23/1319)** — the gap it routes
  around via lookup (~99%) + run_code (100% on v7).
- **Base models on disk:** `models/smollm-135m-instruct` (default),
  `models/smollm-360m-instruct`, and `models/smollm-1.7b-instruct` (all
  downloaded for the size sweep; no external APIs at runtime).
- **Cost tracking live**: total **$0.0932** across 32 passes (README banner).
  Refresh: `python -c "from scripts.passdb import PassDB as D; D().cost_report()"`
- Everything committed + pushed to `origin/main` (`git@192.168.0.4:james/smol-lab.git`).
  No background jobs running. Ollama is OFF for this project (user's choice;
  full 8GB P4 available).

## Environment (verified)
- Python: `python3` (3.13.5). **Use `source .venv/bin/activate`** — a uv venv
  with torch 2.5.1+cu121, transformers, peft, trl, datasets, accelerate,
  bitsandbytes. PEP 668 — do NOT `pip install` outside the venv.
- GPU: Tesla P4, 8GB, compute 6.1. For eval (big batches) export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `~/llm_eval/datasets/gsm8k_train.jsonl` (7473) + `gsm8k_test.jsonl` (1319)
  + `eval.db` are mined by the synthesis script. Don't move them.

## How to resume / run
```bash
cd /home/aec/smol
source .venv/bin/activate

# re-evaluate an adapter (chunked, no OOM):
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/eval_toolcall.py --model adapters/v4 --data data/raw/flashcards2.jsonl

# train a new adapter:
python -u scripts/train_adapter.py --data data/raw/flashcards2.jsonl \
        --out adapters/vN --epochs 3 --lr 2e-4 --batch 8 --max-len 256

# regenerate the synthetic training set (Type A lookup / B answer / C run_code):
python scripts/build_synth_cards.py

# gsm8k math baseline (batched HF, fast):
python scripts/eval_gsm8k_hf.py --data ~/llm_eval/datasets/gsm8k_test.jsonl --batch 32
```

## Key conventions / gotchas (learned the hard way — see wiki/BUGS.md)
- **BUG-005/006**: per-call eval pegged CPU; batched `generate_all` (left-pad +
  slice at full seq len) is required. Don't revert to per-call generate.
- **BUG-007**: `generate_all` must CHUNK (chunk=16). Batching all 827 in one
  forward OOMs the P4 (7.35GB).
- **BUG-008**: well_formed was a MEASUREMENT bug — `max_new_tokens=64` truncated
  long queries before the closing quote, so the strict `CALL_RE` scored correct
  calls as malformed. Fix = `CALL_OPEN_RE` (open-quote accepted) + training data
  capped to MAX_Q=180 so the model learns to CLOSE the quote. v2's "0.488" was
  phantom; v3's 0.970 is real. **Diagnose before fixing.**
- The priming cue string in `train_adapter.py` and `eval_toolcall.py` MUST stay
  byte-identical or the call habit won't transfer.
- `flashcards2.jsonl` is GENERATED (not hand-authored) — regenerate, don't edit.
- `adapters/`, `models/`, `logs/`, `benchmarks/*.db` are gitignored (artifacts).

## Open next step (PHASE 5 DONE + base-size sweep DONE, 2026-07-11)
The model now has TWO tools. `lookup` (fetch) and `run_code` (compute) both
resolve end-to-end. Verified across a 3-way base sweep (135m / 360m / 1.7B):
- `lookup`: emits `TOOL lookup query="..."` → sovereign KB resolver
  (`scripts/tool_resolver.py`, 8892 entries, zero external APIs) → **~99%**
  resolved-correct on gsm8k_test at every size (v7 1253/1266 = 99.0%).
- `run_code`: emits `TOOL run_code code="<expr>"` → restricted executor
  (`scripts/sandbox.py`) computes it. Coverage scales with base size:
  v5b (135m) 89.0% (266/299) → v6 (360m) 96.7% (289/299) → **v7 (1.7B) 100%
  (300/300)**, 0 misses. Sandbox is defense-in-depth: AST-scan rejects
  imports/open/defs/dunders, runs in a separate `-I` subprocess with a CPU
  rlimit + timeout kill.

Run it:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lookup loop on gsm8k (end-to-end):
python -u scripts/eval_resolver.py --model adapters/v7 \
        --data ~/llm_eval/datasets/gsm8k_test.jsonl --kind gsm8k
# run_code end-to-end (Type-C cards scored vs gold):
python -u scripts/eval_resolver.py --model adapters/v7 \
        --data data/raw/flashcards2.jsonl --kind flashcard
# toolcall habit eval (A/B/C scored):
python -u scripts/eval_toolcall.py --model adapters/v7 --data data/raw/flashcards2.jsonl
# resolver standalone smoke test (no GPU):
python scripts/tool_resolver.py --stats
python scripts/tool_resolver.py --tool run_code "51 + 99"
```
- Every eval writes a FULL per-item JSONL log to `logs/`
  (`eval_resolver_*.jsonl`, `eval_toolcall_*.jsonl`) — inspectable after the fact.
- **Where to stop on size**: 1.7B already hits 100% run_code on the synthetic set,
  so the next lever is NOT a bigger base — it's task diversity. The residual gsm8k
  lookup misses (13/1266 on v7) are KB re-wording gaps, not habit misses.
- **Next real capability step (open):** Phase 6 — LLM-wiki / disk-backed wiki as the
  lookup source + an orchestration layer (pi/hermes/opencode-shaped) to dispatch
  ToMoC calls AND feed results back so the model emits a final answer. The "emit a
  call → resolve → produce final answer" loop is the missing half of ToMoC.

## Phase 6 — the closing ToMoC loop (2026-07-11, 360m/v6 default base)
User locked in **360m (v6)** as the production base (1.7B too slow for the
marginal gain). Built `scripts/orchestrate.py`: the two-turn loop
`q → turn1 (model emits TOOL call) → resolve → turn2 (feed result back) →
model emits FINAL ANSWER`. Modes: `--ask "<q>"` (live single) and
`--data <jsonl> --kind flashcard|gsm8k` (batched two-pass scoring). Batched
turn-1-all then turn-2-all (chunk=16) to keep the P4 at ~97% util; per-item
JSONL to `logs/orchestrate_*.jsonl`; passdb-logged (passes 31-32).

**Result — the loop mechanically works, and exposes ONE clean gap:**
- gsm8k (1319): call 0.986, resolved_hit 0.992, **canonical_correct 1280/1319
  = 97.0%** (the tool gets the right answer), but the model's own turn-2
  **final_answer_correct = 708/1319 = 53.7%**.
- **Diagnosis (measured, not guessed):** the 53.7% is NOT a reasoning failure.
  Of 1301 rows that called a tool, **571 (~44%) emitted an EMPTY turn-2** (just
  EOS after `Final answer:`). Of the 730 rows where it DID answer,
  **708 = 97.0% were correct** — the tiny model faithfully echoes the tool
  result WHEN it continues. v6 was never trained on the two-turn
  `Tool result: X\nFinal answer:` format, so it often just stops.
- flashcard run_code loop: canonical 289/300 = 96.3%, final 251/300 = 83.7%
  (183/821 empty turn-2). (Note: flashcard join shows gold=None for many C rows
  in the quick diagnostic — gsm8k is the trustworthy end-to-end signal.)
- **Open next (Phase 6b):** add Type-D two-turn cards (question + call + tool
  result → final answer) to the training set and retrain v6→v8. Target: drive
  empty-turn-2 → ~0 so end-to-end tracks the tool's ~97% ceiling. This is a
  training-format fix, not a bigger-base or better-resolver problem.

Run it:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# live single-question ToMoC loop:
python -u scripts/orchestrate.py --model adapters/v6 --ask "48 - 5 + 20"
# batched end-to-end final-answer scoring:
python -u scripts/orchestrate.py --model adapters/v6 \
        --data data/raw/flashcards2.jsonl --kind flashcard
python -u scripts/orchestrate.py --model adapters/v6 \
        --data ~/llm_eval/datasets/gsm8k_test.jsonl --kind gsm8k
```

## Card schema (build_synth_cards.py)
- Type A (lookup):  `a = TOOL lookup query="<verbatim q>"`
- Type B (answer):  `a = <answer>` (model answers directly, no tool)
- Type C (run_code):`a = TOOL run_code code="<expr>"`, carries `answer` + `code`
  for training/eval scoring. Disjoint from A (no same-question contradiction).

## Commit / push
- Branch `main`, remote `origin` (local GitLab 192.168.0.4). Identity
  `James <james@homelab.local>`. Fast-forward push is normal; force-push is
  enabled by user preference if history needs rewriting.
- Keep commits KISS; update `wiki/JOURNAL.md` (pass table) + README cost banner
  with every training/eval run.
