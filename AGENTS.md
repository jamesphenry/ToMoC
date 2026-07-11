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

## Current state (as of 2026-07-11, after 360m comparison — v6)
- **Best adapters (same 2-tool format, 1127-card clean-balanced set):**
  - `adapters/v6/` — LoRA on **smolLM-360m-instruct** (hidden 960, 32 layers).
    **Best overall**: gsm8k lookup **99.2%** (1280/1290, call_rate 0.986) AND
    run_code **96.7%** (289/299) on the 300-card hard set. The 360m crushes the
    135m's operator-confusion ceiling (residual error 11% -> 3%).
  - `adapters/v5b/` — LoRA on **smolLM-135m** (cleaner balanced Type-C, 300 cards).
    run_code **89.0% (266/299)**, lookup 98.5%. Good if you MUST stay at 135m.
  - `adapters/v4/` — 135m, 2 tools; on the hard 300-card set run_code only 71.1%
    (it never trained on division). Lookup 98.4%.
  - (v5, skewed sub/mixed dist, 87.6% — superseded by v5b. Don't use.)
  - Dataset: 1127 cards (527 lookup / 300 answer / 300 run_code).
- **`base` model scores math gsm8k_test = 1.74% (23/1319)** — the gap it routes
  around via lookup (~99%) + run_code (~97% on v6).
- **Base models on disk:** `models/smollm-135m-instruct` (default) and
  `models/smollm-360m-instruct` (downloaded for the comparison).
- **Cost tracking live**: total **$0.0566** across 27 passes (README banner).
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

## Open next step (PHASE 5 DONE, 2026-07-11)
The model now has TWO tools. `lookup` (fetch) and `run_code` (compute) both
resolve end-to-end:
- `lookup`: emits `TOOL lookup query="..."` → sovereign KB resolver
  (`scripts/tool_resolver.py`, 8892 entries, zero external APIs) → **98.4%**
  resolved-correct on gsm8k_test (pass 17, call_rate 0.995).
- `run_code`: emits `TOOL run_code code="<expr>"` → restricted executor
  (`scripts/sandbox.py`) computes it → **94.7% correct** (142/150, pass 16).
  Sandbox is defense-in-depth: AST-scan rejects imports/open/defs/dunders, runs
  in a separate `-I` subprocess with a CPU rlimit + timeout kill.

Run it:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# lookup loop on gsm8k (end-to-end):
python -u scripts/eval_resolver.py --model adapters/v4 \
        --data ~/llm_eval/datasets/gsm8k_test.jsonl --kind gsm8k
# run_code end-to-end (Type-C cards scored vs gold):
python -u scripts/eval_resolver.py --model adapters/v4 \
        --data data/raw/flashcards2.jsonl --kind flashcard
# toolcall habit eval (A/B/C scored):
python -u scripts/eval_toolcall.py --model adapters/v4 --data data/raw/flashcards2.jsonl
# resolver standalone smoke test (no GPU):
python scripts/tool_resolver.py --stats
python scripts/tool_resolver.py --tool run_code "51 + 99"
```
- Every eval writes a FULL per-item JSONL log to `logs/`
  (`eval_resolver_*.jsonl`, `eval_toolcall_*.jsonl`) — inspectable after the fact.
- **Next real capability step (open):** Phase 6 — LLM-wiki / disk-backed wiki as the
  lookup source + an orchestration layer (pi/hermes/opencode-shaped) to dispatch
  ToMoC calls. OR: push run_code coverage higher — the residual ~11% on the 300-card
  hard set is the 135m's own +/−/×/÷ verb confusion (NOT a sandbox bug). Levers: more
  clean Type-C variety / a bigger base (360m). Note v5b already beats v4 on a matched
  set (89% vs 71%) and adds division; the easy "more variety" lever backfired (v5
  skewed dist → 87.6%), so the next attempt should be 360m or constrained decoding.

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
