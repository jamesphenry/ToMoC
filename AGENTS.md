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

## Current state (as of 2026-07-10, commit 3baa7a0)
- **Best adapter: `adapters/v3/`** — LoRA trained on 827 synthetic cards.
  - call_rate_when_should **0.970**, well_formed **0.970** (REAL — closes quote),
    correct_tool 1.000, over_call 0.027. (pass 10)
  - This is the milestone: a 135m model that calls lookup 97% of the time it
    should, with the right tool, correctly formatted, barely over-calling.
- **`base` model scores math gsm8k_test = 1.74% (23/1319)** — the gap it must
  route around. (pass 5)
- **Cost tracking live**: total **$0.0161** across 10 passes (README banner).
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
python -u scripts/eval_toolcall.py --model adapters/v3 --data data/raw/flashcards2.jsonl

# train a new adapter:
python -u scripts/train_adapter.py --data data/raw/flashcards2.jsonl \
        --out adapters/vN --epochs 3 --lr 2e-4 --batch 8 --max-len 256

# regenerate the 827-card synthetic training set (caps query to MAX_Q=180):
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

## Open next step (DIRECTION B — not started)
The lookup is still a **stub**: the model emits `TOOL lookup query="..."` but
nothing resolves it. Wire a real resolver so calls actually compute:
- **run_code** first (smallest, most sovereign): route math lookups to a sandboxed
  Python exec; returns the answer. Directly fixes the gsm8k 1.74% gap.
- Later: LLM-wiki / SearXNG for factual lookups. See `future.md` (ToMoC).

## Commit / push
- Branch `main`, remote `origin` (local GitLab 192.168.0.4). Identity
  `James <james@homelab.local>`. Fast-forward push is normal; force-push is
  enabled by user preference if history needs rewriting.
- Keep commits KISS; update `wiki/JOURNAL.md` (pass table) + README cost banner
  with every training/eval run.
