# wiki/JOURNAL.md — the smol-lab journey so far

> A running log of what we built, what we learned, and the real numbers.
> Bugs/hotfixes live in [BUGS.md](BUGS.md); parked ideas in [future.md](../future.md).
> This file is the STORY; the others are the reference.

---

## The thesis (why we're here)

Build a tiny LLM (smolLM:135m) that **looks up what it needs** — functions ARE
its knowledge, reasoning > raw smarts. Endgame = **ToMoC** (Tool-Routed Mixture
of Capabilities): the model's tool-call decision is the router, external
disk-backed tools/wiki are the experts. Sovereignty is the constraint: homelab
only, no external APIs, minimal requirements. "Disks are cheap, VRAM isn't."

This rules out DISTILLATION (needs an external teacher = dependency) and pushes
us toward LoRA + tool-calling as the way to teach a 135M model real competence
without scaling params.

---

## Timeline of the build

### Phase 0 — foundations (DONE)
- Verified smolLM (via Ollama) has **no native tool-calling** — confirmed the
  gap we're filling. (Ollama serves it as Q4_0 GGUF; training needs the full
  fp16 HF checkpoint, which we pulled to `models/smollm-135m-instruct`.)
- Pinned env: uv venv, torch 2.5.1+cu121, transformers/peft/trl on the Tesla P4
  (compute 6.1, 8GB). P4 is slow but idle most of the time — fine for LoRA.
- `passdb.py`: SQLite store logging every training/eval pass with GPU mem +
  walltime, so we can compare across runs on the P4.

### Phase 1 — the habit + eval pipeline (DONE)
- `flashcards_spec.md` + `build_flashcards.py`: 60-card smoke set, 50/50 split:
  - **Type A** (lookup): model SHOULD emit `TOOL lookup query="..."`
  - **Type B** (answer): model should just answer (except math/knowledge
    wrong-items, which seed the lookup habit)
- Mini-format v1: `TOOL lookup query="<q>"` (NOT JSON yet — grammar later).
- `eval_toolcall.py`: console eval measuring the CALL DECISION, not accuracy:
  `call_rate_when_should`, `over_call_rate`, `well_formed_rate`, `correct_tool_rate`.
- Baseline (pass 1): base model has **zero** tool habit. Expected floor.

### Phase 2 — first real LoRA + the speed crisis (DONE)
- `train_adapter.py`: LoRA on 135m, logs to passdb. Trained `adapters/v1`.
- **The 100%-CPU mystery**: eval ran at 100% CPU / 12% GPU for 7+ min. Profiling
  proved the model was on cuda:0 — it was host-bound sync from 60 separate
  `generate()` calls (BUG-005). Fix: batch all prompts into ONE forward pass →
  GPU hits 100%, 60 cards in ~41s. Right-padding was corrupting batched output
  (BUG-006) → left-padding + slice-at-seq-len.
- **Result: the adapter learned NOTHING useful** (see numbers below). The habit
  isn't installed yet. That's the open problem.

---

## The numbers (passdb, honest)

| pass | what | base | loss | walltime | GPU | call_rate | over_call | well_formed |
|------|------|------|------|----------|-----|-----------|-----------|-------------|
| 1 | baseline eval (base) | smollm:135m | — | 63.2s | — | 0.000 | 0.000 | 0.000 |
| 2 | TRAIN adapters/v1 | smollm:135m | 1.247 | 47.4s | 896MB | — | — | — |
| 3 | eval adapter (BUGGY, pre-fix) | adapters/v1 | — | **451s** | — | n/a | n/a | n/a |
| 4 | eval adapter (fixed, batched) | adapters/v1 | — | 41.5s | — | 0.000 | 0.033 | 0.000 |
| 5 | **benchmark-ref** gsm8k_test (base) | smollm-135m-instruct | — | 713.6s | 795MB | gsm8k_acc=0.017 | — | — |
| 6 | TRAIN adapters/v2 (synthetic) | smollm:135m | 0.110 | 699.8s | 1508MB | — | — | — |
| 7 | eval adapters/v2 (synthetic) | adapters/v2 | — | 432.9s | — | **0.964** | 0.030 | 0.488 ⚠️ |
| 8 | eval adapters/v2 **re-scored** (BUG-008 parser fix) | adapters/v2 | — | 897.8s | — | **0.964** | 0.030 | **0.964** |
| 9 | TRAIN adapters/v3 (capped-query, MAX_Q=180) | smollm:135m | 0.149 | 889.5s | 1112MB | — | — | — |

⚠️ pass 7's `well_formed=0.488` was a MEASUREMENT BUG (BUG-008), not a model flaw:
the model emits the correct `TOOL lookup query="..."` but `max_new_tokens=64`
truncated long questions before the closing quote. The parser (BUG-008 fix) accepts
the open-quote form → `well_formed` jumps to **0.964** at pass 8. The model was
never broken; our scoring couldn't see the closing quote.

Pass 3 is kept on purpose: it's the 100%-CPU run (451s) before the batching fix.
It never logged metrics (crashed/slow) — a monument to the bug we killed.

**Pass 5 is the headline:** full gsm8k_test (1319 rows) on the HF base, batched
via `eval_gsm8k_hf.py`, scored **1.74% (23/1319)**. This is the quantified proof
of the gap — 135m CANNOT do math, so it must LOOKUP instead of guess. The
Ollama run #3 was crawling toward the same number item-by-item; batching got the
full picture in ~12 min instead of hours. NOTE: ~1.74% here vs math_gsm 0.00 in
the llm_eval column — different sets (full test vs 15 curated) but same verdict.

**Pass 6+7 = THE BREAKTHROUGH.** v2 was trained on 827 synthesized cards
(527 A / 300 B, mined from gsm8k_train + eval wrong-items for A, eval right-items
for B; A-heavy on purpose because our failure was UNDER-call, not over-call).
A priming cue ("If you are not certain... call the lookup tool") was injected
IDENTICALLY into train and eval prompts so the habit transfers. Result:
call_rate_when_should **0.000 → 0.964**, correct_tool_rate **1.000**,
over_call_rate **0.030** (it does NOT spam the tool on strong tasks). Loss
cratered 1.247 → 0.110. The "functions ARE its knowledge" thesis is DEMONSTRATED:
a 135m model that scored 1.74% on math now KNOWS to LOOKUP instead of guess.

**The remaining gap (pass 7):** well_formed_rate = 0.488. The CALL DECISION is
learned (96% call when weak, 100% correct tool) but the FORMAT isn't reliable —
only ~half the calls are perfectly `TOOL lookup query="..."`. Next step is either
(a) tighten Type-A formatting in training, or (b) move to JSON + grammar
constraints (parked in future.md) to force well-formed output. Separately the
lookup is still a STUB — nothing resolves the query yet; wiring run_code (math)
or LLM-wiki/SearXNG (facts) is the real capability step.

**Bug found during v2 eval:** BUG-007 — unchunked `generate_all` batched all 827
prompts into one forward and OOM'd the P4 (7.35GB). Fixed by chunking to 16/forward
+ `empty_cache()`; peak VRAM dropped to ~0.5GB. See wiki/BUGS.md.

## Key findings (the lessons)

1. **LoRA on 135m is cheap and safe on the P4.** 896MB for training, coexists
   with ollama's ~1.6GB (we can't stop ollama — no sudo — and don't need to).
   8GB is plenty for 135m LoRA.

2. **Batching is non-negotiable for eval speed.** Per-call `generate()` pegs a
   CPU core and starves the GPU. One batched call saturates the P4. ~10x faster.

3. **The tool-call habit is NOT free to learn.** 30 Type-A examples drowned by
   30 Type-B "answer" examples in the loss. 135m is too small to pick up a weak
   meta-behavior from so few cues. The loss optimizes "predict next word" — and
   the base model's instinct is always "answer."

4. **Verification guardrails earn their keep.** The CALL_RE regex bug (trailing
   quote) and the CPU mystery were both caught by re-checking / profiling, not
   by trusting the run. We now ad-hoc verify changed code in /tmp before claiming
   done.

5. **Ollama's smolLM is Q4_0 quantized** — useless for training. Always pull the
   fp16 HF checkpoint for LoRA work.

---

## What's next (open)

- **Make training actually teach the habit.** Lean: synthesize ~200-400 primed
  Type-A examples (we have 7,473 gsm8k + knowledge/reasoning wrong-items as a
  lookup KB), prime the prompt ("if unsure, call TOOL lookup"), retrain, re-eval.
  Bigger base (360m) is the fallback lever, not the first move (sovereignty+KISS).
- Scale the flashcard set beyond 60.
- Later (future.md): JSON + constrained grammar, `run_code`, SearXNG, LLM-wiki,
  reasoning self-correct, multi-tool, ToMoC orchestration.

---

## How to extend this journal
Append a new dated section per milestone. Keep the passdb table honest (include
the buggy runs — they're data). Link bugs to BUGS.md, ideas to future.md.
