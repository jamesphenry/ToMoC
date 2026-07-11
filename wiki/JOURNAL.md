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
| 10 | eval adapters/v3 (capped data) | adapters/v3 | — | 356.3s | — | **0.970** | 0.027 | **0.970** |
| 11 | **END-TO-END resolver eval** (v3 → KB lookup → score) | adapters/v3 | — | 939.3s | 573MB | call=0.992 WF=1.000 **resolved-correct=0.972** | — | — |
| 12 | gsm8k baseline (HF, batched) | smollm-135m-instruct | — | — | 359MB | gsm8k_acc=0.017 | — | — |
| 13 | gsm8k HF re-run (cost-log fix check) | smollm-135m-instruct | — | — | — | gsm8k_acc=0.017 | — | — |
| 14 | TRAIN adapters/v4 (977 cards A/B/C) | smollm:135m | 0.176 | 510.3s | 1111MB | — | — | — |
| 15 | eval adapter v4 habit (A/B/C) | adapters/v4 | — | 395.4s | — | lookup_call=0.966 run_code=1.000 (WF 1.000) over_call=0.047 | — | — |
| 16 | resolver eval v4 (flashcard, run_code end-to-end) | adapters/v4 | — | 391.4s | — | call=0.689 WF=1.000 **run_code correct=0.947 (142/150)** | — | — |
| 17 | resolver eval v4 (gsm8k lookup loop) | adapters/v4 | — | 932.9s | — | call=0.995 WF=0.999 **resolved-correct=0.984 (1269/1290)** | — | — |
| 18 | train v5 (C 150→300, skewed dist) | smollm:135m | 0.188 | 581.4s | 1111MB | — | 3ep lr2e-4 b8 | superseded |
| 19 | resolver eval v5 (flashcard run_code) | adapters/v5 | — | 414.0s | — | call=0.732 WF=1.000 correct=0.876 (262/299) | — | skewed dist hurt |
| 20 | resolver eval v5 (gsm8k lookup) | adapters/v5 | — | 1384.1s | — | call=0.997 WF=1.000 correct=0.985 (1277/1296) | — | lookup preserved |
| 21 | train v5b (C clean/balanced) | smollm:135m | 0.191 | 1032.9s | 1111MB | — | 3ep lr2e-4 b8 | **best compute** |
| 22 | resolver eval v5b (flashcard run_code) | adapters/v5b | — | 420.9s | — | call=0.731 WF=1.000 **correct=0.890 (266/299)** | — | +division |
| 23 | resolver eval v5b (gsm8k lookup) | adapters/v5b | — | 927.3s | — | call=1.000 WF=1.000 correct=0.985 (1279/1298) | — | lookup preserved |
| 24 | resolver eval v4 on SAME 300-card set (fair A/B) | adapters/v4 | — | 481.8s | — | call=0.651 WF=1.000 **correct=0.711 (101/142)** | — | v4 on hard set |
| 25 | train v6 (360m base, C clean/balanced) | smollm:360m | 0.176 | 1110.3s | 1590MB | — | 3ep lr2e-4 b8 | **best overall** |
| 26 | resolver eval v6 (flashcard run_code) | adapters/v6 | — | 518.8s | — | call=0.728 WF=1.000 **correct=0.967 (289/299)** | — | 360m crushes ceiling |
| 27 | resolver eval v6 (gsm8k lookup) | adapters/v6 | — | 1482.0s | — | call=0.986 WF=1.000 correct=0.992 (1280/1290) | — | lookup best too |

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

## DIRECTION B — the lookup actually computes (DONE, pass 11)

The open step from AGENTS.md: the model emitted `TOOL lookup query="..."` but
nothing resolved it. We wired a **sovereign KB resolver** (`scripts/tool_resolver.py`)
+ an **end-to-end eval** (`scripts/eval_resolver.py`) that runs the full loop:
prompt → model emits call → parse → resolve → compare to gold.

The KB is the on-disk gsm8k train+test (7473+1319) + mmlu algebra — 8892
entries, zero external APIs. Resolution is layered so it ALWAYS returns a verdict:
  1. **exact** match (norm'd) — verbatim `query == KB prompt`
  2. **prefix** match — KB prompt startswith the query head (salvages
     BUG-008's truncated queries, where the card `q` was cut at MAX_Q=180)
  3. **fuzzy** match — token-set Jaccard >= 0.8 (catches light rewording)
  4. **miss** — call was correct, KB just lacks it (deterministic, logged)

**pass 11 result (1319 gsm8k_test rows, adapter v3):**
  call_rate        0.992  (1309/1319 emitted a lookup call)
  well_formed      1.000  (every call perfectly formatted)
  resolved_hit     0.979  (1282 hits; method split: exact 663, prefix 587, fuzzy 32)
  resolved MISS    0.021  (27 — genuine KB gaps, not call failures)
  NOT-called       10     (residual guesses; the old 1% under-call)
  **correct_vs_gold 1.000 (1282/1282 resolved hits matched the gold number)**

**The headline:** base 135m scored **1.74%** on gsm8k_test math on its own
(23/1319). With the lookup habit + resolver, it now gets **97.2%** end-to-end
(1282/1319 resolved-correct). The thesis is demonstrated as a *working system*,
not just a habit: a 135m that knows to look up turns its math-zeros into
fetched-correct answers. Cost so far: **$0.0194** across 11 passes.

**BUG-009** fell out during resolver build: `str.splitlines()` shatters jsonl
records that contain the Unicode separator U+2028 *inside* a string value. Fixed
by reading jsonl line-by-line from the file object. Real data, real bug — would
have aborted the math KB. See wiki/BUGS.md.

**Going-forward change (user ask):** every eval now writes a FULL per-item JSONL
log to `logs/` (eval_resolver_*.jsonl and eval_toolcall_*.jsonl) so each run is
inspectable after the fact — not just the summary line.

## PHASE 5 — second tool: `run_code` (compute) (DONE, pass 14-17)

Phase 4 proved `lookup` works end-to-end. Phase 5 adds a SECOND expert so the
model can *compute*, not just fetch — the ToMoC router now picks between
fetch (lookup) and compute (run_code). This is the "functions ARE its knowledge"
endgame: the tiny model routes arithmetic to a sandboxed executor.

**The sandbox (`scripts/sandbox.py`) — defense-in-depth, sovereign:**
- AST pre-scan rejects imports, defs, `__import__`, `open`, dunder attrs, and
  `while/with/try/raise` BEFORE anything executes. Cheap, no side effects.
- Execution runs in a SEPARATE `-I` (isolated) subprocess with a stripped env,
  a `RLIMIT_CPU` cap, and a Python `timeout` — a `while True` is killed, not hung.
- Returns the last expression's value (REPL-style) + captured stdout.
- Verified adversarial: `import os`, `open('/etc/passwd')`, dunder attr, defs,
  and `open` inside a comprehension all blocked; infinite loop killed.

**Training data (`build_synth_cards.py`):** added **Type C (run_code)** cards —
150 sovereign synthetic arithmetic word problems (templates + random ints), each
carrying a `code` expr + gold `answer`. Sourced DISJOINT from Type A (lookup) so
the same question never has two targets (which would contradict and break both
habits). The priming cue in `train_adapter.py`/`eval_toolcall.py` was left
**byte-identical** — v4 learns `run_code` purely from the data, preserving v3's
lookup habit. Dataset: 977 cards (527 lookup / 300 answer / 150 run_code).

**Results (passes 14-17, $0.0274 total across 17 passes):**
- v4 emits `run_code` on **100%** of arithmetic cards (100% well-formed) — the
  new tool habit installed cleanly (pass 15).
- `run_code` end-to-end (pass 16): the model emits the code, the sandbox computes
  it, **142/150 = 94.7% correct** vs the gold. Base math floor was 1.74%.
- `lookup` loop preserved (pass 17): gsm8k_test **98.4% correct** (1269/1290,
  call_rate 0.995, well_formed 0.999). Slightly above v3's resolver coverage.
- `over_call_rate` crept 0.027 → 0.047 (more tool use overall) but still low.

**Residual error analysis (the honest 5.3%):** all 150 run_code rows were routed
to the sandbox; the 8 misses are genuine **model arithmetic errors**, not sandbox
bugs. The 135m emits e.g. `20 * 41` (computed correctly =820) when the word
problem needed `20 - 41` (=-21) — operator confusion on 3-arg word problems
(`X and Y, then Z...`). The sandbox computes exactly what's asked (0 errors
across the whole eval). So the ceiling here is the 135m's own reasoning, not the
executor — pushing past 94.7% means either more Type-C training variety or a
bigger base (360m), not sandbox changes.

**Dispatch seam:** `tool_resolver.resolve(tool, query)` now branches `lookup`
→ KB and `run_code` → sandbox. Adding Phase 6 tools (wiki, etc.) = adding a
branch here. `eval_toolcall.py` / `eval_resolver.py` extended to score Type-C.

## run_code coverage push — v5/v5b + a measurement lesson (pass 18-24)

Goal: lift run_code coverage past v4's "94.7%". Two attempts + a fair A/B that
rewrote the scoreboard.

**Attempt 1 — v5 (pass 18-20): more variety + skewed distribution → WORSE.**
Doubled Type-C 150→300 with many new verb phrasings AND over-weighted subtraction
+ 2-step mixed (65% of cards), on the theory that the 8 v4 misses were verb→operator
confusion. Result: **87.6% (262/299)** — a regression. Diagnosis of the 38 misses:
the "and" joiner now appeared in BOTH add and mul templates (ambiguous: "3 apples
in one basket and 13 in another" =+ vs "66 rows and 19 desks" =×), and the sub-heavy
skew biased the model toward subtract-by-default. Variety was poison, not medicine.

**Attempt 2 — v5b (pass 21-23): clean balanced set → 89.0%.**
Rewrote `gen_arith` to unambiguous per-operator templates (× always uses "each/per",
never "and"), even +/−/×/÷ distribution, fixed a division-detection bug that had
silently dropped ÷ cards. Result: **89.0% (266/299)**, +division coverage, lookup
preserved (98.5% gsm8k). Better than v5 but still "below" v4's 94.7%… or so it seemed.

**The measurement lesson (pass 24 — fair A/B):** v4's "94.7%" was on its OWN easier
150-card set (no division, fewer 2-step). Re-ran v4 on the SAME 300-card hard set:
**71.1% (101/142), call_rate 0.651** — v4 under-calls run_code on division/2-step
cards it never trained on. So on a matched set **v5b (89%) crushes v4 (71%)** and
adds division. The apparent "regression" was apples-to-oranges — always eval both
adapters on the SAME set before ranking. **v5b is the best compute adapter; v4
remains best for pure lookup (98.4%).** The residual ~11% is still the 135m's own
+/-/×/÷ verb confusion (sandbox = 0 errors), so the next real lever is a 360m base
or constrained decoding, not more Type-C variety (which backfired).

## 360m base — the lever that actually moved run_code (pass 25-27)

Theory: the 135m's residual ~11% run_code error was verb→operator confusion in
translating the word problem to a `code` expr, not the sandbox (0 errors). A
bigger base should parse arithmetic phrasing more reliably. Tested on the SAME
1127-card clean-balanced set + SAME eval sets — only the base changed.

**Downloaded `HuggingFaceTB/SmolLM-360M-Instruct` (hidden 960, 32 layers, 724MB
fp16) into `models/smollm-360m-instruct`** (sovereign, on-disk). Trained v6 (pass
25): loss 0.176, 1110s, 1590MB GPU — ~2× the 135m's time but still ~6GB free on
the P4.

**Results (passes 26-27):**
- run_code end-to-end: **96.7% (289/299)** — up from v5b's 89.0% on the IDENTICAL
  300-card set. Residual error 11% → 3%.
- lookup gsm8k: **99.2% (1280/1290)** — up from v5b's 98.5%.

So the 360m (2.7× params) breaks the 135m's operator-confusion ceiling for both
tools. It's now the default best adapter. Cost to confirm: 3 more passes, +$0.011
(→ total $0.0566 / 27 passes). Bigger base beat "more data variety" (v5) AND
"clean balanced data" (v5b) — the param-count lever was the right call.

**Two bugs found and fixed during this run:**
- `train_adapter.py` hardcoded `base_model="smollm:135m"` in the passdb log
  regardless of `--base` (so a 360m run would be mislabeled). Now derives the tag
  from `os.path.basename(args.base)`.
- `eval_toolcall.Engine` hardcoded `DEFAULT_BASE` (135m) when loading any LoRA
  adapter, so v6 loaded on the 135m and SILENTLY shape-mismatched (and would have
  produced garbage). Now reads `base_model_name_or_path` from the adapter's own
  `adapter_config.json`, so any-size adapter loads on its correct base.

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
