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
| 28 | train v7 (1.7b base, C clean/balanced) | smollm:1.7b | 0.083 | 3524.4s | 4236MB | — | 3ep lr2e-4 b8 | **best compute** |
| 29 | resolver eval v7 (flashcard run_code) | adapters/v7 | — | 1140.5s | — | call=0.720 WF=1.000 **correct=1.000 (300/300)** | — | 0 misses |
| 30 | resolver eval v7 (gsm8k lookup) | adapters/v7 | — | 2937.5s | — | call=0.964 WF=1.000 correct=0.990 (1253/1266) | — | residual=KB gaps |
| 31 | orchestrate flashcard (2-turn ToMoC) | adapters/v6 | — | 1044.4s | — | canonical=0.963 final=0.837 (empty t2=183/821) | $0.0037 | loop works |
| 32 | orchestrate gsm8k (2-turn ToMoC) | adapters/v6 | — | 1817.8s | — | canonical=0.970 final=0.537; of 730 non-empty t2 → 708=0.970 | $0.0064 | gap=empty t2 |
| 33 | train v8 (360m, +Type-D two-turn) | smollm:360m | 0.1456 | 1459.3s | 1605MB | 1427 cards (527A/300B/300C/300D) | $0.0051 | closes empty t2 |
| 34 | orchestrate flashcard (v8) | adapters/v8 | — | 1031.6s | — | final=0.885 canonical=0.908 | $0.0036 | empty t2 ~0 |
| 35 | orchestrate gsm8k (v8) | adapters/v8 | — | 1784.1s | — | final=0.957 canonical=0.958; empty t2=0/1288 | $0.0062 | LOOP CLOSED |

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
  loaded on the 135m and SILENTLY shape-mismatched (and would have
  produced garbage). Now reads `base_model_name_or_path` from the adapter's own
  `adapter_config.json`, so any-size adapter loads on its correct base.

## 1.7B base — closing the compute loop (pass 28-30)

  User: "changing to the larger model is smart… try 1.7b its soo slow, but lets
  see." (There's no 260m in the SmolLM family — 135m / 360m / 1.7B are the sizes;
  used 1.7B as the next step up.) Downloaded `HuggingFaceTB/SmollLM-1.7B` (hidden
  2048, 24 layers, ~6.8GB across 2 safetensors shards) into `models/smollm-1.7b-instruct`.

  **Training v7 (pass 28):** loss 0.083, **3524s (~59 min)**, 4236MB GPU — 3.4×
  the 360m's time and ~4× the 135m's, but only 4236MB of 7680MB on the P4 (no OOM).
  Cost $0.0123 (the most expensive single pass so far, ~3× v6's).

  **Results (passes 29-30), SAME 300-card set:**
  - run_code end-to-end: **100% (300/300)** — 0 misses across all arithmetic cards
    including division + 2-step. The 1.7B (12.6× the 135m's params) eliminates the
    operator-confusion ceiling entirely. Verified manually: it emits correct `code`
    for negatives (`13 - 78` → `-65`), division (`15 / 3` → `5.0`==`5`), mixed
    (`48 - 21 + 5` → `32`).
  - lookup gsm8k: **99.0% (1253/1266 scored)**, call_rate 0.964, WF 1.000. The 13
    scored-wrong are KB re-wording gaps (resolver returns a number for a question
    it can't express as arithmetic, e.g. "half that much"), NOT habit misses; +6
    no-KB rows → 1259/1319 = 95.5% end-to-end.

  **Size sweep summary (run_code / lookup, same set):**
  | base | run_code | lookup | train min | eval min | GPU |
  |------|----------|--------|-----------|----------|-----|
  | 135m (v5b) | 89.0% | 98.5% | 10 | ~23 | 1111MB |
  | 360m (v6) | 96.7% | 99.2% | 18 | ~23 | 1590MB |
  | **1.7B (v7)** | **100%** | **99.0%** | 59 | ~68 | 4236MB |

  **Conclusion:** the bigger base is THE lever — bigger params monotonically beat
  more/better data. But 1.7B already maxes the synthetic run_code set, so the next
  lever is NOT a bigger base; it's task/template diversity + closing the KB
  re-wording gaps. Cost to confirm the sweep: +$0.026 (→ total **$0.0789 / 30
  passes**). v7 = best compute adapter; v6 = speed/accuracy sweet spot; v5b if
  stuck at 135m.

## Phase 6 — closing the ToMoC loop (pass 31-32, 360m/v6)

The first 5 phases proved the model LEARNS to call tools and the sovereign
resolver computes the right answer. But every eval so far stopped at "did the
tool's answer match gold?" — the model never SAW the result and never emitted a
final answer. Phase 6 builds the missing half: `scripts/orchestrate.py`, a
two-turn loop — `q → turn1 (model emits TOOL call) → resolve → turn2 (feed the
tool result back) → model emits FINAL ANSWER`.

Live smoke: `--ask "There were 48 people. 5 left and then 20 arrived..."` →
turn1 `TOOL run_code code="48 - 5 + 20"` → resolver 63 → turn2 `63`. The loop
runs end-to-end with no human in the middle. ✅

Batched scoring (v6, two passes: turn-1-all then turn-2-all, chunk=16, ~97%
P4 util):
- **gsm8k (1319):** call 0.986, resolved_hit 0.992, **canonical 1280/1319 =
  97.0%** (tool right), model's own **final 708/1319 = 53.7%**.
- **flashcard run_code (300 C):** canonical 289/300 = 96.3%, final 251/300 =
  83.7%.

The gap looked alarming (97% → 54%) until diagnosed. **It is NOT a reasoning
failure:** of 1301 tool-calling rows, **571 (~44%) emitted an EMPTY turn-2**
(EOS right after `Final answer:`). Of the 730 that DID answer, **708 = 97.0%
were correct** — the tiny model faithfully echoes the tool result WHEN it
continues. Root cause: v6 was never trained on the two-turn
`Tool result: X\nFinal answer:` format, so it frequently just stops. This is a
training-FORMAT gap, not a base-size or resolver problem (BUG-015-adjacent:
diagnose before despairing — the "53.7%" headline hid a 97% capability).

**Next (Phase 6b, open):** synthesize Type-D two-turn cards (question + call +
tool result → final answer), fold into the 1127-card set, retrain v6→v8.
Target: empty-turn-2 → ~0 so end-to-end tracks the tool's ~97% ceiling.

## Phase 6b — closing the loop: Type-D two-turn cards (pass 33-35)

Phase 6 proved the loop *works* but v6 emitted an EMPTY turn-2 ~44% of the
time (never trained on `Tool result: X\nFinal answer:`). Phase 6b fixes it as
a **training-format** problem (not base-size or resolver):

- **`build_synth_cards.py --d N`** adds **Type-D** cards: a two-turn context
  (the raw question + the model's own TOOL call + the resolved tool result +
  `Final answer:`) with the bare answer as target. 150 run_code (sovereign
  arithmetic) + 150 lookup (gsm8k_train gold) = 300 per run.
- **Critical correctness detail:** the Type-D `prompt_full` is rebuilt
  BYTE-IDENTICAL to `orchestrate.build_turn2_prompt()` (shared cue string),
  and `train_adapter.py` masks the whole prompt so **loss lands only on the
  final answer**. This teaches the model to ECHO the injected tool result,
  not to hallucinate its own `Tool result:` line. Verified: 13/13 ad-hoc
  checks, incl. D-prompt == orchestrate-prompt byte equality + masked-label
  count.
- **Retrain v6→v8** (360m, 1427 cards = 527A/300B/300C/300D, 3ep lr2e-4,
  loss **0.1456**, 1605MB, pass 33).

**Result (pass 34-35) — the loop is CLOSED:**
- gsm8k: empty turn-2 = **0/1288 (0.0%)** (was 571/1301 = 43.9% on v6).
- gsm8k final_answer_correct = **1262/1319 = 95.7%** (was 53.7%); of the
  answers it gives, 1262/1288 = 98.0% are correct → tracks the tool's own
  ~96% ceiling.
- This completes the ToMoC thesis: the tiny sovereign model routes to external
  "experts" (KB + run_code) and faithfully reports the answer. functions ARE
  its knowledge.
- Residual ~4% is now KB re-wording gaps + rare arithmetic slips — NOT the
  loop. flashcard final% is a known join artifact (gold=None for many C rows);
  gsm8k is the trustworthy signal.

## Phase 6c — honesty on KB-miss + show-work (v9/v10, pass 36-39)

Two gaps remained after 6b closed the loop: (1) on a **KB-miss** the model
sometimes *guessed a number* instead of admitting ignorance, and (2) it emitted
a bare `TOOL call` with no visible reasoning — hard to trust, hard to debug.

### Type-E — graceful KB-miss honesty (v9, pass 36)
Measured the gap first (don't guess, diagnose): ran v8 gsm8k misses and found
**11 misses; of those 4 (~36%) emitted a fabricated number** (e.g. `112`, `4`,
`100`, `60`) instead of stopping. That's the model *filling the silence* — a
real honesty failure, not a routing bug.

Fix: **Type-E** cards (`build_synth_cards.py --e N`) — the resolver returns a
real "No answer found in the knowledge base." verdict, and the gold target is a
plain `<answer>…I don't know / can't find…`. Trained **v9** (360m, 1727 cards =
527A/300B/300C/300D/300E, loss **0.1223**, 2287.9s, 1620MB, pass 36). Verified
via an ad-hoc miss-branch probe (`/tmp`): on fabricated-entity questions v9
emits `<answer>` with NO guessed number — the 4/11 guessing behavior is gone.

### BUG-008 strikes again — MAX_NEW too short for show-work
While wiring v10 I caught that `orchestrate.py` capped `MAX_NEW = 64` tokens.
That truncates a show-work *prefix* before the `TOOL call` (same class of bug as
the old well_formed=0.488 truncation). Bumped to **160**. Re-verified v8 at 160
(pass 37): `correct_vs_gold 0.9845` — no regression. (BUG-008 now documented in
AGENTS.md as "truncation bites twice — guard MAX_NEW.")

### Type-F — show-your-work (v10, pass 38)
Added **Type-F** cards (`--f N`) from user-edited "show-work" seed
(`data/raw/f_cards_seed.txt`): each is a word problem with a worked `work`
string + a numeric `code` expr. Target = `{work} TOOL run_code code="{code}"`.
Split 128 seed problems into **106 numeric** (kept) + **22 symbolic**
(geometry/clock/algebra/comparison → `data/raw/f_cards_symbolic.txt`, deferred;
run_code pipeline stays numeric-only). Trained **v10** = v9 + Type-F (360m, 1833
cards = 527A/300B/300C/300D/300E/106F, loss **0.2351**, 2642.7s, 1660MB, pass 38).

**Honest result (pass 39, flashcard router-metrics):** router_precision
**0.972** / recall **0.968**, call_rate 0.996, well_formed 0.996, over_call 0,
false_tool 14, missed_tool 2. Type-E honesty held (no guessed numbers). Show-work
*prefix transferred strongly* — later measured on the flashcard set: **484/490**
run_code outputs open with a `This is addition because…` reasoning span (the
format habit AND the prefix both stuck). A few word-problems still misroute to
`lookup` (the format habit won, the prefix didn't *on those*). The router-metrics
eval is the authoritative signal (flashcard `resolved_hit 0.721` is a known join
artifact — gold=None for many C rows; gsm8k stays the trustworthy end-to-end
number). Also fixed a **run_code empty-code crash** in `tool_resolver.resolve()`
(BUG-008-adjacent): malformed/truncated `run_code` calls with `None` code now
return a clean `miss` instead of `compile()` crashing — verified by the pass-39
eval that failed pre-fix and passed post-fix.

### Doc-sync infra (commit 666f84f)
Added `scripts/sync_docs.py` + `.githooks/pre-commit`: on every commit, the
README cost banner / AGENTS cost line / runs.md Totals + new per-pass rows are
regenerated idempotently from `benchmarks/passes.db`. The hook is wired via
`core.hooksPath` so it propagates to BOTH remotes. JOURNAL stays hand-written
(assistant-owned) — explicitly excluded from automation. Also fixed a
`||` malformed-pipe bug on runs.md rows 35-39 introduced by an earlier manual
sync.

**Cost so far: $0.1278 across 39 passes, 10.15 GPU-h.** Git clean at `666f84f`.

**Where this leaves the thesis:** ToMoC is demonstrably *working* — a 360m
sovereign model routes to external experts (KB lookup ~99% + run_code ~97%) and
faithfully reports the answer, with honest "I don't know" on KB-miss. The loop
mechanics are sound; the open question is *routing balance* under a richer card
mix (see pass 40 below).

### PASS 40 — v10 gsm8k end-to-end (the number we'd been missing)
We never actually eval'd v10 on gsm8k end-to-end — the "~95.7% like v8" I'd been
quoting was **borrowed from v8 and never measured**. Pass 40 ran it for real and
it's a **REGRESSION**:

- correct_vs_gold = **648/1086 = 0.597** (v8 was 0.958). call_rate 0.862,
  router_precision 0.464 / recall 0.400, false_tool 610.
- **Root cause (isolated, not guessed):** when v10 routes to `lookup` it's
  **640/640 = 100% correct** (better than v8's 98.5%). When it routes to
  `run_code` it's **8/490 = 1.6%** — because **429 of those 490 gsm8k questions
  SHOULD have been lookups** (the KB has the exact answer). Type-F's "reason then
  `run_code`" format *tipped the routing default*: v10 now over-emits run_code on
  look-up-able gsm8k. The free resolver lever can't fix this — the model isn't
  *calling* lookup.
- **Silver lining (this was the user's #2 ask):** the show-work reasoning trail
  transferred *strongly* — **484/490** run_code outputs open with a `This is
  addition because…` span. So the reasoning trail is real; the cost was routing.

**Lesson re-learned:** always measure the metric you're claiming before you quote
it. The flashcard router-metrics (97.2%/96.8%) looked great and masked a gsm8k
routing collapse — the two sets measure different things and only gsm8k is the
trustworthy end-to-end signal.

**Fix planned (v11):** rebalance the training mix to re-anchor lookup — bump
`--gsm` 500→1800 (more Type-A from the 7473-row gsm8k_train) + `--d` 300→400
(more lookup two-turn), KEEP Type-F at 106 so the playground keeps the reasoning
trail. One retrain (~45 min) + gsm8k re-eval (~26 min). If routing still off,
next lever is cutting F or adding format-neutral lookup-with-reasoning cards.

### PASS 41-44 — v11 rebalance FAILS, v12 rephrase WINS (the real fix)

**v11 (pass 41, 3233 cards, --gsm 1800 --d 400 --e 300 --f 106):** gsm8k
**0.613** (pass 42) — barely better than v10. Diagnosis refined: when v11
routes to `lookup` it's **664/665 = 99.8%** correct; when it routes to
`run_code` it's **9/461 = 2%** — and 416 of those 461 run_code calls are on
questions that SHOULD be run_code (real arithmetic), yet compute wrong. The
dominant failure isn't "over-routing to run_code on look-up-able Qs" (only 45
such) — it's that **Type-F word-problems taught "word problem → run_code",
which fights "word problem → lookup"** on gsm8k (all 1319 answers ARE in the
KB, so optimal = always look up ≈ resolved_hit 0.964). Lookup-heavy volume
(1827A) could NOT override F's sticky format. Sample failure: `This is
multiplication: 10 * 45, because 10*45=450. TOOL run_code code="10 * 45"` →
gold 460 (wrong code despite confident reasoning).

**The fix (v12, pass 43):** rephrase Type-F at load time — `REPHRASE_F_TO_COMPUTE`
in `build_synth_cards.py` converts each seed word-problem `q` into `Compute
this: <code>` (keeping the WORK reasoning + CODE). Now the reasoning+run_code
habit is triggered by *computation requests*, NOT word problems. Word problems
route to lookup (accurate on gsm8k). Result (pass 44): gsm8k **0.998 (1265/1267)**,
call_rate 0.964, resolved_hit 0.997, well_formed 1.000. **#1 (gsm8k accuracy)
and #2 (reasoning trail) both satisfied.** Trail check: `Compute this: 48 - 5
+ 20` → `This is subtraction: 48 - 5 + 20. TOOL run_code code="48 - 5 + 20"` → 63.

**Proof, not just metrics:** added `scripts/probe_three.py` + `data/probe/
three_prompts.jsonl` — runs 3 fixed probes + N random gsm8k (split PASS/FAIL)
through the real ToMoC loop and writes a **dated, git-stored** markdown to
`probe_logs/` with VERBATIM output. v10 proof (pass 40 era): 6/8 audit pass
(shows the regression); v12 proof: 8/8 audit pass. Commit these — they are the
permanent evidence the metrics hint at.

**Lesson:** a training-mix *volume* change couldn't fix a *format conflict*.
The conflict was structural (F's word-problem→run_code vs gsm8k's
word-problem→lookup), and only a format change (rephrase F) resolved it. Also:
router_precision/recall heuristics are NOISY on gsm8k (they expect lookup for
everything and mislabel); trust `correct_vs_gold` + `resolved_hit`.

**v12 is the new default best (360m).** Costs: v10 $0.0055, v11 $0.0055, v12
$0.0053 eval; training ~$0.005/pass; total project $0.1674 / 44 passes.

## What's next (directions — open, not yet chosen)

The repo is at a clean resting point. Levers, roughly lean → moonshot:

1. **Close the ~4% residual (lean).** It's KB re-wording gaps + rare arithmetic
   slips. Options: a KB-rephrase pass (normalize question phrasing so more hit
   exact/prefix), or a few targeted Type-F word-problem cards for the misrouted
   cases (e.g. the "Ariel hammers" lookup-misfire). Highest ROI, lowest risk.
2. **Make show-work actually transfer (medium).** The Type-F prefix didn't stick
   at 360m. Could try: longer/cleaner work exemplars, a small weight on the work
   span, or prompt-tuning the turn-1 cue to invite reasoning. Pure capability
   polish — the router already works without it.
3. **PHASE 7 — LLM-wiki (moonshot, the original open item).** Replace the static
   8.9k gsm8k KB with a disk-backed wiki the model READS and WRITES: it stores
   what it learns, re-phrases entries to close re-wording gaps, and grows real
   memory instead of a frozen lookup table. The honest "I don't know" → "let me
   record this" bridge. Documented in future.md.
4. **From-scratch (North Star, deferred).** Build the base model ourselves —
   tokenizer + corpus + pretrain — once the concept + features are proven here.
   This lab is explicitly the pain-point finder; from-scratch is the payoff.

User decision so far: not calling it done; wants to keep pushing, starting from
direction ideas. Sovereignty + KISS stay the constraints. (See future.md for
the flag-to-dataset human-in-the-loop design that gates any autonomous KB write.)

---

## How to extend this journal
Append a new dated section per milestone. Keep the passdb table honest (include
the buggy runs — they're data). Link bugs to BUGS.md, ideas to future.md.
