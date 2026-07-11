# wiki/BUGS.md — real bugs hit in smol-lab (with root cause + fix)

> Lessons learned the hard way. Each entry: symptom, root cause, fix, and the
> commit/pass where it was caught. The model's knowledge lives on disk — so do
> ours. Add new bugs here as they're found.

---

## BUG-001 — eval parse_call scored correct calls as malformed
- **Symptom**: `eval_toolcall.py` `well_formed_rate` / `correct_tool_rate`
  would be 0 even when the model emitted a perfect `TOOL lookup query="..."`.
- **Root cause**: `CALL_RE = r'TOOL\s+lookup\s+query="(.+?)"\s*"$'` required a
  SECOND trailing quote. Real format has one closing quote. Baseline hid it
  (0 calls to parse); would have silently broken eval once the adapter learned
  to call.
- **Fix**: `CALL_RE = r'TOOL\s+lookup\s+query="(.*)"'` (matches the actual
  single-quote format). Verified with 4 cases (well-formed / plain / malformed
  tool / long query).
- **Caught**: fresh ad-hoc verification (hermes-verify) before commit, pass 1
  baseline. Commit `ac1227e`.

## BUG-002 — Trainer crashed: "labels excessive nesting" / batched tensors
- **Symptom**: `train_adapter.py` died at step 0 with
  `ValueError: Unable to create tensor ... 'labels' ... excessive nesting`.
- **Root cause**: `DataCollatorForLanguageModeling(mlm=False)` pads
  `input_ids`/`attention_mask` but does NOT pad `labels` the same way, so
  variable-length `labels` lists can't stack into a tensor. (Only bites when
  batch has unequal-length sequences — i.e. always, with mixed card lengths.)
- **Fix**: replaced with a custom `PadCollator` that pads `input_ids`,
  `attention_mask`, and `labels` together; `labels` padded with -100 (loss
  ignores). Caps at `max_len`.
- **Caught**: first training run (pass 2 attempt), crashed before GPU use. Fixed
  in place, re-run pending.

## ENV-001 — HF_TOKEN is NOT in ~/.bashrc
- **Symptom**: `source ~/.bashrc` did not expose `HF_TOKEN`.
- **Root cause**: user believed the token lived there; it does not (or is set
  elsewhere / not exported). Irrelevant for public models — `smollm-135m-instruct`
  downloaded fine unauthenticated (just a rate-limit warning).
- **Action**: don't rely on ~/.bashrc for HF_TOKEN. If a private model is ever
  needed, ask the user where the token actually lives.

## BUG-003 — eval loaded LoRA adapter as a base model → 100% CPU, garbage output
- **Symptom**: `eval_toolcall.py --model adapters/v1` ran at 100% CPU, 12% GPU,
  took 7+ min for 60 cards, and produced meaningless output. No error raised.
- **Root cause**: `adapters/v1` is a LoRA adapter only (adapter_config.json +
  adapter_model.safetensors, NO base weights). `AutoModelForCausalLM.from_pretrained`
  on an adapter dir can't find a base, silently falls back to CPU, and decodes
  noise. Training worked because it loaded the real base then attached LoRA;
  eval didn't replicate that.
- **Fix**: `Engine` now detects `adapter_config.json` in the dir and loads
  `base (DEFAULT_BASE) + PeftModel.from_pretrained(base, adapter_path)`. Eval
  runs on cuda:0 now.
- **Caught**: user observed 100% CPU in nvidia-smi during the pass-3 eval run.
  Fixed before re-running.

## BUG-004 — eval reloaded the model once PER CARD (60x)
- **Symptom**: even the baseline (pass 1) took 63s; would get worse with bigger
  datasets. Each `generate()` call reloaded weights from disk.
- **Root cause**: `evaluate()` called `generate(model, prompt)` per card, and
  `generate()` instantiated the model every time.
- **Fix**: introduced an `Engine` class that loads the model ONCE in __init__
  and is callable per card. `evaluate()` now reuses one Engine. (See BUG-001
  era — this was the dominant cost, not training.)
- **Caught**: during pass-3 diagnosis; fixed alongside BUG-003.

## ENV-003 — `torch_dtype` deprecated; `temperature` invalid with do_sample=False
- **Symptom**: transformers warnings: `torch_dtype is deprecated! Use dtype`
  and `generation flags not valid: ['temperature']`.
- **Root cause**: API drift in transformers 5.x. Temperature is ignored when
  do_sample=False anyway.
- **Fix**: use `dtype=torch.float16` and drop the `temperature` arg in eval's
  local generate. (train_adapter.py still uses torch_dtype — works but warns;
  left for a later cleanup pass.)
- **Action**: harmless, cosmetic. Note for future: pin transformers or migrate.

## BUG-005 — eval pegged 1 CPU core / 12% GPU on 60 separate generate() calls
- **Symptom**: `eval_toolcall.py` (pre-batch) ran at 100% on a SINGLE CPU core,
  12% GPU, ~7 min for 60 cards. The model was proven on cuda:0 (probe showed
  every layer on cuda:0), so it was NOT CPU inference — it was host-bound sync.
- **Root cause**: cProfile proved 50.6s/52.7s of a 5-card run was inside
  `modeling_llama.py` forward, but split across 5 separate `model.generate()`
  calls. Each call launches ~7680 tiny layer kernels serially; the GPU never
  saturates and Python spins in the sync loop → one core pegged, GPU idle.
- **Fix**: `Engine.generate_all(prompts)` batches ALL prompts into ONE
  `model.generate()` (batched sequences saturate the P4). Eval now does 60
  cards in ~41s with GPU at 100% (user visually confirmed). ~10x faster.
- **Caught**: user reported "100% cpu" twice; profiling localized it.
- **Verified**: ad-hoc /tmp check + pass-4 ran at 100% GPU.

## BUG-006 — right-padding corrupts batched decoder-only generation
- **Symptom**: after batching (BUG-005), outputs were garbage (`Question\nQuestion\n...`
  repetition) and a transformers warning fired: "right-padding was detected!
  For correct generation results, please set padding_side='left'".
- **Root cause**: decoder-only LMs generate left-to-right; right-padding puts
  PAD tokens inside the prompt region, so the model attends to pads and emits
  noise. Must use LEFT padding for batched generation.
- **Fix**: set `tok.padding_side = "left"` around the encode; generated tokens
  then start at the full padded seq length S (NOT per-row prompt len L, which
  would leak the prompt). Slice `out[i][S:]`. Verified no leak on real prompts.
- **Caught**: warning + garbage output after batching.
- **Verified**: real-prompt leak check → NO LEAK; synthetic-marker "leak" was a
  false positive (tiny model echoes novel tokens — model behavior, not a bug).

## ENV-002 — Ollama is a system service; user has NO sudo
- **Symptom**: `kill`/`pkill` on `ollama serve` fails or respawns; `ollama stop`
  unloads the model but the service stays up.
- **Root cause**: ollama runs under systemd; this user is not root, so it
  cannot be stopped to free the P4.
- **Action**: NEVER try to kill ollama. Training must coexist with ollama's
  ~1.6GB VRAM footprint. smolLM:135m (269MB fp16) trains fine alongside it on
  the 8GB P4. If a bigger base ever OOMs, that's a model-size problem, not a
  "stop ollama" fix.

## BUG-007 — unchunked batched eval OOMs the 8GB P4
- **Symptom**: `eval_toolcall.py --model adapters/v2 --data flashcards2.jsonl`
  (827 cards) died with `torch.OutOfMemoryError: tried to allocate 374 MiB`
  while 7.35 GiB was "in use" (5.55 allocated + 1.67 reserved-but-unallocated).
- **Root cause**: `Engine.generate_all` batched ALL prompts into ONE forward
  pass (correct for BUG-005's CPU fix, but at 827×256 tokens the single forward
  exceeded the P4). The caching allocator had also eagerly reserved a 5.5GB pool
  from training, leaving <100MB free. Fragmentation, not live tensors.
- **Fix**: chunk `generate_all` into `chunk=16` prompts per forward pass (loop +
  `torch.cuda.empty_cache()` between chunks). Peak VRAM dropped 7.35GB → ~0.5GB.
  Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fight
  fragmentation. Batching is preserved (still not one-reload-per-call), just
  bounded per forward.
- **Caught**: eval run after v2 training (pass 6) completed.
- **Verified**: re-run with chunk=16 → GPU mem 553 MiB, process alive, no OOM.

## BUG-008 — eval scored well-formed calls as malformed (truncation artifact)
- **Symptom**: v2 showed `well_formed_rate=0.488` while `call_rate_when_should=0.964`
  and `correct_tool_rate=1.000` — i.e. it called, but half the calls were "malformed."
- **Root cause**: every malformed call was `TOOL lookup query="<long text` with ONE
  quote — the model emits the correct format, but `max_new_tokens=64` truncated the
  generation BEFORE the closing quote on long questions. `CALL_RE` required the
  closing quote, so the call scored as malformed even though the format intent was
  correct. It was a MEASUREMENT bug, not a model flaw.
- **Fix (2 parts)**:
  1. Parser (correctness of measurement): `CALL_OPEN_RE = r'TOOL\s+lookup\s+query="(.*)'`
     accepts the open-quote (truncated) form as well-formed. Re-scoring v2 with it
     lifted `well_formed_rate` 0.488 → **0.964** (pass 8) — the model was never broken.
  2. Training data (real format completeness): `build_synth_cards.py` now caps the
     query to `MAX_Q=180` chars (word-boundary cut + ellipsis) so the full
     `query="..."` (with closing quote) fits inside the 64-token budget. v3 trains on
     this so the model learns to CLOSE the quote, not just start it.
- **Lesson**: diagnose before fix. The 0.488 "gap" was phantom — almost retrained to
  fix a non-problem. One regex change proved the model was already 96% correct.
- **Caught**: `hermes-formcat` ad-hoc capture of real v2 outputs on A-cards.
- **Verified**: parser fix re-scored (pass 8: well_formed 0.964); mk_A cap verified
  strict (≤180) across query lengths; v3 training launched on capped data.

## BUG-009 — jsonl KB loader shattered by Unicode line separators (U+2028)
- **Symptom**: `scripts/tool_resolver.py` `load_kb()` raised
  `json.decoder.JSONDecodeError: Unterminated string` on gsm8k_train.jsonl
  line 2382 (record "Clive opens a box full of different colored balls..."),
  even though every line parsed fine when read line-by-line from the file.
- **Root cause**: the loader did `text = open(...).read()` then
  `text.splitlines()`. `str.splitlines()` splits on MANY boundaries — not just
  `\n`, but also Unicode line/paragraph separators U+2028 and U+2029, which
  can legitimately appear INSIDE a JSON string value. So a perfectly valid
  record containing U+2028 got chopped into two "lines," the second of which
  was an unterminated fragment → JSONDecodeError. (The earlier per-line check
  used the file iterator, which only breaks on real `\n`, so it saw no error —
  that's why the two checks disagreed.)
- **Fix**: read `.jsonl` line-by-line from the FILE OBJECT (the iterator only
  breaks on real newlines, never on in-string U+2028). Genuinely broken lines
  are now caught individually and skipped + counted (data-hygiene) instead of
  aborting the whole KB. `.json` wrapped files keep `json.loads(fh.read())`.
- **Caught**: standalone `tool_resolver.py --stats` smoke test before any GPU
  run. Real data, real bug — would have silently dropped/aborted the math KB.
- **Lesson**: never `splitlines()` JSONL you didn't author. Use the file
  iterator (or a JSONL reader) so in-string Unicode separators can't shatter
  a record. `textwrap`/`splitlines` family is for display text, not records.

## BUG-010 — resolver missed valid lookups on Unicode punctuation drift (spot-check)
- **Symptom**: pass-11 spot-check found 27 `resolved_miss` rows that were NOT
  real KB gaps — re-running the emitted query against the KB often matched a
  prefix but scored `miss`. Two distinct causes:
  1. **Curly-quote drift (4/27)**: KB prompts contain typographic `’`/`“`/`”`
     but the model emits ASCII `'`/`"`. `norm()` lowercased but did NOT fold
     Unicode punctuation, so `kp.startswith(nq)` failed at that one codepoint.
  2. **Re-wording drift (~13/27)**: the model re-tells the problem instead of
     quoting verbatim — "stolen"→"stole", "enormous"→"immense", "brother"→
     "sister", "none"→"50 birds". The prefix tier can't catch substitution
     (and shouldn't blindly, or it false-matches near-duplicate gsm8k problems).
- **Fix**: `norm()` now folds Unicode punctuation to ASCII (curly quotes/dashes,
  nbsp, U+2028/2029) BEFORE matching, and `FUZZY_THRESH` lowered 0.8 → 0.7.
  Replaying the real pass-11 log through the updated resolver recovered **14 of
  27** misses with **0 false positives** against the 1282 established hits →
  end-to-end 97.2% → **98.3%** (1296/1319). The residual 13 are genuine model
  verbatim-drift the resolver deliberately does NOT guess at (risky false matches).
- **Caught**: pass-11 per-item log spot-check (`logs/eval_resolver_*.jsonl`).
- **Verified**: ad-hoc log-replay (no GPU) — 14 recovered, 0 false positives.

## BUG-011 — gen_arith used `c` before assignment in 3-arg templates
- **Symptom**: `build_synth_cards.py` (with the new Type-C run_code generator)
  crashed `UnboundLocalError: cannot access local variable 'c' where it is not
  associated with a value` when a 3-arg word-problem template was selected
  before any 2-arg one (the `c = rng.randint(...)` lived only in the else branch).
- **Root cause**: `c` was assigned inside the 3-arg branch but referenced in the
  shared `tmpl.format(a=a, b=b, c=c)` call below. If the first template drawn was
  3-arg, `c` was fine; but the loop hit a 2-arg template first on some seeds and
  `c` was never bound.
- **Fix**: always define `c = 0` up front (and only overwrite in the 3-arg
  branch). Also tightened division templates to keep the divisor small (2-12).
- **Caught**: running the generator before training (no GPU wasted).
- **Verified**: regenerated 977 cards; all 150 Type-C `code` strings accepted by
  the sandbox AND compute to their stored `answer` (0 mismatches).

---

## Sandbox design notes (scripts/sandbox.py, Phase 5)
`run_code` executes untrusted model output. Defense-in-depth (not a general REPL):
- AST pre-scan (`_scan`) rejects imports, defs, `__import__`, `open`, dunder
  attrs, and `while/with/try/raise/...` BEFORE spawning anything.
- Execution runs in a SEPARATE `-I` (isolated) subprocess, stripped env, with a
  `RLIMIT_CPU` cap + Python `timeout` — a `while True` is killed, not hung.
- Verified adversarial: `import os`, `from os import`, dunder attr, `open()`,
  function defs, and `open()` inside a comprehension all rejected at the AST scan.
  Math (incl. big ints) computes correctly. This is the compute half of ToMoC.

## How to add a bug
Copy the template, bump the number, fill it in, commit with the others.
Keep fixes minimal and note the commit hash so we can bisect later.
