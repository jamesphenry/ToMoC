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

---

## How to add a bug
Copy the template, bump the number, fill it in, commit with the others.
Keep fixes minimal and note the commit hash so we can bisect later.
