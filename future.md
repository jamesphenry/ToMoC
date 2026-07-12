# future.md — out of scope for the initial runs

> Parking lot for ideas that are GOOD but not for v1. Keep them here so we
> don't lose them and don't let them bloat the baby-steps build.
> Anything here is speculative until pulled into a real plan.

## Tool-call format evolution
- Train on custom mini-format first (`TOOL lookup query="..."`).
- LATER: proper JSON tool calling. Achievable via CONSTRAINED DECODING at
  inference (llama.cpp GBNF grammar, Ollama `format:"json"`, or outlines) —
  the grammar forces valid JSON regardless of model size. The LoRA only has to
  learn the HABIT; the grammar guarantees the shape.
- Key insight: don't train harder to fix malformed JSON — constrain output.

## Reasoning + self-correction of JSON
- Idea: add a reasoning step so the model can self-correct a malformed tool call
  before emitting it. (Spawned from the JSON-training discussion.)
- NOT the point for v1. Revisit after the basic call habit is reliable.

## Knowledge-base / lookup sources (longer-term)
- gsm8k_train (from llm_eval) repurposed as a seed "knowledge base" the
  `lookup` tool searches — offline, deterministic, perfect for first adapter.
- SearXNG for REAL web lookup later (live, needs a server; adds flakiness —
  keep out of v1 training/eval).
- LLM-wiki-style knowledge base: a structured wiki the model can query.
  **STARTED 2026-07-12 (Phase 7 substrate):** `data/wiki/wiki.jsonl` + `WikiKB`
  in `tool_resolver.py`. READ path live (`lookup` falls through to wiki, no
  retrain); WRITE path is human-in-the-loop (`--wiki-add`) for now. Model does
  NOT yet emit `TOOL wiki` — that needs a LoRA capability + retrain (see below).

## Baked-in wiki that DEFINES tools (hypothetical, user curiosity)
- Idea: an "LLM wiki" knowledge base that is itself baked into the model, from
  which TOOLS are auto-derived/defined for the model to use.
  i.e. the wiki isn't just searched — its structure generates the tool schemas.
- Plausible: a curated structured wiki (entries with typed fields) -> a build
  step emits one tool per entry-type or per queryable surface. The model learns
  the call format; the wiki is the source of truth + the tool catalog.
- Why interesting: collapses "knowledge" and "capability" into one artifact the
  model carries. Fits the "functions are its knowledge" thesis perfectly.
- Status: curiosity only. Investigate in a later iteration, not v1.
- General direction confirmed by user: "functions ARE its knowledge" — the model
  looks up what it needs rather than storing it.

## Math via execution
- Idea: since 135m excels at coding tasks (100% coding_func), have it solve
  math by WRITING + RUNNING python instead of computing in-text.
- Could be its own tool: `run_code(code)` that executes and returns stdout.
- Open question: overcomplicated for v1? Park until base lookup habit works.
  (Note: this overlaps with the calculator idea — python subsumes calc.)

## Multi-tool
- Start with ONE tool (`lookup`). Later: lookup + calculate/run_code + others.
- Don't design the multi-tool schema yet — premature.

## Eval framework
- Decision: a SMALL CUSTOM eval framework, SEPARATE from llm_eval, no UI,
  console output (verbose optional). Lives in this lab (scripts/eval_*.py).
- Measures: did the model emit a tool call when it should? correct tool? correct
  arg? (Not raw accuracy — the call decision is the metric.)
- llm_eval remains the benchmark for base-model ranking only.

## Destination / model-feature map (user curiosity, ELI5)
Where the project COULD grow. None is v1; tool-call is priority. Captured so
the end-goal is documented, not lost.

GUIDING PRINCIPLE (user, 2026-07-10): homelab-first, wean off LLM providers,
no external services whenever possible, keep requirements LOW. "Disks are
cheap, VRAM isn't." Functions are knowledge.

- MoE (Mixture of Experts): team of specialists + a router picking 1-2 per
  question; only those wake up -> small/fast but smart. Classic MoE bakes
  experts into the MODEL WEIGHTS (needs them in VRAM).
- ToMoC (Tool-Routed Mixture of Capabilities) — USER-COINED, the chosen path:
  the "router" is the model's decision to CALL A TOOL; the "experts" are
  EXTERNAL (LLM-wiki, calculator, lookup) living on DISK, not in weights.
  -> 135m stays tiny forever; capability scales by ADDING TOOLS, not params.
  -> No weight-MoE needed; the tooling framework (below) IS the orchestration.
  -> v1 LoRA (lookup-or-not) is already the first router. Scaling to "MoE" =
     add tools + orchestration, not retrain architecture. Matches "disk cheap,
     VRAM isn't" exactly.
- Reasoning: model thinks out-loud (scratchpad) before answering. = the parked
  "self-correct JSON" idea. Upgrade that makes tool calls cleaner, post-v1.
- Distillation: small model (student) copies a big model's (teacher) behavior.
  REJECTED for this project — not a params problem, a DEPENDENCY problem: the
  teacher is usually an external/API model, which violates the homelab-sovereignty
  goal. Out. (Your instinct was right for a deeper reason than params.)
- Speculative/draft decoding: write several tokens, verify, rewind if wrong.
  SPEED only, irrelevant to the tool mission, but nice on the slow P4.
- RAG (retrieval-augmented): look up facts from a store before answering. OUR
  `lookup` tool IS micro-RAG. The LLM-wiki idea = RAG-with-benefits.

### Goals beyond v1 (user, captured)
- CORRECT-AND-UPDATE-KB: ability to correct the model with VERIFIED facts; it
  should update its own knowledge base (the disk-backed wiki), not its weights.
  -> Embodies "disks cheap, VRAM isn't": knowledge lives on disk, fix it there.
  -> **WRITE path live 2026-07-12 (Phase 7):** `tool_resolver.py --wiki-add` upserts
     human-authored entries atomically. Model autonomy (auto-write from a
     `wiki_write` tool call) is NOT yet built — that needs a LoRA capability +
     retrain + a verification gate so bad corrections don't poison the wiki.
- FLAG-TO-DATASET (user idea, 2026-07-11 — human-in-the-loop correction loop):
  When the model gets something wrong (or is unsure), instead of it auto-writing
  the KB, the USER tells the model something like "flag that" so it can be
  reviewed. The human then goes in and CREATES A TRAINING DATASET ENTRY teaching
  the correct response (e.g. a Type-F "show your work" card: problem + the right
  operation + correct code/answer). That card is folded into the next retrain.
  -> Key distinction from plain CORRECT-AND-UPDATE-KB: the correction is expressed
     as SUPERVISED TRAINING DATA the human authors, not an autonomous KB write.
     The model doesn't correct itself — it surfaces the failure, the human
     authors the fix, retraining installs it. Keeps the human as the verifier
     (sovereignty + no poisoned-KB risk) while still closing the loop.
  -> Natural pairing: the playground "flag" could tag a turn (like the existing
     /mark seen|fixed on chat exports) and emit a stub card (Q + placeholder
     WORK/CODE/A) for the human to fill in. Feeds directly into the f_cards
     format used for Type-F reasoning training. Post-v9, post-Type-F feasibility.
- BUILD FROM SCRATCH (end-state ambition): after this experiment, user wants to
  create everything from scratch with 100%-own data (tokenizer, corpus, training
  set) — full sovereignty, no borrowed base. Long-term; v1 still uses smolLM:135m
  as a pragmatic bootstrap. Note the tension: "from scratch" is a big lift; the
  ToMoC tool-layer softens it (capability comes from tools, not the base model).

## Resources for from-scratch (collected, not yet used)
- **SmolLM corpus** (HuggingFaceTB): https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
  — the exact pretraining corpus used to build the smolLM base models
  (cosmopedia / fineweb-edu / etc.). When we eventually pretrain our own base
  (North Star), this is the reference corpus to bootstrap from / compare against.
  Saved 2026-07-12 on user's pointer.
- **SmolLM repo** (huggingface/smollm): https://github.com/huggingface/smollm
  — official repo: training code, tokenizer, data recipes for the smolLM family.
  Reference for the full from-scratch build pipeline.
- **OpenHermes-2.5** (teknium): https://huggingface.co/datasets/teknium/OpenHermes-2.5
  — general instruct/SFT corpus (1M+ multi-turn). For the instruction-tuning stage.
- **self-oss-instruct-sc2-exec-filter-50k** (bigcode):
  https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k
  — self-instruct code corpus w/ exec-filter. Relevant if we want code capability.
- **Magpie-Pro-300K-Filtered** (Magpie-Align):
  https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered
  — large filtered alignment/instruct corpus. For the alignment/SFT mix.
- **smollm2-135-implementation** (abi2024):
  https://github.com/abi2024/smollm2-135-implementation
  — minimal from-scratch 135m impl (architecture + training). Good study reference
  for the eventual own-base build (matches our 135m target size).
- All above are **reference only** for the eventual from-scratch pretrain + SFT +
  alignment stack. Not used by the current LoRA tool-calling experiments.
  Saved 2026-07-12 on user's pointer.

Possible end-state: a tiny 135m router (v1 LoRA habit) that dispatches to
external ToMoC experts (lookup, calculate, LLM-wiki) via a homelab tooling
framework; a reasoning scratchpad self-corrects calls; the KB is user-correctable
and disk-backed; eventually retrained from 100% own data. All small, all on the
P4, zero external dependencies. "Functions ARE its knowledge" as an ARCHITECTURE.

## Tooling framework (far-future, user note)
- The project will eventually need a TOOLING framework (orchestration / agent
  loop around the model's tool calls), motivated by the LLM-wiki + baked-in
  knowledge ideas + ToMoC dispatch. User name-dropped pi, hermes, opencode as
  shape refs.
- NOT locked in; not v1. Discuss much later — after the base lookup habit
  and the wiki-as-tool-catalog idea are proven. Just captured so it's not lost.
