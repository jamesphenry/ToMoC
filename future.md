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
- LLM-wiki-style knowledge base: a structured wiki the model can query. Details
  TBD (corpus shape, retrieval method). Discuss later.

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
  -> Implies the KB needs a write path (not just lookup), plus a verification
  gate so bad corrections don't poison it. Post-v1, post-tooling-framework.
- BUILD FROM SCRATCH (end-state ambition): after this experiment, user wants to
  create everything from scratch with 100%-own data (tokenizer, corpus, training
  set) — full sovereignty, no borrowed base. Long-term; v1 still uses smolLM:135m
  as a pragmatic bootstrap. Note the tension: "from scratch" is a big lift; the
  ToMoC tool-layer softens it (capability comes from tools, not the base model).

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
