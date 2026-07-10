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
