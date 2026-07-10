# smol-lab

> Mission: a *very small* LLM that knows how to LOOK UP what it needs to know.
> It doesn't have to be smart — it has to REASON well enough to call the right
> function with the right argument. The functions ARE its knowledge.
> Everything here is experimental, fast-and-loose, villain-coded.

> 🔌 **Total electricity cost so far: $0.0148** across 9 training/eval passes
> (1.18 GPU-hrs @ 14¢/kWh, ~90W over server idle). Sovereign compute is cheap.
> Refresh: `python -c "from scripts.passdb import PassDB as D; D().cost_report()"`

## The thesis
A small model can't store much. So instead of memorizing facts, it learns to
emit a tool/function call when it hits a gap: "I don't know from memory → go
look it up." The model's *functions become its knowledge*. Reasoning quality
matters more than raw knowledge — we want it to pick the right tool, not to
already know the answer.

## Vision (where this is going)
This lab is a learning experiment with a real goal: **wean off LLM providers**
and run sovereign intelligence on homelab hardware. Principles:
- **Homelab-first, no external services.** If it needs an API call, it's out.
- **Disks are cheap, VRAM isn't.** Knowledge lives on disk (tools, a wiki), not
  in model weights. The model stays tiny; capability scales by *adding tools*,
  not parameters.
- **Functions ARE its knowledge.** The model doesn't memorize — it decides *where
  to look*, then calls that tool.

The endgame architecture is **ToMoC** (Tool-Routed Mixture of Capabilities):
the model's tool-call decision is the *router*, and the "experts" are external,
disk-backed tools (lookup, calculator, a user-correctable LLM-wiki). v1 just
teaches the first router habit (call `lookup` when stuck); scaling to "MoE" means
adding tools + orchestration, never retraining a bigger model. Longer-term
ambitions (correct-the-KB-with-verified-facts, build-everything-from-scratch
with 100% own data) are parked in [future.md](future.md) — tool-calling is the
priority for now.

SmolLM (135M / 360M / 1.7B) is a chat model with NO tool support —
confirmed against local Ollama (`does not support tools`). The lab's job is to
teach it tool calling anyway via a LoRA adapter that emits a tool-call "script."

## Benchmark results (llm_eval, run-20260710-040720, all done)
Task scores averaged across 8 tasks. Lower latency / higher tok/s = better fit
for the Tesla P4.

| Model                  | Avg   | Avg lat (ms) | tok/s | Notes                         |
|------------------------|-------|---------------|-------|-------------------------------|
| ornith:9b             | 92.1% | 38989         | 5.1   | smart but 39s/item — too slow |
| RefinedNeuro/vibethinker-3b | 89.3% | 46027    | 8.8   | 46s/item — too slow          |
| sam860/VibeThinker:1.5b-Q8 | 86.7% | 19481    | 25.8  | reasoning-specialized, slow   |
| qwen2.5:1.5b         | 83.3% | 4229          | 45.3  | fast control; NATIVE tool support |
| smollm:360m           | 81.8% | 2989          | 106.4 | strong + fast                 |
| smollm:latest (1.7b)  | 79.0% | 7946          | 34.4  | 3x latency of small ones     |
| smollm:135m           | 72.0% | 3067          | 114.0 | fastest; weakest knowledge    |

smolLM per-task (the lookup-shaped holes):
  task            135m   360m   1.7b
  brainteasers    100%   100%   66.7%
  coding_func     100%   100%   100%
  mmlu_algebra     96%    98%    97%
  hallucination    90%    90%    95%
  knowledge_qa     60%   86.7%  86.7%
  math_gsm          0%    20%   46.7%
  reasoning_logic  30%    60%    40%
  summarization   100%   100%   100%

qwen2.5:1.5b is the notable control: it scores well AND supports Ollama tools
natively (smolLM doesn't) — useful as a comparison for whether our LoRA
adapter matches native behavior.

## Why smollm:135m is the pick
Not the smartest — deliberately. The mission wants a model that REASONS well
and leans on functions for knowledge, not one that already knows everything.
135m is the purest test of that thesis:
- Fastest of all (114 tok/s, ~3s latency) — great for P4 training iteration.
- Reasonably good reasoning where it counts: 100% brainteasers + coding_func,
  60% knowledge_qa, 100% summarization.
- Its WEAKNESSES are exactly the lookup-shaped holes: math_gsm 0%, reasoning_logic
  30%, knowledge_qa 60%. A 135m that learns "I can't do this from memory →
  call lookup" turns those zeros into fetched-correct answers. That's the win.
- 360m is the safer/more robust base (higher floor) if 135m can't reliably emit
  well-formed calls — fallback option, not the target.

## Two-project setup
Kept SEPARATE (per user decision):
- `~/smol-lab` (this dir) — training / hacking lab. venv + PyTorch + PEFT/TRL.
  LoRA adapters get built here.
- `~/llm_eval` — benchmark harness (FastAPI + SQLite + web UI). Drives Ollama's
  `/api/generate`, P4-tuned (concurrency=1, unloads models between phases).
  Picks the winner before we spend GPU on training; will later evaluate adapters.

Planned (not yet wired) flow:
  1. llm_eval ranks smolLM sizes on real tasks  [DONE — 135m chosen]
  2. LoRA base = smollm:135m
  3. lab trains tool-call adapter → adapters/
  4. point llm_eval at the adapter; measure if tool calling works

## Hardware reality (Tesla P4)
- Pascal, compute 6.1, 8 GB VRAM, ~5.5 TFLOPS. SLOW but sits idle a lot.
- Driver 580.x / CUDA 13.0; PyTorch 2.5.1+cu121 runs on it (sm_61 kernels).
- LoRA ONLY. Full fine-tune of 1.7B won't fit 8 GB.
- Ollama's `llama-server` also uses the P4 (~1.6 GB). Free VRAM before a run:
  `ollama stop` or kill the process if training OOMs.

## Activate
    source /home/aec/smol/.venv/bin/activate

## Rebuild from scratch
    uv venv .venv --python 3.13
    source .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    uv pip install -r requirements.lock.txt

## Layout
    data/raw        raw "flashcards" (tool-call training samples)
    data/processed  tokenized / formatted training files
    models/         base model weights (if downloaded locally)
    adapters/       trained LoRA adapters land here
    configs/        training / eval configs (yaml or json)
    scripts/        runnable training / probe scripts
    logs/           run logs
    benchmarks/     eval results (gsm8k + mmlu already in HF cache)

## Quick GPU probe
    python scripts/probe_env.py

## Current stack
    torch 2.5.1+cu121, transformers 5.x, peft 0.19, trl 1.8, datasets 5.x, accelerate 1.14

## Roadmap
Phased, KISS, baby-steps. Each phase ends with a passdb entry so we can compare.
Detailed parked ideas live in [future.md](future.md).
The build journey, real numbers, and lessons live in [wiki/JOURNAL.md](wiki/JOURNAL.md);
bugs and hotfixes in [wiki/BUGS.md](wiki/BUGS.md).

- [x] **Phase 0 — foundations (DONE)**: P4 env verified; smolLM sizes benchmarked
  via llm_eval; smolLM:135m picked as LoRA base; vision + ToMoC documented.
- [x] **Phase 1 — habit pipeline (DONE)**: flashcard generator + 60-card smoke
  set (50/50 A:B, diverse); `passdb.py` metrics store; `eval_toolcall.py` console
  harness; **baseline pass 1** logged (0% call / 0% over-call floor, 63s/60 cards).
- [ ] **Phase 2 — first real adapter (NOW)**: `train_adapter.py` trains a LoRA on
  smolLM:135m from the flashcards; produce pass 2; watch `call_rate_when_should`
  climb above 0 while `over_call_rate` stays low. (The core proof of the thesis.)
- [ ] **Phase 3 — scale + tune balance**: bump `A_RATIO` / synthesize more Type B
  cards; sweep epochs/lr/r to maximize call-rate without over-calling.
- [ ] **Phase 4 — proper JSON tool calling**: switch output to JSON + constrained
  decoding (GBNF/grammar) so calls are always valid. Habit transfers from mini-format.
- [ ] **Phase 5 — second tool (`calculate`/`run_code`)**: use 135m's 100% coding
  strength for math via execution; ToMoC grows to 2 experts.
- [ ] **Phase 6 — LLM-wiki + tooling framework**: disk-backed wiki as the lookup
  source; orchestration layer (pi/hermes/opencode-shaped) dispatches ToMoC calls.
- [ ] **Phase 7 — correct-and-update-KB**: feed verified facts; model updates its
  disk-backed KB (not weights) behind a verification gate.
- [ ] **Phase 8 — endgame**: reasoning scratchpad self-corrects calls; retrain from
  100%-own data (full sovereignty). "Functions ARE its knowledge" as architecture.

REJECTED / parked: distillation (needs external teacher → breaks homelab
sovereignty). Speculative decoding = speed-only, optional later.

## Status / TODO
- [x] env + P4 verification
- [x] training stack installed
- [x] benchmark smolLM sizes (llm_eval) — 135m selected as LoRA base
- [x] flashcard generator (scripts/build_flashcards.py) + v1 smoke set
      (data/raw/flashcards.jsonl, 60 cards, 50/50 A:B, diverse)
- [x] passdb metrics store (scripts/passdb.py) — logs every training/eval pass
- [ ] LoRA training script in scripts/ (train_adapter.py)
- [ ] evaluate adapter via scripts/eval_toolcall.py (uses passdb)
- [ ] baseline (base model, no adapter) eval pass for comparison
