# TOMOC — Tool-Routed Mixture of Capabilities

> The router owns the *decision of where knowledge lives* — not the knowledge
> itself. The tiny model is the smallest, most stable component in a sovereign
> AI system whose intelligence lives in disk-backed capabilities, not weights.

This document is the canonical architecture reference for the smol-lab project.
It deliberately contains **no experiments, no benchmarks, no bug lists** — those
live in `wiki/JOURNAL.md` and `wiki/BUGS.md`. Here we describe *how the machine
works* and *why it is built this way*.

---

## 1. Philosophy

The project is a learning lab: a place to find the pain points of a specific
architecture before committing to it. The guiding principles (derived from the
project's stated ultimate goal and the efficiency-over-scale thesis):

- **Disks are cheap, VRAM isn't.** Always ask *"can we move this responsibility
  onto disk?"* — knowledge, memory, tools, indexes, planning — rather than
  *"can we fit more into VRAM?"*
- **Functions ARE its knowledge.** The model does not memorize. It decides
  *where to look*, then calls that tool. Reasoning quality matters more than raw
  knowledge.
- **Efficiency over scale.** Parameter efficiency (how small can the router be?),
  memory efficiency (knowledge on disk), compute efficiency (wake a capability
  only when needed), training efficiency (update the KB, not the weights), and
  energy efficiency (every run is costed in electricity).
- **Build everything from scratch — eventually.** The long-term goal is to own
  the tokenizer, corpus, pretraining, instruction tuning, tool syntax, eval
  suite, knowledge format, and orchestration. This is the *destination*, not
  the next step. The lab earns it by collecting the gotchas first.

The research question the architecture is designed to answer: **which
capabilities must live in neural weights, and which can live in software, data
structures, and tools?** We measure the boundary; we don't assume it.

---

## 2. Terminology

| Term            | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| **Router**      | The tiny LLM + LoRA. Emits a tool call. Owns *which* tool, not *how*. |
| **Tool call**   | A mini script: `TOOL <name> <arg>="<value>"`.                        |
| **Expert / Capability** | An external, disk-backed function (lookup, run_code, …).    |
| **Resolver**    | Dispatches a parsed call to the right expert and returns a result.   |
| **Orchestration** | The two-turn loop: emit call → resolve → feed result → final answer. |
| **KB / Wiki**   | Disk-backed knowledge the router reads (and, later, writes).         |
| **ToMoC**       | Tool-Routed Mixture of Capabilities — this architecture.             |

---

## 3. Architecture

```
            User Question
                 │
                 ▼
   ┌─────────────────────────────┐
   │  Tiny Router (LoRA on 360M)  │   only decides WHICH capability
   └─────────────────────────────┘
                 │  emits: TOOL <name> arg="..."
                 ▼
   ┌─────────────────────────────┐
   │  Orchestration (2-turn loop) │   call → resolve → result → final
   └─────────────────────────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
  TOOL lookup         TOOL run_code
       │                    │
       ▼                    ▼
  Sovereign KB         Sandboxed exec
  (8892 entries)       (restricted Python)
       │                    │
       └─────────┬──────────┘
                 ▼
          Tool result → fed back
                 │
                 ▼
          Final answer
```

The model never computes or recalls the answer itself. It routes. The
capabilities and the knowledge evolve **independently** of the router — adding a
new expert requires no retraining of the router's size, only more routing
examples.

### Capability Independence (design principle)
The router must not know *how* a capability is implemented. `lookup` is today a
sovereign SQLite/resolver table; it could become a local vector index or a
wiki query without changing a single router weight. The resolver is the only
component that knows the implementation, and it is swappable behind a stable
dispatch seam (`tool_resolver.resolve(tool, query, kb)`).

---

## 4. Routing

Training teaches the router *when* to call, via a LoRA that emits the call
script under a fixed priming cue:

```
If you are not certain of the answer, call the lookup tool instead of guessing.
Question: <q>
Answer or call a tool:
```

Card types (see `scripts/build_synth_cards.py`):

- **A — lookup**: `TOOL lookup query="<verbatim q>"` (fetch knowledge)
- **B — answer**: `<answer>` directly (model answers, no tool)
- **C — run_code**: `TOOL run_code code="<expr>"` (compute)
- **D — two-turn report**: question + emitted call + tool result → final answer
  (closes the empty-turn-2 gap; the model learns to *echo* the result, not
  hallucinate it)
- **E — KB-miss honesty**: when the lookup misses, emit a graceful "no answer
  found" instead of guessing a number (the real fix for the echo weakness)
- **F — show-your-work**: emit a short reasoning prefix, then the run_code call
  (teaches the model to *translate the word problem into code* — the residual
  run_code error is semantic parsing, not arithmetic)

---

## 5. Experts (current)

| Expert      | Implementation                         | Sovereign? | Notes                         |
|-------------|----------------------------------------|-----------|-------------------------------|
| `lookup`    | `tool_resolver.py` (exact→prefix→fuzzy→miss), 8892-entry KB | yes | zero external APIs |
| `run_code`  | `sandbox.py` (AST-scan rejects imports/open/defs/dunders; subprocess + CPU rlimit + timeout) | yes | computes arithmetic the model can't |

Future experts (deferred — see `future.md`): a disk-backed **wiki** (read in
Phase 7A, writable in 7C) that can *define* new tools from its schema; planner,
compiler, simulator as the capability layer grows.

---

## 6. Comparisons

**vs RAG** — RAG retrieves *facts* into the prompt. ToMoC retrieves *abilities*:
the router calls a capability, it does not stuff retrieved text into context.
Knowledge and computation are externalized, not inlined.

**vs MoE** — MoE gates *internal* expert sub-networks with a learned router
inside one big model. ToMoC's "experts" are *external, disk-backed tools*; the
router is a single hard route (no gating network), and the experts are not
neural. The label "Mixture of Capabilities" is metaphor, not mechanism.

**vs MCP / agent frameworks** — Those standardize *how* a model talks to tools.
ToMoC is narrower and more deterministic: a fixed call syntax, a sovereign
resolver, and a closed two-turn loop with no autonomous multi-step planning.
It is an architecture for *sovereign, reproducible, costed* routing — not a
general agent runtime.

---

## 7. Current state (snapshot)

- **Best adapters** (same 2-tool format): `v8` (360m, default — closed the
  empty-turn-2 loop, 95.7% gsm8k end-to-end), `v9` (360m + Type-E honesty),
  `v10` (360m + Type-E + Type-F show-your-work, in training).
- **Loop**: call → resolve → final answer, verified end-to-end.
- **Cost**: ~$0.12 of electricity across 37 passes (homelab, Tesla P4).
- **Open**: Phase 7 — the LLM-wiki (read/write disk-backed memory). That is the
  next *earned* capability; the router-size question ("can it be 30M?") is
  explicitly deferred until after Phase 7, per the staged progression.

---

## 8. Why this is a research direction, not just a trick

The measured progression —

```
135m base (1.74% math)
  ↓ learns WHEN to call
  ↓ tool executes
  ↓ model accepts result
  ↓ final answer matches tool
```

— proves that **reasoning can be decomposed into routing plus execution**. That
is a stronger claim than "small models can use tools." It reframes the model as
a *decision-maker over capabilities*, and makes the question *"how little model
do I actually need if I build the right system around it?"* experimentally
answerable.
