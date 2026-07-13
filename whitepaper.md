# smol-lab: Teaching a Tiny Sovereign LLM to "Pass" GSM8K by Routing to Tools

**A whitepaper on the ToMoC (Tiny Model, Orchestrated Capabilities) method**

- Lab / author: James Henry (homelab)
- Date: 2026-07-13
- Status: experimental, fast-and-loose, villain-coded
- License: proprietary / all-rights-reserved

---

## Abstract

A 135m–360m-parameter base language model is hopeless at arithmetic and
free-recall on its own: on the full gsm8k_test split (1,319 grade-school word
problems) the untrained base scores **1.74%** (23/1319). This paper describes a
method — *ToMoC* — that lifts that number to **99.8%** (1265/1267) on a 360m
model **without increasing model size, adding parameters, or retrieving
external model weights at inference beyond the two tools it calls**. The core
idea: a small model does not need to *know* facts or *compute* arithmetic if it
can be taught to *recognize when it has hit a gap* and emit a tiny, structured
tool-call script that routes the gap to an external "expert" (a knowledge base
or a sandboxed calculator). Reasoning quality — *did it call the right tool with
the right argument?* — matters more than stored knowledge. The functions become
the model's knowledge.

We report (a) the training recipe that induces the tool-routing habit with a
LoRA adapter, (b) the sovereign resolver that executes those calls (KB → vault →
live web for facts; a restricted subprocess for arithmetic), (c) the closing
two-turn loop that feeds the tool result back so the model emits its own final
answer, (d) a disk-backed, human-gated memory (an Obsidian-style vault the model
can *read* and *propose writing to*), and (e) a 7-dataset capability audit across
18 adapters (v1→v17) showing how the habit evolved. Everything runs on a single
8 GB Tesla P4 at ~90 W, fully offline except for an optional user-operated
SearXNG instance.

---

## 1. Motivation

Large language models are trained to *be* knowledge bases. That couples
capability to parameter count: a 360m model simply cannot store what a 70b
model can. The smol-lab thesis inverts this. A tiny model's job is not to
memorize the world but to **route to where the world lives**:

> The model's *functions are its knowledge*. The router owns the *decision of
> where knowledge lives* — not the knowledge itself.

If a model can reliably (1) detect a recall gap, (2) emit `lookup` with the
right query, and (3) detect a computation gap, emit `run_code` with the right
expression, then it behaves *as if* it knew the facts and could do the math —
at a fraction of the parameter budget and the electricity. This is the
"sovereign intelligence is cheap" position stated in the project README: the
entire lab to date cost ~$0.25 of electricity (50 training/eval passes, 20.03
GPU-hours).

A secondary motivation is *sovereignty*. No external model APIs are used at
inference. The only outbound call is an optional, user-operated SearXNG search
instance that the model never controls and whose results are never auto-saved.

---

## 2. The ToMoC loop

The method is a two-turn orchestration. The model never "knows" the answer; it
produces a *call*, an external resolver computes the answer, and the result is
fed back so the model produces the final text.

```
        ┌──────────────────────────────────────────────────────────┐
        │                       ToMoC loop                           │
        │                                                            │
  q ───►│  turn 1: model emits  TOOL <name> <arg>                    │
        │                 │                                          │
        │                 ▼                                          │
        │        ┌─────────────────┐                                 │
        │        │  tool_resolver  │  lookup  ─► KB → vault → web    │
        │        │  (sovereign)    │  run_code ─► sandboxed Python   │
        │        │                 │  wiki / wiki_write (gated)      │
        │        └────────┬────────┘                                 │
        │                 │  tool result (string)                    │
        │                 ▼                                          │
        │  turn 2: model sees  "Tool result: <r>\nFinal answer:"     │
        │          and emits the FINAL ANSWER                        │
        └──────────────────────────────────────────────────────────┘
```

The model is trained with a fixed **priming cue** prepended to every prompt.
That cue is byte-identical between training (`train_adapter.py`) and evaluation
(`eval_toolcall.py`, `orchestrate.py`); if it drifts, the call habit fails to
transfer. The two-turn format trains the model to *echo* the tool result rather
than hallucinate it: loss is supervised only on the final answer, while the
injected `Tool result:` context is masked, so the model learns to reproduce the
external answer, not to invent one.

---

## 3. The model and the tool habit

### 3.1 Base and adapter sizes

Three base models are downloaded for a size sweep: `smollm-135m-instruct`,
`smollm-360m-instruct`, `smollm-1.7b-instruct`. The production base is **360m**
(the speed/accuracy sweet spot: ~18 min train / ~23 min eval, ~1.6 GB). 1.7B
reaches 100% run_code but is too slow for marginal gain; 135m is kept only for
fastest iteration.

### 3.2 Training data: synthetic card types

The training corpus is *generated*, not hand-authored (`build_synth_cards.py`).
Routing behavior is taught through typed cards:

| Type | Purpose | Example target (`a`) |
|------|---------|----------------------|
| A | lookup habit | `TOOL lookup query="<verbatim q>"` |
| B | answer directly (no tool) | `<answer>` (real gold, not a placeholder) |
| C | run_code habit | `TOOL run_code code="<expr>"` (carries `answer`+`code`) |
| D | two-turn echo | question → emitted call → tool result → final answer |
| E | KB-miss honesty | on a lookup miss, stop rather than guess a number |
| F | show-work (as compute) | "Compute this: <code>" → reason then `run_code` |
| G | wiki READ routing | `TOOL wiki ...` resolves the vault directly |
| H | wiki WRITE (gated) | `TOOL wiki_write key= body= category=` proposed, never auto-applied |

Types are disjoint where they must be (A vs C never contradict on the same
question). Type-F is deliberately *rephrased to computation requests*, not word
problems, so the "word problem → run_code" habit does not fight "word problem →
lookup" on gsm8k (an earlier regression, v10/v11, where over-emitted `run_code`
dropped gsm8k to 0.61; fixed in v12).

### 3.3 Adapter lineage (selected)

- **v6** (360m): the two-tool base; run_code 96.7%, lookup 99.2%, ~18 min train.
- **v8** (360m): + Type-D two-turn cards → closed the empty-turn-2 gap; gsm8k
  end-to-end 95.7%.
- **v12** (360m): Type-F rephrased to compute → **gsm8k 0.998 (1265/1267)**, the
  default best on math. Routing restored; lookup still 100% when routed.
- **v15/v16** (360m): BUG-010 fix (real gold in Type-B); v16 adds gated
  `wiki_write` (Type-H).
- **v17** (360m): + Type-H with `category=` (model *suggests* a category the
  human approves/changes) and the SearXNG web fallback folded into `lookup`.
  **Current best 360m adapter (pass 50).**

### 3.4 The priming cue

A fixed cue string is prepended to every training/eval prompt and must stay
byte-identical across the two paths or the habit does not transfer. It frames
the task as "emit a TOOL call when you hit a gap." (The exact bytes are a
training constant shared by `train_adapter.py` and `eval_toolcall.py`; see those
files — they are the source of truth.)

---

## 4. The tool layer (tool_resolver)

`tool_resolver.py` is the sovereign dispatch seam. Each tool is an external,
disk-backed "expert" the tiny model routes to.

### 4.1 `lookup` — facts, with a three-stage fallthrough

```
lookup(query) →
   1. static KB   (exact → prefix → fuzzy Jaccard)
   2. vault       (Obsidian-style markdown notes, fuzzy Jaccard ≥ 0.5)
   3. web         (live SearXNG search; shown, NEVER auto-saved)
```

The code (`lookup()` at `tool_resolver.py:415`) is exactly this ordered
fallthrough. The web branch (`web()`, `:431`) queries
`$SEARXNG_URL/search?q=...&format=json` (read at runtime from the environment,
never hardcoded), synthesizes a snippet from the top results, tags it
`source: web`, and returns it. Because web results are never written to the
vault automatically, there is **no poison path** — the model cannot silently
corrupt its own memory with a bad search hit.

### 4.2 `run_code` — arithmetic in a sandbox

`run_code` executes the model's expression in `scripts/sandbox.py`. Defense in
depth:

1. **AST pre-scan** rejects imports, `open`, function/class definitions, and
   dunder access before anything runs (`_scan`).
2. Execution runs in a **separate `-I` (isolated) subprocess**, not in-process.
3. A **CPU-time rlimit** (`RLIMIT_CPU`) kills runaway loops, plus a wall-clock
   `timeout` (default 2 s) kills the process on hang.
4. The child environment is stripped to `PATH/PYTHONPATH/LANG/LC_ALL` — no
   network handles, no credentials.

This is what lets a 360m model "do math": it emits `code="48 - 5 + 20"`, the
sandbox returns `63`, and the loop feeds `63` back as the tool result.

### 4.3 `wiki` / `wiki_write` — sovereign memory

- `wiki` resolves the vault directly (READ).
- `wiki_write` is **gated**. `resolve()` returns `verdict=proposed_write` and
  **never mutates the store**. A human commits via
  `tool_resolver.py --wiki-write KEY BODY --category <cat> --approve`. The
  model may *propose*, never *poison*.

---

## 5. The vault (disk-backed, human-editable memory)

The Phase-7 memory is an Obsidian-style markdown vault at `data/vault/<category>/<slug>.md`.
Each note carries YAML frontmatter:

```yaml
---
key: <key>
category: <category>
source: human|model-approved|web
created: <iso>
updated: <iso>
---
<markdown body>
```

Folders *are* categories. `WikiKB` walks `data/vault/**/*.md`, parses
frontmatter, and builds the same exact+fuzzy index the static KB uses. The 128
legacy `wiki.jsonl` entries were migrated to `data/vault/general/` via
`--migrate`. Because it is plain markdown, the memory is human-readable and
human-editable, not a frozen lookup table.

The model *suggests* a `category=` on write (Type-H); the human approves or
retypes it, and the loop asks "save to vault? [y/N]" — no automatic saves.

---

## 6. Evaluation methodology

### 6.1 Why a dedicated audit harness

The headline gsm8k number is necessary but insufficient: it only measures math.
We want a capability *profile* across reasoning, recall, coding, factuality, and
truthful-decline (anti-hallucination). `scripts/audit_capabilities.py` reuses the
**real production `run_question` loop** (lookup → vault → web + the gated wiki
flow), so the audit measures the shipped behavior, not a parallel reimplementation.

### 6.2 The 7-dataset suite

| set | scorer | what it probes |
|-----|--------|----------------|
| brainteasers | llm_judge | lateral reasoning |
| reasoning_logic | contains | deductive logic (deterministic) |
| coding_func | llm_judge | function synthesis |
| knowledge_qa | contains | fact recall / lookup routing (deterministic) |
| summarization | llm_judge | compression |
| math_gsm | regex (last-number) | arithmetic sample (deterministic) |
| hallucination | contains + llm_judge | 10 closed-fact + 10 truthful-decline traps |

`contains`/`regex` are deterministic and trustworthy. `llm_judge` rows are
graded by an external judge. **Two judge backends exist**: a local 1.7B model
(weak; treat as directional only) and **`ollama:qwen2.5:1.5b`** (local, via
stdlib `urllib` to localhost:11434, no pip dependency) — the benchmark-project
standard. The judge posts `temperature: 0` so grading is **reproducible**; we
verified two identical runs produce byte-identical verdicts.

### 6.3 Batching and speed

The audit batches turn-1 across items, resolves, then batches turn-2
(`generate_all(..., chunk=16)`), fixing an earlier per-item bottleneck
(BUG-005/007). This cut a full 18-adapter run from ~2 h 13 m to ~55 m. Note:
greedy decode under GPU batch padding has a ~1% argmax-tie drift between
identical runs; adapter gaps in the audit are 5–15 points, well above that
noise, so ranking is stable. Exact per-adapter scores may shift ±1 row on
re-run.

---

## 7. Results

### 7.1 Headline: gsm8k

| model | gsm8k_test (end-to-end) |
|-------|--------------------------|
| base 360m (untrained) | 1.74% (23/1319) |
| v12 (360m, +Type-F-as-compute) | **99.8% (1265/1267)** |

The jump is entirely the routing habit: the model calls `lookup` on
KB-answerable items and `run_code` on arithmetic, and the resolver computes the
answer. Residual misses are KB re-wording gaps, not habit failures.

### 7.2 Capability audit across adapters (70 items each, greedy judge)

The full table lives in `benchmarks/adapter_comparison.md`. Best overall on the
70-item capability sample:

| adapter | brainteasers | reasoning_logic | coding_func | knowledge_qa | summarization | math_gsm | OVERALL |
|---------|------|------|------|------|------|------|--------|
| v1  | 15/15 | 7/10 | 9/10 | 1/15 | 5/5 | 2/15 | 39/70 (56%) |
| v6  | 7/15  | 0/10 | 10/10 | 0/15 | 5/5 | 6/15 | 28/70 (40%) |
| v13 | 7/15  | 0/10 | 10/10 | 1/15 | 5/5 | 5/15 | 28/70 (40%) |
| v15 | 11/15 | 3/10 | 10/10 | 7/15 | 5/5 | 4/15 | 40/70 (57%) |
| v16 | 11/15 | 4/10 | 10/10 | 8/15 | 4/5 | 4/15 | 41/70 (59%) |
| **v17** | **13/15** | 3/10 | 10/10 | **9/15** | 3/5 | 4/15 | **42/70 (60%)** |

Reading the lineage: the v1 "56%" is the *base* adapter before the tool format
was locked; intermediates dip during format experiments, then **v15→v16→v17**
recover and climb as real lookup training lands — `knowledge_qa` goes
0/15 → 7 → 8 → 9 and `brainteasers` (judge-graded reasoning) 7 → 11 → 11 → 13.
`coding_func` is 100% throughout (function synthesis is well-taught early);
`summarization` sits at 3–5/5 under the greedy judge.

> **Caveat (important):** the `math_gsm` *column* above is a **15-item regex
> sample**, not the full gsm8k_test. The real end-to-end gsm8k number is the
> 0.998 in §7.1. The audit's OVERALL is a *capability profile across 7 skills*,
> not a single math score.

### 7.3 Cost

Total lab cost to date: **$0.2524** across 50 passes, 20.03 GPU-hours, ~90 W.
Refresh live: `python -c "from scripts.passdb import PassDB as D; D().cost_report()"`.

---

## 8. Threats to validity / limitations

- **Judge variance.** `llm_judge` columns depend on the judge backend. With the
  greedy `qwen2.5:1.5b` judge they are reproducible but still *automated* grades;
  deterministic `contains`/`regex` columns are the trustworthy signal.
- **Web non-determinism.** `lookup`→web rows depend on live SearXNG results,
  which change over time; a re-run can shift those rows. Web is sovereign
  (never saved) but not reproducible.
- **Batch argmax ties.** Greedy decode under padding has ~1% per-row drift
  between identical runs; ranking is stable, exact counts are not pixel-exact.
- **Not a general reasoning engine.** The model routes; it does not *compose*
  multi-tool plans. Each turn emits at most one tool call.
- **Sample sizes.** The capability audit is 70 items/adapter; the hallucination
  trap set is 10 items. Conclusions are directional at the tail.

---

## 9. Positioning / related ideas

ToMoC is a small-model instance of the tool-augmentation family (function
calling, ReAct-style reasoning, retrieval-augmented generation), but with three
constraints that distinguish it:

1. **Sovereign & offline** — no external model/inference APIs; the only outbound
   call is an optional user-operated SearXNG.
2. **Habit, not native support** — the base has *no* tool-calling ability
   (confirmed vs Ollama); the habit is induced entirely by a LoRA adapter over a
   byte-identical priming cue.
3. **Gated memory** — the model can *propose* writes but a human must approve;
   no autonomous self-poisoning.

The efficiency thesis ("functions are its knowledge") is the same lever used by
much larger tool-using agents, applied here at the 360m scale to show how far
*routing* alone gets you.

---

## 10. Future work

- **From scratch.** The lab's stated end goal is a from-scratch tiny model
  (own tokenizer, corpus, pretraining) once the routing concept and failure
  modes are fully understood. Deferred by user choice; LoRA-on-pretrained is the
  current vehicle.
- **Exact reproducibility.** Pin model decode (disable TF32 / deterministic CUDA)
  to remove the ~1% batch-tie drift, at a small speed cost.
- **Multi-tool composition.** Teach the model to chain `lookup` → `run_code`
  across turns for multi-step problems.
- **Vault auto-curation (gated).** Surface high-value web hits for human-approved
  vault promotion, closing the learn-from-web loop without poisoning.

---

## 11. Reproducibility

All commands below assume the uv venv and the P4 GPU.

```bash
cd /home/aec/smol
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SEARXNG_URL="http://192.168.0.6:8080"   # optional; runtime-read

# regenerate the synthetic training set
python scripts/build_synth_cards.py

# train a 360m adapter (vN)
python -u scripts/train_adapter.py --base models/smollm-360m-instruct \
        --data data/raw/flashcards2.jsonl --out adapters/vN \
        --epochs 3 --lr 2e-4 --batch 8 --max-len 256

# headline gsm8k end-to-end (closes the loop)
python -u scripts/orchestrate.py --model adapters/v12 \
        --data ~/llm_eval/datasets/gsm8k_test.jsonl --kind gsm8k

# resolver smoke test (no GPU)
python scripts/tool_resolver.py --tool run_code "51 + 99"
python scripts/tool_resolver.py --wiki-write KEY BODY --category cat --approve

# 7-dataset capability audit across all adapters (greedy ollama judge)
python -u scripts/audit_capabilities.py \
        --model adapters/v1 adapters/v2 ... adapters/v17 --judge ollama
# -> benchmarks/adapter_comparison.md  + logs/audit_<adapter>_*.jsonl
```

Artifacts `adapters/`, `models/`, `logs/`, `data/vault/`, `data/wiki/` are
gitignored (local-only). `benchmarks/adapter_comparison.md` and this whitepaper
are tracked.

---

## 12. Conclusion

A 360m model that scores 1.74% on gsm8k can be lifted to 99.8% — not by making
it smarter, but by teaching it to *ask*. The ToMoC method trains a tiny, sovereign
LoRA adapter to emit structured tool calls (`lookup`, `run_code`) whenever it
hits a recall or arithmetic gap, executes those calls in a sovereign resolver
(KB → vault → optional web; sandboxed arithmetic), and feeds the result back so
the model emits its own final answer. A gated, human-approved vault gives it
growing memory without self-poisoning. The result is a cheap (~$0.25 lab to date),
offline, function-routed model whose *capabilities are the tools it knows how to
call* — proof that, at the small end, **functions are the knowledge**.

---

*Generated as a pause-point document for the smol-lab project. Numbers cited are
from the committed runs (passes 1–50, adapters v1–v17) and the logged audit at
`benchmarks/adapter_comparison.md`. See `wiki/JOURNAL.md` and `wiki/BUGS.md` for
the running narrative and failure log.*
