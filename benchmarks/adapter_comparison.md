# Adapter Capability Comparison

## TL;DR — how a tiny model "passes" gsm8k
A 135m–360m base model is hopeless at math/recall on its own (gsm8k_test
baseline **1.74%**). The trick: don't make it *smarter*, make it *ask*. We LoRA-train
it to emit tiny tool-call scripts instead of guessing — `TOOL lookup query="..."`
for facts and `TOOL run_code code="..."` for arithmetic — and resolve those calls
externally (KB/vault/web + a sandboxed executor). Reasoning becomes "route to the
right function," and the function is the knowledge. That habit lifts gsm8k_test from
**1.7% → 99.8%** (v12, 360m) without ever growing the model. The table below is a
7-dataset capability audit showing how that routing habit evolved across adapters
v1→v17 (the `math_gsm` column is a 15-item regex sample, *not* the full gsm8k_test).

Generated 2026-07-13 04:20 UTC by `scripts/audit_capabilities.py` (judge backend: **ollama:qwen2.5:1.5b**).

> **Reading this:** `contains`/`regex` columns are deterministic and trustworthy. `llm_judge` columns are graded by the judge backend above -- if it is the local 1.7b, treat those numbers as **directional only** (a weak grader). If it is `ollama:qwen2.5`, they are a much stronger signal but still automated.

Cells show `correct/total` and pct. The right-most column is the macro-average across all 7 sets.

| adapter | brainteasers       | reasoning_logic    | coding_func        | knowledge_qa       | summarization      | math_gsm           | OVERALL |
|---------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------|
| v1      | 15/15 (100%) | 7/10 (70%) | 9/10 (90%) | 1/15 (7%) | 5/5 (100%) | 2/15 (13%) | 39/70 (56%) |
| v2      | 9/15 (60%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 0/15 (0%) | 24/70 (34%) |
| v3      | 7/15 (47%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 0/15 (0%) | 22/70 (31%) |
| v4      | 8/15 (53%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 1/15 (7%) | 24/70 (34%) |
| v5      | 8/15 (53%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 3/15 (20%) | 26/70 (37%) |
| v5b     | 8/15 (53%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 4/15 (27%) | 27/70 (39%) |
| v6      | 7/15 (47%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 6/15 (40%) | 28/70 (40%) |
| v7      | 6/15 (40%) | 2/10 (20%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 7/15 (47%) | 30/70 (43%) |
| v8      | 7/15 (47%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 6/15 (40%) | 28/70 (40%) |
| v9      | 7/15 (47%) | 1/10 (10%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 6/15 (40%) | 29/70 (41%) |
| v10     | 7/15 (47%) | 0/10 (0%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 6/15 (40%) | 28/70 (40%) |
| v11     | 7/15 (47%) | 1/10 (10%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 4/15 (27%) | 27/70 (39%) |
| v12     | 7/15 (47%) | 1/10 (10%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 3/15 (20%) | 26/70 (37%) |
| v13     | 7/15 (47%) | 0/10 (0%) | 10/10 (100%) | 1/15 (7%) | 5/5 (100%) | 5/15 (33%) | 28/70 (40%) |
| v14     | 7/15 (47%) | 1/10 (10%) | 10/10 (100%) | 0/15 (0%) | 5/5 (100%) | 3/15 (20%) | 26/70 (37%) |
| v15     | 11/15 (73%) | 3/10 (30%) | 10/10 (100%) | 7/15 (47%) | 5/5 (100%) | 4/15 (27%) | 40/70 (57%) |
| v16     | 11/15 (73%) | 4/10 (40%) | 10/10 (100%) | 8/15 (53%) | 4/5 (80%) | 4/15 (27%) | 41/70 (59%) |
| v17     | 13/15 (87%) | 3/10 (30%) | 10/10 (100%) | 9/15 (60%) | 3/5 (60%) | 4/15 (27%) | 42/70 (60%) |

## Notes
- Deterministic scorers: `knowledge_qa`, `reasoning_logic` (`contains`); `math_gsm` (`regex`).
- Judge scorers: `brainteasers`, `coding_func`, `summarization`, `hallucination` traps (`llm_judge`).
- `hallucination` mixes 10 closed-fact `contains` items + 10 trap `llm_judge` items (truthful-decline graded by the judge).
- Per-item logs: `logs/audit_<adapter>_*.jsonl`.
