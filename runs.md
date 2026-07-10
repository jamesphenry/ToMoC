# runs.md — every training / eval pass, with cost breakdown

> Auto-logged in `benchmarks/passes.db` via `scripts/passdb.py`. This file is a
> human-readable mirror; regenerate the live totals with
> `python -c "from scripts.passdb import PassDB as D; D().cost_report()"`.
> Cost model: `watts/1000 * hours * $0.14/kWh`, watts = ~90W over server idle
> (box idles ~120W, ~210W under GPU load). See AGENTS.md / wiki/BUGS.md.

## Totals (all 11 passes)
| metric | value |
|--------|-------|
| total cost | **$0.0194** |
| total GPU time | 1.537 h |
| avg cost / pass | $0.00176 |
| electricity rate | $0.14 / kWh |
| assumed draw | 90 W over idle |

## Per-pass detail
Sorted by pass id. `cost` is electricity only. `wall` = wall-clock seconds.

| pass | when (UTC) | type | model | cards | loss | wall (s) | GPU MB | cost $ |
|------|-----------|------|-------|-------|------|----------|--------|--------|
| 1 | 06:17 | eval | smollm:135m (base) | 60 | — | 63.2 | — | 0.00022 |
| 2 | 06:29 | train | smollm:135m → v1 | 45 | 1.247 | 47.4 | 896 | 0.00017 |
| 3 | 06:48 | eval | adapters/v1 (BUGGY) | 60 | — | 451.0 | — | 0.00158 |
| 4 | 07:04 | eval | adapters/v1 (fixed) | 60 | — | 41.5 | — | 0.00014 |
| 5 | 09:07 | benchmark-ref | smollm-135m-instruct (gsm8k) | 1319 | — | 713.6 | 795 | 0.00250 |
| 6 | 09:20 | train | smollm:135m → v2 (synthetic) | 827 | 0.110 | 699.8 | 1508 | 0.00245 |
| 7 | 09:30 | eval | adapters/v2 (synthetic) | 827 | — | 432.9 | — | 0.00152 |
| 8 | 09:55 | eval | adapters/v2 **re-score** (BUG-008) | 827 | — | 897.8 | — | 0.00314 |
| 9 | 09:56 | train | smollm:135m → v3 (capped query) | 827 | 0.149 | 889.5 | 1112 | 0.00311 |
| 10 | 10:07 | eval | adapters/v3 (capped data) | 827 | — | 356.3 | — | 0.00125 |
| 11 | 20:43 | resolver-eval | adapters/v3 → KB → score | 1319 | — | 939.3 | 573 | 0.00330 |

## Cost by category
| category | passes | sum cost $ | sum GPU-h |
|----------|--------|-----------|-----------|
| training (v1/v2/v3) | 2, 6, 9 | 0.00573 | 0.455 |
| eval (incl. buggy + rescore) | 1, 3, 4, 7, 8, 10 | 0.00825 | 0.623 |
| benchmark-ref (gsm8k base) | 5 | 0.00250 | 0.198 |

> Note: pass 8 is a *re-score* of v2 (BUG-008 parser fix) — it re-ran the eval but
> measured the same adapter. Pass 3 is the pre-fix 451 s CPU-pegged run kept as a
> monument to BUG-005.

## Key eval metrics (call-rate arc)
| pass | model | call_rate | well_formed | correct_tool | over_call | note |
|------|-------|-----------|-------------|--------------|-----------|------|
| 1 | base | 0.000 | 0.000 | 0.000 | 0.000 | no habit at all |
| 4 | v1 | 0.000 | 0.000 | 0.000 | 0.033 | LoRA trained, still 0 calls |
| 7 | v2 | 0.964 | 0.488 ⚠️ | 1.000 | 0.030 | well_formed was a measurement bug |
| 8 | v2 (re-scored) | 0.964 | 0.964 | 1.000 | 0.030 | BUG-008 fix → real number |
| 10 | v3 | 0.970 | 0.970 | 1.000 | 0.027 | capped query → model closes quote |

## The headline
A 135m model went from **0% tool-calling** (base, v1) to **97% correct,
well-formed calls** (v3) — for **$0.0161 total electricity** across 10 runs.
Sovereign compute is cheap. Next: wire `lookup` to a real resolver (run_code) so
the calls actually compute (direction B, see AGENTS.md / future.md).

## The NEW headline (DIRECTION B, pass 11)
The lookup now **computes**. End-to-end on gsm8k_test (1319 math problems):
base 135m solved **1.74%** on its own; with the v3 lookup habit + sovereign
KB resolver it resolves **97.2% correct** (1282/1319). call_rate 0.992,
well_formed 1.000, resolved-correct 1.000 of hits. Cost across all 11 passes:
**$0.0194**. "Functions ARE its knowledge" is now a working loop, not a slogan.
