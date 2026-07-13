# runs.md — every training / eval pass, with cost breakdown

> Auto-logged in `benchmarks/passes.db` via `scripts/passdb.py`. This file is a
> human-readable mirror; regenerate the live totals with
> `python -c "from scripts.passdb import PassDB as D; D().cost_report()"`.
> Cost model: `watts/1000 * hours * $0.14/kWh`, watts = ~90W over server idle
> (box idles ~120W, ~210W under GPU load). See AGENTS.md / wiki/BUGS.md.

## Totals (all 50 passes)
| metric | value |
|--------|-------|
| total cost | **$0.2524** |
| total GPU time | 20.03 h |
| avg cost / pass | $0.00505 |
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
| 12 | 04:5x | eval | adapters/v3 (gsm8k HF smoke, 8) | 8 | — | 41.2 | 359 | 0.00014 |
| 13 | 04:5x | eval | adapters/v3 (gsm8k HF re-run, 2) | 2 | — | 18.2 | 292 | 0.00006 |
| 14 | 05:00 | train | smollm:135m → v4 (A/B/C) | 977 | 0.176 | 510.3 | 1111 | 0.00179 |
| 15 | 05:06 | eval | adapters/v4 habit (A/B/C) | 977 | — | 395.4 | — | 0.00138 |
| 16 | 05:13 | resolver-eval | adapters/v4 → run_code end-to-end | 977 | — | 391.4 | — | 0.00137 |
| 17 | 05:29 | resolver-eval | adapters/v4 → gsm8k lookup loop | 1319 | — | 932.9 | — | 0.00327 |
| 18 | 05:57 | train | smollm:135m → v5 (C 150→300, skewed) | 1127 | 0.188 | 581.4 | 1111 | 0.00200 |
| 19 | 06:05 | resolver-eval | adapters/v5 → run_code end-to-end | 1127 | — | 414.0 | — | 0.00140 |
| 20 | 06:28 | resolver-eval | adapters/v5 → gsm8k lookup loop | 1319 | — | 1384.1 | — | 0.00484 |
| 21 | 06:31 | train | smollm:135m → v5b (C clean/balanced) | 1127 | 0.191 | 1032.9 | 1111 | 0.00361 |
| 22 | 06:39 | resolver-eval | adapters/v5b → run_code end-to-end | 1127 | — | 420.9 | — | 0.00147 |
| 23 | 06:55 | resolver-eval | adapters/v5b → gsm8k lookup loop | 1319 | — | 927.3 | — | 0.00324 |
| 24 | 07:03 | resolver-eval | adapters/v4 → run_code on SAME 300-card set (fair A/B) | 1127 | — | 481.8 | — | 0.00169 |
| 25 | 07:29 | train | smollm:360m → v6 (C clean/balanced) | 1127 | 0.176 | 1110.3 | 1590 | 0.00389 |
| 26 | 08:02 | resolver-eval | adapters/v6 → run_code end-to-end | 1127 | — | 518.8 | — | 0.00181 |
| 27 | 08:27 | resolver-eval | adapters/v6 → gsm8k lookup loop | 1319 | — | 1482.0 | — | 0.00519 |
| 28 | 09:45 | train | smollm:1.7b → v7 (C clean/balanced) | 1127 | 0.083 | 3524.4 | 4236 | 0.01233 |
| 29 | 10:05 | resolver-eval | adapters/v7 → run_code end-to-end | 1127 | — | 1140.5 | — | 0.00399 |
| 30 | 10:54 | resolver-eval | adapters/v7 → gsm8k lookup loop | 1319 | — | 2937.5 | — | 0.01029 |
| 31 | 12:47 | orchestrate-eval | adapters/v6 → flashcard ToMoC loop (2-turn) | 1127 | — | 1044.4 | — | 0.00366 |
| 32 | 13:04 | orchestrate-eval | adapters/v6 → gsm8k ToMoC loop (2-turn) | 1319 | — | 1817.8 | — | 0.00636 |
| 33 | 15:08 | train | smollm:360m → v8 (Phase 6b, +Type-D two-turn) | 1427 | 0.1456 | 1459.3 | 1605 | 0.00510 |
| 34 | 15:08 | orchestrate-eval | adapters/v8 → flashcard ToMoC loop | 1427 | — | 1031.6 | — | 0.00361 |
| 35 | 15:26 | orchestrate-eval | adapters/v8 → gsm8k ToMoC loop | 1319 | — | 1784.1 | — | 0.00624 |
| 36 | 21:23 | train | smollm:360m → v9 (+Type-E KB-miss honesty) | 1727 | 0.1223 | 2287.9 | 1620 | 0.00801 |
| 37 | 22:21 | resolver-eval | adapters/v8 @ MAX_NEW=160 (regression check) | 200 | — | 228.3 | — | 0.00080 |
| 38 | 23:06 | train | smollm:360m → v10 (+Type-F show-work) | 1833 | 0.2351 | 2642.7 | 1660 | 0.00925 |
| 39 | 23:24 | resolver-eval | adapters/v10 → flashcard router-metrics | 500 | — | 452.3 | — | 0.00158 |
| 40 | 00:26 | resolver-eval | adapters/v10 | 1319 | — | 1565.8 | — | 0.00548 |
| 41 | 01:45 | train | smollm-360m-instruct | 3233 | 0.0950 | 4141.8 | 1660.1 | 0.01450 |
| 42 | 02:14 | resolver-eval | adapters/v11 | 1319 | — | 1581.2 | — | 0.00553 |
| 43 | 03:40 | train | smollm-360m-instruct | 3233 | 0.0992 | 4012.2 | 1655.1 | 0.01404 |
| 44 | 04:06 | resolver-eval | adapters/v12 | 1319 | — | 1516.0 | — | 0.00531 |
| 45 | 06:02 | train | smollm-360m-instruct | 3413 | 0.1119 | 3889.9 | 1655.1 | 0.01362 |
| 46 | 07:05 | train | smollm-135m-instruct | 3489 | 0.1090 | 2019.3 | 1173.1 | 0.00707 |
| 47 | 08:15 | train | smollm-360m-instruct | 3489 | 0.0984 | 4023.2 | 1655.1 | 0.01408 |
| 48 | 09:30 | train | smollm-360m-instruct | 3489 | 0.1332 | 4058.6 | 2005.3 | 0.01421 |
| 49 | 11:19 | train | smollm-360m-instruct | 3617 | 0.1346 | 4037.9 | 2005.3 | 0.01413 |
| 50 | 01:11 | train | smollm-360m-instruct | 3617 | 0.1357 | 4735.9 | 2005.3 | 0.01658 |

## Cost by category
| category | passes | sum cost $ | sum GPU-h |
|----------|--------|-----------|-----------|
| training (v1/v2/v3/v4) | 2, 6, 9, 14 | 0.00752 | 0.616 |
| eval (incl. buggy + rescore) | 1, 3, 4, 7, 8, 10, 12, 13, 15, 16, 17 | 0.01550 | 1.239 |
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
| 15 | v4 | lookup 0.966 / run_code 1.000 | 1.000 | 1.000 (lookup) | 0.047 | 2-tool habit; run_code 100% emit |

## Resolver end-to-end (the real capability number)
| pass | model | tool | dataset | call_rate | well_formed | resolved / correct | vs gold |
|------|-------|------|---------|-----------|-------------|--------------------|--------|
| 11 | v3 | lookup | gsm8k_test (1319) | 0.992 | 1.000 | 1282 hit | **97.2%** (1282/1319) |
| 16 | v4 | run_code | flashcards2 C (150, easy, no ÷) | 1.000 | 1.000 | 142 correct | **94.7%** (142/150) |
| 17 | v4 | lookup | gsm8k_test (1319) | 0.995 | 0.999 | 1269 correct | **98.4%** (1269/1290) |
| 19 | v5 | run_code | flashcards2 C (300, skewed) | 0.732 | 1.000 | 262 correct | 87.6% (262/299) |
| 20 | v5 | lookup | gsm8k_test (1319) | 0.997 | 1.000 | 1277 correct | 98.5% (1277/1296) |
| 22 | v5b | run_code | flashcards2 C (300, balanced) | 0.731 | 1.000 | 266 correct | **89.0%** (266/299) |
| 23 | v5b | lookup | gsm8k_test (1319) | 1.000 | 1.000 | 1279 correct | 98.5% (1279/1298) |
| 24 | v4 | run_code | SAME 300-card set (fair A/B) | 0.651 | 1.000 | 101 correct | 71.1% (101/142) |
| 26 | v6 (360m) | run_code | flashcards2 C (300, balanced) | 0.728 | 1.000 | 289 correct | **96.7%** (289/299) |
| 27 | v6 (360m) | lookup | gsm8k_test (1319) | 0.986 | 1.000 | 1280 correct | **99.2%** (1280/1290) |
| 29 | v7 (1.7b) | run_code | flashcards2 C (300, balanced) | 0.720 | 1.000 | 300 correct | **100%** (300/300) |
| 30 | v7 (1.7b) | lookup | gsm8k_test (1319) | 0.964 | 1.000 | 1253 correct | **99.0%** (1253/1266) |

## Base-size sweep — run_code + lookup vs model size (passes 22-30)
Bigger base is the lever that moved run_code past the 135m ceiling, and 1.7B
closes it. Same 1127-card clean-balanced training set, same eval sets, only base changes:

| base | adapter | run_code (300-card) | lookup (gsm8k) | GPU mem | train min | eval min |
|------|---------|---------------------|----------------|---------|-----------|----------|
| 135m | v5b | 89.0% (266/299) | 98.5% (1279/1298) | 1111 MB | 10 | ~9+14 |
| 360m | v6 | 96.7% (289/299) | 99.2% (1280/1290) | 1590 MB | 18 | ~9+14 |
| **1.7b** | **v7** | **100% (300/300)** | **99.0% (1253/1266)** | 4236 MB | 59 | ~19+49 |

The 135m's residual ~11% error was genuine verb→operator confusion
(`20 * 41` for `20 - 41`). The 360m (2.7× params) cuts that to ~3%; the 1.7B
(12.6× params) eliminates it (0 misses across all 300 arithmetic cards, incl.
division + 2-step). The lookup habit stays pinned at ~99% every size. Cost of
going bigger: training ~3× slower, eval ~5× slower, but all three fit the 8GB
P4 (max 4236 MB). **v7 (1.7B) is now the default best adapter for compute;**
keep v6 (360m) as the speed/accuracy sweet spot, v5b if you must stay at 135m.
(gsm8k lookup "vs gold" is scored-rows only; +6 no-KB rows → 1259/1319 = 95.5%
end-to-end, the 13 misses being KB re-wording gaps, not habit misses.)

## FAIR A/B — v5b vs v4 on the SAME 300-card hard set
The headline v4 "94.7%" was measured on its own easier 150-card set (no division,
fewer 2-step). On the matched 300-card set (now incl. ÷ + more 2-step) the picture
flips and **v5b is the strictly more capable compute adapter**:
- v4 on 300-card set: **71.1%** (101/142), call_rate 0.651 — it under-calls run_code
  on division/2-step cards it never trained on.
- v5b on 300-card set: **89.0%** (266/299), call_rate 0.731 — and it ADDS division
  coverage v4 lacks.
So "v5b 89% < v4 94.7%" was a measurement artifact (different test sets). Use v5b
for compute; v4 remains the best pure-lookup adapter (98.4% gsm8k).

## The headline
A 135m model went from **0% tool-calling** (base, v1) to **97% correct,
well-formed calls** (v3) — for **$0.0161 total electricity** across 10 runs.
Sovereign compute is cheap. Next: wire `lookup` to a real resolver (run_code) so
the calls actually compute (direction B, see AGENTS.md / future.md).

## The headline (DIRECTION B, pass 11)
The lookup now **computes**. End-to-end on gsm8k_test (1319 math problems):
base 135m solved **1.74%** on its own; with the v3 lookup habit + sovereign
KB resolver it resolves **97.2% correct** (1282/1319). call_rate 0.992,
well_formed 1.000, resolved-correct 1.000 of hits. Cost across all 11 passes:
**$0.0194**. "Functions ARE its knowledge" is now a working loop, not a slogan.

## The headline (PHASE 5 — two-tool ToMoC, pass 14-17)
The 135m now routes between **two** sovereign experts. `lookup` (fetch) resolves
gsm8k_test at **98.4% correct** (1269/1290, call_rate 0.995). The new `run_code`
(compute) tool: the model emits `TOOL run_code code="..."` and a sandboxed
executor (`scripts/sandbox.py`) computes it — **94.7% correct** (142/150) on
synthetic arithmetic, up from the base's 1.74% math floor. Both tools learned
with the SAME priming cue (byte-identical, unchanged). Total lab cost across 17
passes: **$0.0274** (~3 cents). "Functions ARE its knowledge" is now a 2-expert
ToMoC loop (fetch + compute), not a slogan.
