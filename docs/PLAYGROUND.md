# Playground — `scripts/orchestrate.py --chat`

An interactive, sovereign ToMoC loop you can actually talk to. No Ollama, no
external APIs — it drives the same 360m `v8` LoRA + HF engine + local KB that
the batch scorers use, just one question at a time, with the live "watch it
think" view.

The model (q) → emits a `TOOL` call → the resolver computes/runs it → the
result is fed back → the model emits a final answer. `--chat` shows every step.

## Run it

```bash
cd /home/aec/smol
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/orchestrate.py --chat
```

> `--model` defaults to `adapters/v8` (the closed-loop production adapter).
> Override with `--model adapters/v6` etc. if you want to compare.

## What you see

For each question the playground shows the model's *reasoning* (its tool-call
choice) and the tool's result as a thinking step, then the final answer as a
clean reply:

```
you> A baker made 3 trays of 12 cookies each. How many cookies?
  reasoning:
    • model called: TOOL run_code code="3 * 12"
    • tool returned: 36

  answer: 36
```

- **reasoning** — the model's actual thinking. It decided to call `run_code`
  and wrote the exact expression (`3 * 12`). For a lookup question this shows
  `TOOL lookup query="..."`; if the model answers directly it says so.
- **tool returned** — what the resolver/sandbox produced (`hit` → the value;
  otherwise `(no answer found in the knowledge base)`).
- **answer** — the model's final reply, echoing the tool result when it's right.

> Note: this is a *surface* of the reasoning the model already does. The 360m
> adapter was trained to emit the tool call + echo the result; it does not
> narrate a natural-language chain-of-thought. (Training real CoT is a separate,
> later step — see the share-and-review loop / future phases.)

## Commands (inside the chat)

| command | what it does |
|---------|--------------|
| `quit` / `exit` / `q` / `/quit` / `/exit` / `/bye` | leave; auto-saves the session (see below). Slash form added so you don't need ctrl-c. |
| `/export [path]` | write the whole conversation as markdown. Defaults to `logs/chat_export.md`. Pass a path to choose your own (e.g. `/export /tmp/session1.md`). |
| `/mark <n> <seen|fixed>` | flip turn `n`'s review status tag. Used for the share-and-review handoff. |

Empty input is ignored. Anything else is treated as a question.

## Exporting a conversation for review

`/export` writes a markdown file, one `## Turn N` block per exchange:

```markdown
# smol ToMoC conversation

- model: `.../adapters/v8`
- exported: 2026-07-11 20:06 UTC
- turns: 2

## Turn 1  _[status: new]_

**You:** A baker made 3 trays of 12 cookies each. How many cookies?

**Model (turn-1 call):** `TOOL run_code code="3 * 12"`
**Tool:** hit → `36`
**Final answer:** 36

## Turn 2  _[status: new]_

**You:** 48 - 5 + 20
...
```

Each turn carries a `_[status: new]_` tag. Use `/mark 1 fixed` (or just edit
the file by hand) to flip it to `seen` or `fixed`. The footer tells the
reviewer to do exactly that.

**Share-and-review loop:**

1. Chat → `/export /tmp/session1.md`
2. Hand the file to the assistant (paste it back, or point at the path).
3. The assistant marks turns `seen` / `fixed` and tells you what to correct.
4. Next session, repeat.

On `quit`, the session is **auto-saved** to `logs/chat_last.md` so nothing is
lost even if you forget `/export`.

## How it works (for the curious)

`--chat` reuses the exact same code path as the batch scorer:

- `run_question(engine, kb, q, gold=None, verbose=True)` — the two-turn loop:
  turn 1 emit the call, resolve via `tool_resolver.resolve()`, turn 2 feed the
  result back and emit the final answer.
- `KB.get()` loads the sovereign lookup table; `run_code` queries go through
  `scripts/sandbox.py` (restricted executor).
- The transcript is just an in-memory list of dicts, rendered to markdown by
  `render_md()` when you `/export` or quit.

The playground is a thin REPL over the verified orchestration loop — it adds
no model behavior, it only makes the existing ToMoC loop talkable.

## Known limitation: KB-miss recovery

When a question is **not in the knowledge base**, the resolver can't answer and
`run_question` feeds back the literal string *"No answer found in the knowledge
base."* The 360m adapter was trained to **echo** the tool result, so facing a
non-numeric string it either echoes that text or guesses from its weak weights
— which can produce a wrong number (and thus look like the "wrong operation").

Measured on the gsm8k eval logs: lookup misses dominate (~370 rows), and among
tool-call misses the chosen operator is **not** skewed to multiplication
(addition 320×, multiplication only 196× across all calls; multiplication just 6
of the misses). So there is **no** "prefers to multiply" bias — the symptom is
really *poor recovery on a KB miss*, not an arithmetic preference.

**Not yet fixed** (would need either a retrain on miss-recovery cards, or a
graceful-miss prompt that tells the model to answer from its own reasoning).
Tracked as a future improvement; see the share-and-review loop above.

## Related

- Batch end-to-end scoring: `python -u scripts/orchestrate.py --data <jsonl> --kind gsm8k`
- Single live question (no REPL): `python -u scripts/orchestrate.py --ask "48 - 5 + 20"`
- Build/loop background: [wiki/JOURNAL.md](../wiki/JOURNAL.md) (Phase 6 / 6b).
