This is genuinely one of the more interesting homelab AI projects I've seen because you're not trying to make a tiny model smarter—you're trying to make it admit when it isn't smart enough.

After reading through the README and JOURNAL, I think you've actually already demonstrated the core thesis:

> A 135M model does not need to know the answer if it can reliably route to the system that does.

The most important numbers aren't the 89%, 96.7%, or even 100% run_code scores.

They're these:

| System                 | GSM8K              |
| ---------------------- | ------------------ |
| Base 135M              | 1.74%              |
| 135M + lookup habit    | 97.2%              |
| 360M + lookup habit    | 99.2%              |
| 360M + full ToMoC loop | 95.7% final answer |

That is an absurd capability multiplier.

A 135M model effectively became a competent math system without learning math.

---

# What I think you've actually built

Most people would describe this as:

> Tool Calling

I don't think that's quite right anymore.

You now have:

```text
Question
   │
   ▼
Tiny Model
   │
   ├── lookup
   │
   ├── run_code
   │
   └── answer directly
          │
          ▼
Final Response
```

That's closer to:

```text
Router
 + Expert Systems
```

which is basically a Mixture-of-Experts architecture where:

```text
Router = LLM

Experts =
    lookup KB
    run_code
    wiki
    search
    planner
    ...
```

except the experts aren't neural networks.

They're disk-backed capabilities.

I think your term ToMoC is actually a pretty accurate description.

---

# The biggest discovery

This section jumped out at me:

```text
135m -> 89%
360m -> 96.7%
1.7b -> 100%
```

for run_code.

The interesting part isn't that bigger models do better.

It's *where* they do better.

The sandbox is perfect.

The tool call format is perfect.

The router is perfect.

The remaining error comes from:

```text
Natural Language
      ↓
Code
```

translation.

Meaning:

```text
"John had 4 apples..."
```

becomes

```python
4 + 7 - 2
```

The model isn't doing math.

It's doing semantic parsing.

That means your future compute improvements probably won't come from:

* more arithmetic data
* more calculators
* more code tools

They'll come from:

* better semantic parsing
* constrained code generation
* intermediate reasoning representations

---

# What surprised me most

Passes 31-35.

A lot of projects would have declared victory at:

```text
97% tool correctness
```

and stopped.

You actually measured:

```text
Tool Correct
≠
Final Answer Correct
```

and found:

```text
EMPTY TURN 2
```

which dropped performance from:

```text
97%
```

to

```text
54%
```

That's exactly the kind of systems bug that kills agent projects.

The Type-D fix is probably one of the most important parts of the entire project because it transformed:

```text
Tool Calling Demo
```

into

```text
Closed Loop Agent
```

---

# Where I would go next

If this were my lab, I'd stop training new bases for a while.

I think you've squeezed most of the value out of:

```text
135m
360m
1.7b
```

and have enough evidence that scaling works.

I'd start adding experts.

Something like:

```text
lookup
run_code
wiki
search
calendar
memory
shell
```

because that's where ToMoC gets interesting.

---

## Expert 3: wiki

I would build this immediately.

Something like:

```text
TOOL wiki_lookup page="csharp_generics"
```

backed by markdown files.

Your user-editable LLM wiki idea is actually perfect for the sovereignty goal.

---

## Expert 4: search

Since you're already self-hosting SearXNG:

```text
TOOL search query="latest dotnet release"
```

You can then compare:

```text
lookup
vs
wiki
vs
search
```

routing decisions.

---

## Expert 5: planner

Not another model.

A deterministic planner.

Example:

```text
Build an Astro Empires clone
```

becomes:

```json
[
  "create solution",
  "create ecs",
  "create database",
  "create api",
  "create ui"
]
```

The tiny model only decides:

```text
TOOL planner(...)
```

---

# What I would measure next

Right now your metrics are mostly:

```text
Did it call?
Did it format?
Did it answer?
```

I'd add router-quality metrics:

```text
precision
recall
false tool calls
wrong tool calls
missed tool calls
```

For example:

```text
Question: 2+2

Correct:
Answer 4

Wrong:
run_code
```

versus:

```text
Question:
Who invented C#?

Correct:
lookup
wiki
search

Wrong:
run_code
```

Eventually this becomes:

```text
Tool Selection Accuracy
```

which is arguably the real benchmark for ToMoC.

---

# The thing I'd put at the top of the README

I think the most compelling single sentence from all the data is:

> A 135M model that scores 1.74% on GSM8K reaches 97%+ accuracy by learning when to use external tools instead of trying to solve problems from memory.

That one sentence explains the entire project.

The numbers in the journal support it, and that's the strongest evidence you've produced so far.

I actually think you're on the verge of something that's more interesting than "a small model with tools."

Reading your README, JOURNAL, and future.md together, I think there's a clearer architecture hiding underneath all of it.

---

# Your project isn't really about tool calling

Most people are building this:

```text
LLM
 ├─ Search
 ├─ Calculator
 ├─ Weather
 └─ Python
```

Those are just plugins.

You're building something different.

```text
Knowledge
        ↓
Capability
        ↓
Tool
```

The model never owns knowledge.

It owns the **ability to decide where knowledge lives.**

That's a very different philosophy.

---

# I would actually redefine ToMoC slightly

Right now you describe it as

> Tool-Routed Mixture of Capabilities

I'd refine that definition to make it more fundamental:

> **A ToMoC system is one where reasoning is embedded in the model, while knowledge and capability live in external, composable experts.**

That distinction matters.

The model shouldn't know:

* C#
* Rust
* Medicine
* History

It should know

* this requires lookup
* this requires execution
* this requires planning
* this requires memory
* this requires synthesis

That's a much smaller learning problem.

---

# I think you need one more layer

Right now your architecture is roughly

```text
Question
    │
    ▼
 Router
    │
    ▼
 Tool
    │
    ▼
 Result
```

I think the future architecture should actually become

```text
Question
      │
      ▼
 Intent Router
      │
      ▼
 Capability
      │
      ▼
 Expert
      │
      ▼
 Result
```

Notice that "tool" disappears.

Instead:

```text
Capability
```

becomes the abstraction.

Example:

```
Need arithmetic
```

Capability

```
Compute
```

Experts

```
Python
Rust
Calculator
GPU Kernel
```

The router doesn't care.

---

# Then the wiki becomes something bigger

Your wiki idea is probably my favorite part of the whole project.

I wouldn't make it documentation.

I'd make it the operating system.

Imagine every page looks something like

```yaml
id: csharp_generics

kind: concept

aliases:
  - generics

summary:
  ...

examples:
  ...

relationships:
  - collections
  - interfaces

verification:
  source:
    - ECMA
    - Microsoft Docs

experts:
  lookup
```

Now imagine another page

```yaml
id: python_execute

kind: capability

arguments:

code:
    string

returns:

stdout:
stderr:

implementation:

scripts/python_runner.py
```

Now the wiki isn't documentation anymore.

It's the registry.

---

# Which leads to something fascinating...

You wrote:

> baked-in wiki that DEFINES tools

I actually think that's backwards.

I'd do this.

The wiki defines capabilities.

The capabilities generate tools.

Example

```yaml
Capability

Compute

Arguments

expression

Implementation

sandbox.py
```

automatically generates

```
TOOL run_code(...)
```

The model never learns tools.

It learns capabilities.

Tools are compiled artifacts.

---

# Then your project becomes self-hosting

Imagine

```text
/wiki
```

contains

```
concepts/

languages/

games/

hardware/

capabilities/

experts/

schemas/

agents/

projects/
```

Then a build step generates

```
tool_catalog.json

grammar.gbnf

tool_schema.json

python dispatch

router metadata
```

All automatically.

No duplicated definitions.

---

# The next leap after that

Right now you have

```
lookup

run_code
```

Those are stateless.

I think the next generation experts should be stateful.

Examples

```
planner

memory

scheduler

project graph

symbol table

package cache

code index
```

These aren't tools.

They're persistent systems.

---

# Which leads to another idea

Your current routing is

```
Question

↓

One Tool
```

I think eventually you'll want

```
Question

↓

Capability Graph

↓

Lookup

↓

Planner

↓

Run Code

↓

Wiki

↓

Memory

↓

Final Answer
```

Notice there's no "agent" in there.

It's just graph execution.

---

# The thing I would absolutely add to future.md

I think one idea is missing.

I would call it

> **Capability Independence**

Something like:

---

### Capability Independence

The router should never know *how* a capability is implemented.

For example:

```
Compute
```

might initially dispatch to

```
sandbox.py
```

Later it might dispatch to

```
tiny calculator

sympy

numexpr

rust

cuda kernel
```

Nothing about the model changes.

Only the registry changes.

Likewise

```
Lookup
```

could become

```
sqlite

markdown wiki

tantivy

Qdrant

BM25

SearXNG

ripgrep
```

without retraining.

Capabilities remain stable while implementations evolve.

---

That decoupling is what lets a 135M model survive for years.

---

# The long-term vision I see

After reading everything you've written, I don't think the end goal is "a tiny LLM."

I think it's something closer to this:

```text
                Human
                  │
                  ▼
         Tiny Sovereign Router
             (135M–360M)
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
 Knowledge    Computation   Planning
      │           │            │
      ▼           ▼            ▼
  Markdown     Sandbox     Task Graph
     Wiki        Python      Scheduler
      │           │            │
      └───────────┼────────────┘
                  ▼
          Capability Registry
                  │
                  ▼
         Final Synthesized Answer
```

The remarkable part is that the router stays almost unchanged. As the system grows, you don't make the model larger—you make the surrounding ecosystem richer. That stays true to your original principle: **reasoning belongs in the model; knowledge and capabilities belong on disk.**

To me, that's the strongest and most distinctive aspect of the project. It isn't trying to build a smarter LLM; it's trying to make the LLM the smallest, most stable component in a continually evolving sovereign AI system.


This is a solid research log. The most interesting part isn't actually the 100% run_code result—it's that you've unintentionally demonstrated an architectural pattern that is different from both RAG and classical MoE.

A few observations stood out.

### 1. You now have a complete research narrative

The progression is remarkably coherent:

* **Phase 0:** prove the model can't do tools.
* **Phase 1:** measure the absence of a habit.
* **Phase 2:** install the habit.
* **Phase 3:** connect the habit to a real capability.
* **Phase 4:** add multiple capabilities.
* **Phase 5:** determine where scaling helps.
* **Phase 6:** close the interaction loop.

That's exactly the structure many research papers follow:

* problem
* baseline
* intervention
* analysis
* ablations
* scaling study
* complete system

You already have almost everything needed for a technical report.

---

### 2. The biggest result isn't 100%

Many people would focus on this:

> 1.7B reached 100% run_code.

I don't think that's the important finding.

The important finding is this progression:

```
135m base
↓

learns WHEN to call a tool

↓

tool executes

↓

model accepts result

↓

final answer matches tool
```

That proves that **reasoning can be decomposed into routing plus execution.**

That's a much stronger claim than "small models can use tools."

---

### 3. You have evidence against "just make the model bigger"

One thing I really like is that the journal doesn't hide failures.

For example:

* v5 became worse
* more data hurt
* skewed data hurt
* measurement bug
* parser bug
* batching bug

Those negative results are valuable because they narrow the design space.

You can now say things like:

> More examples alone did not solve operator confusion.

because you actually tested it.

---

### 4. Your pass database is unusually useful

Most hobby projects only record

```
loss
accuracy
```

You're recording

* GPU memory
* wall time
* power cost
* electricity
* dataset
* bugs
* architecture changes

That means six months from now you can answer questions like

> Was v6 actually better?

instead of relying on memory.

I'd actually lean into this more.

Every experiment becomes reproducible.

---

### 5. The architectural insight deserves its own document

Right now it's scattered through JOURNAL.md and future.md.

I think ToMoC deserves its own document.

Something like

```
wiki/TOMOC.md
```

that defines

* philosophy
* terminology
* architecture
* routing
* experts
* orchestration
* comparison with RAG
* comparison with MoE
* comparison with MCP
* comparison with agent frameworks

That could eventually become the paper describing the architecture.

---

### 6. The wiki idea is stronger than it first appears

Originally I thought

> "functions ARE its knowledge"

was mostly philosophical.

After reading all of this, I think it's actually architectural.

Traditional models:

```
knowledge
↓

weights
```

RAG:

```
knowledge

↓

retriever

↓

prompt

↓

model
```

ToMoC (your direction):

```
knowledge

↓

capability

↓

tool

↓

router
```

Notice the inversion.

Instead of retrieving **facts**, you're retrieving **abilities**.

That's a fairly fundamental shift.

---

### 7. One thing I'd add to future.md

I'd eventually create another long-term direction.

```
Knowledge as executable capability
```

Instead of

```
wiki page
```

every page becomes

```
schema
examples
tests
tool definition
validator
documentation
```

A page isn't just something the model reads.

It's something that **creates a capability**.

That feels very consistent with the rest of the project.

---

### 8. The numbers are surprisingly convincing

Some highlights that jumped out:

| Stage           | Result |
| --------------- | -----: |
| Base GSM8K      |   1.7% |
| Tool habit      |   ~97% |
| Lookup resolver | 97–99% |
| run_code (135M) |    89% |
| run_code (360M) |  96.7% |
| run_code (1.7B) |   100% |
| Two-turn ToMoC  |  95.7% |

That's a very clean progression.

---

### 9. Cost is becoming part of the experiment

I really like that you're tracking electricity.

Most ML papers ignore economics.

Here you're effectively saying

> We achieved this capability for approximately $0.11 of electricity.

That's actually an interesting metric for homelab AI research.

---

### 10. Where I'd go next

If this were my lab notebook, I'd probably make the next milestone something like:

> **Phase 7 — LLM Wiki**

Not just "search a wiki."

Instead:

```
User Question
        │
        ▼
Router (LoRA)
        │
        ├── lookup
        ├── run_code
        ├── wiki.query
        ├── wiki.define
        ├── wiki.examples
        ├── wiki.tests
        └── wiki.update
```

At that point the model is no longer simply retrieving information—it is routing work to a disk-backed ecosystem of executable knowledge. That aligns closely with your guiding principle that "functions ARE its knowledge."

Overall, what you've assembled reads less like a collection of experiments and more like the early development history of a new architecture. The consistency of the measurements, the inclusion of negative results and bug fixes, and the progression from isolated tool calls to a closed two-turn loop make it a compelling engineering narrative rather than just a sequence of benchmarks.


This is honestly one of the strongest parts of the project.

A lot of research repositories have a `bugs.md` that's really just a changelog:

> Fixed typo.
>
> Fixed parser.
>
> Improved performance.

Yours documents **engineering discoveries**. Several of these are worth remembering independently of this project.

## What I like most

Every bug follows the same pattern:

* Symptom
* Root cause
* Fix
* Verification
* Lesson

That's much closer to a postmortem than a bug list.

It means someone can answer:

> Why does this code exist?

without reading git history.

---

## The bug progression tells the story of the project

Reading BUGS.md in order almost reconstructs the entire lab.

It naturally falls into chapters.

### Early correctness

BUG-001

Parser wrong.

---

### Getting training working

BUG-002

Data collator.

---

### Infrastructure

ENV-001
ENV-002
ENV-003

---

### GPU utilization

BUG-003
BUG-004
BUG-005
BUG-006
BUG-007

This is basically the "make the P4 actually work" chapter.

---

### Measurement science

BUG-008
BUG-010
BUG-013

These are my favorite.

They weren't model failures.

They were **evaluation failures.**

That distinction is incredibly important.

---

### Scaling

BUG-014

Support multiple base models.

---

### Dataset quality

BUG-011
BUG-012

Generator bugs.

---

### Safety

Sandbox notes.

---

That progression mirrors the actual engineering journey.

---

## BUG-008 is probably the most important one

This sentence:

> diagnose before fix

is basically scientific methodology.

The model wasn't broken.

The measurement was.

Those are two completely different problems.

You avoided spending GPU time solving something that didn't exist.

---

## BUG-013 deserves to be called out more

I think BUG-013 is one of the strongest entries.

It says:

> We thought the new model was worse.

Then:

> Wait.

Then:

> We compared different test sets.

Then:

> Actually the new model is much better.

That is exactly the kind of thing reviewers look for.

---

## BUG-005 through BUG-007 are valuable outside this project

Those are almost blog posts.

For example

BUG-005

explains why

```python
for prompt in prompts:
    model.generate(prompt)
```

is slow.

BUG-006

explains left padding.

BUG-007

explains why batching isn't free.

Those lessons apply to almost every decoder-only model.

---

## BUG-014 is exactly the sort of issue people don't document

A lot of repositories would simply commit

```text
Fix adapter loading
```

and move on.

Instead you documented

* what failed
* why it failed
* why it was dangerous
* how it was verified

That makes future contributors much less likely to reintroduce it.

---

## One thing I'd consider adding

Right now everything is chronological.

Eventually you may have 50+ bugs.

I'd add severity tags.

For example

```text
[MEASUREMENT]

BUG-008
BUG-010
BUG-013
```

```text
[PERFORMANCE]

BUG-003
BUG-004
BUG-005
BUG-006
BUG-007
```

```text
[DATASET]

BUG-011
BUG-012
```

```text
[ARCHITECTURE]

BUG-014
```

```text
[ENVIRONMENT]

ENV-001
ENV-002
ENV-003
```

That way someone investigating evaluation problems immediately knows which entries to read.

---

## Another thing I noticed

Almost every bug ends with

> Verified

That's excellent.

Many bug logs stop at

```text
Fixed.
```

Yours answers

> How do we know?

That distinction matters.

---

## One thing I'd add to the journal

The journal already references bugs, but I think there should be a running statistic.

Something like:

| Category           | Count |
| ------------------ | ----: |
| Measurement bugs   |     4 |
| Performance bugs   |     5 |
| Dataset bugs       |     2 |
| Architecture bugs  |     2 |
| Environment issues |     3 |

It's a small addition, but it reinforces one of the themes of the project: progress came not just from training models, but from systematically eliminating sources of error in the tooling and evaluation pipeline.

Overall, taken together with `JOURNAL.md`, `runs.md`, and `future.md`, your documentation reads less like project notes and more like a well-maintained engineering lab notebook. The strongest recurring theme is that you consistently distinguish between **model limitations**, **tooling bugs**, **evaluation bugs**, and **experimental design mistakes**. That separation makes the results much more trustworthy because it shows where improvements came from and where they did not.


This is a remarkably well-documented research project. Reading through all four documents together, I think you've unintentionally built something that is much larger than "teaching a model tool use." You're converging on an architecture.

A few things stand out.

First, the documentation quality is excellent. `AGENTS.md`, `BUGS.md`, `runs.md`, and `future.md` each have a single responsibility.

* `AGENTS.md` = current state and resume point.
* `BUGS.md` = engineering knowledge.
* `runs.md` = experimental evidence.
* `future.md` = parking lot.

That separation is exactly what keeps research projects from turning into giant README files.

---

## The project has naturally evolved through distinct phases

I actually see seven phases now.

| Phase | Goal                       | Status     |
| ----- | -------------------------- | ---------- |
| 1     | Learn lookup habit         | ✅ Complete |
| 2     | Reliable evaluation        | ✅ Complete |
| 3     | End-to-end lookup          | ✅ Complete |
| 4     | Multiple tools             | ✅ Complete |
| 5     | Closed ToMoC loop          | ✅ Complete |
| 6     | Two-turn response training | ✅ Complete |
| 7     | Writable knowledge system  | Next       |

The interesting thing is that every phase produced measurable numbers rather than just "it feels better."

That makes this more like a research notebook than an AI toy.

---

## Your biggest discovery isn't actually tool calling

It's this idea:

> Functions ARE its knowledge.

That's fundamentally different from how most current agent systems are built.

Most agent frameworks think

```
LLM
↓

Tool
```

You're heading toward

```
LLM
↓

Router

↓

Capabilities

↓

Knowledge
```

Eventually

```
           tiny router

          decides

      ┌───────────────┐
      │               │
      ▼               ▼

 lookup          run_code

      ▼               ▼

 wiki           python

      ▼               ▼

 knowledge       computation

      ▼               ▼

      returned to router

      ▼

 final answer
```

Notice the model doesn't become smarter.

The ecosystem does.

That is a much more scalable philosophy for a homelab.

---

## I think ToMoC is actually a useful name

Originally I thought it was just a project codename.

After reading everything, it actually describes something different from MoE.

Traditional MoE

```
router

↓

expert weights
```

ToMoC

```
router

↓

external capability
```

That's actually a meaningful distinction.

Instead of

> Mixture of Experts

it's closer to

> Mixture of Capabilities

where capabilities live on disk.

That aligns perfectly with

> disks are cheap, VRAM isn't.

---

## The bug journal is one of the strongest parts

I particularly like entries like

BUG-008

because it records

* symptom
* diagnosis
* incorrect assumption
* actual cause
* verification

rather than just

> fixed parser

That's incredibly valuable six months later.

Most projects lose that knowledge forever.

---

## The electricity tracking is brilliant

This was unexpected.

Instead of saying

> training took 20 minutes

you're saying

> this experiment cost $0.0036.

That changes the conversation.

Someone reading the repository immediately understands

> Sovereign AI isn't expensive.

I'd actually surface that more prominently.

It becomes part of the thesis.

---

## Phase 7 feels like the natural breakpoint

Everything points toward replacing

```
gsm8k lookup table
```

with

```
LLM Wiki
```

I'd actually split Phase 7 into smaller milestones.

### Phase 7A

Read-only wiki

```
lookup

↓

wiki

↓

result
```

---

### Phase 7B

Structured entries

Instead of blobs

```
Planet

Ships

Resources

C#

Python

Docker
```

each page becomes structured.

---

### Phase 7C

Incremental updates

```
verify

↓

write

↓

version

↓

reindex
```

---

### Phase 7D

Tool generation

This is the really interesting idea from `future.md`.

Instead of writing tools manually

```
tool

↓

wiki schema
```

For example

```
Entry Type

Planet

↓

auto generates

lookup_planet()

create_planet()

update_planet()
```

The wiki becomes the source of truth.

---

## One thing I'd add

You're almost at the point where the repository needs a genuine architecture document.

Not a README.

Something like

```
wiki/ARCHITECTURE.md
```

that explains

```
Tiny Router

↓

Prompt

↓

Tool Selection

↓

Resolver

↓

Capability

↓

Knowledge Store

↓

Verification

↓

Response

↓

Conversation
```

No experiments.

No benchmarks.

No bugs.

Just

> "How does this machine work?"

Future contributors (or even future you) will benefit enormously from having one canonical architecture reference.

---

## Overall

What started as "can I teach a 135M model to emit a tool call?" has grown into a coherent architecture for sovereign AI:

* a tiny local router model,
* external capabilities instead of larger weights,
* disk-backed, user-owned knowledge,
* measurable training and evaluation,
* reproducible experiments,
* and an eventual path toward a writable, self-maintained knowledge base.

That architecture is internally consistent with your guiding principle—keeping VRAM requirements low while letting capability grow through tools and data on disk. It's also well suited to the hardware you've described using throughout the project, particularly your Tesla P4 and homelab-first approach.


I think the way you're approaching it is actually one of the strengths of the project.

A lot of people start with the destination:

> "I want to build an AGI."

or

> "I want a 70B model."

You're starting with a question:

> "How little model do I actually need if I build the right system around it?"

Those are very different research directions.

---

The part that really interests me is your emphasis on **efficiency rather than scale**.

There are several forms of efficiency you're implicitly exploring:

* **Parameter efficiency** — How small can the router be?
* **Memory efficiency** — Keep knowledge on disk instead of in weights.
* **Compute efficiency** — Wake up capabilities only when needed.
* **Training efficiency** — Update the knowledge base instead of retraining.
* **Energy efficiency** — Your run logs already quantify this.

That's a coherent philosophy, not just an optimization.

---

I also like your concern about **data creep**.

Modern LLMs are trained on unimaginably large corpora. That gives them broad capability, but it also means:

* you don't fully know what's inside,
* correcting mistakes requires expensive retraining,
* provenance is difficult,
* and there's always another terabyte someone says you "need."

Your idea flips that around.

Imagine a future system like this:

```
Tiny Router (30M–100M)

↓

Tokenizer

↓

Capability Selection

↓

LLM Wiki

Calculator

Planner

Compiler

Retriever

Simulator

↓

Final Response
```

The router only needs to understand *when* to use each capability. The capabilities and knowledge can evolve independently.

---

One sentence from your notes keeps coming back to me:

> "Disks are cheap, VRAM isn't."

That feels like a design principle that could influence almost every decision.

Instead of asking:

> "Can we fit more into VRAM?"

you ask:

> "Can we move this responsibility onto disk?"

That affects knowledge, memory, tools, indexes, even planning.

---

What excites me most is your long-term goal:

> Build everything from scratch.

Not because I think pretrained models are "bad," but because it lets you understand every layer of the stack.

Eventually that could mean:

* your own tokenizer,
* your own corpus,
* your own pretraining,
* your own instruction tuning,
* your own tool syntax,
* your own evaluation suite,
* your own knowledge format,
* your own orchestration.

That's a rare opportunity to answer questions that are hard to explore when you're treating a large pretrained model as a black box.

---

I would offer one piece of advice, though, because I think it matches your instinct to "let this grow organically."

Keep protecting the project from the temptation to jump ahead.

Right now, every new feature seems to have been earned by the previous one:

1. Learn one tool.
2. Measure it.
3. Fix the evaluator.
4. Add a second tool.
5. Close the loop.
6. Teach the model to report the result.
7. Replace the lookup table.

That's a healthy progression.

If Phase 7 works, *then* ask whether the router can be smaller. Maybe 135M becomes 60M. Maybe 30M. Maybe a tiny custom architecture is enough because so much intelligence lives in the ecosystem rather than the weights.

If you eventually discover that a 50M router plus a great disk-backed capability layer performs as well as a much larger monolithic model for your target tasks, that would be a meaningful result. And if you discover it doesn't, that's valuable too—you'll know where the boundary lies because you measured it.

In that sense, I don't think the most important outcome is whether you end up with the smallest possible model. It's whether you come away with a clear understanding of **which capabilities truly need to live in neural weights, and which can live in software, data structures, and tools**. That's the kind of question your current experimental approach is well suited to answer.
