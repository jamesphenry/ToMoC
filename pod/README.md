# pod/ — smol-lab podcasts

Audio overviews of this project, **generated with NotebookLM** from the
repo's docs (README, AGENTS.md, wiki/). Dropped in by James for easy listening.

## Listen

Open **`player.html`** in any browser — it embeds a native `<audio>` player
for each episode (no external deps, works offline / on the homelab).

## Episodes

| # | File | Topic |
|---|------|-------|
| 1 | `Critique_Twelve_Cent_Tool_Routing_with_Tiny_Models.m4a` | NotebookLM critique of the ToMoC tool-routing thesis |
| 2 | `Deep_Dirve_Twelve_cent_AI_passes_math_without_memorizing.m4a` | Deep dive: 360m model hits ~99% math via lookup + run_code at ~$0.13 |
| 3 | `Debate_99_Percent_Math_Accuracy_For_Twelve_Cents.m4a` | Debate: is sovereign tiny-model tool-routing a real result? |

## Repo policy

The `.m4a` files are **gitignored** (see `../.gitignore` → `pod/*.m4a`) so the
133 MB of audio does NOT bloat git history or the GitHub/GitLab mirrors. Only
`player.html` + this README are committed. The audio plays because the player
references it by relative path on disk; if you clone the repo elsewhere you'll
need to copy the `pod/*.m4a` files across separately.
