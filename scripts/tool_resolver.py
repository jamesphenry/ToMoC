#!/usr/bin/env python3
"""tool_resolver — sovereign resolver for the model's `TOOL lookup query="..."`.

DIRECTION B (see AGENTS.md / future.md). The LoRA adapter (adapters/v3) emits
a mini tool call:

    TOOL lookup query="<verbatim question>"

This module RESOLVES that call into an answer, entirely offline / homelab-only.
No external APIs. The "knowledge base" is the on-disk gsm8k / mmlu corpora we
already mined for training (models/KB-files never leave the box). The thesis:
functions ARE its knowledge — the model looks up rather than guesses.

Resolution strategy (KISS, layered so we ALWAYS return something: even a
"not found" verdict is deterministic and inspectable):
    1. exact match   : query == a KB prompt (hash/set lookup)
    2. prefix match  : KB prompt startswith/normalized-startswith query
                        (covers BUG-008 truncated queries — card `q` was cut at
                         MAX_Q=180, so the emitted query is a head of the prompt)
    3. fuzzy match   : token-set Jaccard >= FUZZY_THRESH over normalized tokens
    4. miss          : return MISS verdict (call was correct; KB just lacks it)

The resolver is tool-agnostic at the dispatch seam: `resolve(tool, query)` is
the entry point other tools (run_code, wiki) can plug into later. Today only
`lookup` is implemented; everything else resolves as UNKNOWN_TOOL.

Usage (standalone smoke test, no GPU):
    python scripts/tool_resolver.py "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    python scripts/tool_resolver.py --miss "What is the capital of France?"
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LLM_EVAL = os.path.expanduser("~/llm_eval")
DATASETS = os.path.join(LLM_EVAL, "datasets")

# KB corpora: gsm8k train+test (7473+1319) + mmlu abstract algebra.
# All use {prompt, expected, ...}. The .json files (knowledge_qa etc.) wrap
# {items:[{prompt, expected}]} and are NOT used YET (future KB expansion) —
# they don't overlap cleanly with the gsm8k-trained lookup habit, so we keep
# the resolver scoped to the math KB it was trained against for now.
KB_FILES = [
    os.path.join(DATASETS, "gsm8k_train.jsonl"),
    os.path.join(DATASETS, "gsm8k_test.jsonl"),
    os.path.join(DATASETS, "mmlu_abstract_algebra_test.jsonl"),
]

FUZZY_THRESH = 0.7  # token-set Jaccard above this counts as a hit
                     # (0.7 recovers light re-wording; verified 0 false-pos vs hits)

# ---- text normalization (deterministic, no deps) -------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Unicode punctuation the model often emits differently from the KB
# (curly quotes/dashes). Fold to ASCII so verbatim-ish queries align.
_CURLY = {
    "’": "'", "‘": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    "\u00a0": " ",   # nbsp
    "\u2028": " ", "\u2029": " ",  # line/para separators
}


def _fold_unicode(t: str) -> str:
    for k, v in _CURLY.items():
        t = t.replace(k, v)
    return t


def norm(t: str) -> str:
    """Collapse whitespace + lowercase. Fast equality/prefix key.

    Folds Unicode punctuation (curly quotes, dashes, nbsp, U+2028/2029)
    to ASCII first so a query emitted with ' matches a KB prompt with ’.
    """
    if not t:
        return ""
    return _WS.sub(" ", _fold_unicode(t).strip()).lower()


def toks(t: str) -> set:
    """Word tokens (punctuation stripped). For fuzzy Jaccard."""
    t = _PUNCT.sub(" ", t.lower())
    return set(w for w in _WS.sub(" ", t).split() if w)


# ---- KB loading ---------------------------------------------------------
def load_kb(paths=KB_FILES):
    """Return (exact_map, prefix_index, fuzzy_records).

    exact_map   : norm(prompt) -> answer string
    prefix_index: list of (norm_prompt, answer) for prefix scan
    fuzzy_records: list of (token_set, answer) for Jaccard scan

    NOTE: read line-by-line from the file OBJECT (not text.splitlines()).
    splitlines() also splits on Unicode separators like U+2028 that can
    legitimately appear INSIDE a JSON string, which would shatter a valid
    record into unparseable fragments. The file iterator only breaks on real
    newlines. Genuinely broken lines are skipped + counted (data-hygiene).
    """
    exact, prefixes, fuzz = {}, [], []
    skipped = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        recs = []
        if p.endswith(".jsonl"):
            with open(p, encoding="utf-8") as fh:
                for line in fh:               # file iterator: real newlines only
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        skipped += 1
        else:
            try:
                with open(p, encoding="utf-8") as fh:
                    obj = json.loads(fh.read())
                recs = obj.get("items", []) or []
            except Exception:
                recs = []
        for r in recs:
            q = (r.get("prompt") or r.get("question") or "").strip()
            a = r.get("expected")
            if not q:
                continue
            a = "" if a is None else str(a)
            nq = norm(q)
            if nq not in exact:          # first wins; training set overlaps test
                exact[nq] = a
            prefixes.append((nq, a))
            fuzz.append((toks(q), a))
    return exact, prefixes, fuzz


class KB:
    """Lazily-loaded singleton KB with the 3-tier resolver."""
    _inst = None

    def __init__(self):
        self.exact, self.prefixes, self.fuzzy = load_kb()

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = KB()
        return cls._inst

    def resolve(self, query: str):
        """Return dict: {verdict, answer, matched, method}."""
        if not query or not query.strip():
            return {"verdict": "empty", "answer": None,
                    "matched": None, "method": None}
        nq = norm(query)
        # 1. exact
        if nq in self.exact:
            return {"verdict": "hit", "answer": self.exact[nq],
                    "matched": nq, "method": "exact"}
        # 2. prefix (KB prompt starts with query head) — handles truncated q
        for np, a in self.prefixes:
            if np.startswith(nq) and len(nq) >= 20:
                return {"verdict": "hit", "answer": a,
                        "matched": np, "method": "prefix"}
        # 3. fuzzy (token-set Jaccard)
        qt = toks(query)
        if qt:
            best, best_a = 0.0, None
            for ft, a in self.fuzzy:
                if not ft:
                    continue
                inter = len(qt & ft)
                if not inter:
                    continue
                j = inter / len(qt | ft)
                if j > best:
                    best, best_a = j, a
            if best >= FUZZY_THRESH:
                return {"verdict": "hit", "answer": best_a,
                        "matched": f"jaccard={best:.2f}", "method": "fuzzy"}
        return {"verdict": "miss", "answer": None,
                "matched": None, "method": None}


# ---- Phase 7: disk-backed, read/write wiki (Obsidian-style markdown vault) -
# A second knowledge source the model can READ (lookup falls through to it
# after the frozen gsm8k/mmlu KB misses, and after that to a live web search)
# and the human can WRITE (sovereign, no poison risk). Lives as a vault of
# markdown notes: data/vault/<category>/<slug>.md, each with YAML frontmatter
# (key, category, source, created, updated) + a markdown body. Obsidian-friendly.
WIKI_PATH = os.path.join(ROOT, "data", "wiki", "wiki.jsonl")   # legacy (migrated)
VAULT_ROOT = os.path.join(ROOT, "data", "vault")
DEFAULT_CATEGORY = "general"
WIKI_FUZZY_THRESH = 0.5  # looser than KB: vault notes are few + human-curated


def slugify(s: str) -> str:
    """Filesystem-safe slug from a key (lowercase, hyphenated, ascii)."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:80] or "note"


def parse_frontmatter(text: str):
    """Split a markdown note into (meta:dict, body:str). Reads a leading
    YAML `---` block if present; otherwise meta={} and body=whole text."""
    text = text.lstrip("\ufeff")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            try:
                import yaml
                meta = yaml.safe_load(text[3:end]) or {}
                if not isinstance(meta, dict):
                    meta = {}
            except Exception:
                meta = {}
            return meta, text[end + 4:].lstrip("\n")
    return {}, text


def render_note(key, category, body, source, created, updated):
    """Build a markdown note string with frontmatter."""
    import yaml
    meta = {"key": key, "category": category, "source": source,
            "created": created, "updated": updated}
    fm = yaml.safe_dump(meta, sort_keys=True, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body.strip()}\n"

class WikiKB:
    """Editable, disk-backed markdown VAULT. Notes live under
    data/vault/<category>/<slug>.md. Same resolve() contract as before
    (exact key + fuzzy token Jaccard) so the rest of the stack is unchanged.
    """

    _inst = None

    @classmethod
    def get(cls, root=VAULT_ROOT):
        if cls._inst is None:
            cls._inst = WikiKB(root)
        return cls._inst

    def __init__(self, root=VAULT_ROOT):
        self.root = root
        self.entries = []          # list of dicts (key, body, source, category,
                                   # created, updated, path)
        self.load()

    def load(self):
        self.entries = []
        if not os.path.isdir(self.root):
            self._idx = {}
            return
        for dirpath, _, files in os.walk(self.root):
            for fn in files:
                if not fn.endswith(".md"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, encoding="utf-8") as fh:
                        meta, body = parse_frontmatter(fh.read())
                except Exception:
                    continue
                cat = meta.get("category", DEFAULT_CATEGORY)
                key = meta.get("key", fn[:-3])
                self.entries.append({
                    "key": key, "body": body.strip(),
                    "source": meta.get("source", "unknown"),
                    "category": cat,
                    "created": meta.get("created"),
                    "updated": meta.get("updated"),
                    "path": full,
                })
        self._idx = {norm(e["key"]): e for e in self.entries}

    def resolve(self, query: str):
        """Exact key match, else fuzzy token Jaccard over keys+bodies."""
        if not query or not query.strip():
            return {"verdict": "empty", "answer": None,
                    "matched": None, "method": "wiki"}
        nq = norm(query)
        if nq in self._idx:
            e = self._idx[nq]
            return {"verdict": "hit", "answer": e["body"],
                    "matched": e["key"], "method": "wiki-exact",
                    "category": e.get("category"), "path": e.get("path")}
        qt = toks(query)
        if qt:
            best, best_e, best_m = 0.0, None, None
            for e in self.entries:
                kt = toks(e["key"])
                bt = toks(e["body"])
                jk = len(qt & kt) / len(qt | kt) if kt else 0.0
                jb = len(qt & bt) / len(qt | bt) if bt else 0.0
                j = max(jk, jb * 0.9)
                if j > best:
                    best, best_e, best_m = j, e, ("wiki-key" if jk >= jb
                                                   else "wiki-body")
            if best >= WIKI_FUZZY_THRESH:
                return {"verdict": "hit", "answer": best_e["body"],
                        "matched": best_e["key"],
                        "method": f"{best_m} jaccard={best:.2f}",
                        "category": best_e.get("category"),
                        "path": best_e.get("path")}
        return {"verdict": "miss", "answer": None,
                "matched": None, "method": "wiki"}

    def write(self, key: str, body: str, source: str = "human",
              category: str = DEFAULT_CATEGORY):
        """Upsert a note. Returns ('created'|'updated', entry)."""
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nk = norm(key)
        existing = self._idx.get(nk)
        if existing is not None:
            entry = existing
            entry["body"] = body
            entry["source"] = source
            entry["category"] = category or entry.get("category", DEFAULT_CATEGORY)
            entry["updated"] = now
            action = "updated"
        else:
            entry = {"key": key, "body": body, "source": source,
                     "category": category or DEFAULT_CATEGORY,
                     "created": now, "updated": now, "path": None}
            self.entries.append(entry)
            action = "created"
        self._persist(entry)
        self._idx[norm(entry["key"])] = entry
        return action, entry

    def _persist(self, entry):
        """Write a single note file under data/vault/<category>/<slug>.md."""
        cat = (entry.get("category") or DEFAULT_CATEGORY).strip() or DEFAULT_CATEGORY
        cat = re.sub(r"[^a-zA-Z0-9 _-]", "", cat).strip() or DEFAULT_CATEGORY
        d = os.path.join(self.root, cat)
        os.makedirs(d, exist_ok=True)
        slug = slugify(entry["key"])
        path = os.path.join(d, slug + ".md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_note(
                entry["key"], cat, entry["body"], entry["source"],
                entry.get("created") or entry["updated"],
                entry["updated"]))
        entry["path"] = path

    def propose_write(self, key: str, body: str, source: str = "model",
                      category: str = DEFAULT_CATEGORY):
        """Phase 7 #1: build a proposed write WITHOUT mutating the store.

        `category` is the MODEL'S SUGGESTION (shown to the human for
        approval/change). Returns verdict='proposed_write'. Sovereign: the
        model can *propose*, never *poison*.
        """
        nk = norm(key)
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        exists = nk in self._idx
        action = "updated" if exists else "created"
        entry = {"key": key, "body": body, "source": source,
                 "category": category or DEFAULT_CATEGORY,
                 "created": now, "updated": now}
        return {"verdict": "proposed_write", "answer": body,
                "matched": key, "method": "wiki_write",
                "action": action, "exists": exists,
                "category": category or DEFAULT_CATEGORY,
                "entry": entry, "needs_approval": True}

    def commit_write(self, entry: dict, source: str = "model-approved"):
        """GATED write: only call after explicit human approval."""
        return self.write(entry.get("key", ""), entry.get("body", ""),
                          source=source,
                          category=entry.get("category", DEFAULT_CATEGORY))

    # ---- legacy jsonl bridge (used by the one-time migration only) ----
    @staticmethod
    def migrate_jsonl(jsonl_path=WIKI_PATH, root=VAULT_ROOT):
        """Convert a legacy wiki.jsonl into the markdown vault. Idempotent:
        skips keys that already exist in the vault."""
        if not os.path.exists(jsonl_path):
            return 0
        created = 0
        vault = WikiKB(root)
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                key = e.get("key", "")
                if not key:
                    continue
                if norm(key) in vault._idx:
                    continue
                vault.write(key, e.get("body", ""),
                            source=e.get("source", "migrated"),
                            category=DEFAULT_CATEGORY)
                created += 1
        return created


def lookup(query: str, kb: KB = None, wiki: WikiKB = None):
    """Phase 7 READ path: static KB -> vault -> live web (SearXNG)."""
    if kb is None:
        kb = KB.get()
    r = kb.resolve(query)
    if r["verdict"] == "hit":
        return r
    if wiki is None:
        wiki = WikiKB.get()
    r = wiki.resolve(query)
    if r["verdict"] == "hit":
        return r
    # final fallback: live web search (never auto-saved -> no poison)
    return web(query)


def web(query: str, max_results: int = 3):
    """Phase 7 web fallback: query a SearXNG JSON endpoint (env SEARXNG_URL).

    Returns a 'hit' whose answer is a synthesized snippet from the top
    results. Sovereign: web answers are shown but NEVER written to the vault
    automatically — the human may save them via the gated wiki_write flow.
    """
    import urllib.parse
    import urllib.request
    base = os.environ.get("SEARXNG_URL", "").rstrip("/")
    if not base:
        return {"verdict": "no_web", "answer": None, "matched": None,
                "method": "web", "error": "SEARXNG_URL not set"}
    url = f"{base}/search?q={urllib.parse.quote(query)}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smol-tomoc/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"verdict": "web_error", "answer": None, "matched": None,
                "method": "web", "error": str(e)}
    results = data.get("results", [])[:max_results]
    if not results:
        return {"verdict": "miss", "answer": None, "matched": None,
                "method": "web"}
    snippets = []
    for r in results:
        title = r.get("title", "").strip()
        content = re.sub(r"\s+", " ", r.get("content", "")).strip()
        if content:
            snippets.append(f"- {title}: {content}")
    answer = "\n".join(snippets) if snippets else "(web returned no snippets)"
    return {"verdict": "hit", "answer": answer, "matched": query,
            "method": "web", "source": "web", "urls":
            [r.get("url", "") for r in results]}
def resolve(tool: str, query: str, kb: KB = None):
    """Entry point. tool-agnostic: route to the right backend by name.

    - `lookup`   -> KB resolver (exact -> prefix -> fuzzy -> miss)
    - `run_code` -> sandboxed Python exec (math/expression answers)
    UNKNOWN_TOOL returned otherwise.

    This is the ToMoC dispatch seam: each tool is an external, disk-backed
    "expert" the 135m routes to. Adding tools = adding branches here.
    """
    if kb is None:
        kb = KB.get()
    if tool == "lookup":
        return lookup(query, kb=kb)          # Phase 7: KB -> wiki fallthrough
    if tool == "wiki":
        return WikiKB.get().resolve(query)
    if tool == "wiki_write":
        # Phase 7 #1 (FLAG-TO-DATASET): the model proposes a wiki edit. This is
        # GATED — resolve() NEVER mutates the store. It returns a proposed write
        # (verdict="proposed_write") that a human must approve via
        # WikiKB.commit_write() (CLI --wiki-write --approve). Sovereign: no
        # silent self-poisoning of the knowledge store.
        if query is None or "\u0001" not in query:
            return {"verdict": "malformed_write", "answer": None,
                    "matched": None, "method": "wiki_write",
                    "error": "wiki_write needs key + body"}
        parts = query.split("\u0001", 2)
        key, body = parts[0].strip(), parts[1].strip()
        category = parts[2].strip() if len(parts) > 2 else ""
        if not key or not body:
            return {"verdict": "malformed_write", "answer": None,
                    "matched": None, "method": "wiki_write",
                    "error": "empty key or body"}
        return WikiKB.get().propose_write(key, body, category=category)
    if tool == "run_code":
        # Guard: a run_code call with empty/None/non-string code must not reach
        # ast.parse (compile(None) crashes). Model sometimes emits
        # `TOOL run_code code=""` (truncated/empty) -> treat as a miss, not a crash.
        if not isinstance(query, str) or not query.strip():
            return {"verdict": "miss", "answer": None,
                    "matched": None, "method": "run_code",
                    "error": "empty code argument"}
        try:
            from sandbox import run as _run
        except Exception as e:
            return {"verdict": "error", "answer": None,
                    "matched": None, "method": "run_code",
                    "error": f"sandbox unavailable: {e}"}
        r = _run(query)
        if r["ok"]:
            return {"verdict": "hit", "answer": str(r["value"]),
                    "matched": "run_code", "method": "run_code",
                    "stdout": r["stdout"]}
        return {"verdict": "error", "answer": None,
                "matched": None, "method": "run_code",
                "error": r["error"]}
    return {"verdict": "unknown_tool", "answer": None,
            "matched": tool, "method": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="query text to resolve")
    ap.add_argument("--tool", default="lookup",
                    help="tool backend to route to (lookup | wiki | run_code)")
    ap.add_argument("--miss", dest="force_miss", action="store_true",
                    help="also show a guaranteed-miss example")
    ap.add_argument("--stats", action="store_true",
                    help="print KB size stats and exit")
    # Phase 7 wiki write path (human-in-the-loop, sovereign)
    ap.add_argument("--wiki-add", nargs=2, metavar=("KEY", "BODY"),
                    help="add a wiki entry (key + body); upserts if key exists")
    ap.add_argument("--wiki-set", nargs=2, metavar=("KEY", "BODY"),
                    help="alias for --wiki-add (explicit upsert)")
    ap.add_argument("--wiki-stats", action="store_true",
                    help="print vault note count and exit")
    # Phase 7 #1: model-proposed write, gated behind --approve (no silent poison)
    ap.add_argument("--wiki-write", nargs=2, metavar=("KEY", "BODY"),
                    help="PROPOSE a wiki write (key + body); requires --approve "
                         "to actually commit to the vault")
    ap.add_argument("--category", default=DEFAULT_CATEGORY,
                    help="category folder for --wiki-write/--wiki-add "
                         f"(default: {DEFAULT_CATEGORY})")
    ap.add_argument("--approve", dest="approve", action="store_true",
                    help="commit a --wiki-write proposal to the vault")
    # Phase 7 web fallback (SearXNG)
    ap.add_argument("--web", metavar="QUERY",
                    help="query the live SearXNG web search (SEARXNG_URL env)")
    ap.add_argument("--web-stats", action="store_true",
                    help="print whether SEARXNG_URL is configured")
    ap.add_argument("--migrate", action="store_true",
                    help="one-time: convert legacy data/wiki/wiki.jsonl -> vault")
    args = ap.parse_args()

    if args.migrate:
        n = WikiKB.migrate_jsonl()
        print(json.dumps({"migrated": n, "vault": VAULT_ROOT},
                         ensure_ascii=False))
        return

    if args.wiki_add or args.wiki_set:
        key, body = (args.wiki_add or args.wiki_set)
        action, e = WikiKB.get().write(key, body, source="human",
                                       category=args.category)
        print(json.dumps({"action": action, **e}, ensure_ascii=False))
        return

    if args.wiki_write:
        key, body = args.wiki_write
        prop = WikiKB.get().propose_write(key, body, source="model",
                                          category=args.category)
        if not args.approve:
            # gate: show the proposal, do NOT mutate
            print(json.dumps({**prop,
                              "committed": False,
                              "note": "add --approve to commit this write"},
                             ensure_ascii=False))
            return
        action, e = WikiKB.get().commit_write(prop["entry"], source="model-approved")
        print(json.dumps({"action": action, "committed": True, **e},
                         ensure_ascii=False))
        return

    if args.web:
        r = web(args.web)
        print(json.dumps({"tool": "web", "query": args.web, **r},
                         ensure_ascii=False))
        return

    if args.web_stats:
        print(f"SEARXNG_URL : {os.environ.get('SEARXNG_URL', '(not set)')}")
        return

    if args.wiki_stats:
        w = WikiKB.get()
        cats = {}
        for e in w.entries:
            cats[e.get("category", DEFAULT_CATEGORY)] = \
                cats.get(e.get("category", DEFAULT_CATEGORY), 0) + 1
        print(f"vault notes : {len(w.entries)}")
        print(f"categories  : {cats}")
        return

    if args.stats:
        kb = KB.get()
        print(f"KB exact entries : {len(kb.exact)}")
        print(f"KB prefix index : {len(kb.prefixes)}")
        print(f"KB fuzzy records: {len(kb.fuzzy)}")
        return

    if not args.query and not args.force_miss:
        ap.print_help()
        return

    if args.query:
        r = resolve(args.tool, args.query)
        print(json.dumps({"tool": args.tool, "query": args.query, **r},
                         ensure_ascii=False))

    if args.force_miss:
        r = resolve("lookup", "What is the capital of France?")
        print(json.dumps({"tool": "lookup",
                          "query": "What is the capital of France?", **r},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
