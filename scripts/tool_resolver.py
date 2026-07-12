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


# ---- Phase 7: disk-backed, read/write wiki --------------------------------
# A second knowledge source the model can READ (lookup falls through to it
# after the frozen gsm8k/mmlu KB misses) and the human can WRITE (no model
# autonomy yet — sovereign, no poison risk). Lives at data/wiki/wiki.jsonl.
WIKI_PATH = os.path.join(ROOT, "data", "wiki", "wiki.jsonl")
WIKI_FUZZY_THRESH = 0.5  # looser than KB: wiki entries are few + human-curated


class WikiKB:
    """Editable, disk-backed wiki. {key, body, source, created, updated}."""
    _inst = None

    @classmethod
    def get(cls, path=WIKI_PATH):
        if cls._inst is None:
            cls._inst = WikiKB(path)
        return cls._inst

    def __init__(self, path=WIKI_PATH):
        self.path = path
        self.entries = []          # list of dicts (in file order)
        self.load()

    def load(self):
        self.entries = []
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.entries.append(json.loads(line))
                except Exception:
                    continue
        # index by normalized key for exact lookups
        self._idx = {norm(e.get("key", "")): e for e in self.entries}

    def resolve(self, query: str):
        """Exact key match, else fuzzy token Jaccard over keys+bodies."""
        if not query or not query.strip():
            return {"verdict": "empty", "answer": None,
                    "matched": None, "method": "wiki"}
        nq = norm(query)
        # 1. exact key
        if nq in self._idx:
            e = self._idx[nq]
            return {"verdict": "hit", "answer": e["body"],
                    "matched": e.get("key"), "method": "wiki-exact"}
        # 2. fuzzy over key tokens (and body tokens as a weaker signal)
        qt = toks(query)
        if qt:
            best, best_e, best_m = 0.0, None, None
            for e in self.entries:
                kt = toks(e.get("key", ""))
                bt = toks(e.get("body", ""))
                jk = len(qt & kt) / len(qt | kt) if kt else 0.0
                jb = len(qt & bt) / len(qt | bt) if bt else 0.0
                j = max(jk, jb * 0.9)   # body match weighted slightly lower
                if j > best:
                    best, best_e, best_m = j, e, ("wiki-key" if jk >= jb else "wiki-body")
            if best >= WIKI_FUZZY_THRESH:
                return {"verdict": "hit", "answer": best_e["body"],
                        "matched": best_e.get("key"),
                        "method": f"{best_m} jaccard={best:.2f}"}
        return {"verdict": "miss", "answer": None,
                "matched": None, "method": "wiki"}

    def write(self, key: str, body: str, source: str = "human"):
        """Upsert an entry by key. Returns ('created'|'updated', entry)."""
        nk = norm(key)
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in self.entries:
            if norm(e.get("key", "")) == nk:
                e["body"] = body
                e["source"] = source
                e["updated"] = now
                self._save()
                return "updated", e
        e = {"key": key, "body": body, "source": source,
             "created": now, "updated": now}
        self.entries.append(e)
        self._save()
        return "created", e

    def propose_write(self, key: str, body: str, source: str = "model"):
        """Phase 7 #1: build a proposed write WITHOUT mutating the store.

        Returns a dict with verdict='proposed_write' and the would-be entry.
        The caller must gate this through a human (commit_write) before it
        lands in data/wiki/wiki.jsonl. Sovereign by design: the model can
        *propose*, never *poison*.
        """
        nk = norm(key)
        now = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        exists = nk in self._idx
        action = "updated" if exists else "created"
        entry = {"key": key, "body": body, "source": source,
                 "created": now, "updated": now}
        return {"verdict": "proposed_write", "answer": body,
                "matched": key, "method": "wiki_write",
                "action": action, "exists": exists,
                "entry": entry, "needs_approval": True}

    def commit_write(self, entry: dict, source: str = "model-approved"):
        """GATED write: only call this after explicit human approval.

        Appends/updates the entry in the store. Returns ('created'|'updated', e).
        """
        return self.write(entry.get("key", ""), entry.get("body", ""),
                          source=source)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for e in self.entries:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)   # atomic


def lookup(query: str, kb: KB = None, wiki: WikiKB = None):
    """Phase 7 READ path: static KB first, then fall through to the wiki."""
    if kb is None:
        kb = KB.get()
    r = kb.resolve(query)
    if r["verdict"] == "hit":
        return r
    if wiki is None:
        wiki = WikiKB.get()
    return wiki.resolve(query)
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
        key, body = query.split("\u0001", 1)
        key, body = key.strip(), body.strip()
        if not key or not body:
            return {"verdict": "malformed_write", "answer": None,
                    "matched": None, "method": "wiki_write",
                    "error": "empty key or body"}
        return WikiKB.get().propose_write(key, body)
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
                    help="print wiki entry count and exit")
    # Phase 7 #1: model-proposed write, gated behind --approve (no silent poison)
    ap.add_argument("--wiki-write", nargs=2, metavar=("KEY", "BODY"),
                    help="PROPOSE a wiki write (key + body); requires --approve "
                         "to actually commit to data/wiki/wiki.jsonl")
    ap.add_argument("--approve", dest="approve", action="store_true",
                    help="commit a --wiki-write proposal to the store")
    args = ap.parse_args()

    if args.wiki_add or args.wiki_set:
        key, body = (args.wiki_add or args.wiki_set)
        action, e = WikiKB.get().write(key, body, source="human")
        print(json.dumps({"action": action, **e}, ensure_ascii=False))
        return

    if args.wiki_write:
        key, body = args.wiki_write
        prop = WikiKB.get().propose_write(key, body, source="model")
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

    if args.wiki_stats:
        w = WikiKB.get()
        print(f"wiki entries : {len(w.entries)}")
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
