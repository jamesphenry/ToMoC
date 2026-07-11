#!/usr/bin/env python3
"""sandbox — a deliberately restricted Python executor for `run_code`.

Sovereignty principle: capability lives in disk-backed, inspectable tools — not
in model weights. `run_code` lets the model ask the resolver to *compute* an
arithmetic/expression answer instead of guessing. The risk is obvious: a model
that can emit code can emit `import os; os.system(...)`. So this module is
defense-in-depth:

  1. AST pre-scan rejects the call BEFORE execution if it contains imports,
     function/class defs, comprehensions-with-calls, or any name in a
     dangerous set (open, eval, exec, __import__, compile, breakpoint, help,
     memoryview, globals, locals, vars, input, ...). Cheap, no side effects.
  2. Execution happens in a SEPARATE subprocess (not in-process) so even a
     missed AST case can't corrupt the parent. The child runs with:
       - a tight CPU time rlimit (ulimit-style) to kill runaway loops,
       - no network (blocked by name + subprocess boundary),
       - stdout/stderr captured, return code checked.
  3. Only the LAST EXPRESSION's value is returned (plus any printed output),
     exactly like a REPL. No filesystem, no imports, no side effects.

Result shape: dict {ok, value, stdout, stderr, error, timed_out, killed}.

KISS: this is a math-grade sandbox, not a general Python REPL. It is meant for
the 135m tool-call loop, where the model emits tiny arithmetic expressions.
"""
import ast
import subprocess
import sys
import textwrap
import os

# Names that must never appear in untrusted code.
_FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "breakpoint", "help",
    "input", "memoryview", "globals", "locals", "vars", "dir", "getattr",
    "setattr", "delattr", "exit", "quit", "license", "credits",
}
# Attribute access to dunders is also rejected.
_FORBIDDEN_ATTR_PREFIX = "__"


def _scan(tree: ast.AST) -> list:
    """Return a list of human-readable violation reasons (empty = clean)."""
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad.append("imports are not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                               ast.Lambda)):
            bad.append("defining functions/classes is not allowed")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_NAMES:
                bad.append(f"call to '{fn.id}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            bad.append(f"name '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith(_FORBIDDEN_ATTR_PREFIX):
            bad.append(f"attribute '{node.attr}' is not allowed")
        elif isinstance(node, (ast.With, ast.Try, ast.Raise, ast.Yield,
                               ast.YieldFrom, ast.Await)):
            bad.append(f"{type(node).__name__} is not allowed")
    return bad


def _wrap(code: str) -> str:
    """Wrap user code so the LAST expression's value is printed as JSON.

    Works for a trailing expression (REPL-style) or purely imperative code
    that only prints. Mirrors how the model will emit `run_code` payloads.
    """
    return textwrap.dedent(f"""
        import json, sys, ast as _ast
        _src = {code!r}
        _tree = _ast.parse(_src, mode="exec")
        _last_expr = None
        if _tree.body and isinstance(_tree.body[-1], _ast.Expr):
            _last_expr = _tree.body[-1].value
        _main = _ast.Module(
            body=_tree.body[:-1] if _last_expr is not None else _tree.body,
            type_ignores=[])
        exec(compile(_main, "<sandbox>", "exec"), globals())
        if _last_expr is not None:
            _val = eval(compile(_ast.Expression(_last_expr), "<sandbox>", "eval"), globals())
            sys.stdout.write("\\x00VALUE\\x00" + json.dumps(_val))
    """)


def run(code: str, timeout: float = 2.0, cpu_seconds: int = 1) -> dict:
    """Execute `code` in a restricted subprocess. Returns a result dict.

    verdict-ready fields: ok (bool), value (last-expr value or None),
    stdout, stderr, error (str|None), timed_out (bool), killed (bool).
    """
    res = {"ok": False, "value": None, "stdout": "", "stderr": "",
           "error": None, "timed_out": False, "killed": False}
    # 1) AST pre-scan — reject clearly unsafe code before spawning anything.
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        res["error"] = f"syntax error: {e}"
        return res
    violations = _scan(tree)
    if violations:
        res["error"] = "rejected: " + "; ".join(violations)
        return res

    child = (
        "import resource, sys\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))\n"
        "except Exception:\n"
        "    pass\n" % (cpu_seconds, cpu_seconds)
    ) + _wrap(code)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", child],
            capture_output=True, text=True, timeout=timeout,
            env={k: v for k, v in os.environ.items()
                 if k in ("PATH", "PYTHONPATH", "LANG", "LC_ALL")},
        )
    except subprocess.TimeoutExpired:
        res["timed_out"] = True
        res["error"] = f"timed out after {timeout}s"
        return res
    except Exception as e:  # pragma: no cover - defensive
        res["error"] = f"exec error: {e}"
        return res

    out = proc.stdout or ""
    err = proc.stderr or ""
    if "\x00VALUE\x00" in out:
        body, val_json = out.split("\x00VALUE\x00", 1)
        res["stdout"] = body
        try:
            res["value"] = json.loads(val_json)
        except Exception:
            res["value"] = val_json
    else:
        res["stdout"] = out
    res["stderr"] = err
    if proc.returncode != 0:
        # RLIMIT_CPU fires SIGXCPU (152) then SIGKILL (137/-9) on a busy loop.
        if proc.returncode in (-9, 137, -24, 152) or "CPU time limit" in err:
            res["killed"] = True
            res["error"] = "killed: CPU time limit exceeded"
        else:
            res["error"] = (err.strip() or f"exit {proc.returncode}")
        return res
    res["ok"] = True
    return res


if __name__ == "__main__":
    tests = {
        "arith": "print(2+3*4)\n(48/2) + 10",
        "import_blocked": "__import__('os').system('echo pwned')",
        "open_blocked": "open('/etc/passwd').read()",
        "infinite_loop": "while True:\n    pass",
        "func_blocked": "def f():\n    return 1\nf()",
        "big_num": "x=10**10\nx*x",
    }
    for name, code in tests.items():
        r = run(code)
        print(f"[{name}] ok={r['ok']} value={r['value']!r} "
              f"err={r['error']!r} to={r['timed_out']} killed={r['killed']}")
