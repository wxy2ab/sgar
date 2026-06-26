"""Git-plumbing change detection for the code-task audit.

Computes, from the working tree vs a base ref, the set of changed ``.py`` files
(classified prod vs test, new vs modified) and the *added top-level symbols*
(``def`` / ``class``) per file. Used by :mod:`core.ccx.audit.wiring` (criterion
A) and :mod:`core.ccx.audit.code_task` (self-gating + criterion B/C scoping).

All git calls go through :func:`_git`; the module never assumes a shell and
never writes to the index or working tree (read-only plumbing only) — so it is
safe to invoke even when the feature flag is off (it simply isn't called then).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field

#: The well-known empty-tree object hash (``git hash-object -t tree /dev/null``).
#: Used as the diff base when the repo has no commits yet, so the very first
#: change set is auditable instead of erroring on ``git diff HEAD``.
EMPTY_TREE_HASH = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

_DEF_RE = re.compile(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)")
_CLASS_RE = re.compile(r"^class\s+([A-Za-z_]\w*)")


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def is_inside_work_tree(cwd: str) -> bool:
    p = _git(["rev-parse", "--is-inside-work-tree"], cwd)
    return p.returncode == 0 and p.stdout.strip() == "true"


def resolve_base(cwd: str, base: str) -> str:
    """Resolve the diff base, substituting the empty tree for a commit-less repo."""
    if base == "HEAD":
        p = _git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd)
        if p.returncode != 0:
            return EMPTY_TREE_HASH
    return base


def is_test_path(path: str) -> bool:
    """Heuristic: does this path belong to the test surface (never required to be wired)?"""
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if any(seg in ("tests", "test") for seg in parts[:-1]):
        return True
    base = parts[-1]
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
    )


@dataclass
class FileChange:
    path: str
    status: str  # "A" | "M" | "R" | "D" | "untracked"
    is_py: bool
    is_test: bool
    is_new: bool
    added_symbols: list[str] = field(default_factory=list)
    syntax_error: bool = False


@dataclass
class ChangeSet:
    files: list[FileChange]
    base_ref: str

    @property
    def py(self) -> list[FileChange]:
        return [f for f in self.files if f.is_py and f.status != "D"]

    @property
    def prod_py(self) -> list[FileChange]:
        return [f for f in self.py if not f.is_test]

    @property
    def test_py(self) -> list[FileChange]:
        return [f for f in self.py if f.is_test]

    @property
    def new_prod_modules(self) -> list[FileChange]:
        """Brand-new, non-test, non-``__init__`` ``.py`` files (the wiring hard gate)."""
        return [
            f
            for f in self.prod_py
            if f.is_new and not f.path.endswith("__init__.py")
        ]


def _mk_change(cwd: str, path: str, *, status: str, is_new: bool) -> FileChange:
    is_py = path.endswith(".py")
    return FileChange(
        path=path,
        status=status,
        is_py=is_py,
        is_test=is_test_path(path) if is_py else False,
        is_new=is_new,
    )


def collect_changes(cwd: str, base: str) -> ChangeSet:
    """Collect the working-tree change set vs ``base`` (resolved)."""
    base_ref = resolve_base(cwd, base)
    files: dict[str, FileChange] = {}

    # Tracked changes vs base, NUL-delimited so paths with spaces survive.
    diff = _git(["diff", "--name-status", "-z", base_ref, "--"], cwd)
    tokens = diff.stdout.split("\0")
    i = 0
    while i < len(tokens):
        st = tokens[i]
        if not st:
            i += 1
            continue
        code = st[0]
        if code == "R":  # rename: <Rxxx> \0 <old> \0 <new>
            new = tokens[i + 2] if i + 2 < len(tokens) else ""
            path, i = new, i + 3
        else:  # <code> \0 <path>
            path, i = (tokens[i + 1] if i + 1 < len(tokens) else ""), i + 2
        if not path:
            continue
        files[path] = _mk_change(
            cwd, path, status=code, is_new=code in ("A", "R"),
        )

    # Untracked, not-ignored files = entirely new.
    others = _git(["ls-files", "-o", "--exclude-standard", "-z"], cwd)
    for path in others.stdout.split("\0"):
        if path and path not in files:
            files[path] = _mk_change(cwd, path, status="untracked", is_new=True)

    for fc in files.values():
        if not fc.is_py or fc.status == "D":
            continue
        if fc.is_new:
            fc.added_symbols, fc.syntax_error = _symbols_new_file(cwd, fc.path)
        else:
            fc.added_symbols = _added_symbols_modified(cwd, base_ref, fc.path)

    return ChangeSet(files=list(files.values()), base_ref=base_ref)


def _added_symbols_modified(cwd: str, base: str, path: str) -> list[str]:
    """Top-level ``def``/``class`` names added (``+`` hunk lines) to a modified file."""
    p = _git(["diff", "-U0", base, "--", path], cwd)
    out: list[str] = []
    for line in p.stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        content = line[1:]
        if content[:1] in (" ", "\t"):  # indented ⇒ not a top-level definition
            continue
        m = _DEF_RE.match(content) or _CLASS_RE.match(content)
        if m:
            out.append(m.group(1))
    return out


def _symbols_new_file(cwd: str, path: str) -> tuple[list[str], bool]:
    """Top-level symbols of a brand-new file via AST (regex fallback on syntax error)."""
    full = os.path.join(str(cwd), path)
    try:
        with open(full, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return [], False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        syms = []
        for line in src.splitlines():
            m = _DEF_RE.match(line) or _CLASS_RE.match(line)
            if m:
                syms.append(m.group(1))
        return syms, True
    syms = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return syms, False


__all__ = [
    "ChangeSet",
    "EMPTY_TREE_HASH",
    "FileChange",
    "collect_changes",
    "is_inside_work_tree",
    "is_test_path",
    "resolve_base",
]
