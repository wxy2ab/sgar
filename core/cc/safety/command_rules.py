from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class CommandClassification:
    category: str
    is_destructive: bool = False
    touches_network: bool = False
    touches_workspace: bool = False


READ_ONLY_PATTERNS = (
    "ls",
    "dir",
    "pwd",
    "cat ",
    "type ",
    "rg ",
    "find ",
    "git status",
    "git diff",
    "git log",
    "sed ",
    "awk ",
    "head ",
    "tail ",
    "wc ",
    "grep ",
    "sort ",
    "cut ",
    "uniq ",
    "diff ",
    "less ",
    "more ",
    "strings ",
    "file ",
    "stat ",
)

NETWORK_PATTERNS = (
    "curl ",
    "wget ",
    "Invoke-WebRequest".lower(),
    "Invoke-RestMethod".lower(),
    "pip install ",
    "npm install ",
)

DESTRUCTIVE_PATTERNS = (
    # Unambiguous multi-token signatures. These are safe to match anywhere as a
    # substring (incl. embedded in interpreter payloads, e.g. shutil.rmtree()),
    # because they don't collide with ordinary flags or prose. `rm` with
    # force+recursive (any flag order) is handled by ``_is_recursive_force_rm``;
    # the English words ``format``/``truncate`` are handled by
    # ``_has_destructive_command_word`` (command position only) so that
    # ``git log --format`` and ``git commit -m "truncate table"`` don't false-positive.
    "del /f /q",
    "rmdir ",
    "remove-item -recurse -force",
    "remove-item -force -recurse",
    "remove-item -r -force",
    "git reset --hard",
    "git checkout --",
    "mkfs",
    "find / -delete",
    "dd of=",
    "chmod 777",
    "rmtree(",
)

# English words that are destructive as commands but common as flags/args.
_AMBIGUOUS_DESTRUCTIVE_WORDS = ("format", "truncate")

# In-place stream editors (sed/perl/ruby with -i / --in-place) mutate files.
_INPLACE_EDIT_RE = re.compile(r"\b(?:sed|perl|ruby)\b[^;&|]*(?:\s-i\b|--in-place\b)")

# Command / process substitution can smuggle arbitrary commands past token
# classification, so it is always downgraded to ``unknown`` (ask).
_SUBSTITUTION_MARKERS = ("$(", "<(", ">(")

_POWERSHELL_READ_ONLY_COMMANDS = (
    "get-childitem",
    "gci",
    "get-content",
    "gc",
    "get-location",
    "gl",
    "get-item",
    "gi",
    "get-process",
    "gps",
    "get-service",
    "select-string",
    "where-object",
)

_POWERSHELL_WRITE_COMMANDS = (
    "set-content",
    "add-content",
    "out-file",
    "new-item",
    "move-item",
    "copy-item",
    "rename-item",
    "set-item",
    "set-itemproperty",
    "set-acl",
)

_POWERSHELL_DESTRUCTIVE_COMMANDS = (
    "remove-item",
    "ri",
    "del",
    "erase",
    "rm",
    "rmdir",
    "remove-itemproperty",
    "clear-item",
    "clear-content",
    "format-volume",
    "clear-disk",
    "initialize-disk",
    "stop-computer",
    "restart-computer",
    "stop-process",
)

WORKSPACE_WRITE_PATTERNS = (
    "mv ",
    "cp ",
    "touch ",
    "echo ",
    "tee ",
    "sed -i",
    "python ",
    "node ",
    "powershell ",
)

INTERPRETER_PREFIXES = (
    "python ",
    "python3 ",
    "py ",
    "node ",
    "bash ",
    "sh ",
    "powershell ",
    "pwsh ",
)

INTERPRETER_FLAG_PATTERNS = (
    " -c ",
    " -e ",
    " -command ",
)


def _normalize_command(command: str) -> str:
    normalized = re.sub(r"\s+", " ", command.strip()).lower()
    return normalized


def _contains_pattern(text: str, pattern: str) -> bool:
    if text.startswith(pattern):
        return True
    return f" {pattern}" in text or pattern in text


def _is_interpreter_wrapper(command: str) -> bool:
    if not any(command.startswith(prefix) for prefix in INTERPRETER_PREFIXES):
        return False
    return any(flag in command for flag in INTERPRETER_FLAG_PATTERNS)


def _has_destructive_command_word(command: str) -> bool:
    # Match ``format``/``truncate`` only at command position (start of the
    # command or whitespace-preceded), never as a flag (``--format``) or right
    # after a quote (``git commit -m "truncate table"``).
    return any(
        re.search(rf"(?:^|\s){word}\b", command) is not None
        for word in _AMBIGUOUS_DESTRUCTIVE_WORDS
    )


def _is_recursive_force_rm(command: str) -> bool:
    # ``rm`` is destructive when it carries BOTH a recursive and a force flag,
    # in any order or combination (``-rf``, ``-fr``, ``-r -f``, ``--recursive
    # --force``, ``-f --recursive``).
    for match in re.finditer(r"(?:^|[\s;&|(])rm\b([^;&|]*)", command):
        has_recursive = has_force = False
        for token in match.group(1).split():
            if token.startswith("--"):
                name = token[2:]
                if name == "recursive":
                    has_recursive = True
                elif name == "force":
                    has_force = True
            elif token.startswith("-") and len(token) > 1:
                letters = token[1:]
                if "r" in letters:
                    has_recursive = True
                if "f" in letters:
                    has_force = True
            else:
                break  # first positional argument — stop scanning flags
            if has_recursive and has_force:
                return True
    return False


def _is_inplace_edit(command: str) -> bool:
    return _INPLACE_EDIT_RE.search(command) is not None


def _has_command_name(command: str, names: tuple[str, ...]) -> bool:
    alternatives = "|".join(re.escape(name) for name in names)
    return re.search(rf"(?:^|[;|&{{(]\s*)(?:{alternatives})\b", command) is not None


def _classification_from_patterns(
    command: str,
    *,
    destructive_patterns: tuple[str, ...],
    network_patterns: tuple[str, ...],
    read_only_patterns: tuple[str, ...],
    workspace_write_patterns: tuple[str, ...],
) -> CommandClassification:
    if (
        _is_recursive_force_rm(command)
        or _has_destructive_command_word(command)
        or any(_contains_pattern(command, pattern) for pattern in destructive_patterns)
    ):
        return CommandClassification(category="destructive", is_destructive=True, touches_workspace=True)
    if any(_contains_pattern(command, pattern) for pattern in network_patterns):
        return CommandClassification(category="network", touches_network=True)
    if _is_inplace_edit(command) or any(
        _contains_pattern(command, pattern) for pattern in workspace_write_patterns
    ):
        return CommandClassification(category="workspace_write", touches_workspace=True)
    if any(command.startswith(pattern) or _contains_pattern(command, pattern) for pattern in read_only_patterns):
        return CommandClassification(category="read_only")
    return CommandClassification(category="unknown")


def classify_command(command: str, *, shell_kind: str) -> CommandClassification:
    normalized = _normalize_command(command)

    # Check command / process substitution FIRST, before base classification —
    # otherwise a read-only or workspace-write outer token (``cat $(...)``,
    # ``echo `...```) returns early and the substituted command is never seen.
    substitution_markers = _SUBSTITUTION_MARKERS
    if shell_kind != "powershell":
        substitution_markers = substitution_markers + ("`",)
    if any(marker in normalized for marker in substitution_markers):
        return CommandClassification(category="unknown", touches_workspace=True)

    if shell_kind == "powershell":
        if _has_command_name(normalized, _POWERSHELL_DESTRUCTIVE_COMMANDS):
            return CommandClassification(
                category="destructive",
                is_destructive=True,
                touches_workspace=True,
            )
        if _has_command_name(normalized, _POWERSHELL_WRITE_COMMANDS):
            return CommandClassification(category="workspace_write", touches_workspace=True)
        if _has_command_name(normalized, _POWERSHELL_READ_ONLY_COMMANDS):
            return CommandClassification(category="read_only")

    destructive_patterns = DESTRUCTIVE_PATTERNS
    workspace_write_patterns = WORKSPACE_WRITE_PATTERNS
    if shell_kind == "powershell":
        destructive_patterns = destructive_patterns + ("clear-item",)
        workspace_write_patterns = workspace_write_patterns + (
            "set-content",
            "add-content",
            "out-file",
            "set-itemproperty",
        )

    base = _classification_from_patterns(
        normalized,
        destructive_patterns=destructive_patterns,
        network_patterns=NETWORK_PATTERNS,
        read_only_patterns=READ_ONLY_PATTERNS,
        workspace_write_patterns=workspace_write_patterns,
    )
    if base.category != "unknown":
        return base

    if _is_interpreter_wrapper(normalized):
        wrapped = _classification_from_patterns(
            normalized,
            destructive_patterns=destructive_patterns,
            network_patterns=NETWORK_PATTERNS,
            read_only_patterns=(),
            workspace_write_patterns=workspace_write_patterns,
        )
        if wrapped.category != "unknown":
            return wrapped
        return CommandClassification(category="workspace_write", touches_workspace=True)

    return CommandClassification(category="unknown")
