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
    "rm -rf",
    "rm -r -f",
    "del /f /q",
    "rmdir ",
    "remove-item -recurse -force",
    "remove-item -force -recurse",
    "remove-item -r -force",
    "git reset --hard",
    "git checkout --",
    "format ",
    "mkfs",
    "find / -delete",
    "truncate ",
    "dd of=",
    "chmod 777",
    "rmtree(",
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


def _classification_from_patterns(
    command: str,
    *,
    destructive_patterns: tuple[str, ...],
    network_patterns: tuple[str, ...],
    read_only_patterns: tuple[str, ...],
    workspace_write_patterns: tuple[str, ...],
) -> CommandClassification:
    if any(_contains_pattern(command, pattern) for pattern in destructive_patterns):
        return CommandClassification(category="destructive", is_destructive=True, touches_workspace=True)
    if any(_contains_pattern(command, pattern) for pattern in network_patterns):
        return CommandClassification(category="network", touches_network=True)
    if any(_contains_pattern(command, pattern) for pattern in workspace_write_patterns):
        return CommandClassification(category="workspace_write", touches_workspace=True)
    if any(command.startswith(pattern) or _contains_pattern(command, pattern) for pattern in read_only_patterns):
        return CommandClassification(category="read_only")
    return CommandClassification(category="unknown")


def classify_command(command: str, *, shell_kind: str) -> CommandClassification:
    normalized = _normalize_command(command)
    destructive_patterns = DESTRUCTIVE_PATTERNS
    workspace_write_patterns = WORKSPACE_WRITE_PATTERNS
    if shell_kind == "powershell":
        destructive_patterns = destructive_patterns + ("clear-item", "set-itemproperty")
        workspace_write_patterns = workspace_write_patterns + ("set-content", "add-content", "out-file")

    base = _classification_from_patterns(
        normalized,
        destructive_patterns=destructive_patterns,
        network_patterns=NETWORK_PATTERNS,
        read_only_patterns=READ_ONLY_PATTERNS,
        workspace_write_patterns=workspace_write_patterns,
    )
    if base.category != "unknown":
        return base

    if "$(" in normalized or "`" in normalized:
        return CommandClassification(category="unknown", touches_workspace=True)

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
