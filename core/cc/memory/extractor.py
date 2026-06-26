from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..conversation.session import QuerySession
from .models import MemoryFact, MemoryWriteCandidate
from .policy import infer_room, trim_text
from .serializer import outline_to_details, repository_subject


_STRUCTURE_HINTS = (
    "architecture",
    "component",
    "module",
    "repository",
    "structure",
    "仓库",
    "模块",
    "架构",
    "目录",
    "结构",
)


def build_turn_write_candidates(
    *,
    session: QuerySession,
    assistant_text: str,
    max_chars: int,
) -> list[MemoryWriteCandidate]:
    prompt_context = dict(session.metadata.state.get("system_prompt_context") or {})
    outline = prompt_context.get("repository_outline")
    outline_text = str(prompt_context.get("repository_outline_text") or "")
    repo_subject = repository_subject(session.cwd)
    candidates: list[MemoryWriteCandidate] = []

    if outline_text.strip():
        entries = _parse_outline_entries(outline_text)
        facts = _outline_facts(repo_subject, entries)
        top_level_names = _top_level_outline_entries(entries)
        entrypoint_names = _entrypoint_names(entries)
        candidates.append(
            MemoryWriteCandidate(
                memory_kind="repo_structure",
                subject=repo_subject,
                summary=f"{repo_subject} repository outline",
                text=outline_text,
                details=outline_to_details(outline),
                sources=[str(Path(session.cwd).resolve())],
                room=infer_room("repo_structure"),
                facts=facts,
            )
        )
        if top_level_names:
            candidates.append(
                MemoryWriteCandidate(
                    memory_kind="module_map",
                    subject=repo_subject,
                    summary=f"{repo_subject} module map",
                    text="\n".join(f"- {name}" for name in top_level_names),
                    details={
                        "top_level_entries": top_level_names,
                        "entrypoint_candidates": entrypoint_names,
                    },
                    sources=[str(Path(session.cwd).resolve())],
                    room=infer_room("module_map"),
                    facts=_module_map_facts(repo_subject, top_level_names, entrypoint_names),
                )
            )

    normalized_assistant = trim_text(assistant_text, max_chars)
    if normalized_assistant and _looks_structural(assistant_text):
        candidates.append(
            MemoryWriteCandidate(
                memory_kind="architecture",
                subject=repo_subject,
                summary=f"{repo_subject} architecture summary",
                text=normalized_assistant,
                details={"session_id": session.session_id},
                sources=[],
                room=infer_room("architecture"),
            )
        )

    return _dedupe_candidates(candidates)


def build_compaction_candidate(
    *,
    session: QuerySession,
    compact_summary: str,
    max_chars: int,
) -> MemoryWriteCandidate | None:
    normalized = trim_text(compact_summary, max_chars)
    if not normalized:
        return None
    subject = repository_subject(session.cwd)
    return MemoryWriteCandidate(
        memory_kind="compact_summary",
        subject=subject,
        summary=f"{subject} compact summary",
        text=normalized,
        details={"session_id": session.session_id},
        sources=[],
        room=infer_room("compact_summary"),
    )


def _looks_structural(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in _STRUCTURE_HINTS)


def _outline_facts(subject: str, entries: list[dict[str, Any]]) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        predicate = "has_component" if entry["depth"] == 1 else "contains"
        fact_subject = subject if entry["depth"] == 1 else str(entry.get("parent") or subject)
        key = (fact_subject, predicate, str(entry["name"]))
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            MemoryFact(
                subject=fact_subject,
                predicate=predicate,
                object=str(entry["name"]),
                metadata={"depth": entry["depth"], "kind": entry["kind"]},
            )
        )
        if entry["kind"] == "file" and _looks_like_entrypoint(str(entry["name"])):
            entry_key = (str(entry["name"]), "entrypoint_for", subject)
            if entry_key not in seen:
                seen.add(entry_key)
                facts.append(
                    MemoryFact(
                        subject=str(entry["name"]),
                        predicate="entrypoint_for",
                        object=subject,
                        metadata={"depth": entry["depth"]},
                    )
                )
            parent = str(entry.get("parent") or "")
            if parent:
                parent_key = (parent, "has_entrypoint", str(entry["name"]))
                if parent_key not in seen:
                    seen.add(parent_key)
                    facts.append(
                        MemoryFact(
                            subject=parent,
                            predicate="has_entrypoint",
                            object=str(entry["name"]),
                            metadata={"depth": entry["depth"]},
                        )
                    )
            repo_key = (subject, "has_entrypoint", str(entry["name"]))
            if repo_key not in seen:
                seen.add(repo_key)
                facts.append(
                    MemoryFact(
                        subject=subject,
                        predicate="has_entrypoint",
                        object=str(entry["name"]),
                        metadata={"depth": entry["depth"]},
                    )
                )
        if len(facts) >= 16:
            break
    return facts


def _top_level_outline_entries(entries: list[dict[str, Any]]) -> list[str]:
    return [str(entry["name"]) for entry in entries if entry["depth"] == 1][:12]


def _module_map_facts(subject: str, top_level_names: list[str], entrypoint_names: list[str]) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    for name in top_level_names[:10]:
        predicate = "has_module" if "." not in name else "has_entry_file"
        facts.append(MemoryFact(subject=subject, predicate=predicate, object=name))
    for entrypoint in entrypoint_names[:6]:
        facts.append(MemoryFact(subject=entrypoint, predicate="entrypoint_for", object=subject))
        facts.append(MemoryFact(subject=subject, predicate="has_entrypoint", object=entrypoint))
    return facts[:16]


def _entrypoint_names(entries: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for entry in entries:
        if entry["kind"] != "file":
            continue
        name = str(entry["name"])
        if not _looks_like_entrypoint(name):
            continue
        if name in names:
            continue
        names.append(name)
        if len(names) >= 6:
            break
    return names


def _parse_outline_entries(outline_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    stack: list[str] = []
    for raw_line in outline_text.splitlines()[1:]:
        match = re.match(r"^(?P<indent>\s*)- (?P<label>.+)$", raw_line)
        if not match:
            continue
        label = match.group("label").rstrip()
        if not label or label.startswith("..."):
            continue
        depth = (len(match.group("indent")) // 2) or 1
        name = label.rstrip("/")
        kind = "directory" if label.endswith("/") else "file"
        while len(stack) >= depth:
            stack.pop()
        parent = stack[-1] if stack else None
        entries.append(
            {
                "depth": depth,
                "name": name,
                "kind": kind,
                "parent": parent,
            }
        )
        if kind == "directory":
            stack.append(name)
    return entries


def _looks_like_entrypoint(name: str) -> bool:
    lowered = name.lower()
    return lowered in {
        "__main__.py",
        "__init__.py",
        "main.py",
        "app.py",
        "api.py",
        "cli.py",
        "query_engine.py",
    }


def _dedupe_candidates(candidates: list[MemoryWriteCandidate]) -> list[MemoryWriteCandidate]:
    deduped: list[MemoryWriteCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.memory_kind, candidate.subject, candidate.summary)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
