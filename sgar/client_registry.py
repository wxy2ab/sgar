from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CORE_LLM_DIR = ROOT_DIR / "core" / "llms"
KNOWN_LLM_BASES = {
    "LLMApiClient",
    "MoonShotClient",
    "OpenAIClient",
    "SimpleDeepSeekClient",
}
MODEL_ATTR_NAMES = {"MODEL_CONFIG_KEYS"}


@dataclass(frozen=True)
class ClientMetadata:
    client_name: str
    source_file: str
    credential_keys: tuple[str, ...]
    model_keys: tuple[str, ...]

    @property
    def one_liner(self) -> str:
        return f"sgar config set --client {self.client_name}"


def list_client_metadata() -> list[ClientMetadata]:
    metadata: dict[str, ClientMetadata] = {}
    for file_path in sorted(CORE_LLM_DIR.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        for item in _parse_client_metadata(file_path):
            metadata[item.client_name] = item
    return sorted(metadata.values(), key=lambda item: item.client_name.lower())


def get_client_metadata_map() -> dict[str, ClientMetadata]:
    return {item.client_name: item for item in list_client_metadata()}


def _parse_client_metadata(file_path: Path) -> list[ClientMetadata]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    items: list[ClientMetadata] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_llm_client_class(node):
            continue
        credential_keys = sorted(_extract_credential_keys(node))
        model_keys = sorted(_extract_model_keys(node))
        items.append(
            ClientMetadata(
                client_name=node.name,
                source_file=file_path.name,
                credential_keys=tuple(credential_keys),
                model_keys=tuple(model_keys),
            )
        )
    return items


def _is_llm_client_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = _dotted_name(base)
        if name and name.split(".")[-1] in KNOWN_LLM_BASES:
            return True
    return False


def _extract_credential_keys(node: ast.ClassDef) -> set[str]:
    keys = set()
    for value in _iter_string_literals_from_config_calls(node):
        if _looks_like_credential_key(value):
            keys.add(value)
    return keys


def _extract_model_keys(node: ast.ClassDef) -> set[str]:
    keys = set()
    for value in _iter_string_literals_from_config_calls(node):
        if _looks_like_model_key(value):
            keys.add(value)
    for stmt in node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        target_names = {
            target.id for target in stmt.targets if isinstance(target, ast.Name)
        }
        if not target_names.intersection(MODEL_ATTR_NAMES):
            continue
        keys.update(_extract_string_sequence(stmt.value))
    return keys


def _iter_string_literals_from_config_calls(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        owner_name, method_name = _call_parts(child.func)
        if owner_name != "config":
            continue
        if method_name == "get":
            for arg in child.args[:1]:
                values.extend(_extract_string_sequence(arg))
        elif method_name == "resolve_value":
            for arg in child.args[1:2]:
                values.extend(_extract_string_sequence(arg))
    return values


def _extract_string_sequence(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[str] = []
        for elt in node.elts:
            values.extend(_extract_string_sequence(elt))
        return values
    return []


def _looks_like_model_key(value: str) -> bool:
    lowered = value.lower()
    return "model" in lowered or "deployment_name" in lowered


def _looks_like_credential_key(value: str) -> bool:
    lowered = value.lower()
    if _looks_like_model_key(value):
        return False
    markers = ("key", "token", "secret", "endpoint", "url", "host", "port")
    return any(marker in lowered for marker in markers)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted_name(node.value)
        if left is None:
            return None
        return f"{left}.{node.attr}"
    return None


def _call_parts(node: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Attribute):
        return _dotted_name(node.value), node.attr
    if isinstance(node, ast.Name):
        return None, node.id
    return None, None
