"""P3 — guardrails for fixed declarative DAGs (the actual product).

A draft exported by ``fixed_dag_export`` is a topology sketch; what makes it
safe to *run* on a schedule is the structured discipline here. Four guards, each
addressing a way a "just re-run the DAG every night" pipeline silently goes
wrong:

(a) ``mark_requires_approval`` — a human reviews the draft and marks every
    side-effect node ``requires_approval``. v5 then halts the run at that node
    (``WAITING_APPROVAL``) instead of firing the side effect unattended.

(b) ``schema_preflight_capability`` — a fail-loud data-contract node (column set
    / value enum / row bounds) wired ahead of the work. Data drift raises
    (→ ``TOOL_ERROR``) and stops the DAG, instead of the downstream nodes
    silently producing a wrong or empty report.

(c) ``OnceGuard`` / ``once_per_period`` — a CROSS-RUN once-guard backed by an
    external store. A scheduled re-run of a non-idempotent side effect (send the
    report, post the numbers) is skipped unless it is a new period. This does
    NOT use ``ToolSpec.idempotent`` — that flag only governs in-run crash
    recovery and is false safety across runs.

(d) ``NamedSpecRegistry`` — dispatch by EXACT name only. No fuzzy / substring /
    similarity lookup (a substring dispatch once fired a destructive ``init``).
    A caller — or the scheduler's ``request_template`` — must name the DAG
    exactly; an unknown name fails loud.

All four are additive helpers a caller opts into; none changes any existing run.
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.deepstack_v5 import NodeSpec, ToolSpec


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# (a) require approval on side-effect nodes
# --------------------------------------------------------------------------- #

def mark_requires_approval(
    specs: list[NodeSpec],
    *,
    side_effect_tools: Iterable[str] = (),
    predicate: Callable[[NodeSpec], bool] | None = None,
) -> list[NodeSpec]:
    """Return copies of ``specs`` with ``requires_approval=True`` on side-effect
    nodes — a node whose ``tool`` is in ``side_effect_tools`` or that ``predicate``
    selects. v5 halts the run at such a node pending an explicit
    ``engine.approve`` (see WAITING_APPROVAL), so the side effect never fires
    unattended. Pure: non-matching specs are returned unchanged.
    """
    tools = set(side_effect_tools)
    out: list[NodeSpec] = []
    for spec in specs:
        is_side_effect = spec.tool in tools or bool(predicate and predicate(spec))
        if is_side_effect and not spec.requires_approval:
            out.append(dataclasses.replace(spec, requires_approval=True))
        else:
            out.append(spec)
    return out


# --------------------------------------------------------------------------- #
# (b) fail-loud schema contract preflight
# --------------------------------------------------------------------------- #

class SchemaContractError(RuntimeError):
    """Raised when observed data violates a `SchemaContract`. In a tool node the
    dispatcher turns this into a ``TOOL_ERROR`` — loud, not a silent bad result."""


@dataclass(slots=True)
class SchemaContract:
    """A data-shape contract checked before the real work runs.

    ``required_columns`` — every row must contain (at least) these keys.
    ``allowed_values`` — column → the closed set of values it may take (e.g. the
    funnel stages ``{"visit","signup","purchase"}``); a value outside the set is
    drift. ``min_rows`` / ``max_rows`` — inclusive row-count bounds.
    """

    required_columns: frozenset[str] | None = None
    allowed_values: Mapping[str, frozenset[str]] | None = None
    min_rows: int | None = None
    max_rows: int | None = None


def check_schema(rows: list[Mapping[str, Any]], contract: SchemaContract) -> None:
    """Raise `SchemaContractError` on the first contract violation; else return."""
    n = len(rows)
    if contract.min_rows is not None and n < contract.min_rows:
        raise SchemaContractError(
            f"CCX_SCHEMA_ROW_UNDERFLOW: {n} rows < min {contract.min_rows}"
        )
    if contract.max_rows is not None and n > contract.max_rows:
        raise SchemaContractError(
            f"CCX_SCHEMA_ROW_OVERFLOW: {n} rows > max {contract.max_rows}"
        )
    if contract.required_columns is not None:
        for i, row in enumerate(rows):
            missing = contract.required_columns - set(row.keys())
            if missing:
                raise SchemaContractError(
                    f"CCX_SCHEMA_MISSING_COLUMNS: row {i} missing "
                    f"{sorted(missing)} (has {sorted(row.keys())})"
                )
    if contract.allowed_values:
        for col, allowed in contract.allowed_values.items():
            for i, row in enumerate(rows):
                if col in row and row[col] not in allowed:
                    raise SchemaContractError(
                        f"CCX_SCHEMA_BAD_VALUE: row {i} {col}={row[col]!r} "
                        f"not in {sorted(allowed)}"
                    )


def schema_preflight_capability(
    name: str,
    contract: SchemaContract,
    load_rows: Callable[..., Iterable[Mapping[str, Any]]],
) -> ToolSpec:
    """Build a preflight `ToolSpec` that loads rows via ``load_rows(**params)``
    and enforces ``contract``, raising (→ TOOL_ERROR) on drift.

    Wire the resulting node as a ``depends_on`` of the real work so a schema
    drift halts the DAG loudly before any downstream node runs on bad data.
    """
    def fn(**params: Any) -> dict[str, Any]:
        rows = list(load_rows(**params))
        check_schema(rows, contract)
        return {"final_text": f"schema contract {name!r} OK", "row_count": len(rows)}

    return ToolSpec(name=name, fn=fn)


# --------------------------------------------------------------------------- #
# (c) cross-run once-guard / idempotent write
# --------------------------------------------------------------------------- #

class OnceGuard:
    """External, cross-run at-most-once guard keyed on an arbitrary string.

    Backed by its own SQLite file so it survives process restarts and is shared
    across scheduled runs — unlike ``ToolSpec.idempotent``, which only affects
    in-run crash-recovery and provides no cross-run protection at all.

    ``claim`` is atomic (``INSERT OR IGNORE`` + rowcount): exactly one caller
    across any number of concurrent/sequential runs gets ``True`` for a given
    key. ``once_per_period`` claims BEFORE the effect and ``release``s the key
    if the effect raises, so a transient failure (which v5 retries) is re-run
    rather than silently skipped. The one residual window is a process CRASH
    after claim but before completion: the key stays claimed and the effect is
    skipped next run — the safe default for "don't double-send". A node that
    would rather be idempotent should use an upsert-key in its own write.
    """

    #: Generous busy timeout so concurrent claims across processes serialize
    #: (retry on the write lock) instead of raising OperationalError.
    _CONNECT_TIMEOUT_S = 30.0

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        con = self._connect()
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS once_guard "
                "(key TEXT PRIMARY KEY, claimed_at_ms INTEGER)"
            )
            con.commit()
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        # timeout sets sqlite's busy handler: concurrent writers wait on the
        # lock and retry rather than immediately raising OperationalError.
        return sqlite3.connect(self._path, timeout=self._CONNECT_TIMEOUT_S)

    def claim(self, key: str) -> bool:
        """Atomically claim ``key``. Returns True if newly claimed (the caller
        may run the side effect), False if a prior run already claimed it."""
        con = self._connect()
        try:
            cur = con.execute(
                "INSERT OR IGNORE INTO once_guard(key, claimed_at_ms) VALUES (?, ?)",
                (key, int(time.time() * 1000)),
            )
            con.commit()
            return cur.rowcount == 1
        finally:
            con.close()

    def release(self, key: str) -> None:
        """Drop ``key`` so a subsequent claim can succeed again. Used to undo a
        claim whose effect failed, so a retry re-runs instead of being skipped."""
        con = self._connect()
        try:
            con.execute("DELETE FROM once_guard WHERE key = ?", (key,))
            con.commit()
        finally:
            con.close()

    def seen(self, key: str) -> bool:
        con = self._connect()
        try:
            row = con.execute(
                "SELECT 1 FROM once_guard WHERE key = ?", (key,)
            ).fetchone()
            return row is not None
        finally:
            con.close()


def once_per_period(
    fn: Callable[..., Any],
    *,
    key: str,
    guard: OnceGuard,
    on_skip: Any | None = None,
) -> Callable[..., Any]:
    """Wrap a side-effect ``fn`` so it runs at most once per ``key`` across runs.

    On a repeat (the key is already claimed) the wrapper returns ``on_skip``
    (default a skip marker) WITHOUT calling ``fn`` — so a scheduler re-firing the
    same period is a no-op, not a double side effect. Use a period-stamped key,
    e.g. ``"nightly_sales_report:2026-07-02"``.

    If ``fn`` RAISES, the claim is released and the error re-raised, so a
    transient failure that v5 retries (or a later scheduled run) re-runs the
    effect rather than being silently skipped with a green verdict. The residual
    at-most-once window is a process CRASH between claim and completion (the key
    stays claimed); a node needing stronger guarantees should use an
    idempotent upsert-key in its own write.
    """
    def wrapped(**params: Any) -> Any:
        if not guard.claim(key):
            logger.info("ccx once-guard: key %r already applied; skipping", key)
            return on_skip if on_skip is not None else {
                "final_text": f"skipped: {key!r} already applied this period",
                "ccx_once_skipped": True,
            }
        try:
            return fn(**params)
        except Exception:
            # A (retryable) error — the effect did not complete, so undo the
            # claim and let a retry re-run it instead of skipping a never-applied
            # side effect. A KeyboardInterrupt / SystemExit is deliberately NOT
            # caught: on a crash-like interrupt the claim stays (at-most-once,
            # "don't double-send"), matching the guard's documented contract.
            guard.release(key)
            raise

    return wrapped


# --------------------------------------------------------------------------- #
# (d) explicit, exact-match named dispatch (no fuzzy / substring)
# --------------------------------------------------------------------------- #

class NamedSpecRegistry:
    """Explicit registry of named fixed DAGs, resolved by EXACT name only.

    There is deliberately no fuzzy / substring / similarity lookup: a substring
    dispatch has historically matched the wrong DAG and fired a destructive
    ``init``. A caller — or the scheduler's ``request_template`` — must name the
    DAG exactly; an unknown name raises. This is the whole point of the class:
    it is the safe dispatch surface, not a convenience matcher.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, list[NodeSpec]] = {}

    def register(self, name: str, specs: list[NodeSpec], *, replace: bool = False) -> None:
        if not name:
            raise ValueError("CCX_NAMED_DAG_EMPTY_NAME: a fixed DAG needs a name")
        if name in self._by_name and not replace:
            raise ValueError(
                f"CCX_NAMED_DAG_DUPLICATE: {name!r} already registered "
                "(pass replace=True to overwrite)"
            )
        self._by_name[name] = list(specs)

    def get(self, name: str) -> list[NodeSpec]:
        """Return a copy of the specs registered under EXACTLY ``name``.

        Raises ``KeyError`` on any miss — including a name that is a substring of
        a registered one. No fuzzy matching, ever.
        """
        try:
            specs = self._by_name[name]
        except KeyError:
            raise KeyError(
                f"CCX_NAMED_DAG_UNKNOWN: no fixed DAG named {name!r}; dispatch is "
                f"exact-match only (no fuzzy/substring). Known: {self.names()}"
            ) from None
        return list(specs)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


__all__ = [
    "NamedSpecRegistry",
    "OnceGuard",
    "SchemaContract",
    "SchemaContractError",
    "check_schema",
    "mark_requires_approval",
    "once_per_period",
    "schema_preflight_capability",
]
