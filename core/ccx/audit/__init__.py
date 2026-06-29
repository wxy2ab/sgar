"""Cross-mode, machine-verifiable CODE-task definition-of-done for ccx.

A *code task* (a node/run that edits production ``.py``) must not self-report
"done" before there is machine evidence of (A) wiring, (C) a green scoped test
suite, plus advisory signals (B test-accompaniment, D additive/gating, E an
honest ledger). The evidence is produced by a single deterministic command,
``python -m core.ccx.audit.code_task`` (see :mod:`core.ccx.audit.code_task`),
which every governed mode injects as one ``[check:]`` exit criterion via
:func:`build_code_task_contract`.

Everything here is **opt-in, default OFF**. With ``CCX_CODE_TASK_AUDIT`` unset,
:func:`code_task_audit_enabled` returns ``False`` and no mode injects anything,
so behaviour is byte-equivalent to before. This module deliberately imports
nothing heavy at load time (only ``os``); the contract builder and CLI are
imported lazily so a flag-off cold path pays no import cost.
"""

from __future__ import annotations

import os
from typing import Any

#: Master switch. Unset (or a non-truthy value) ⇒ the whole feature is inert.
CODE_TASK_AUDIT_ENV = "CCX_CODE_TASK_AUDIT"

#: Optional explicit test command for criterion C. When unset, the audit
#: derives a scoped pytest command from the changed files.
CODE_TASK_TEST_CMD_ENV = "CCX_CODE_TASK_TEST_CMD"

#: Master switch for the regression-capture ratchet auto-injection (default
#: OFF). The promotion library itself is inert unless called; this gate only
#: governs any auto-wiring of it into a run, mirroring the code-task switch.
REGRESSION_CAPTURE_ENV = "CCX_REGRESSION_CAPTURE"

#: Master switch for the debug binding's behavioural additions (cheap/heavy
#: stimulus split + anti-theater advisory consumption). Default OFF ⇒
#: ``agent_mode="debug"`` degrades to a plain goal loop, byte-equivalent.
DEBUG_MODE_ENV = "CCX_DEBUG_MODE"

#: Master switch for the hard-feedback audit agent (design
#: ``docs/audit_agent_design_2026-06-28.md``). Default OFF ⇒ ``agent_mode="audit"``
#: degrades to a plain goal loop, byte-equivalent, and the post-gate
#: claim↔evidence fidelity enrichment never fires. INFORM-only when on: audit
#: output is a snapshot side-channel, never a term in the goal ``met`` gate.
AUDIT_MODE_ENV = "CCX_AUDIT_MODE"

_TRUTHY = frozenset({"1", "true", "on", "yes", "all"})


def _flag_enabled(env_name: str) -> bool:
    """Shared default-OFF gate: unset / non-truthy ⇒ ``False`` (inert)."""
    raw = os.environ.get(env_name)
    return raw is not None and raw.strip().lower() in _TRUTHY


def code_task_audit_enabled() -> bool:
    """Whether cross-mode code-task auditing is enabled (default OFF).

    ``CCX_CODE_TASK_AUDIT`` unset (or set to anything other than
    ``1``/``true``/``on``/``yes``/``all``) ⇒ ``False`` ⇒ every injection site
    is byte-equivalent to the pre-feature behaviour.
    """
    return _flag_enabled(CODE_TASK_AUDIT_ENV)


def regression_capture_enabled() -> bool:
    """Whether regression-capture auto-injection is enabled (default OFF)."""
    return _flag_enabled(REGRESSION_CAPTURE_ENV)


def debug_mode_enabled() -> bool:
    """Whether the debug binding's behavioural additions are enabled (default OFF).

    Unset ⇒ ``agent_mode="debug"`` behaves byte-equivalently to a plain goal
    loop (no cheap/heavy stimulus split, no advisory consumption); the mode is
    still accepted so a flag-off debug request never silently falls through to
    the cc fallback.
    """
    return _flag_enabled(DEBUG_MODE_ENV)


def audit_mode_enabled() -> bool:
    """Whether the hard-feedback audit agent's enrichment is enabled (default OFF).

    Unset ⇒ ``agent_mode="audit"`` behaves byte-equivalently to a plain goal loop
    (no post-gate claim↔evidence fidelity audit, no advisory channel); the mode is
    still accepted so a flag-off audit request never silently falls through to the
    cc fallback. When on, the audit is INFORM-only — it writes
    ``snapshot['audit_advisories']`` and never enters the ``met`` gate (design §4).
    """
    return _flag_enabled(AUDIT_MODE_ENV)


def build_code_task_contract(shape: str = "contract", **kwargs: Any) -> Any:
    """Lazy proxy to :func:`core.ccx.audit.contract.build_code_task_contract`.

    Imported here so a flag-off path never imports the contract module / sgar
    models. See that function for the ``shape`` values and return shapes.
    """
    from .contract import build_code_task_contract as _impl

    return _impl(shape, **kwargs)


__all__ = [
    "CODE_TASK_AUDIT_ENV",
    "CODE_TASK_TEST_CMD_ENV",
    "REGRESSION_CAPTURE_ENV",
    "DEBUG_MODE_ENV",
    "AUDIT_MODE_ENV",
    "build_code_task_contract",
    "code_task_audit_enabled",
    "regression_capture_enabled",
    "debug_mode_enabled",
    "audit_mode_enabled",
]
