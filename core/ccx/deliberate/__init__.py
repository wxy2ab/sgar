"""Adversarial proposal-deliberation mode for ccx (``agent_mode="deliberate"``).

A *deliberation* takes a **proposal** (a free-text proposition, NOT a landed
task) and runs a bounded, read-only adversarial study: decompose it into
sub-claims, gather *supporting* and *refuting* evidence per sub-claim with two
stance-committed read-only research agents, then synthesize an **evidence-
weighted, INFORM-only for/against report**. Design:
``docs/deliberate_mode_design_2026-06-30.md``.

This is **decision-support, not an audit**: the output is INFORM-only by
construction (it never gates a ``met`` / ``goal_verdict`` / contract). The
consumer is a human or a planner *deciding whether to pursue a proposal*, so the
consumer is the fidelity source — "no deterministic gate" is the correct design,
not a defect. The deliberation's own "teeth" are **cite-or-discount**: the
verdict is computed deterministically from each side's *grounded* evidence
anchors (file:line), never from rhetorical strength (see
:mod:`core.ccx.agents.deliberate_runner`).

Everything here is **opt-in, default OFF**. With ``CCX_DELIBERATE_MODE`` unset,
:func:`deliberate_mode_enabled` returns ``False`` and the mode is still
*accepted* (never silently falls through to the cc fallback) but runs inert —
``api._run_deliberate_loop`` returns a cheap "disabled" result without spinning
up any LLM/research. Adding the mode is otherwise additive: no other agent_mode
changes behaviour.
"""

from __future__ import annotations

import os

#: Master switch. Unset (or a non-truthy value) ⇒ the whole feature is inert.
DELIBERATE_MODE_ENV = "CCX_DELIBERATE_MODE"

#: Operator params for ``agent_mode="deliberate"`` (max_subclaims / scope), read
#: from ``request.metadata``. Mirrors the audit/debug request keys; the
#: producer/DAG never writes it.
CCX_DELIBERATE_REQUEST_METADATA_KEY = "ccx_deliberate"

_TRUTHY = frozenset({"1", "true", "on", "yes", "all"})


def deliberate_mode_enabled() -> bool:
    """Whether adversarial proposal-deliberation is enabled (default OFF).

    ``CCX_DELIBERATE_MODE`` unset (or set to anything other than
    ``1``/``true``/``on``/``yes``/``all``) ⇒ ``False`` ⇒ ``agent_mode="deliberate"``
    runs inert (a cheap disabled result), never spinning up an LLM or research
    worker.
    """
    raw = os.environ.get(DELIBERATE_MODE_ENV)
    return raw is not None and raw.strip().lower() in _TRUTHY


__all__ = [
    "DELIBERATE_MODE_ENV",
    "CCX_DELIBERATE_REQUEST_METADATA_KEY",
    "deliberate_mode_enabled",
]
