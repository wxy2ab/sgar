"""Single source of truth for parent → child metadata inheritance.

When a cc_query_loop agent's turn produces buffered spawn / research /
sgar requests, ``CcAgentRunner._run_async`` drains those buffers and
converts each into a ``SubagentInvocation`` whose metadata blends two
sources:

1. **Child-supplied keys** — anything the LLM passed via the
   ``ccx_spawn`` / ``ccx_research`` / ``ccx_sgar`` tool call. These
   always win.
2. **Inherited parent keys** — the small set of context keys below
   that should follow the workflow even if the LLM forgot to thread
   them through every tool call. Missing these silently breaks
   multi-stage workflows: e.g. an SGAR run where the LLM omits
   ``sgar_session`` on a child spawn would have the child do its work
   against a fresh, unrelated session, and the bug would only show
   up at validate-stage time when the artifact paths don't match.

Codex's ``codex_delegate.rs`` makes this list explicit by listing the
"inherited services" (mcp, skills, plugins, exec_policy, ...) at the
fork point. We do the equivalent here for ccx's flatter metadata
shape: one ``frozenset`` of keys, one docstring per key, one place
to review when adding a new one.

**Adding a new key**:

1. Add it to ``INHERITABLE_METADATA_KEYS`` below.
2. Add a one-line entry in the per-key table comment explaining what
   downstream consumer reads it and why every child needs to see it.
3. Confirm no existing tests rely on the key being *excluded* — the
   safer-by-default direction is "inherit" but adding a key that some
   sibling deliberately overrides per-child can be surprising.
4. ``test_cc_agent_metadata_propagation`` and
   ``test_module_boundaries`` both already cover this set; rerun them.

The set is intentionally a ``frozenset`` (not a tuple) so:

* membership tests in ``_inherit()`` are O(1).
* the order is irrelevant — there's no "first wins" semantics to
  preserve, and a frozenset makes that absence of ordering visible.
* the value is immutable, so a typo like
  ``INHERITABLE_METADATA_KEYS.add("foo")`` fails loudly at the call
  site rather than silently mutating shared state.
"""

from __future__ import annotations


#: Parent invocation metadata keys auto-propagated into spawned children.
#:
#: Per-key semantics:
#:
#: * ``sgar_session`` — the SGAR session id this workflow operates
#:   against. Every child that touches ``.sgar/`` artifacts needs to
#:   know which session they belong to or they'll write to the wrong
#:   directory tree. Required for the entire SGAR command surface
#:   (init, set-blueprint, start-stage, verify, close-stage, ...).
#: * ``session_id`` — the cc-side session id for the parent turn.
#:   Children inherit it so cross-child memory / mempalace lookups
#:   resolve to the same session as the parent.
#: * ``cwd`` — the working directory the parent run was started in.
#:   Children inherit so relative paths (``Read("src/foo.py")``) mean
#:   the same thing across the whole workflow. A child that ends up
#:   with a different ``cwd`` than its parent is almost always a bug.
#: * ``request_metadata`` — the original ``AgentRunRequest.metadata``
#:   passed by the caller. Some callers thread custom keys (e.g.
#:   ``docs_output_path``, R1's ``preferred_model``) here that should
#:   reach every child.
#: * ``stage_id`` — the active SGAR stage id. Used by validate /
#:   verify operations to know which stage's exit criteria are
#:   applicable. Without it a verify command can't locate its
#:   acceptance criteria.
#: * ``mission_id`` — the active SGAR mission id. Mirrors ``stage_id``
#:   for mission-level operations (mission manifests, mission-local
#:   result files).
INHERITABLE_METADATA_KEYS: frozenset[str] = frozenset({
    "sgar_session",
    "session_id",
    "cwd",
    "request_metadata",
    "stage_id",
    "mission_id",
})


#: Metadata key counting how many ccx_spawn generations sit above an
#: invocation. Deliberately NOT in ``INHERITABLE_METADATA_KEYS``: the
#: inherit rule there is "copy parent's value if the child has none",
#: but depth must *increment* at each spawn hop and must be stamped
#: authoritatively (an LLM-supplied value on a spawn request must not
#: be able to reset the counter). ``CcAgentRunner._run_async`` stamps
#: ``parent_depth + 1`` on every drained child; ``runtime._make_mode_tool``
#: propagates the parent's value unchanged through non-spawning mode
#: hops (plan → spec → agent) so the count can't be laundered away by
#: routing a spawn through an intermediate mode.
SPAWN_DEPTH_METADATA_KEY: str = "ccx_spawn_depth"

#: Default ceiling on ccx_spawn recursion. A cc-agent turn whose own
#: ``ccx_spawn_depth`` is >= this value has child-agent spawning
#: refused (the ccx_spawn tool returns a clear error and any buffered
#: spawn entries are dropped). 3 generations of recursive
#: decomposition is generous for real workloads while bounding the
#: worst case; override per-runner via ``CcAgentRunner.max_spawn_depth``
#: or per-build via ``build_runtime(cc_max_spawn_depth=...)``.
DEFAULT_MAX_SPAWN_DEPTH: int = 3

#: Default ceiling on ccx_spawn fan-out WIDTH per cc-agent turn — the total
#: number of ordinary (recursive) spawn-mode children one turn may enqueue
#: across all its ``ccx_spawn`` calls. Depth bounds how deep recursion goes;
#: this bounds how wide a single generation gets, so a single runaway turn
#: can't buffer thousands of siblings (the only prior backstop was v5's
#: ~10k-node ``max_loop_iterations`` catastrophe ceiling). 32 is generous for
#: real decomposition while staying finite; ``research`` / ``sgar`` entries
#: are terminal and NOT counted. Override per-runner via
#: ``CcAgentRunner.max_spawn_fanout`` or per-build via
#: ``build_runtime(cc_max_spawn_fanout=...)``; ``None`` disables the cap.
DEFAULT_MAX_SPAWN_FANOUT: int = 32


def coerce_spawn_depth(value: object) -> int:
    """Parse a metadata depth value defensively.

    Metadata travels through JSON round-trips and LLM-authored tool
    arguments, so the value may be an int, a numeric string, a float,
    or garbage. Unparseable or negative values collapse to 0 (treat as
    a root-level turn) rather than raising — a malformed counter must
    never crash a run, only make the guard conservative one way or the
    other.
    """
    try:
        depth = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, depth)


__all__ = [
    "DEFAULT_MAX_SPAWN_DEPTH",
    "DEFAULT_MAX_SPAWN_FANOUT",
    "INHERITABLE_METADATA_KEYS",
    "SPAWN_DEPTH_METADATA_KEY",
    "coerce_spawn_depth",
]
